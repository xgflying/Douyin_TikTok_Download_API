import os
import shutil
import subprocess
import hashlib
from typing import Optional, Tuple

import aiofiles
import httpx
import numpy as np
from fastapi import APIRouter, Body, Query, Request, HTTPException  # 导入FastAPI组件
from app.api.models.APIResponseModel import ResponseModel, ErrorResponseModel  # 导入响应模型

from crawlers.bilibili.web.web_crawler import BilibiliWebCrawler  # 导入哔哩哔哩web爬虫
import tempfile


router = APIRouter()
BilibiliWebCrawler = BilibiliWebCrawler()


# 在部分运行方式下（IDE/服务进程）可能拿不到交互式 shell 的 PATH，这里做兜底探测
def _find_bin(name: str) -> Optional[str]:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidates = [
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
        f"/bin/{name}",
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# 获取 ffmpeg 的可执行文件路径（优先 PATH，兜底常见安装路径）
def _ffmpeg_path() -> str:
    p = _find_bin("ffmpeg")
    if not p:
        raise RuntimeError(
            "ffmpeg not found: watermark removal requires ffmpeg. "
            "Please ensure ffmpeg is in PATH or installed to /opt/homebrew/bin or /usr/local/bin."
        )
    return p


# 获取 ffprobe 的可执行文件路径（优先 PATH，兜底常见安装路径）
def _ffprobe_path() -> str:
    p = _find_bin("ffprobe")
    if not p:
        raise RuntimeError(
            "ffprobe not found: watermark removal requires ffprobe. "
            "Please ensure ffprobe is in PATH or installed to /opt/homebrew/bin or /usr/local/bin."
        )
    return p


async def _fetch_data_stream(url: str, request: Request, headers: dict, file_path: str) -> bool:
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            async with aiofiles.open(file_path, "wb") as out_file:
                async for chunk in response.aiter_bytes():
                    if await request.is_disconnected():
                        await out_file.close()
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                        return False
                    await out_file.write(chunk)
    return True


async def _merge_bilibili_video_audio(
    video_url: str,
    audio_url: str,
    request: Request,
    output_path: str,
    headers: dict,
) -> bool:
    # Bilibili DASH 会分离视频/音频流：先分别下载到临时文件，再用 ffmpeg 无损合并
    video_temp_path = None
    audio_temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".m4v", delete=False) as video_temp:
            video_temp_path = video_temp.name
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as audio_temp:
            audio_temp_path = audio_temp.name

        video_success = await _fetch_data_stream(video_url, request, headers=headers, file_path=video_temp_path)
        audio_success = await _fetch_data_stream(audio_url, request, headers=headers, file_path=audio_temp_path)
        if not video_success or not audio_success:
            return False

        ffmpeg_cmd = [
            _ffmpeg_path(),
            "-y",
            "-i",
            video_temp_path,
            "-i",
            audio_temp_path,
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-f",
            "mp4",
            output_path,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        return result.returncode == 0
    finally:
        for tmp_path in (video_temp_path, audio_temp_path):
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _pick_dash_urls(dash: dict, resolution: int) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    video_list = dash.get("video") or []
    audio_list = dash.get("audio") or []

    def _video_url(v: dict) -> Optional[str]:
        return v.get("baseUrl") or v.get("base_url") or v.get("url")

    def _audio_url(a: dict) -> Optional[str]:
        return a.get("baseUrl") or a.get("base_url") or a.get("url")

    target_height = {0: None, 1: 720, 2: 1080}.get(resolution, None)
    candidates = []
    for v in video_list:
        url = _video_url(v)
        if not url:
            continue
        height = v.get("height")
        candidates.append((height, url))

    picked_video_url = None
    picked_height = None
    if candidates:
        heighted = [(h, u) for h, u in candidates if isinstance(h, int)]
        if target_height is None:
            picked_height, picked_video_url = max(heighted, key=lambda x: x[0]) if heighted else candidates[0]
        else:
            leq = [(h, u) for h, u in heighted if h <= target_height]
            if leq:
                picked_height, picked_video_url = max(leq, key=lambda x: x[0])
            else:
                picked_height, picked_video_url = min(heighted, key=lambda x: x[0]) if heighted else candidates[0]

    picked_audio_url = None
    if audio_list:
        audio_candidates = []
        for a in audio_list:
            url = _audio_url(a)
            if not url:
                continue
            bandwidth = a.get("bandwidth") if isinstance(a.get("bandwidth"), int) else -1
            audio_candidates.append((bandwidth, url))
        if audio_candidates:
            _, picked_audio_url = max(audio_candidates, key=lambda x: x[0])

    return picked_video_url, picked_audio_url, picked_height


def _ffprobe_video_size(input_path: str) -> Tuple[int, int]:
    # 用 ffprobe 获取原视频宽高，用于把采样坐标映射回原图坐标
    ffprobe = _ffprobe_path()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip() or "ffprobe failed")
    text = (result.stdout or "").strip()
    if "x" not in text:
        raise RuntimeError("ffprobe output invalid")
    w_str, h_str = text.split("x", 1)
    return int(w_str), int(h_str)


def _sha256_file(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    # 计算文件 SHA256，用于确认“去水印”确实对输出产生了变化
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extract_gray_frames(
    input_path: str,
    frame_count: int = 8,
    sample_w: int = 320,
    sample_h: int = 180,
    fps: int = 1,
    start_seconds: int = 2,
) -> np.ndarray:
    # 抽帧并缩放到较小尺寸，后续基于灰度帧做水印区域检测（降低 CPU 开销）
    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        input_path,
        "-vf",
        f"fps={fps},scale={sample_w}:{sample_h}",
        "-vframes",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or "ffmpeg frame extraction failed")

    raw = result.stdout or b""
    frame_size = sample_w * sample_h
    available = len(raw) // frame_size
    if available <= 0:
        raise RuntimeError("no frames extracted")
    used = min(available, frame_count)
    arr = np.frombuffer(raw[: used * frame_size], dtype=np.uint8)
    return arr.reshape((used, sample_h, sample_w))


def _detect_watermark_box(
    input_path: str,
    sample_w: int = 320,
    sample_h: int = 180,
) -> Tuple[int, int, int, int]:
    # 基于“跨帧稳定性 + 边缘强度”粗定位文字水印区域，并偏向右上角（bilibili 常见位置）
    frames = _extract_gray_frames(input_path, frame_count=12, sample_w=sample_w, sample_h=sample_h, fps=1, start_seconds=1)
    orig_w, orig_h = _ffprobe_video_size(input_path)

    f = frames.astype(np.float32)
    std = f.std(axis=0)
    median = np.median(f, axis=0)
    dx = np.abs(median[:, 1:] - median[:, :-1])
    dy = np.abs(median[1:, :] - median[:-1, :])
    grad = np.zeros_like(median, dtype=np.float32)
    grad[:, 1:] += dx
    grad[1:, :] += dy

    cw = int(sample_w * 0.40)
    ch = int(sample_h * 0.40)
    cw2 = int(sample_w * 0.52)
    ch2 = int(sample_h * 0.46)
    bw = int(sample_w * 0.55)
    bh = int(sample_h * 0.28)
    regions = {
        "tl": (0, 0, cw, ch),
        "tr": (sample_w - cw, 0, cw, ch),
        "tr_wide": (sample_w - cw2, 0, cw2, ch2),
        "bl": (0, sample_h - ch, cw, ch),
        "br": (sample_w - cw, sample_h - ch, cw, ch),
        "bc": ((sample_w - bw) // 2, sample_h - bh, bw, bh),
    }

    best = None
    best_score = -1.0
    best_mask = None
    best_region = None
    for name, (rx, ry, rw, rh) in regions.items():
        reg_std = std[ry : ry + rh, rx : rx + rw]
        reg_grad = grad[ry : ry + rh, rx : rx + rw]
        thr_std = float(np.percentile(reg_std, 25.0))
        thr_grad = float(np.percentile(reg_grad, 90.0))
        mask = (reg_std <= thr_std) & (reg_grad >= thr_grad)
        score = int(mask.sum())
        if score < 20:
            thr_std2 = float(np.percentile(reg_std, 12.0))
            reg_med = median[ry : ry + rh, rx : rx + rw]
            thr_med = float(np.percentile(reg_med, 70.0))
            mask2 = (reg_std <= thr_std2) & (reg_med >= thr_med)
            score2 = int(mask2.sum())
            if score2 > score:
                mask = mask2
                score = score2
        bias = 1.8 if name in ("tr", "tr_wide") else 1.0
        weighted_score = float(score) * bias
        if weighted_score > best_score:
            best_score = weighted_score
            best = name
            best_mask = mask
            best_region = (rx, ry, rw, rh)

    rx, ry, rw, rh = best_region
    ys, xs = np.where(best_mask)
    if ys.size == 0 or xs.size == 0:
        pad_w = int(sample_w * 0.14)
        pad_h = int(sample_h * 0.10)
        margin_x = int(sample_w * 0.02)
        margin_y = int(sample_h * 0.02)
        if best == "tl":
            sx, sy = margin_x, margin_y
        elif best in ("tr", "tr_wide"):
            sx, sy = sample_w - pad_w - margin_x, margin_y
        elif best == "bl":
            sx, sy = margin_x, sample_h - pad_h - margin_y
        elif best == "br":
            sx, sy = sample_w - pad_w - margin_x, sample_h - pad_h - margin_y
        else:
            sx, sy = (sample_w - pad_w) // 2, sample_h - pad_h - margin_y
        ex, ey = sx + pad_w, sy + pad_h
    else:
        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        pad = 6
        sx = max(rx + x1 - pad, 0)
        sy = max(ry + y1 - pad, 0)
        ex = min(rx + x2 + pad, sample_w - 1)
        ey = min(ry + y2 + pad, sample_h - 1)

    sx_o = int(round(sx * orig_w / sample_w))
    sy_o = int(round(sy * orig_h / sample_h))
    ex_o = int(round(ex * orig_w / sample_w))
    ey_o = int(round(ey * orig_h / sample_h))
    x = max(0, min(sx_o, orig_w - 2))
    y = max(0, min(sy_o, orig_h - 2))
    w = max(2, min(ex_o - x, orig_w - x))
    h = max(2, min(ey_o - y, orig_h - y))
    pad_x = max(2, int(w * 0.15))
    pad_y = max(2, int(h * 0.15))
    x2 = max(0, x - pad_x)
    y2 = max(0, y - pad_y)
    w2 = min(orig_w - x2, w + 2 * pad_x)
    h2 = min(orig_h - y2, h + 2 * pad_y)
    x, y, w, h = x2, y2, max(2, w2), max(2, h2)
    if best in ("tr", "tr_wide"):
        # 右上角文字水印通常较小：设定最小框，避免识别框过大造成明显遮挡
        min_w = int(orig_w * 0.16)
        min_h = int(orig_h * 0.11)
        if w < min_w or h < min_h:
            margin_x = max(4, int(orig_w * 0.01))
            margin_y = max(4, int(orig_h * 0.01))
            w = max(w, min_w)
            h = max(h, min_h)
            x = max(0, orig_w - w - margin_x)
            y = margin_y
        if w < int(orig_w * 0.05) or h < int(orig_h * 0.03):
            margin_x = max(4, int(orig_w * 0.01))
            margin_y = max(4, int(orig_h * 0.01))
            w = int(orig_w * 0.16)
            h = int(orig_h * 0.11)
            x = max(0, orig_w - w - margin_x)
            y = margin_y
    return x, y, w, h


def _remove_watermark_ffmpeg(input_path: str, output_path: str, box: Tuple[int, int, int, int]) -> None:
    # 先尝试 delogo 去水印 + 模糊覆盖增强；失败再用相邻区域贴片兜底（保证水印消失）
    ffmpeg = _ffmpeg_path()
    x, y, w, h = box
    video_encoders = [
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"],
        ["-c:v", "h264_videotoolbox", "-b:v", "4000k"],
        ["-c:v", "mpeg4", "-q:v", "3"],
    ]
    last_err = None
    for enc in video_encoders:
        filter_complex = (
            f"[0:v]delogo=x={x}:y={y}:w={w}:h={h}[clean];"
            f"[clean]split=2[base][wm];"
            f"[wm]crop=w={w}:h={h}:x={x}:y={y},boxblur=luma_radius=22:luma_power=2[blur];"
            f"[base][blur]overlay=x={x}:y={y}[v]"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:a",
            "copy",
            *enc,
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        last_err = (result.stderr or result.stdout or "").strip()

    patch_x = max(0, x - w)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-filter_complex",
        # 兜底：把水印左侧同尺寸区域裁出并覆盖到水印位置（可能有贴片感，但能保证文字消失）
        f"[0:v]split=2[base][src];[src]crop=w={w}:h={h}:x={patch_x}:y={y}[patch];[base][patch]overlay=x={x}:y={y}[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    last_err2 = (result.stderr or result.stdout or "").strip()
    raise RuntimeError(last_err2 or last_err or "ffmpeg watermark removal failed")


# 获取单个视频详情信息
@router.get("/fetch_one_video", response_model=ResponseModel, summary="获取单个视频详情信息/Get single video data")
async def fetch_one_video(request: Request,
                          bv_id: str = Query(example="BV1M1421t7hT", description="作品id/Video id")):
    """
    # [中文]
    ### 用途:
    - 获取单个视频详情信息
    ### 参数:
    - bv_id: 作品id
    ### 返回:
    - 视频详情信息

    # [English]
    ### Purpose:
    - Get single video data
    ### Parameters:
    - bv_id: Video id
    ### Return:
    - Video data

    # [示例/Example]
    bv_id = "BV1M1421t7hT"
    """
    try:
        data = await BilibiliWebCrawler.fetch_one_video(bv_id)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取视频流地址
@router.get("/fetch_video_playurl", response_model=ResponseModel, summary="获取视频流地址/Get video playurl")
async def fetch_one_video(request: Request,
                          bv_id: str = Query(example="BV1y7411Q7Eq", description="作品id/Video id"),
                          cid:str = Query(example="171776208", description="作品cid/Video cid")):
    """
    # [中文]
    ### 用途:
    - 获取视频流地址
    ### 参数:
    - bv_id: 作品id
    - cid: 作品cid
    ### 返回:
    - 视频流地址

    # [English]
    ### Purpose:
    - Get video playurl
    ### Parameters:
    - bv_id: Video id
    - cid: Video cid
    ### Return:
    - Video playurl

    # [示例/Example]
    bv_id = "BV1y7411Q7Eq"
    cid = "171776208"
    """
    try:
        data = await BilibiliWebCrawler.fetch_video_playurl(bv_id, cid)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取用户发布视频作品数据
@router.get("/fetch_user_post_videos", response_model=ResponseModel,
            summary="获取用户主页作品数据/Get user homepage video data")
async def fetch_user_post_videos(request: Request,
                                 uid: str = Query(example="178360345", description="用户UID"),
                                 pn: int = Query(default=1, description="页码/Page number"),):
    """
    # [中文]
    ### 用途:
    - 获取用户发布的视频数据
    ### 参数:
    - uid: 用户UID
    - pn: 页码
    ### 返回:
    - 用户发布的视频数据

    # [English]
    ### Purpose:
    - Get user post video data
    ### Parameters:
    - uid: User UID
    - pn: Page number
    ### Return:
    - User posted video data

    # [示例/Example]
    uid = "178360345"
    pn = 1
    """
    try:
        data = await BilibiliWebCrawler.fetch_user_post_videos(uid, pn)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取用户所有收藏夹信息
@router.get("/fetch_collect_folders", response_model=ResponseModel,
            summary="获取用户所有收藏夹信息/Get user collection folders")
async def fetch_collect_folders(request: Request,
                                uid: str = Query(example="178360345", description="用户UID")):
    """
    # [中文]
    ### 用途:
    - 获取用户收藏作品数据
    ### 参数:
    - uid: 用户UID
    ### 返回:
    - 用户收藏夹信息

    # [English]
    ### Purpose:
    - Get user collection folders
    ### Parameters:
    - uid: User UID
    ### Return:
    - user collection folders

    # [示例/Example]
    uid = "178360345"
    """
    try:
        data = await BilibiliWebCrawler.fetch_collect_folders(uid)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定收藏夹内视频数据
@router.get("/fetch_user_collection_videos", response_model=ResponseModel,
            summary="获取指定收藏夹内视频数据/Gets video data from a collection folder")
async def fetch_user_collection_videos(request: Request,
                                       folder_id: str = Query(example="1756059545",
                                                              description="收藏夹id/collection folder id"),
                                       pn: int = Query(default=1, description="页码/Page number")
                                       ):
    """
    # [中文]
    ### 用途:
    - 获取指定收藏夹内视频数据
    ### 参数:
    - folder_id: 用户UID
    - pn: 页码
    ### 返回:
    - 指定收藏夹内视频数据

    # [English]
    ### Purpose:
    - Gets video data from a collection folder
    ### Parameters:
    - folder_id: collection folder id
    - pn: Page number
    ### Return:
    - video data from collection folder

    # [示例/Example]
    folder_id = "1756059545"
    pn = 1
    """
    try:
        data = await BilibiliWebCrawler.fetch_folder_videos(folder_id, pn)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定用户的信息
@router.get("/fetch_user_profile", response_model=ResponseModel,
            summary="获取指定用户的信息/Get information of specified user")
async def fetch_collect_folders(request: Request,
                                uid: str = Query(example="178360345", description="用户UID")):
    """
    # [中文]
    ### 用途:
    - 获取指定用户的信息
    ### 参数:
    - uid: 用户UID
    ### 返回:
    - 指定用户的个人信息

    # [English]
    ### Purpose:
    - Get information of specified user
    ### Parameters:
    - uid: User UID
    ### Return:
    - information of specified user

    # [示例/Example]
    uid = "178360345"
    """
    try:
        data = await BilibiliWebCrawler.fetch_user_profile(uid)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取综合热门视频信息
@router.get("/fetch_com_popular", response_model=ResponseModel,
            summary="获取综合热门视频信息/Get comprehensive popular video information")
async def fetch_collect_folders(request: Request,
                                pn: int = Query(default=1, description="页码/Page number")):
    """
    # [中文]
    ### 用途:
    - 获取综合热门视频信息
    ### 参数:
    - pn: 页码
    ### 返回:
    - 综合热门视频信息

    # [English]
    ### Purpose:
    - Get comprehensive popular video information
    ### Parameters:
    - pn: Page number
    ### Return:
    - comprehensive popular video information

    # [示例/Example]
    pn = 1
    """
    try:
        data = await BilibiliWebCrawler.fetch_com_popular(pn)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定视频的评论
@router.get("/fetch_video_comments", response_model=ResponseModel,
            summary="获取指定视频的评论/Get comments on the specified video")
async def fetch_collect_folders(request: Request,
                                bv_id: str = Query(example="BV1M1421t7hT", description="作品id/Video id"),
                                pn: int = Query(default=1, description="页码/Page number")):
    """
    # [中文]
    ### 用途:
    - 获取指定视频的评论
    ### 参数:
    - bv_id: 作品id
    - pn: 页码
    ### 返回:
    - 指定视频的评论数据

    # [English]
    ### Purpose:
    - Get comments on the specified video
    ### Parameters:
    - bv_id: Video id
    - pn: Page number
    ### Return:
    - comments of the specified video

    # [示例/Example]
    bv_id = "BV1M1421t7hT"
    pn = 1
    """
    try:
        data = await BilibiliWebCrawler.fetch_video_comments(bv_id, pn)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取视频下指定评论的回复
@router.get("/fetch_comment_reply", response_model=ResponseModel,
            summary="获取视频下指定评论的回复/Get reply to the specified comment")
async def fetch_collect_folders(request: Request,
                                bv_id: str = Query(example="BV1M1421t7hT", description="作品id/Video id"),
                                pn: int = Query(default=1, description="页码/Page number"),
                                rpid: str = Query(example="237109455120", description="回复id/Reply id")):
    """
    # [中文]
    ### 用途:
    - 获取视频下指定评论的回复
    ### 参数:
    - bv_id: 作品id
    - pn: 页码
    - rpid: 回复id
    ### 返回:
    - 指定评论的回复数据

    # [English]
    ### Purpose:
    - Get reply to the specified comment
    ### Parameters:
    - bv_id: Video id
    - pn: Page number
    - rpid: Reply id
    ### Return:
    - Reply of the specified comment

    # [示例/Example]
    bv_id = "BV1M1421t7hT"
    pn = 1
    rpid = "237109455120"
    """
    try:
        data = await BilibiliWebCrawler.fetch_comment_reply(bv_id, pn, rpid)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定用户动态
@router.get("/fetch_user_dynamic", response_model=ResponseModel,
            summary="获取指定用户动态/Get dynamic information of specified user")
async def fetch_collect_folders(request: Request,
                                uid: str = Query(example="16015678", description="用户UID"),
                                offset: str = Query(default="", example="953154282154098691",
                                                    description="开始索引/offset")):
    """
    # [中文]
    ### 用途:
    - 获取指定用户动态
    ### 参数:
    - uid: 用户UID
    - offset: 开始索引
    ### 返回:
    - 指定用户动态数据

    # [English]
    ### Purpose:
    - Get dynamic information of specified user
    ### Parameters:
    - uid: User UID
    - offset: offset
    ### Return:
    - dynamic information of specified user

    # [示例/Example]
    uid = "178360345"
    offset = "953154282154098691"
    """
    try:
        data = await BilibiliWebCrawler.fetch_user_dynamic(uid, offset)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取视频实时弹幕
@router.get("/fetch_video_danmaku", response_model=ResponseModel, summary="获取视频实时弹幕/Get Video Danmaku")
async def fetch_one_video(request: Request,
                          cid: str = Query(example="1639235405", description="作品cid/Video cid")):
    """
    # [中文]
    ### 用途:
    - 获取视频实时弹幕
    ### 参数:
    - cid: 作品cid
    ### 返回:
    - 视频实时弹幕

    # [English]
    ### Purpose:
    - Get Video Danmaku
    ### Parameters:
    - cid: Video cid
    ### Return:
    - Video Danmaku

    # [示例/Example]
    cid = "1639235405"
    """
    try:
        data = await BilibiliWebCrawler.fetch_video_danmaku(cid)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定直播间信息
@router.get("/fetch_live_room_detail", response_model=ResponseModel,
            summary="获取指定直播间信息/Get information of specified live room")
async def fetch_collect_folders(request: Request,
                                room_id: str = Query(example="22816111", description="直播间ID/Live room ID")):
    """
    # [中文]
    ### 用途:
    - 获取指定直播间信息
    ### 参数:
    - room_id: 直播间ID
    ### 返回:
    - 指定直播间信息

    # [English]
    ### Purpose:
    - Get information of specified live room
    ### Parameters:
    - room_id: Live room ID
    ### Return:
    - information of specified live room

    # [示例/Example]
    room_id = "22816111"
    """
    try:
        data = await BilibiliWebCrawler.fetch_live_room_detail(room_id)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定直播间视频流
@router.get("/fetch_live_videos", response_model=ResponseModel,
            summary="获取直播间视频流/Get live video data of specified room")
async def fetch_collect_folders(request: Request,
                                room_id: str = Query(example="1815229528", description="直播间ID/Live room ID")):
    """
    # [中文]
    ### 用途:
    - 获取指定直播间视频流
    ### 参数:
    - room_id: 直播间ID
    ### 返回:
    - 指定直播间视频流

    # [English]
    ### Purpose:
    - Get live video data of specified room
    ### Parameters:
    - room_id: Live room ID
    ### Return:
    - live video data of specified room

    # [示例/Example]
    room_id = "1815229528"
    """
    try:
        data = await BilibiliWebCrawler.fetch_live_videos(room_id)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取指定分区正在直播的主播
@router.get("/fetch_live_streamers", response_model=ResponseModel,
            summary="获取指定分区正在直播的主播/Get live streamers of specified live area")
async def fetch_collect_folders(request: Request,
                                area_id: str = Query(example="9", description="直播分区id/Live area ID"),
                                pn: int = Query(default=1, description="页码/Page number")):
    """
    # [中文]
    ### 用途:
    - 获取指定分区正在直播的主播
    ### 参数:
    - area_id: 直播分区id
    - pn: 页码
    ### 返回:
    - 指定分区正在直播的主播

    # [English]
    ### Purpose:
    - Get live streamers of specified live area
    ### Parameters:
    - area_id: Live area ID
    - pn: Page number
    ### Return:
    - live streamers of specified live area

    # [示例/Example]
    area_id = "9"
    pn = 1
    """
    try:
        data = await BilibiliWebCrawler.fetch_live_streamers(area_id, pn)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 获取所有直播分区列表
@router.get("/fetch_all_live_areas", response_model=ResponseModel,
            summary="获取所有直播分区列表/Get a list of all live areas")
async def fetch_collect_folders(request: Request,):
    """
    # [中文]
    ### 用途:
    - 获取所有直播分区列表
    ### 参数:
    ### 返回:
    - 所有直播分区列表

    # [English]
    ### Purpose:
    - Get a list of all live areas
    ### Parameters:
    ### Return:
    - list of all live areas

    # [示例/Example]
    """
    try:
        data = await BilibiliWebCrawler.fetch_all_live_areas()
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 通过bv号获得视频aid号
@router.get("/bv_to_aid", response_model=ResponseModel, summary="通过bv号获得视频aid号/Generate aid by bvid")
async def fetch_one_video(request: Request,
                          bv_id: str = Query(example="BV1M1421t7hT", description="作品id/Video id")):
    """
    # [中文]
    ### 用途:
    - 通过bv号获得视频aid号
    ### 参数:
    - bv_id: 作品id
    ### 返回:
    - 视频aid号

    # [English]
    ### Purpose:
    - Generate aid by bvid
    ### Parameters:
    - bv_id: Video id
    ### Return:
    - Video aid

    # [示例/Example]
    bv_id = "BV1M1421t7hT"
    """
    try:
        data = await BilibiliWebCrawler.bv_to_aid(bv_id)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())


# 通过bv号获得视频分p信息
@router.get("/fetch_video_parts", response_model=ResponseModel, summary="通过bv号获得视频分p信息/Get Video Parts By bvid")
async def fetch_one_video(request: Request,
                          bv_id: str = Query(example="BV1vf421i7hV", description="作品id/Video id")):
    """
    # [中文]
    ### 用途:
    - 通过bv号获得视频分p信息
    ### 参数:
    - bv_id: 作品id
    ### 返回:
    - 视频分p信息

    # [English]
    ### Purpose:
    - Get Video Parts By bvid
    ### Parameters:
    - bv_id: Video id
    ### Return:
    - Video Parts

    # [示例/Example]
    bv_id = "BV1vf421i7hV"
    """
    try:
        data = await BilibiliWebCrawler.fetch_video_parts(bv_id)
        return ResponseModel(code=200,
                             router=request.url.path,
                             data=data)
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(code=status_code,
                                    message=str(e),
                                    router=request.url.path,
                                    params=dict(request.query_params),
                                    )
        raise HTTPException(status_code=status_code, detail=detail.dict())

# 动态水印：BV15TrHBpEn5   静态水印：BV1qCrYB4E28
@router.get("/down_video", response_model=ResponseModel, summary="下载视频/Download video")
async def down_video(
    request: Request,
    bv_id: str = Query(example="BV1qCrYB4E28", description="作品id/Video id"),
    remove_watermark: int = Query(default=0, description="是否移除水印(0/1)"),
    resolution: int = Query(default=0, description="视频分辨率(0:原分辨率,1:720p,2:1080p)"),
):
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        video_dir = os.path.join(project_root, "video")
        os.makedirs(video_dir, exist_ok=True)
        file_path = os.path.join(video_dir, f"{bv_id}.mp4")

        existed_before = os.path.exists(file_path)
        if existed_before and not remove_watermark:
            return ResponseModel(
                code=200,
                router=request.url.path,
                data={"bv_id": bv_id, "file_path": file_path, "exists": True},
            )

        picked_height = None
        cid = None
        if not os.path.exists(file_path):
            video_detail = await BilibiliWebCrawler.fetch_one_video(bv_id)
            cid = (video_detail.get("data") or {}).get("cid")
            if not cid:
                raise ValueError("Failed to get cid from bilibili video detail")

            qn_map = {1: "64", 2: "80"}
            qn = qn_map.get(resolution, "120")
            playurl_data = await BilibiliWebCrawler.fetch_video_playurl(bv_id, str(cid), qn=qn)
            data = playurl_data.get("data") or {}

            kwargs = await BilibiliWebCrawler.get_bilibili_headers()
            headers = kwargs.get("headers") or {}

            dash = data.get("dash") or {}
            video_url, audio_url, picked_height = _pick_dash_urls(dash, resolution=resolution)
            if video_url and audio_url:
                if shutil.which("ffmpeg") is None:
                    playurl_progressive = await BilibiliWebCrawler.fetch_video_playurl(
                        bv_id,
                        str(cid),
                        qn=qn,
                        fnval="0",
                    )
                    progressive_data = playurl_progressive.get("data") or {}
                    durl_list = progressive_data.get("durl") or []
                    if durl_list and (durl_list[0] or {}).get("url"):
                        ok = await _fetch_data_stream(
                            (durl_list[0] or {}).get("url"),
                            request,
                            headers=headers,
                            file_path=file_path,
                        )
                        if not ok:
                            raise RuntimeError("Download interrupted")
                    else:
                        raise RuntimeError("ffmpeg not found: cannot merge bilibili video/audio streams")
                else:
                    ok = await _merge_bilibili_video_audio(
                        video_url=video_url,
                        audio_url=audio_url,
                        request=request,
                        output_path=file_path,
                        headers=headers,
                    )
                    if not ok:
                        raise RuntimeError("Failed to merge bilibili video and audio streams")
            else:
                durl_list = data.get("durl") or []
                direct_url = None
                if durl_list:
                    direct_url = (durl_list[0] or {}).get("url")
                direct_url = direct_url or video_url
                if not direct_url:
                    raise RuntimeError("Failed to get bilibili video url")
                ok = await _fetch_data_stream(direct_url, request, headers=headers, file_path=file_path)
                if not ok:
                    raise RuntimeError("Download interrupted")

        watermark_removed = False
        watermark_box = None
        if remove_watermark:
            _ = _ffmpeg_path()
            _ = _ffprobe_path()
            sha_before = _sha256_file(file_path)
            box = _detect_watermark_box(file_path)
            watermark_box = {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}
            tmp_out = os.path.join(video_dir, f"{bv_id}.nowm.tmp.mp4")
            try:
                _remove_watermark_ffmpeg(file_path, tmp_out, box)
                os.replace(tmp_out, file_path)
            finally:
                if os.path.exists(tmp_out):
                    try:
                        os.remove(tmp_out)
                    except OSError:
                        pass
            sha_after = _sha256_file(file_path)
            if sha_after == sha_before:
                raise RuntimeError("watermark removal produced identical output (no changes applied)")
            watermark_removed = True

        return ResponseModel(
            code=200,
            router=request.url.path,
            data={
                "bv_id": bv_id,
                "file_path": file_path,
                "exists": existed_before,
                "remove_watermark": 1 if remove_watermark else 0,
                "watermark_removed": watermark_removed,
                "watermark_box": watermark_box,
                "resolution": resolution,
                "picked_height": picked_height,
            },
        )
    except Exception as e:
        status_code = 400
        detail = ErrorResponseModel(
            code=status_code,
            message=str(e),
            router=request.url.path,
            params=dict(request.query_params),
        )
        raise HTTPException(status_code=status_code, detail=detail.dict())
