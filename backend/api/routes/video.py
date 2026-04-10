"""
Video Processing Routes
"""

import os
import sys
import asyncio
import shutil
import json
import hashlib
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File

from backend.models.schemas import VideoProcessRequest, VideoInfoResponse, StatusResponse, GPUStatusResponse
from backend.services.video_service import VideoService
from backend.services.clip_service import ClipService
from backend.services.export_service import ExportService
from backend.modules.downloader import sanitize_filename
from backend.core.state import app_state
from backend.core.config import settings

router = APIRouter(prefix="/api/video", tags=["video"])

# Service instances
video_service = VideoService()
clip_service = ClipService()
export_service = ExportService()


def get_transcription_cache_path() -> str:
    """Get cache path for transcription (single file only)"""
    return os.path.join(settings.DATA_DIR, "transcription_cache.json")


def load_cached_transcription() -> dict:
    """Load transcription from cache if exists"""
    cache_path = get_transcription_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CACHE] Failed to load transcription cache: {e}")
    return None


def save_transcription_cache(video_title: str, video_path: str, transcription: dict):
    """Save transcription to cache with video info"""
    cache_path = get_transcription_cache_path()
    try:
        cache_data = {
            'video_title': video_title,
            'video_path': video_path,
            'video_filename': os.path.basename(video_path),
            'saved_at': __import__('datetime').datetime.now().isoformat(),
            'transcription': transcription
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[CACHE] Transcription saved for: {video_title}")
    except Exception as e:
        print(f"[CACHE] Failed to save transcription cache: {e}")


def clear_transcription_cache():
    """Clear transcription cache"""
    cache_path = get_transcription_cache_path()
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"[CACHE] Cleared transcription cache")
            return True
        except Exception as e:
            print(f"[CACHE] Failed to clear cache: {e}")
            return False
    return True


def get_cache_info() -> dict:
    """Get info about cached transcription"""
    cache_data = load_cached_transcription()
    if cache_data and 'video_title' in cache_data:
        return {
            'has_cache': True,
            'video_title': cache_data.get('video_title', 'Unknown'),
            'video_filename': cache_data.get('video_filename', 'Unknown'),
            'saved_at': cache_data.get('saved_at', ''),
            'segment_count': len(cache_data.get('transcription', {}).get('segments', [])),
            'language': cache_data.get('transcription', {}).get('language', 'unknown')
        }
    return {'has_cache': False}


async def process_youtube_video(url: str):
    """Background task to process YouTube video"""
    try:
        app_state.processing.is_processing = True
        app_state.processing.current_stage = "downloading"
        app_state.processing.source_url = url
        app_state.processing.cancel_requested = False

        # Get video info
        app_state.add_log(f"Getting video info for: {url}")

        info = await video_service.get_video_info(url)
        app_state.processing.video_title = info['title']
        app_state.processing.video_duration = info['duration']

        app_state.add_log(f"Video: {info['title']} ({info['duration']:.1f}s)")
        await app_state.broadcast('video_info', info)

        # Download video
        app_state.add_log("Starting download...")
        video_path = await video_service.download_video(url)
        app_state.processing.video_path = video_path

        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        app_state.add_log(f"Downloaded: {os.path.basename(video_path)} ({file_size:.1f}MB)")

        if app_state.processing.cancel_requested:
            app_state.add_log("Processing cancelled after download", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        # Check for cached transcription first
        app_state.processing.current_stage = "transcribing"
        cached_data = load_cached_transcription()

        if cached_data and 'transcription' in cached_data:
            cached_title = cached_data.get('video_title', '')
            current_title = info['title']

            # Check if titles match
            if cached_title == current_title:
                print(f"[PROCESS] ✓ Cache HIT - skipping Whisper for: {cached_title}")
                app_state.add_log(f"✓ Cache match! Skipping Whisper transcription...")
                app_state.add_log(f"Using cached transcription for: {cached_title}")
                transcription = cached_data['transcription']
                segment_count = len(transcription.get('segments', []))
                app_state.add_log(f"Loaded {segment_count} segments, language: {transcription.get('language', 'unknown')}")

                # Skip directly to analyzing stage (no Whisper needed)
                app_state.processing.current_stage = "analyzing"
            else:
                # Title mismatch - notify user and transcribe fresh
                app_state.add_log(f"⚠️ Cache mismatch: cached='{cached_title}' vs current='{current_title}'", "warning")
                app_state.add_log("Transcribing fresh video...")
                transcription = await video_service.transcribe_video(video_path)

                # Save to cache (overwrites old cache)
                save_transcription_cache(current_title, video_path, transcription)

                segment_count = len(transcription.get('segments', []))
                app_state.add_log(f"Transcription complete: {segment_count} segments, language: {transcription.get('language', 'unknown')}")

                # Clear Whisper model from VRAM
                app_state.add_log("Unloading Whisper model to free VRAM...")
                video_service.unload_models()
                app_state.add_log("VRAM freed for face tracking")
        else:
            # No cache - Transcribe with Whisper
            app_state.add_log("Starting transcription with Whisper model...")
            transcription = await video_service.transcribe_video(video_path)

            # Save to cache
            save_transcription_cache(info['title'], video_path, transcription)

            segment_count = len(transcription.get('segments', []))
            app_state.add_log(f"Transcription complete: {segment_count} segments, language: {transcription.get('language', 'unknown')}")

            # Clear Whisper model from VRAM to free space for YOLO
            app_state.add_log("Unloading Whisper model to free VRAM...")
            video_service.unload_models()
            app_state.add_log("VRAM freed for face tracking")

        app_state.processing.transcription = transcription

        if app_state.processing.cancel_requested:
            app_state.add_log("Processing cancelled after transcription", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        # Analyze for clips
        app_state.processing.current_stage = "analyzing"
        app_state.add_log("Analyzing transcript for interesting clips...")
        app_state.add_log(f"Settings: {app_state.settings.get('min_clip_duration', 15)}-{app_state.settings.get('max_clip_duration', 60)}s, finding {app_state.settings.get('clips_to_find', 5)} clips")

        clips = await clip_service.analyze_clips(
            transcription,
            info['duration'],
            min_duration=app_state.settings.get('min_clip_duration', 15),
            max_duration=app_state.settings.get('max_clip_duration', 60),
            num_clips=app_state.settings.get('clips_to_find', 5)
        )

        if app_state.processing.cancel_requested:
            app_state.add_log("Processing cancelled after analysis", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        app_state.set_clips(clips)
        app_state.add_log(f"Found {len(clips)} interesting clips!", "success")

        # Broadcast clips ready
        await app_state.broadcast('clips_ready', {
            'clips': clips
        })

        app_state.processing.current_stage = "idle"
        app_state.processing.is_processing = False
        app_state.add_log("Processing complete! Ready to export clips.", "success")

    except Exception as e:
        error_msg = str(e)
        app_state.add_log(f"Error: {error_msg}", "error")
        await app_state.broadcast('error', {'message': error_msg})
        app_state.processing.is_processing = False
        app_state.processing.current_stage = "idle"
        # Don't re-raise - just log and return gracefully
        # This prevents the ASGI exception and allows queue to continue


async def process_local_video(file_path: str):
    """Background task to process local video file"""
    try:
        print(f"\n[PROCESS] Starting local video: {file_path}")

        if not os.path.exists(file_path):
            print(f"[PROCESS ERROR] File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")

        # Check for incomplete download (.part file) or invalid extension
        valid_extensions = ('.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv')
        if file_path.endswith('.part'):
            print(f"[PROCESS ERROR] File is incomplete (.part): {file_path}")
            raise Exception(f"Download incomplete - file is still .part: {os.path.basename(file_path)}")

        if not file_path.lower().endswith(valid_extensions):
            print(f"[PROCESS ERROR] Invalid file extension: {file_path}")
            raise Exception(f"Invalid video file: {os.path.basename(file_path)}. Must be {valid_extensions}")

        # Check file size - empty or very small files are likely corrupt
        file_size = os.path.getsize(file_path)
        if file_size < 1024:  # Less than 1KB
            print(f"[PROCESS ERROR] File too small ({file_size} bytes): {file_path}")
            raise Exception(f"File too small ({file_size} bytes) - download may have failed")

        app_state.processing.is_processing = True
        app_state.processing.current_stage = "transcribing"
        app_state.processing.video_path = file_path
        app_state.processing.is_local_file = True
        app_state.processing.cancel_requested = False

        # Get video info from file
        import cv2
        print(f"[PROCESS] Loading video with cv2...")
        app_state.add_log(f"Loading video: {os.path.basename(file_path)}")
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print(f"[PROCESS ERROR] Failed to open video with cv2: {file_path}")
            raise Exception(f"Failed to open video file: {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration <= 0:
            print(f"[PROCESS ERROR] Invalid video duration: {duration}")
            raise Exception(f"Invalid video - duration is {duration}s. File may be corrupt or incomplete.")

        print(f"[PROCESS] Video info: {width}x{height}, {fps:.1f}fps, {duration:.1f}s")

        app_state.processing.video_title = os.path.basename(file_path)
        app_state.processing.video_duration = duration
        app_state.update_progress('download', 100, 'Local file loaded')

        # Detailed video info log
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        app_state.add_log(f"Video info: {width}x{height}, {fps:.1f}fps, {duration:.1f}s, {file_size:.1f}MB")

        await app_state.broadcast('video_info', {
            'title': app_state.processing.video_title,
            'duration': duration,
            'is_local': True
        })

        app_state.add_log(f"Local video: {app_state.processing.video_title} ({duration:.1f}s)")

        # Check cancel before transcription
        if app_state.processing.cancel_requested:
            app_state.add_log("Processing cancelled before transcription", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        # Check for cached transcription first
        video_title = os.path.basename(file_path)
        cached_data = load_cached_transcription()

        if cached_data and 'transcription' in cached_data:
            cached_title = cached_data.get('video_title', '')

            # Check if titles match
            if cached_title == video_title:
                print(f"[PROCESS] ✓ Cache HIT - skipping Whisper for: {cached_title}")
                app_state.add_log(f"✓ Cache match! Skipping Whisper transcription...")
                app_state.add_log(f"Using cached transcription for: {cached_title}")
                transcription = cached_data['transcription']
                segment_count = len(transcription.get('segments', []))
                app_state.add_log(f"Loaded {segment_count} segments, language: {transcription.get('language', 'unknown')}")

                # Skip directly to analyzing stage (no Whisper needed)
                app_state.processing.current_stage = "analyzing"
            else:
                # Title mismatch - notify and transcribe fresh
                print(f"[PROCESS] Cache mismatch: cached='{cached_title}' vs current='{video_title}'")
                app_state.add_log(f"⚠️ Cache mismatch: cached='{cached_title}' vs current='{video_title}'", "warning")
                app_state.add_log("Transcribing fresh video...")
                app_state.add_log(f"Using device: {app_state.settings.get('whisper_device', 'cuda')}")

                try:
                    transcription = await video_service.transcribe_video(file_path)
                    print(f"[PROCESS] Transcription complete: {len(transcription.get('segments', []))} segments")

                    # Save to cache (overwrites old cache)
                    save_transcription_cache(video_title, file_path, transcription)

                except Exception as e:
                    print(f"[PROCESS ERROR] Transcription failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

                segment_count = len(transcription.get('segments', []))
                app_state.add_log(f"Transcription complete: {segment_count} segments, language: {transcription.get('language', 'unknown')}")

                # Clear Whisper model from VRAM
                app_state.add_log("Unloading Whisper model to free VRAM...")
                video_service.unload_models()
                app_state.add_log("VRAM freed for face tracking")
        else:
            # No cache - Transcribe with Whisper
            print(f"[PROCESS] Starting transcription...")
            app_state.add_log("Starting transcription with Whisper model...")
            app_state.add_log(f"Using device: {app_state.settings.get('whisper_device', 'cuda')}")

            try:
                transcription = await video_service.transcribe_video(file_path)
                print(f"[PROCESS] Transcription complete: {len(transcription.get('segments', []))} segments")

                # Save to cache
                save_transcription_cache(video_title, file_path, transcription)

            except Exception as e:
                print(f"[PROCESS ERROR] Transcription failed: {e}")
                import traceback
                traceback.print_exc()
                raise

            segment_count = len(transcription.get('segments', []))
            app_state.add_log(f"Transcription complete: {segment_count} segments, language: {transcription.get('language', 'unknown')}")

            # Clear Whisper model from VRAM to free space for YOLO
            app_state.add_log("Unloading Whisper model to free VRAM...")
            video_service.unload_models()
            app_state.add_log("VRAM freed for face tracking")

        app_state.processing.transcription = transcription

        # Check cancel after transcription
        if app_state.processing.cancel_requested:
            app_state.add_log("Processing cancelled after transcription", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        # Analyze for clips
        app_state.processing.current_stage = "analyzing"
        print(f"[PROCESS] Starting AI analysis...")
        app_state.add_log("Analyzing transcript for interesting clips...")
        app_state.add_log(f"Settings: {app_state.settings.get('min_clip_duration', 15)}-{app_state.settings.get('max_clip_duration', 60)}s, finding {app_state.settings.get('clips_to_find', 5)} clips")

        try:
            clips = await clip_service.analyze_clips(
                transcription,
                duration,
                min_duration=app_state.settings.get('min_clip_duration', 15),
                max_duration=app_state.settings.get('max_clip_duration', 60),
                num_clips=app_state.settings.get('clips_to_find', 5)
            )
            print(f"[PROCESS] Analysis complete: {len(clips)} clips found")
        except Exception as e:
            print(f"[PROCESS ERROR] AI analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Check cancel after analysis
        if app_state.processing.cancel_requested:
            app_state.add_log("Processing cancelled after analysis", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        app_state.set_clips(clips)
        app_state.add_log(f"Found {len(clips)} interesting clips!", "success")

        await app_state.broadcast('clips_ready', {
            'clips': clips
        })

        app_state.processing.current_stage = "idle"
        app_state.processing.is_processing = False
        app_state.add_log("Processing complete! Ready to export clips.", "success")

    except Exception as e:
        error_msg = str(e)
        print(f"[PROCESS ERROR] Exception: {error_msg}")
        import traceback
        traceback.print_exc()
        app_state.add_log(f"Error: {error_msg}", "error")
        await app_state.broadcast('error', {'message': error_msg})
        app_state.processing.is_processing = False
        app_state.processing.current_stage = "idle"
        # Don't re-raise - just log and return gracefully
        # This prevents the ASGI exception and allows queue to continue


@router.post("/youtube")
async def process_youtube(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """Start processing a YouTube video"""
    if app_state.processing.is_processing:
        raise HTTPException(status_code=409, detail="Already processing a video")

    video_service.reset_cancel()
    background_tasks.add_task(process_youtube_video, request.url)

    return {"status": "started", "url": request.url}


@router.post("/local")
async def process_local(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """Start processing a local video file"""
    if app_state.processing.is_processing:
        raise HTTPException(status_code=409, detail="Already processing a video")

    if not os.path.exists(request.url):
        raise HTTPException(status_code=404, detail="File not found")

    video_service.reset_cancel()
    background_tasks.add_task(process_local_video, request.url)

    return {"status": "started", "path": request.url}


@router.get("/info", response_model=VideoInfoResponse)
async def get_video_info():
    """Get current video information"""
    if not app_state.processing.video_title:
        raise HTTPException(status_code=404, detail="No video loaded")

    return VideoInfoResponse(
        title=app_state.processing.video_title or "",
        duration=app_state.processing.video_duration,
        is_local=app_state.processing.is_local_file
    )


@router.post("/reanalyze")
async def reanalyze_video(background_tasks: BackgroundTasks):
    """
    Re-analyze current video using cached transcription.
    Useful when tracking failed and user wants to try again without re-transcribing.
    """
    if app_state.processing.is_processing:
        raise HTTPException(status_code=409, detail="Already processing a video")

    if not app_state.processing.video_path:
        raise HTTPException(status_code=400, detail="No video loaded")

    video_path = app_state.processing.video_path

    # Check if cached transcription exists
    cached_data = load_cached_transcription()
    if not cached_data or 'transcription' not in cached_data:
        raise HTTPException(status_code=400, detail="No cached transcription found. Please process video first.")

    transcription = cached_data['transcription']
    cached_title = cached_data.get('video_title', '')

    video_service.reset_cancel()
    background_tasks.add_task(reanalyze_video_task, video_path, transcription, cached_title)

    return {"status": "started", "path": video_path, "using_cache": True, "cached_title": cached_title}


async def reanalyze_video_task(video_path: str, transcription: dict, cached_title: str):
    """Background task to re-analyze video with cached transcription"""
    try:
        app_state.processing.is_processing = True
        app_state.processing.current_stage = "analyzing"
        app_state.processing.cancel_requested = False

        app_state.add_log(f"Re-analyzing using cached transcription from: {cached_title}")
        app_state.processing.transcription = transcription

        # Get video duration
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        # Analyze for clips
        app_state.add_log("Analyzing transcript for interesting clips...")
        app_state.add_log(f"Settings: {app_state.settings.get('min_clip_duration', 15)}-{app_state.settings.get('max_clip_duration', 60)}s, finding {app_state.settings.get('clips_to_find', 5)} clips")

        clips = await clip_service.analyze_clips(
            transcription,
            duration,
            min_duration=app_state.settings.get('min_clip_duration', 15),
            max_duration=app_state.settings.get('max_clip_duration', 60),
            num_clips=app_state.settings.get('clips_to_find', 5)
        )

        if app_state.processing.cancel_requested:
            app_state.add_log("Re-analysis cancelled", "warning")
            app_state.processing.is_processing = False
            app_state.processing.current_stage = "idle"
            return

        app_state.set_clips(clips)
        app_state.add_log(f"Found {len(clips)} interesting clips!", "success")

        await app_state.broadcast('clips_ready', {
            'clips': clips
        })

        app_state.processing.current_stage = "idle"
        app_state.processing.is_processing = False
        app_state.add_log("Re-analysis complete! Ready to export clips.", "success")

    except Exception as e:
        error_msg = str(e)
        app_state.add_log(f"Error: {error_msg}", "error")
        await app_state.broadcast('error', {'message': error_msg})
        app_state.processing.is_processing = False
        app_state.processing.current_stage = "idle"


@router.get("/cache")
async def get_transcription_cache():
    """Get info about cached transcription"""
    return get_cache_info()


@router.delete("/cache")
async def clear_cache():
    """Clear transcription cache"""
    success = clear_transcription_cache()
    if success:
        app_state.add_log("Transcription cache cleared.")
        return {"status": "cleared"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.post("/cancel")
async def cancel_processing():
    """Cancel current processing"""
    if not app_state.processing.is_processing:
        raise HTTPException(status_code=400, detail="No processing in progress")

    # Cancel all services
    video_service.cancel()
    export_service.cancel()
    app_state.processing.cancel_requested = True
    app_state.add_log("Cancellation requested - stopping all processes...", "warning")

    # Reset processing state after a short delay
    app_state.processing.is_processing = False
    app_state.processing.current_stage = "idle"
    app_state.add_log("Processing cancelled by user", "warning")

    return {"status": "cancelled"}


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current processing status"""
    return StatusResponse(**app_state.get_status())


@router.get("/gpu", response_model=GPUStatusResponse)
async def get_gpu_status():
    """Get GPU status information"""
    info = video_service.get_gpu_status()
    return GPUStatusResponse(
        cuda_available=info.get('cuda_available', False),
        gpu_name=info.get('gpu_name'),
        gpu_memory=info.get('gpu_memory'),
        device=info.get('device', 'cpu')
    )


@router.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload and process a video file (supports large files)"""
    if app_state.processing.is_processing:
        raise HTTPException(status_code=409, detail="Already processing a video")

    # Validate file type
    allowed_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")

    # Save uploaded file to temp directory with chunked writing for large files
    # Sanitize filename to remove emojis and special characters
    safe_filename = sanitize_filename(file.filename)
    temp_path = os.path.join(settings.TEMP_DIR, safe_filename)

    try:
        if safe_filename != file.filename:
            app_state.add_log(f"Receiving file: {file.filename} (sanitized to: {safe_filename})")
        else:
            app_state.add_log(f"Receiving file: {file.filename}")

        # Write file in chunks to handle large files
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)

        # Get file size
        file_size = os.path.getsize(temp_path)
        app_state.add_log(f"File saved: {safe_filename} ({file_size / 1024 / 1024:.1f} MB)")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    video_service.reset_cancel()
    background_tasks.add_task(process_local_video, temp_path)

    return {"status": "started", "filename": safe_filename, "path": temp_path}
