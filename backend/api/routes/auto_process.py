"""
Auto Process API Routes
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

from backend.services.auto_process_service import auto_process_service

router = APIRouter(prefix="/api/auto-process", tags=["auto-process"])


class SettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    resolution: Optional[str] = None
    check_interval_hours: Optional[int] = None
    active_hours_start: Optional[int] = None
    active_hours_end: Optional[int] = None
    processing_priority: Optional[str] = None
    genres: Optional[List[str]] = None
    duration_filter: Optional[str] = None
    region_code: Optional[str] = None
    max_videos_per_run: Optional[int] = None
    max_downloads_per_scan: Optional[int] = None
    search_query: Optional[str] = None
    skip_processed: Optional[bool] = None
    show_completed_label: Optional[bool] = None


class MarkProcessedRequest(BaseModel):
    video_id: str
    title: str
    channel: str
    output_path: Optional[str] = None
    clips_count: int = 0


@router.get("/settings")
async def get_settings():
    """Get auto process settings"""
    return auto_process_service.get_settings()


@router.post("/settings")
async def update_settings(updates: SettingsUpdate):
    """Update auto process settings"""
    # Only include non-None values
    update_dict = {k: v for k, v in updates.dict().items() if v is not None}
    return auto_process_service.update_settings(update_dict)


@router.post("/settings/reset")
async def reset_settings():
    """Reset settings to defaults"""
    return auto_process_service.reset_settings()


@router.post("/mark-processed")
async def mark_processed(request: MarkProcessedRequest):
    """Mark a video as processed"""
    success = auto_process_service.mark_video_processed(
        video_id=request.video_id,
        title=request.title,
        channel=request.channel,
        output_path=request.output_path,
        clips_count=request.clips_count
    )
    return {"success": success}


@router.post("/unmark-processed/{video_id}")
async def unmark_processed(video_id: str):
    """Remove a video from processed list"""
    success = auto_process_service.unmark_video_processed(video_id)
    return {"success": success}


@router.get("/processed")
async def get_processed_videos():
    """Get all processed videos"""
    videos = auto_process_service.get_processed_videos()
    return {
        "videos": videos,
        "count": len(videos)
    }


@router.get("/processed/ids")
async def get_processed_ids():
    """Get list of processed video IDs"""
    return {
        "ids": auto_process_service.get_processed_video_ids()
    }


@router.get("/is-processed/{video_id}")
async def is_processed(video_id: str):
    """Check if a video is processed"""
    return {
        "video_id": video_id,
        "processed": auto_process_service.is_video_processed(video_id)
    }


@router.post("/clear-processed")
async def clear_processed():
    """Clear all processed videos"""
    success = auto_process_service.clear_processed_videos()
    return {"success": success}
