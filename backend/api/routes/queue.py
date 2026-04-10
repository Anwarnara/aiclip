"""
Queue API Routes
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from backend.services.queue_service import queue_service

router = APIRouter(prefix="/api/queue", tags=["queue"])


class AddToQueueRequest(BaseModel):
    video_id: str
    video_url: str
    title: str
    channel: str
    thumbnail: str = ""
    resolution: str = "1080"


@router.get("")
async def get_queue():
    """Get all queue items"""
    return {
        "items": queue_service.get_queue(),
        "stats": queue_service.get_stats()
    }


@router.get("/stats")
async def get_stats():
    """Get queue statistics"""
    return queue_service.get_stats()


@router.get("/item/{item_id}")
async def get_item(item_id: str):
    """Get a specific queue item"""
    item = queue_service.get_item(item_id)
    if item:
        return item
    return {"error": "Item not found"}


@router.post("/add")
async def add_to_queue(request: AddToQueueRequest):
    """Add a video to the download queue"""
    item = queue_service.add_to_queue(
        video_id=request.video_id,
        video_url=request.video_url,
        title=request.title,
        channel=request.channel,
        thumbnail=request.thumbnail,
        resolution=request.resolution
    )
    return {
        "success": True,
        "item": item.to_dict()
    }


@router.delete("/item/{item_id}")
async def remove_from_queue(item_id: str, force: bool = False):
    """Remove an item from the queue. Use force=true to cancel active downloads."""
    success = queue_service.remove_from_queue(item_id, force=force)
    return {"success": success}


@router.post("/retry/{item_id}")
async def retry_failed(item_id: str):
    """Retry a failed item"""
    success = queue_service.retry_failed(item_id)
    return {"success": success}


@router.post("/clear-completed")
async def clear_completed():
    """Clear all completed and failed items"""
    count = queue_service.clear_completed()
    return {"success": True, "cleared": count}


@router.post("/pause")
async def pause_queue():
    """Pause all queue operations"""
    success = queue_service.pause()
    return {"success": success, "paused": True}


@router.post("/resume")
async def resume_queue():
    """Resume queue operations"""
    success = queue_service.resume()
    return {"success": success, "paused": False}


@router.get("/paused")
async def is_paused():
    """Check if queue is paused"""
    return {"paused": queue_service.is_queue_paused()}


@router.post("/pause/{item_id}")
async def pause_item(item_id: str):
    """Pause a specific item"""
    success = queue_service.pause_item(item_id)
    return {"success": success}


@router.post("/resume/{item_id}")
async def resume_item(item_id: str):
    """Resume a paused item"""
    success = queue_service.resume_item(item_id)
    return {"success": success}


@router.get("/transcription/{video_id}")
async def check_transcription(video_id: str):
    """Check if there's a saved transcription for this video"""
    has_transcription = queue_service.has_saved_transcription(video_id)
    return {"video_id": video_id, "has_transcription": has_transcription}


@router.delete("/transcription/{video_id}")
async def delete_transcription(video_id: str):
    """Delete saved transcription for a video"""
    success = queue_service.delete_transcription(video_id)
    return {"success": success}
