"""
YouTube Scraper API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os

from backend.services.youtube_service import youtube_service, VIDEO_CATEGORIES, REGION_CODES
from backend.core.state import app_state
from backend.core.config import settings

router = APIRouter(prefix="/api/youtube", tags=["youtube"])


class SearchRequest(BaseModel):
    query: str
    max_results: int = 20
    duration_filter: str = "any"  # any, short, medium, long
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None  # Will be converted to channel_id
    order: str = "relevance"  # relevance, date, viewCount, rating
    published_after: Optional[str] = None
    category_id: Optional[str] = None  # Video category/genre
    region_code: str = "ID"  # Country code


class DownloadRequest(BaseModel):
    video_url: str
    format_id: str = "best"
    output_dir: Optional[str] = None


@router.get("/status")
async def get_status():
    """Check if YouTube API is configured"""
    has_key = bool(youtube_service.api_key)
    return {
        "configured": has_key,
        "api_key_preview": youtube_service.api_key[:10] + "..." if has_key else None
    }


@router.get("/categories")
async def get_categories():
    """Get list of video categories/genres"""
    categories = [
        {"id": k, "name": v}
        for k, v in VIDEO_CATEGORIES.items()
    ]
    # Sort by name, but keep "All" first
    categories.sort(key=lambda x: (x["id"] != "0", x["name"]))
    return {"categories": categories}


@router.get("/regions")
async def get_regions():
    """Get list of available regions/countries"""
    regions = [
        {"code": k, "name": v}
        for k, v in REGION_CODES.items()
    ]
    # Sort by name, but keep "ID" (Indonesia) first
    regions.sort(key=lambda x: (x["code"] != "ID", x["name"]))
    return {"regions": regions}


@router.post("/search")
async def search_videos(request: SearchRequest):
    """Search YouTube videos"""
    channel_id = request.channel_id

    # Convert channel name to ID if provided
    if request.channel_name and not channel_id:
        channel_id = youtube_service.get_channel_id(request.channel_name)
        if not channel_id:
            return {"error": f"Channel not found: {request.channel_name}", "videos": []}

    result = youtube_service.search_videos(
        query=request.query,
        max_results=request.max_results,
        duration_filter=request.duration_filter,
        channel_id=channel_id,
        order=request.order,
        published_after=request.published_after,
        category_id=request.category_id,
        region_code=request.region_code
    )

    return result


@router.get("/formats")
async def get_formats(url: str):
    """Get available download formats for a video"""
    formats = youtube_service.get_video_formats(url)
    return {"formats": formats}


@router.post("/download")
async def download_video(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start video download"""
    output_dir = request.output_dir or settings.DOWNLOAD_DIR

    # Create download directory if needed
    os.makedirs(output_dir, exist_ok=True)

    def progress_callback(percent: float, status: str):
        app_state.add_log(f"YT Download: {status}")

    # Run download
    result = youtube_service.download_video(
        video_url=request.video_url,
        output_dir=output_dir,
        format_id=request.format_id,
        progress_callback=progress_callback
    )

    if result["success"]:
        app_state.add_log(f"Downloaded: {os.path.basename(result['path'])}", "success")
        return {
            "success": True,
            "path": result["path"],
            "filename": os.path.basename(result["path"]) if result["path"] else None
        }
    else:
        app_state.add_log(f"Download failed: {result['error']}", "error")
        raise HTTPException(status_code=500, detail=result["error"])


@router.get("/channel")
async def get_channel_id(name: str):
    """Get channel ID from channel name"""
    channel_id = youtube_service.get_channel_id(name)
    if channel_id:
        return {"channel_id": channel_id, "name": name}
    else:
        raise HTTPException(status_code=404, detail=f"Channel not found: {name}")


class RecommendRequest(BaseModel):
    videos: List[dict]
    purpose: str = "viral clips"


@router.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    """Get AI recommendations for best videos to process"""
    result = youtube_service.get_ai_recommendations(
        videos=request.videos,
        purpose=request.purpose
    )
    return result
