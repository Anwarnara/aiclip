"""
Upload API Routes
Handles auto-upload to social media platforms (YouTube, Facebook)
"""

import os
import json
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Import YouTube OAuth service
from backend.services import youtube_oauth

# Data paths
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
YT_API_KEY_PATH = os.path.join(DATA_DIR, "yt_api_key.txt")
UPLOAD_SETTINGS_PATH = os.path.join(DATA_DIR, "upload_settings.json")
UPLOAD_HISTORY_PATH = os.path.join(DATA_DIR, "upload_history.json")
UPLOAD_QUEUE_PATH = os.path.join(DATA_DIR, "upload_queue.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/api/upload", tags=["upload"])


def load_yt_api_key() -> str:
    """Load YouTube API key from file"""
    try:
        if os.path.exists(YT_API_KEY_PATH):
            with open(YT_API_KEY_PATH, 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


# Default settings
DEFAULT_SETTINGS = {
    "youtube_enabled": False,
    "youtube_api_key": "",
    "youtube_channel_id": "",
    "youtube_privacy": "private",  # private, unlisted, public
    "youtube_category": "22",  # People & Blogs
    "youtube_tags": [],
    "facebook_enabled": False,
    "facebook_access_token": "",
    "facebook_page_id": "",
    "auto_upload_on_complete": False,
    "default_title_template": "{original_title} - Clip {clip_number}",
    "default_description_template": "Auto-generated clip from {original_title}\n\n#shorts #viral #fyp",
    "upload_delay_seconds": 60,  # Delay between uploads to avoid spam
}


def load_settings() -> dict:
    """Load upload settings"""
    settings = DEFAULT_SETTINGS.copy()

    # Always try to load YT API key from file first (auto-detect)
    yt_key = load_yt_api_key()
    if yt_key:
        settings["youtube_api_key"] = yt_key

    # Then load saved settings (overrides defaults but not API key if from file)
    try:
        if os.path.exists(UPLOAD_SETTINGS_PATH):
            with open(UPLOAD_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # Don't override API key if we loaded from file and saved is empty
                if saved.get("youtube_api_key") or not yt_key:
                    settings.update(saved)
                else:
                    # Keep the file API key
                    saved_copy = saved.copy()
                    saved_copy.pop("youtube_api_key", None)
                    settings.update(saved_copy)
    except Exception:
        pass

    return settings


def save_settings(settings: dict) -> bool:
    """Save upload settings"""
    try:
        with open(UPLOAD_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_upload_history() -> List[dict]:
    """Load upload history"""
    try:
        if os.path.exists(UPLOAD_HISTORY_PATH):
            with open(UPLOAD_HISTORY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_upload_history(history: List[dict]) -> bool:
    """Save upload history"""
    try:
        with open(UPLOAD_HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


class UploadSettingsUpdate(BaseModel):
    youtube_enabled: Optional[bool] = None
    youtube_api_key: Optional[str] = None
    youtube_channel_id: Optional[str] = None
    youtube_privacy: Optional[str] = None
    youtube_category: Optional[str] = None
    youtube_tags: Optional[List[str]] = None
    facebook_enabled: Optional[bool] = None
    facebook_access_token: Optional[str] = None
    facebook_page_id: Optional[str] = None
    auto_upload_on_complete: Optional[bool] = None
    default_title_template: Optional[str] = None
    default_description_template: Optional[str] = None
    upload_delay_seconds: Optional[int] = None


class UploadRequest(BaseModel):
    file_path: str
    platform: str  # youtube, facebook
    title: str
    description: Optional[str] = ""
    tags: Optional[List[str]] = []
    privacy: Optional[str] = "private"


class FolderUploadRequest(BaseModel):
    folder_path: str
    platform: str  # youtube, facebook


def parse_seo_content(content: str) -> dict:
    """Parse SEO content from text file with various formats"""
    metadata = {}
    lines = content.strip().split('\n')

    current_section = None
    current_content = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        line_upper = line_stripped.upper()

        # Detect section headers
        if 'JUDUL' in line_upper or ('TITLE' in line_upper and ':' in line):
            # Save previous section
            if current_section == 'description' and current_content:
                metadata['description'] = '\n'.join(current_content).strip()

            # Check if content is on same line after colon
            if ':' in line_stripped:
                after_colon = line_stripped.split(':', 1)[1].strip()
                if after_colon:
                    metadata['title'] = after_colon
                    current_section = None
                else:
                    current_section = 'title'
                    current_content = []
            else:
                current_section = 'title'
                current_content = []

        elif 'DESKRIPSI' in line_upper or ('DESCRIPTION' in line_upper and ':' in line):
            # Save previous section
            if current_section == 'title' and current_content:
                metadata['title'] = ' '.join(current_content).strip()

            if ':' in line_stripped:
                after_colon = line_stripped.split(':', 1)[1].strip()
                if after_colon:
                    metadata['description'] = after_colon
                    current_section = 'description_continue'
                    current_content = [after_colon]
                else:
                    current_section = 'description'
                    current_content = []
            else:
                current_section = 'description'
                current_content = []

        elif 'HASHTAG' in line_upper or 'TAGS' in line_upper:
            # Save previous section
            if current_section == 'title' and current_content:
                metadata['title'] = ' '.join(current_content).strip()
            elif current_section in ('description', 'description_continue') and current_content:
                metadata['description'] = '\n'.join(current_content).strip()

            if ':' in line_stripped:
                after_colon = line_stripped.split(':', 1)[1].strip()
                if after_colon:
                    # Parse hashtags - can be space or comma separated
                    tags = []
                    for part in after_colon.replace(',', ' ').split():
                        tag = part.strip().replace('#', '')
                        if tag:
                            tags.append(tag)
                    if tags:
                        metadata['tags'] = tags
                    current_section = None
                else:
                    current_section = 'hashtags'
                    current_content = []
            else:
                current_section = 'hashtags'
                current_content = []

        elif 'COPY-PASTE READY' in line_upper:
            # Stop parsing at copy-paste section
            if current_section == 'title' and current_content:
                metadata['title'] = ' '.join(current_content).strip()
            elif current_section in ('description', 'description_continue') and current_content:
                metadata['description'] = '\n'.join(current_content).strip()
            elif current_section == 'hashtags' and current_content:
                all_tags = ' '.join(current_content)
                tags = []
                for part in all_tags.replace(',', ' ').split():
                    tag = part.strip().replace('#', '')
                    if tag:
                        tags.append(tag)
                if tags:
                    metadata['tags'] = tags
            break

        elif '===' in line_stripped:
            # Skip separator lines (like === SEO CONTENT FOR VIDEO ===)
            # But stop if we already have content parsed
            if metadata:
                break
            continue

        elif current_section and line_stripped:
            # Collect content for current section
            current_content.append(line_stripped)

    # Handle remaining content
    if current_section == 'title' and current_content:
        metadata['title'] = ' '.join(current_content).strip()
    elif current_section in ('description', 'description_continue') and current_content:
        metadata['description'] = '\n'.join(current_content).strip()
    elif current_section == 'hashtags' and current_content:
        all_tags = ' '.join(current_content)
        tags = []
        for part in all_tags.replace(',', ' ').split():
            tag = part.strip().replace('#', '')
            if tag:
                tags.append(tag)
        if tags:
            metadata['tags'] = tags

    return metadata


def get_video_metadata(folder_path: str, video_filename: str) -> dict:
    """Get metadata for a specific video from its matching .txt file"""
    # Get base name without extension
    base_name = os.path.splitext(video_filename)[0]
    txt_path = os.path.join(folder_path, f"{base_name}.txt")
    json_path = os.path.join(folder_path, f"{base_name}.json")

    metadata = {}

    # Try to read from .txt file first
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                parsed = parse_seo_content(content)
                if parsed:
                    metadata.update(parsed)
        except Exception:
            pass

    # Try JSON file as fallback
    if not metadata and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if 'title' in json_data:
                    metadata['title'] = json_data['title']
                if 'description' in json_data:
                    metadata['description'] = json_data['description']
                if 'tags' in json_data:
                    metadata['tags'] = json_data['tags']
        except Exception:
            pass

    return metadata


def scan_output_folders() -> List[dict]:
    """Scan output directory for video folders with metadata"""
    folders = []

    if not os.path.exists(OUTPUT_DIR):
        return folders

    # Scan each subfolder in output directory
    for folder_name in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Look for video files
        video_files = []
        has_any_metadata = False

        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)

            if not os.path.isfile(filepath):
                continue

            # Check for video files
            if filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
                # Get metadata for this specific video
                video_metadata = get_video_metadata(folder_path, filename)
                if video_metadata:
                    has_any_metadata = True

                video_files.append({
                    "name": filename,
                    "path": filepath,
                    "size": os.path.getsize(filepath),
                    "metadata": video_metadata
                })

        # Add folder to list if it has videos
        if video_files:
            folders.append({
                "folder_name": folder_name,
                "folder_path": folder_path,
                "videos": video_files,
                "video_count": len(video_files),
                "has_metadata": has_any_metadata
            })

    return folders


@router.get("/settings")
async def get_upload_settings():
    """Get upload settings"""
    settings = load_settings()
    # Mask sensitive data
    if settings.get("youtube_api_key"):
        settings["youtube_api_key_set"] = True
        settings["youtube_api_key"] = "***" + settings["youtube_api_key"][-4:] if len(settings["youtube_api_key"]) > 4 else "****"
    if settings.get("facebook_access_token"):
        settings["facebook_access_token_set"] = True
        settings["facebook_access_token"] = "****"
    return settings


@router.post("/settings")
async def update_upload_settings(updates: UploadSettingsUpdate):
    """Update upload settings"""
    settings = load_settings()
    update_dict = updates.dict(exclude_unset=True)

    # Don't overwrite API key with masked value
    if update_dict.get("youtube_api_key", "").startswith("***"):
        del update_dict["youtube_api_key"]
    if update_dict.get("facebook_access_token") == "****":
        del update_dict["facebook_access_token"]

    settings.update(update_dict)
    save_settings(settings)
    return {"success": True}


@router.get("/history")
async def get_upload_history():
    """Get upload history"""
    history = load_upload_history()
    return {
        "history": history,
        "count": len(history)
    }


@router.delete("/history")
async def clear_upload_history():
    """Clear upload history"""
    save_upload_history([])
    return {"success": True}


@router.post("/youtube")
async def upload_to_youtube(request: UploadRequest, background_tasks: BackgroundTasks):
    """Upload video to YouTube"""
    settings = load_settings()

    if not settings.get("youtube_api_key"):
        raise HTTPException(status_code=400, detail="YouTube API key not configured")

    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="File not found")

    # Add to background tasks
    # Note: Full YouTube upload implementation requires OAuth2 authentication
    # This is a placeholder that logs the upload request

    history = load_upload_history()
    history.append({
        "id": len(history) + 1,
        "platform": "youtube",
        "file_path": request.file_path,
        "title": request.title,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "error": None
    })
    save_upload_history(history)

    return {
        "success": True,
        "message": "Upload queued. Note: Full YouTube upload requires OAuth2 setup.",
        "upload_id": len(history)
    }


@router.post("/facebook")
async def upload_to_facebook(request: UploadRequest, background_tasks: BackgroundTasks):
    """Upload video to Facebook"""
    settings = load_settings()

    if not settings.get("facebook_access_token"):
        raise HTTPException(status_code=400, detail="Facebook access token not configured")

    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="File not found")

    history = load_upload_history()
    history.append({
        "id": len(history) + 1,
        "platform": "facebook",
        "file_path": request.file_path,
        "title": request.title,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "error": None
    })
    save_upload_history(history)

    return {
        "success": True,
        "message": "Upload queued. Note: Full Facebook upload requires Page Access Token.",
        "upload_id": len(history)
    }


@router.get("/youtube/status")
async def get_youtube_status():
    """Check YouTube API connection status"""
    settings = load_settings()

    if not settings.get("youtube_api_key"):
        return {
            "connected": False,
            "error": "API key not configured"
        }

    # Test API key with a simple request
    import requests
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={
                "part": "snippet",
                "mine": "true",
                "key": settings["youtube_api_key"]
            },
            timeout=10
        )

        if response.status_code == 200:
            return {
                "connected": True,
                "message": "API key is valid"
            }
        else:
            return {
                "connected": False,
                "error": f"API error: {response.status_code}"
            }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


@router.get("/folders")
async def get_output_folders():
    """Get list of output folders with videos and metadata"""
    folders = scan_output_folders()
    return {
        "folders": folders,
        "count": len(folders),
        "output_dir": OUTPUT_DIR
    }


@router.get("/folder/{folder_name}")
async def get_folder_details(folder_name: str):
    """Get details of a specific output folder"""
    folder_path = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    videos = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        # Video files
        if filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            # Get metadata for this specific video
            video_metadata = get_video_metadata(folder_path, filename)

            videos.append({
                "name": filename,
                "path": filepath,
                "size": os.path.getsize(filepath),
                "size_formatted": f"{os.path.getsize(filepath) / 1024 / 1024:.2f} MB",
                "metadata": video_metadata
            })

    return {
        "folder_name": folder_name,
        "folder_path": folder_path,
        "videos": videos,
        "video_count": len(videos)
    }


@router.post("/folder/upload")
async def upload_folder(request: FolderUploadRequest, background_tasks: BackgroundTasks):
    """Queue all videos in a folder for upload"""
    settings = load_settings()

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    # Get folder details
    folder_name = os.path.basename(request.folder_path)
    videos = []

    for filename in os.listdir(request.folder_path):
        filepath = os.path.join(request.folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            # Get metadata for this specific video
            video_metadata = get_video_metadata(request.folder_path, filename)
            videos.append({
                "name": filename,
                "path": filepath,
                "metadata": video_metadata
            })

    if not videos:
        raise HTTPException(status_code=400, detail="No videos found in folder")

    # Add all videos to upload queue
    history = load_upload_history()
    queued_count = 0

    for video in videos:
        video_name = os.path.splitext(video["name"])[0]
        metadata = video.get("metadata", {})
        title = metadata.get('title', video_name)
        description = metadata.get('description', '')
        tags = metadata.get('tags', [])

        history.append({
            "id": len(history) + 1,
            "platform": request.platform,
            "file_path": video["path"],
            "title": title,
            "description": description,
            "tags": tags,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "error": None,
            "delay_seconds": settings.get("upload_delay_seconds", 60)
        })
        queued_count += 1

    save_upload_history(history)

    return {
        "success": True,
        "message": f"Queued {queued_count} videos for upload",
        "queued_count": queued_count,
        "delay_seconds": settings.get("upload_delay_seconds", 60)
    }


@router.get("/api-key-status")
async def get_api_key_status():
    """Check if API key file exists"""
    file_exists = os.path.exists(YT_API_KEY_PATH)
    key_content = ""
    if file_exists:
        key_content = load_yt_api_key()

    return {
        "file_exists": file_exists,
        "file_path": YT_API_KEY_PATH,
        "key_set": bool(key_content),
        "key_preview": f"...{key_content[-4:]}" if len(key_content) > 4 else "****" if key_content else ""
    }


# =============================================================================
# YouTube OAuth2 Endpoints
# =============================================================================

@router.get("/youtube/oauth-status")
async def get_youtube_oauth_status():
    """Get YouTube OAuth2 authentication status"""
    return youtube_oauth.get_auth_status()


@router.get("/youtube/auth-url")
async def get_youtube_auth_url():
    """Get YouTube OAuth2 authorization URL"""
    auth_url, error = youtube_oauth.get_auth_url()

    if error:
        return {
            "success": False,
            "error": error
        }

    return {
        "success": True,
        "auth_url": auth_url
    }


class OAuthCallbackRequest(BaseModel):
    code: str


@router.post("/youtube/callback")
async def youtube_oauth_callback(request: OAuthCallbackRequest):
    """Handle YouTube OAuth2 callback"""
    success, message = youtube_oauth.handle_callback(request.code)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message
    }


@router.post("/youtube/logout")
async def youtube_logout():
    """Logout from YouTube (clear stored credentials)"""
    youtube_oauth.logout()
    return {"success": True, "message": "Logged out successfully"}


class SingleUploadRequest(BaseModel):
    file_path: str
    title: str
    description: Optional[str] = ""
    tags: Optional[List[str]] = []
    category_id: Optional[str] = "22"
    privacy_status: Optional[str] = "private"


@router.post("/youtube/upload-single")
async def upload_single_to_youtube(request: SingleUploadRequest, background_tasks: BackgroundTasks):
    """Upload a single video to YouTube"""
    # Check authentication
    status = youtube_oauth.get_auth_status()
    if not status["authenticated"]:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login to YouTube first.")

    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="File not found")

    # Add to history as uploading
    history = load_upload_history()
    upload_id = len(history) + 1
    history.append({
        "id": upload_id,
        "platform": "youtube",
        "file_path": request.file_path,
        "title": request.title,
        "description": request.description,
        "tags": request.tags,
        "status": "uploading",
        "created_at": datetime.now().isoformat(),
        "error": None,
        "video_url": None
    })
    save_upload_history(history)

    # Upload in background
    def do_upload():
        result = youtube_oauth.upload_video(
            file_path=request.file_path,
            title=request.title,
            description=request.description,
            tags=request.tags,
            category_id=request.category_id,
            privacy_status=request.privacy_status
        )

        # Update history
        history = load_upload_history()
        for item in history:
            if item["id"] == upload_id:
                if result["success"]:
                    item["status"] = "completed"
                    item["video_url"] = result.get("video_url")
                    item["video_id"] = result.get("video_id")
                else:
                    item["status"] = "failed"
                    item["error"] = result.get("error")
                break
        save_upload_history(history)

    background_tasks.add_task(do_upload)

    return {
        "success": True,
        "message": "Upload started",
        "upload_id": upload_id
    }


@router.post("/youtube/upload-folder")
async def upload_folder_to_youtube(request: FolderUploadRequest, background_tasks: BackgroundTasks):
    """Upload all videos in a folder to YouTube with delay between uploads"""
    # Check authentication
    status = youtube_oauth.get_auth_status()
    if not status["authenticated"]:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login to YouTube first.")

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    settings = load_settings()
    delay_seconds = settings.get("upload_delay_seconds", 60)
    privacy = settings.get("youtube_privacy", "private")
    category = settings.get("youtube_category", "22")

    # Collect videos with metadata
    videos_to_upload = []
    for filename in os.listdir(request.folder_path):
        filepath = os.path.join(request.folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            metadata = get_video_metadata(request.folder_path, filename)
            video_name = os.path.splitext(filename)[0]

            videos_to_upload.append({
                "path": filepath,
                "name": filename,
                "title": metadata.get('title', video_name),
                "description": metadata.get('description', ''),
                "tags": metadata.get('tags', [])
            })

    if not videos_to_upload:
        raise HTTPException(status_code=400, detail="No videos found in folder")

    # Add all to history as queued
    history = load_upload_history()
    upload_ids = []

    for video in videos_to_upload:
        upload_id = len(history) + 1
        upload_ids.append(upload_id)
        history.append({
            "id": upload_id,
            "platform": "youtube",
            "file_path": video["path"],
            "title": video["title"],
            "description": video["description"],
            "tags": video["tags"],
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "error": None,
            "video_url": None
        })

    save_upload_history(history)

    # Upload in background with delay
    def do_batch_upload():
        import time

        for i, (video, upload_id) in enumerate(zip(videos_to_upload, upload_ids)):
            # Update status to uploading
            history = load_upload_history()
            for item in history:
                if item["id"] == upload_id:
                    item["status"] = "uploading"
                    break
            save_upload_history(history)

            # Upload
            result = youtube_oauth.upload_video(
                file_path=video["path"],
                title=video["title"],
                description=video["description"],
                tags=video["tags"],
                category_id=category,
                privacy_status=privacy
            )

            # Update history
            history = load_upload_history()
            for item in history:
                if item["id"] == upload_id:
                    if result["success"]:
                        item["status"] = "completed"
                        item["video_url"] = result.get("video_url")
                        item["video_id"] = result.get("video_id")
                    else:
                        item["status"] = "failed"
                        item["error"] = result.get("error")
                    break
            save_upload_history(history)

            # Wait before next upload (except for last video)
            if i < len(videos_to_upload) - 1:
                time.sleep(delay_seconds)

    background_tasks.add_task(do_batch_upload)

    return {
        "success": True,
        "message": f"Started uploading {len(videos_to_upload)} videos",
        "video_count": len(videos_to_upload),
        "delay_seconds": delay_seconds,
        "upload_ids": upload_ids
    }
