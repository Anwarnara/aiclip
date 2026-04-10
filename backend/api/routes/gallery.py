"""
Gallery API Routes
File Manager functionality for output and download folders
"""

import os
import shutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from backend.core.config import settings

router = APIRouter(prefix="/api/gallery", tags=["gallery"])

# Allowed base directories for security
ALLOWED_DIRS = []


def get_allowed_dirs():
    """Get allowed directories including output subfolders"""
    global ALLOWED_DIRS
    ALLOWED_DIRS = [
        os.path.abspath(settings.OUTPUT_DIR),
        os.path.abspath(settings.DOWNLOAD_DIR),
        os.path.abspath(os.path.join(os.path.dirname(settings.OUTPUT_DIR), "data"))
    ]
    return ALLOWED_DIRS


def is_path_allowed(path: str) -> bool:
    """Check if path is within allowed directories"""
    abs_path = os.path.abspath(path)
    allowed = get_allowed_dirs()
    return any(abs_path.startswith(d) for d in allowed)


class FileItem(BaseModel):
    name: str
    path: str
    size: int
    size_formatted: str
    created: str
    modified: str
    type: str  # folder, video, audio, image, text, other
    is_dir: bool = False
    children_count: int = 0


class FolderItem(BaseModel):
    name: str
    path: str
    is_dir: bool = True
    children_count: int = 0
    modified: str


class RenameRequest(BaseModel):
    old_path: str
    new_name: str


class MoveRequest(BaseModel):
    source_path: str
    dest_folder: str


class CopyRequest(BaseModel):
    source_path: str
    dest_folder: str


class CreateFolderRequest(BaseModel):
    parent_path: str
    folder_name: str


class OpenFolderRequest(BaseModel):
    path: str


def get_file_type(filename: str, is_dir: bool = False) -> str:
    """Determine file type from extension"""
    if is_dir:
        return 'folder'

    ext = filename.lower().split('.')[-1] if '.' in filename else ''

    video_exts = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv', 'wmv']
    audio_exts = ['mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg']
    image_exts = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']
    text_exts = ['txt', 'json', 'md', 'srt', 'vtt', 'log']

    if ext in video_exts:
        return 'video'
    elif ext in audio_exts:
        return 'audio'
    elif ext in image_exts:
        return 'image'
    elif ext in text_exts:
        return 'text'
    return 'other'


def format_size(size: int) -> str:
    """Format file size to readable format"""
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    elif size >= 1_000_000:
        return f"{size / 1_000_000:.2f} MB"
    elif size >= 1_000:
        return f"{size / 1_000:.2f} KB"
    return f"{size} B"


def list_directory(directory: str, file_types: List[str] = None, include_folders: bool = True) -> List[FileItem]:
    """List files and folders in a directory"""
    items = []

    if not os.path.exists(directory):
        return items

    for name in os.listdir(directory):
        filepath = os.path.join(directory, name)
        is_dir = os.path.isdir(filepath)

        if is_dir:
            if not include_folders:
                continue
            try:
                children = len(os.listdir(filepath))
            except:
                children = 0
        else:
            children = 0

        file_type = get_file_type(name, is_dir)

        # Filter by type if specified (folders always included)
        if file_types and not is_dir and file_type not in file_types:
            continue

        try:
            stat = os.stat(filepath)
            items.append(FileItem(
                name=name,
                path=filepath,
                size=stat.st_size if not is_dir else 0,
                size_formatted=format_size(stat.st_size) if not is_dir else f"{children} items",
                created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                type=file_type,
                is_dir=is_dir,
                children_count=children
            ))
        except Exception:
            continue

    # Sort: folders first, then by name
    items.sort(key=lambda x: (not x.is_dir, x.name.lower()))

    return items


@router.get("/browse")
async def browse_directory(path: Optional[str] = None, show_all: bool = False):
    """Browse a directory - returns files and folders"""
    if not path:
        # Return root directories
        return {
            "current_path": None,
            "parent_path": None,
            "items": [
                {
                    "name": "Output",
                    "path": settings.OUTPUT_DIR,
                    "is_dir": True,
                    "type": "folder",
                    "size_formatted": "",
                    "children_count": len(os.listdir(settings.OUTPUT_DIR)) if os.path.exists(settings.OUTPUT_DIR) else 0
                },
                {
                    "name": "Download",
                    "path": settings.DOWNLOAD_DIR,
                    "is_dir": True,
                    "type": "folder",
                    "size_formatted": "",
                    "children_count": len(os.listdir(settings.DOWNLOAD_DIR)) if os.path.exists(settings.DOWNLOAD_DIR) else 0
                }
            ]
        }

    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Path not found")

    # Get parent path
    parent = os.path.dirname(path)
    if not is_path_allowed(parent):
        parent = None

    file_types = None if show_all else ['video', 'audio', 'image', 'text']
    items = list_directory(path, file_types)

    return {
        "current_path": path,
        "parent_path": parent,
        "folder_name": os.path.basename(path),
        "items": [item.dict() for item in items],
        "count": len(items)
    }


@router.get("/output")
async def get_output_files(type: Optional[str] = None):
    """Get files in output folder (legacy endpoint)"""
    file_types = [type] if type else ['video', 'audio']
    items = list_directory(settings.OUTPUT_DIR, file_types, include_folders=True)
    return {
        "folder": "output",
        "path": settings.OUTPUT_DIR,
        "files": [f.dict() for f in items],
        "count": len(items)
    }


@router.get("/download")
async def get_download_files(type: Optional[str] = None):
    """Get files in download folder (legacy endpoint)"""
    file_types = [type] if type else ['video', 'audio']
    items = list_directory(settings.DOWNLOAD_DIR, file_types, include_folders=True)
    return {
        "folder": "download",
        "path": settings.DOWNLOAD_DIR,
        "files": [f.dict() for f in items],
        "count": len(items)
    }


@router.get("/file")
async def get_file(path: str):
    """Get a file by path"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(path)


@router.get("/file-content")
async def get_file_content(path: str):
    """Get text file content"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Access denied")

    # Only allow text files
    if get_file_type(path) != 'text':
        raise HTTPException(status_code=400, detail="Only text files can be read")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"success": True, "content": content, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file-content")
async def save_file_content(path: str, content: str):
    """Save text file content"""
    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/file")
async def delete_file(path: str):
    """Delete a file or folder"""
    if not os.path.exists(path):
        return {"success": False, "error": "File not found"}

    if not is_path_allowed(path):
        return {"success": False, "error": "Access denied"}

    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/rename")
async def rename_file(request: RenameRequest):
    """Rename a file or folder"""
    if not os.path.exists(request.old_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_path_allowed(request.old_path):
        raise HTTPException(status_code=403, detail="Access denied")

    # Validate new name
    if '/' in request.new_name or '\\' in request.new_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    parent_dir = os.path.dirname(request.old_path)
    new_path = os.path.join(parent_dir, request.new_name)

    if os.path.exists(new_path):
        raise HTTPException(status_code=400, detail="A file with this name already exists")

    try:
        os.rename(request.old_path, new_path)
        return {"success": True, "new_path": new_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/move")
async def move_file(request: MoveRequest):
    """Move a file or folder to another location"""
    if not os.path.exists(request.source_path):
        raise HTTPException(status_code=404, detail="Source not found")

    if not is_path_allowed(request.source_path):
        raise HTTPException(status_code=403, detail="Access denied to source")

    if not is_path_allowed(request.dest_folder):
        raise HTTPException(status_code=403, detail="Access denied to destination")

    if not os.path.isdir(request.dest_folder):
        raise HTTPException(status_code=400, detail="Destination must be a folder")

    filename = os.path.basename(request.source_path)
    dest_path = os.path.join(request.dest_folder, filename)

    if os.path.exists(dest_path):
        raise HTTPException(status_code=400, detail="A file with this name already exists in destination")

    try:
        shutil.move(request.source_path, dest_path)
        return {"success": True, "new_path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy")
async def copy_file(request: CopyRequest):
    """Copy a file or folder to another location"""
    if not os.path.exists(request.source_path):
        raise HTTPException(status_code=404, detail="Source not found")

    if not is_path_allowed(request.source_path):
        raise HTTPException(status_code=403, detail="Access denied to source")

    if not is_path_allowed(request.dest_folder):
        raise HTTPException(status_code=403, detail="Access denied to destination")

    if not os.path.isdir(request.dest_folder):
        raise HTTPException(status_code=400, detail="Destination must be a folder")

    filename = os.path.basename(request.source_path)
    dest_path = os.path.join(request.dest_folder, filename)

    # If exists, add number suffix
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(request.dest_folder, f"{base} ({counter}){ext}")
            counter += 1

    try:
        if os.path.isdir(request.source_path):
            shutil.copytree(request.source_path, dest_path)
        else:
            shutil.copy2(request.source_path, dest_path)
        return {"success": True, "new_path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-folder")
async def create_folder(request: CreateFolderRequest):
    """Create a new folder"""
    if not is_path_allowed(request.parent_path):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.isdir(request.parent_path):
        raise HTTPException(status_code=400, detail="Parent must be a folder")

    # Validate folder name
    if '/' in request.folder_name or '\\' in request.folder_name:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    new_path = os.path.join(request.parent_path, request.folder_name)

    if os.path.exists(new_path):
        raise HTTPException(status_code=400, detail="Folder already exists")

    try:
        os.makedirs(new_path)
        return {"success": True, "path": new_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/open-folder")
async def open_folder(request: OpenFolderRequest):
    """Open folder in system file explorer"""
    import subprocess
    import platform

    path = request.path

    # Legacy support for folder names
    if path == "output":
        path = settings.OUTPUT_DIR
    elif path == "download":
        path = settings.DOWNLOAD_DIR

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    try:
        if platform.system() == "Windows":
            subprocess.run(["explorer", path])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux
            subprocess.run(["xdg-open", path])
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/open-file")
async def open_file(request: OpenFolderRequest):
    """Open file with default system application"""
    import subprocess
    import platform

    path = request.path

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux
            subprocess.run(["xdg-open", path])
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
