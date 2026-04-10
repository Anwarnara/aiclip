"""
Settings Routes
"""

import os
import sys
import glob
from fastapi import APIRouter
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.schemas import SettingsModel, SettingsUpdateModel
from backend.core.state import app_state
from backend.core.config import settings as app_config
from backend.modules.subtitle_renderer import get_available_fonts

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsModel)
async def get_settings():
    """Get current settings"""
    return SettingsModel(**app_state.settings)


@router.put("")
async def update_settings(settings: SettingsUpdateModel):
    """Update settings - only provided fields are updated"""
    # Only update fields that are not None
    update_data = {k: v for k, v in settings.model_dump().items() if v is not None}
    if update_data:
        app_state.update_settings(update_data)
    return {"status": "updated", "settings": app_state.settings}


@router.post("/reset")
async def reset_settings():
    """Reset settings to defaults"""
    from backend.core.state import DEFAULT_SETTINGS
    app_state.update_settings(DEFAULT_SETTINGS)
    return {"status": "reset", "settings": app_state.settings}


@router.get("/fonts")
async def get_fonts():
    """Get list of available fonts for subtitles"""
    fonts = get_available_fonts()
    return {"fonts": fonts}


@router.get("/yolo-models")
async def get_yolo_models():
    """Get list of available YOLO models (.pt files) in models directory"""
    models_dir = app_config.MODELS_DIR
    yolo_models = []

    if os.path.exists(models_dir):
        # Scan for .pt files that contain 'yolo' in the name
        for file in os.listdir(models_dir):
            if file.endswith('.pt') and 'yolo' in file.lower():
                yolo_models.append({
                    "name": file,
                    "path": os.path.join(models_dir, file)
                })

    # Sort by name
    yolo_models.sort(key=lambda x: x["name"])

    return {"models": yolo_models}
