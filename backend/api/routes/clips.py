"""
Clips Management Routes
"""

import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from backend.models.schemas import ClipResponse, ExportRequest
from backend.services.export_service import ExportService
from backend.core.state import app_state
from backend.core.config import settings
from backend.api.routes.video import clear_transcription_cache

router = APIRouter(prefix="/api/clips", tags=["clips"])

export_service = ExportService()


@router.get("", response_model=List[ClipResponse])
async def get_clips():
    """Get all discovered clips"""
    clips = []
    for i, clip in enumerate(app_state.processing.clips):
        clips.append(ClipResponse(
            id=i,
            start=clip['start'],
            end=clip['end'],
            start_formatted=clip['start_formatted'],
            end_formatted=clip['end_formatted'],
            duration=clip['duration'],
            title=clip['title'],
            reason=clip['reason']
        ))
    return clips


async def export_clips_task(clip_indices: List[int]):
    """Background task to export clips"""
    try:
        app_state.processing.is_processing = True
        app_state.processing.current_stage = "exporting"
        app_state.processing.cancel_requested = False

        # Get selected clips
        clips_to_export = [app_state.processing.clips[i] for i in clip_indices if i < len(app_state.processing.clips)]

        if not clips_to_export:
            app_state.add_log("No clips to export", "warning")
            return

        # Create output subdirectory based on video title
        video_title = app_state.processing.video_title or "clips"
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        output_dir = os.path.join(settings.OUTPUT_DIR, safe_title)

        app_state.add_log(f"Exporting {len(clips_to_export)} clips to {output_dir}")

        exported = await export_service.export_clips(
            app_state.processing.video_path,
            clips_to_export,
            output_dir
        )

        # Check if cancelled
        if app_state.processing.cancel_requested:
            app_state.add_log(f"Export cancelled. {len(exported)} clips were saved before cancellation.", "warning")
            await app_state.broadcast('export_cancelled', {
                'count': len(exported),
                'output_dir': output_dir
            })
        else:
            app_state.add_log(f"Export complete! {len(exported)} clips saved.", "success")
            await app_state.broadcast('export_complete', {
                'count': len(exported),
                'output_dir': output_dir,
                'files': [os.path.basename(f) for f in exported]
            })

            # Clear transcription cache after successful export
            clear_transcription_cache()
            app_state.add_log("Transcription cache cleared.")

        app_state.processing.is_processing = False
        app_state.processing.current_stage = "idle"

    except Exception as e:
        app_state.add_log(f"Export error: {str(e)}", "error")
        await app_state.broadcast('error', {'message': str(e)})
        app_state.processing.is_processing = False
        app_state.processing.current_stage = "idle"


@router.post("/export")
async def export_clips(request: ExportRequest, background_tasks: BackgroundTasks):
    """Export selected clips"""
    if app_state.processing.is_processing:
        raise HTTPException(status_code=409, detail="Already processing")

    if not app_state.processing.video_path:
        raise HTTPException(status_code=400, detail="No video loaded")

    if not request.clip_ids:
        raise HTTPException(status_code=400, detail="No clips selected")

    export_service.reset_cancel()
    background_tasks.add_task(export_clips_task, request.clip_ids)

    return {"status": "started", "count": len(request.clip_ids)}


@router.post("/export/cancel")
async def cancel_export():
    """Cancel export"""
    if not app_state.processing.is_processing:
        raise HTTPException(status_code=400, detail="No export in progress")

    export_service.cancel()
    app_state.processing.cancel_requested = True
    app_state.add_log("Export cancellation requested - stopping immediately...", "warning")

    # Set state to idle immediately
    app_state.processing.is_processing = False
    app_state.processing.current_stage = "idle"

    return {"status": "cancelled"}
