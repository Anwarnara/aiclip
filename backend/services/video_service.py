"""
Video Processing Service
Handles video download and transcription
"""

import os
import sys
import asyncio
from typing import Optional, Callable, Dict, Any

from backend.modules import YouTubeDownloader, WhisperTranscriber
from backend.core.config import settings
from backend.core.state import app_state


class VideoService:
    """Service for video download and transcription"""

    def __init__(self):
        self.downloader = YouTubeDownloader(settings.TEMP_DIR)
        self.transcriber: Optional[WhisperTranscriber] = None
        self._cancel_flag = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def cancel(self):
        """Request cancellation"""
        self._cancel_flag = True
        app_state.processing.cancel_requested = True

    def reset_cancel(self):
        """Reset cancellation flag"""
        self._cancel_flag = False
        app_state.processing.cancel_requested = False

    def _schedule_broadcast(self, event: str, data: dict):
        """Schedule a broadcast from any thread"""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                app_state.broadcast(event, data),
                self._loop
            )

    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information from YouTube URL"""
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, self.downloader.get_video_info, url)
        return info

    async def download_video(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """Download video from YouTube"""
        self._loop = asyncio.get_event_loop()

        def sync_callback(percent: float, status: str):
            if progress_callback:
                progress_callback(percent, status)
            app_state.update_progress('download', percent, status)
            self._schedule_broadcast('progress', {
                'stage': 'download',
                'value': percent,
                'status': status
            })

        video_path = await self._loop.run_in_executor(
            None,
            lambda: self.downloader.download(url, sync_callback)
        )

        return video_path

    async def transcribe_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Transcribe video audio using Whisper"""
        self._loop = asyncio.get_event_loop()

        # Initialize transcriber if needed
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber(
                model_name=settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE,
                model_path=settings.MODELS_DIR
            )

        def sync_log(msg: str):
            app_state.add_log(msg)
            self._schedule_broadcast('log', {
                'message': msg,
                'timestamp': app_state.processing.logs[-1]['timestamp'] if app_state.processing.logs else ''
            })

        def sync_callback(percent: float, status: str):
            if progress_callback:
                progress_callback(percent, status)
            app_state.update_progress('transcribe', percent, status)
            self._schedule_broadcast('progress', {
                'stage': 'transcribe',
                'value': percent,
                'status': status
            })

        # Load model
        await self._loop.run_in_executor(None, self.transcriber.load_model, sync_log)

        # Transcribe
        result = await self._loop.run_in_executor(
            None,
            lambda: self.transcriber.transcribe(video_path, progress_callback=sync_callback)
        )

        return result

    def format_transcript_for_analysis(self, transcription: Dict[str, Any]) -> str:
        """Format transcription for AI analysis"""
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber()
        return self.transcriber.format_transcript_for_ai(transcription)

    def get_transcript_chunks(self, transcription: Dict[str, Any], chunk_minutes: int = 5):
        """Split transcript into chunks for long videos"""
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber()
        return self.transcriber.format_transcript_chunks(transcription, chunk_minutes)

    def get_transcript_chunks_by_tokens(self, transcription: Dict[str, Any], max_tokens: int = 4000):
        """Split transcript into chunks based on token count"""
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber()
        return self.transcriber.format_transcript_chunks_by_tokens(transcription, max_tokens)

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber()
        return self.transcriber.get_device_info()

    def unload_models(self):
        """Unload models and free VRAM"""
        if self.transcriber:
            self.transcriber.unload_model()
