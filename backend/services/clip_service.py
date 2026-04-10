"""
Clip Analysis Service
Uses AI to find interesting clips from transcription
"""

import os
import sys
import json
import asyncio
from typing import Optional, Callable, Dict, Any, List

from backend.modules import ClipAnalyzer
from backend.core.config import settings
from backend.core.state import app_state

# Data directory for saving transcriptions
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
os.makedirs(DATA_DIR, exist_ok=True)


class ClipService:
    """Service for clip analysis"""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._init_analyzer()

    def _init_analyzer(self):
        """Initialize analyzer based on selected AI"""
        ai_selected = app_state.settings.get('ai_selected', 'A')

        if ai_selected == 'B':
            # AI B: Gemini raw format
            self.analyzer = ClipAnalyzer(
                api_url=settings.AI_B_URL,
                api_key=settings.AI_B_KEY,
                model=settings.AI_B_MODEL,
                ai_type="B"
            )
        else:
            # AI A: Anthropic format (default)
            self.analyzer = ClipAnalyzer(
                api_url=settings.AI_A_URL,
                api_key=settings.AI_A_KEY,
                model=settings.AI_A_MODEL,
                ai_type="A"
            )

    def _schedule_broadcast(self, event: str, data: dict):
        """Schedule a broadcast from any thread"""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                app_state.broadcast(event, data),
                self._loop
            )

    async def analyze_clips(
        self,
        transcription: Dict[str, Any],
        video_duration: float,
        min_duration: int = 15,
        max_duration: int = 60,
        num_clips: int = 5,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze transcription to find interesting clips"""
        self._loop = asyncio.get_event_loop()

        # Re-init analyzer with current AI settings
        self._init_analyzer()
        ai_selected = app_state.settings.get('ai_selected', 'A')
        app_state.add_log(f"Using AI {ai_selected}: {self.analyzer.model}")

        # Store transcription data for precise cut points
        self.analyzer.set_transcription_data(transcription)

        def sync_callback(percent: float, status: str):
            if progress_callback:
                progress_callback(percent, status)
            app_state.update_progress('analyze', percent, status)
            self._schedule_broadcast('progress', {
                'stage': 'analyze',
                'value': percent,
                'status': status
            })

        # Get AI chunk settings from user's panel settings
        ai_auto_chunk = app_state.settings.get('ai_auto_chunk', True)
        ai_chunk_tokens = app_state.settings.get('ai_chunk_tokens', 4000)  # User's manual setting
        ai_chunk_cooldown = app_state.settings.get('ai_chunk_cooldown', 2)
        auto_clip_count = app_state.settings.get('auto_clip_count', False)

        # Import video service for transcript formatting
        from backend.services.video_service import VideoService
        video_service = VideoService()

        # Calculate total transcript tokens
        transcript_text = video_service.format_transcript_for_analysis(transcription)
        total_chars = len(transcript_text)
        total_tokens = total_chars // 4  # Approx 4 chars = 1 token

        # Adaptive token limits for AUTO mode
        # New API tested: 1 MB (~262k tokens) works
        # Using 250,000 tokens as safe limit
        INITIAL_MAX_TOKENS = 250000  # Safe limit based on 1MB test
        TOKEN_DECREMENT = 50000  # Reduce by 50k on failure
        MIN_TOKENS = 10000

        if ai_auto_chunk:
            # Auto chunk with adaptive token limit
            clips = await self._analyze_with_adaptive_chunking(
                transcription=transcription,
                transcript_text=transcript_text,
                total_tokens=total_tokens,
                video_duration=video_duration,
                video_service=video_service,
                min_duration=min_duration,
                max_duration=max_duration,
                num_clips=num_clips,
                auto_clip_count=auto_clip_count,
                ai_chunk_cooldown=ai_chunk_cooldown,
                initial_max_tokens=INITIAL_MAX_TOKENS,
                token_decrement=TOKEN_DECREMENT,
                min_tokens=MIN_TOKENS,
                sync_callback=sync_callback
            )
        else:
            # Manual mode: use user's chunk tokens setting
            user_max = app_state.settings.get('ai_chunk_tokens', 4000)
            chunk_tokens = user_max if user_max > 0 else 4000
            app_state.add_log(f"AI Manual Chunk: Using {chunk_tokens:,} tokens/chunk")

            clips = await self._analyze_with_chunking(
                transcription=transcription,
                transcript_text=transcript_text,
                total_tokens=total_tokens,
                video_duration=video_duration,
                video_service=video_service,
                chunk_tokens=chunk_tokens,
                min_duration=min_duration,
                max_duration=max_duration,
                num_clips=num_clips,
                auto_clip_count=auto_clip_count,
                ai_chunk_cooldown=ai_chunk_cooldown,
                sync_callback=sync_callback
            )

        # Add IDs to clips
        for i, clip in enumerate(clips):
            clip['id'] = i

        # Delete saved transcription on success
        self._delete_saved_transcription()

        # AUTO PROCESS: If enabled, start export immediately
        if app_state.settings.get('auto_process', False) and clips:
            app_state.add_log(f"Auto Processing enabled: Exporting {len(clips)} clips...")

            from backend.services.export_service import ExportService

            async def run_auto_export():
                await asyncio.sleep(2)
                export_service = ExportService()
                video_path = app_state.processing.video_path
                output_dir = os.path.join(settings.OUTPUT_DIR, os.path.splitext(os.path.basename(video_path))[0])

                app_state.processing.current_stage = "exporting"
                await app_state.broadcast('status', app_state.get_status())

                await export_service.export_clips(video_path, clips, output_dir)

                app_state.processing.is_processing = False
                app_state.processing.current_stage = "idle"
                await app_state.broadcast('export_complete', {})

            asyncio.create_task(run_auto_export())

        return clips

    async def _analyze_with_adaptive_chunking(
        self,
        transcription: Dict[str, Any],
        transcript_text: str,
        total_tokens: int,
        video_duration: float,
        video_service,
        min_duration: int,
        max_duration: int,
        num_clips: int,
        auto_clip_count: bool,
        ai_chunk_cooldown: int,
        initial_max_tokens: int,
        token_decrement: int,
        min_tokens: int,
        sync_callback
    ) -> List[Dict[str, Any]]:
        """
        Adaptive chunking: if single request fails, start chunking.
        First try as single request, then chunk into smaller pieces on failure.
        """
        # Step 1: Try sending as single request first
        try:
            app_state.add_log(f"AI Auto Chunk: Trying single request ({total_tokens:,} tokens)...")

            clips = await self._analyze_with_chunking(
                transcription=transcription,
                transcript_text=transcript_text,
                total_tokens=total_tokens,
                video_duration=video_duration,
                video_service=video_service,
                chunk_tokens=0,  # 0 means send as single request
                min_duration=min_duration,
                max_duration=max_duration,
                num_clips=num_clips,
                auto_clip_count=auto_clip_count,
                ai_chunk_cooldown=ai_chunk_cooldown,
                sync_callback=sync_callback
            )

            app_state.add_log(f"AI Auto Chunk: Success with single request!")
            return clips

        except Exception as e:
            error_str = str(e)
            if "500" not in error_str and "API" not in error_str and "403" not in error_str:
                raise  # Non-API error, don't retry

            app_state.add_log(f"AI Auto Chunk: Single request failed, starting chunked approach...")

        # Step 2: Single request failed - try with progressively smaller chunks
        # Start from initial_max_tokens, then reduce by token_decrement
        chunk_sizes = []
        current_chunk = initial_max_tokens
        while current_chunk >= min_tokens:
            chunk_sizes.append(current_chunk)
            current_chunk = current_chunk - token_decrement

        # Ensure we have at least the minimum chunk size
        if not chunk_sizes:
            chunk_sizes = [min_tokens]
        elif chunk_sizes[-1] > min_tokens:
            chunk_sizes.append(min_tokens)

        last_error = None
        for chunk_size in chunk_sizes:
            try:
                num_chunks = (total_tokens // chunk_size) + 1
                app_state.add_log(f"AI Auto Chunk: Trying {num_chunks} chunks ({chunk_size:,} tokens/chunk)...")

                clips = await self._analyze_with_chunking(
                    transcription=transcription,
                    transcript_text=transcript_text,
                    total_tokens=total_tokens,
                    video_duration=video_duration,
                    video_service=video_service,
                    chunk_tokens=chunk_size,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    num_clips=num_clips,
                    auto_clip_count=auto_clip_count,
                    ai_chunk_cooldown=ai_chunk_cooldown,
                    sync_callback=sync_callback
                )

                app_state.add_log(f"AI Auto Chunk: Success with {chunk_size:,} tokens/chunk!")
                return clips

            except Exception as e:
                last_error = str(e)
                if "500" in last_error or "API" in last_error:
                    app_state.add_log(f"AI Auto Chunk: Failed with {chunk_size:,} tokens/chunk, trying smaller...")
                else:
                    raise  # Non-API error

        # All attempts failed
        self._save_transcription_for_retry(transcription, video_duration)
        raise Exception(f"AI Analysis failed after trying all chunk sizes. Last error: {last_error}")

    async def _analyze_with_chunking(
        self,
        transcription: Dict[str, Any],
        transcript_text: str,
        total_tokens: int,
        video_duration: float,
        video_service,
        chunk_tokens: int,
        min_duration: int,
        max_duration: int,
        num_clips: int,
        auto_clip_count: bool,
        ai_chunk_cooldown: int,
        sync_callback
    ) -> List[Dict[str, Any]]:
        """
        Analyze with specified chunk size.
        If chunk_tokens is 0 or total_tokens <= chunk_tokens, send as one request.
        """
        if chunk_tokens > 0 and total_tokens > chunk_tokens:
            # Use token-based chunked analysis
            chunks = video_service.get_transcript_chunks_by_tokens(transcription, max_tokens=chunk_tokens)
            num_chunks = len(chunks)

            mode_str = "AUTO" if auto_clip_count else f"{num_clips} clips"
            app_state.add_log(f"AI Analysis: {num_chunks} chunks ({chunk_tokens:,} tokens/chunk, {ai_chunk_cooldown}s cooldown, {mode_str})")

            clips = await self._loop.run_in_executor(
                None,
                lambda: self.analyzer.analyze_long_video(
                    chunks,
                    video_duration,
                    min_clip_duration=min_duration,
                    max_clip_duration=max_duration,
                    clips_per_chunk=num_clips,
                    cooldown_seconds=ai_chunk_cooldown,
                    auto_clip_count=auto_clip_count,
                    progress_callback=sync_callback
                )
            )
        else:
            # Send all as one request
            mode_str = "AUTO" if auto_clip_count else f"{num_clips} clips"
            app_state.add_log(f"AI Analysis: Sending full transcript as one request ({total_tokens:,} tokens, {mode_str})")

            clips = await self._loop.run_in_executor(
                None,
                lambda: self.analyzer.analyze(
                    transcript_text,
                    video_duration,
                    min_clip_duration=min_duration,
                    max_clip_duration=max_duration,
                    num_clips=num_clips,
                    auto_clip_count=auto_clip_count,
                    progress_callback=sync_callback
                )
            )

        return clips

    def _get_transcription_path(self) -> str:
        """Get path for saving transcription based on current video"""
        video_path = app_state.processing.video_path or "unknown"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Sanitize filename
        safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        return os.path.join(DATA_DIR, f"transcription_{safe_name}.json")

    def _save_transcription_for_retry(self, transcription: Dict[str, Any], video_duration: float):
        """Save transcription to JSON file for retry later"""
        try:
            transcription_path = self._get_transcription_path()
            save_data = {
                "transcription": transcription,
                "video_duration": video_duration,
                "video_path": app_state.processing.video_path,
                "video_title": app_state.processing.video_title
            }
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            app_state.add_log(f"Transcription saved to: {os.path.basename(transcription_path)}")
        except Exception as e:
            app_state.add_log(f"Failed to save transcription: {e}", "error")

    def _delete_saved_transcription(self):
        """Delete saved transcription after successful AI analysis"""
        try:
            transcription_path = self._get_transcription_path()
            if os.path.exists(transcription_path):
                os.remove(transcription_path)
                app_state.add_log(f"Deleted saved transcription: {os.path.basename(transcription_path)}")
        except Exception as e:
            app_state.add_log(f"Failed to delete transcription: {e}", "warning")

    def load_saved_transcription(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Load saved transcription if available"""
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            transcription_path = os.path.join(DATA_DIR, f"transcription_{safe_name}.json")

            if os.path.exists(transcription_path):
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            app_state.add_log(f"Failed to load saved transcription: {e}", "warning")
        return None

    def has_saved_transcription(self, video_path: str) -> bool:
        """Check if there's a saved transcription for this video"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        transcription_path = os.path.join(DATA_DIR, f"transcription_{safe_name}.json")
        return os.path.exists(transcription_path)
