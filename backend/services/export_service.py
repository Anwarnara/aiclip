"""
Export Service
Handles clip export with face tracking
"""

import os
import sys
import asyncio
import requests
import json
import re
from typing import Optional, Callable, Dict, Any, List

from backend.modules import VideoProcessor
from backend.modules.face_tracker import FaceTracker
from backend.modules.face_classifier import FaceClassifier
from backend.core.config import settings
from backend.core.state import app_state


class ExportService:
    """Service for exporting clips with face tracking"""

    def __init__(self):
        self.processor: Optional[VideoProcessor] = None
        self.face_classifier: Optional[FaceClassifier] = None
        self._cancel_flag = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_video_hash: Optional[str] = None  # Track which video was trained

    def cancel(self):
        """Request cancellation"""
        self._cancel_flag = True
        app_state.processing.cancel_requested = True
        # Also cancel processor if it exists
        if self.processor:
            self.processor.cancel()

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

    def _get_processor(self) -> VideoProcessor:
        """Get or create video processor with current settings"""
        s = app_state.settings

        # Cancel checker that checks our flag
        def cancel_checker():
            return self._cancel_flag or app_state.processing.cancel_requested

        # Logging wrapper - ALWAYS logs to app_state
        def log_wrapper(msg):
            app_state.add_log(msg)

        # Log current settings
        prescan_enabled = s.get('use_prescan', True)
        split_enabled = s.get('split_screen', True)
        subtitle_enabled = s.get('subtitle_enabled', True)
        app_state.add_log(f"Processor settings: prescan={prescan_enabled}, split_screen={split_enabled}, subtitle={subtitle_enabled}")

        # Always recreate processor to ensure fresh settings
        yolo_model_name = s.get('yolo_model', 'yolov8n-face.pt')
        self.processor = VideoProcessor(
            output_width=settings.OUTPUT_WIDTH,
            output_height=settings.OUTPUT_HEIGHT,
            yolo_model_path=os.path.join(settings.MODELS_DIR, yolo_model_name),
            classifier_model_path=os.path.join(settings.MODELS_DIR, "face_classifier.pt"),
            confidence=s.get('confidence', 0.5),
            smoothing=s.get('smoothing', 0.2),
            tracking_speed=s.get('tracking_speed', 0.5),
            detection_interval=5,
            enable_split_screen=split_enabled,
            face_padding_single=s.get('single_zoom', 1.0),
            face_padding_split=s.get('split_zoom', 1.0),
            use_prescan=prescan_enabled,
            cinematic_mode=s.get('cinematic_mode', False),
            tracking_method=s.get('tracking_method', 'yolo'),
            deadzone=s.get('deadzone', 40),
            dynamic_tracking=s.get('dynamic_tracking', True),
            tracking_analyzer=s.get('tracking_analyzer', True),
            dynamic_focus=s.get('dynamic_focus', False),
            # Subtitle settings
            subtitle_enabled=subtitle_enabled,
            subtitle_font_size=s.get('subtitle_font_size', 48),
            subtitle_font_path=s.get('subtitle_font_path', ''),
            subtitle_max_words=s.get('subtitle_max_words', 5),
            subtitle_position=s.get('subtitle_position', 85),
            subtitle_style=s.get('subtitle_style', 'uppercase'),
            subtitle_color=s.get('subtitle_color', '#FFFFFF'),
            subtitle_highlight_color=s.get('subtitle_highlight_color', '#FFFF00'),
            subtitle_bg_enabled=s.get('subtitle_bg_enabled', True),
            subtitle_bg_color=s.get('subtitle_bg_color', '#000000'),
            subtitle_bg_opacity=s.get('subtitle_bg_opacity', 0.5),
            cancel_checker=cancel_checker,
            log_callback=log_wrapper,
            debug_mode=s.get('debug_mode', False),
            debug_mode_advanced=s.get('debug_mode_advanced', False)
        )

        return self.processor

    def _get_video_hash(self, video_path: str) -> str:
        """Get a simple hash of video path + size for caching"""
        try:
            stat = os.stat(video_path)
            return f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        except:
            return video_path

    def _generate_seo_content(
        self,
        clip_title: str,
        clip_reason: str,
        clip_transcript: str
    ) -> Dict[str, Any]:
        """
        Generate SEO-friendly title, description, and tags using AI

        Returns:
            Dict with 'title', 'description', 'tags' keys
        """
        api_url = "https://ultyweb.com/account/v1/chat/completions"
        api_key = "gam_master_u7w3k9x2m5q8r1t4y6p0s3v8n2b5j7h"

        system_prompt = """Kamu adalah SEO expert untuk konten TikTok/Reels/Shorts.

TUGAS: Buat konten SEO untuk clip video pendek.

OUTPUT FORMAT (JSON):
{
    "title": "Judul yang catchy, clickbait tapi tidak misleading (max 100 karakter)",
    "description": "Deskripsi menarik yang menjelaskan isi video (max 300 karakter, include call to action)",
    "tags": ["tag1", "tag2", "tag3", ...] (10-15 hashtags relevan tanpa simbol #)
}

TIPS SEO:
1. Title: Gunakan kata-kata power (Rahasia, Ternyata, Gila, Kaget, dll)
2. Description: Mulai dengan hook, jelaskan value, akhiri dengan CTA
3. Tags: Mix antara broad tags (viral, fyp) dan specific tags (topik konten)

OUTPUT HANYA JSON, tanpa penjelasan."""

        user_prompt = f"""CLIP INFO:
- Judul original: {clip_title}
- Alasan menarik: {clip_reason}
- Transkrip clip:
{clip_transcript[:500]}

Buatkan SEO content untuk clip ini."""

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "account_id": "auto"
            }

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                seo_data = json.loads(json_match.group())
                return {
                    'title': seo_data.get('title', clip_title),
                    'description': seo_data.get('description', clip_reason),
                    'tags': seo_data.get('tags', [])
                }
            else:
                raise Exception("No JSON found in response")

        except Exception as e:
            app_state.add_log(f"SEO generation failed: {e}", "warning")
            # Fallback to basic content
            return {
                'title': clip_title,
                'description': clip_reason,
                'tags': ['viral', 'fyp', 'foryou', 'trending']
            }

    def _get_clip_transcript(self, start_time: float, end_time: float) -> str:
        """Extract transcript text for a specific time range"""
        if not app_state.processing.transcription:
            return ""

        transcript_parts = []
        for segment in app_state.processing.transcription.get('segments', []):
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)

            # Check if segment overlaps with clip time range
            if seg_end >= start_time and seg_start <= end_time:
                transcript_parts.append(segment.get('text', '').strip())

        return ' '.join(transcript_parts)

    def _write_seo_file(self, output_path: str, seo_content: Dict[str, Any]):
        """Write SEO content to companion .txt file"""
        txt_path = output_path.rsplit('.', 1)[0] + '.txt'

        tags_formatted = ' '.join([f'#{tag}' for tag in seo_content.get('tags', [])])

        content = f"""=== SEO CONTENT FOR VIDEO ===

📌 JUDUL (Title):
{seo_content.get('title', '')}

📝 DESKRIPSI (Description):
{seo_content.get('description', '')}

🏷️ HASHTAGS:
{tags_formatted}

=== COPY-PASTE READY ===

{seo_content.get('title', '')}

{seo_content.get('description', '')}

{tags_formatted}
"""

        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            app_state.add_log(f"SEO file created: {os.path.basename(txt_path)}")
        except Exception as e:
            app_state.add_log(f"Failed to write SEO file: {e}", "warning")

    async def _train_classifier_for_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Train face classifier for this specific video to distinguish
        human faces from posters/static images.

        Training happens once per video (cached by video hash).
        """
        video_hash = self._get_video_hash(video_path)

        # Skip if already trained for this video
        if self._current_video_hash == video_hash and self.face_classifier and self.face_classifier.is_loaded:
            if progress_callback:
                progress_callback("Using cached face classifier")
            return True

        if progress_callback:
            progress_callback("Training face classifier for this video...")

        # Create classifier with model path
        classifier_path = os.path.join(settings.MODELS_DIR, "face_classifier.pt")
        self.face_classifier = FaceClassifier(model_path=classifier_path)

        # Create a temporary face tracker for training
        yolo_model_name = app_state.settings.get('yolo_model', 'yolov8n-face.pt')
        yolo_path = os.path.join(settings.MODELS_DIR, yolo_model_name)
        temp_tracker = FaceTracker(model_path=yolo_path)

        def train_progress(percent, msg):
            if progress_callback:
                progress_callback(f"Training classifier: {msg}")
            self._schedule_broadcast('training_progress', {
                'percent': percent,
                'status': msg
            })

        try:
            # Run training in executor (CPU-bound)
            success = await self._loop.run_in_executor(
                None,
                lambda: self.face_classifier.auto_train_from_video(
                    video_path,
                    temp_tracker,
                    max_samples=100,
                    progress_callback=train_progress
                )
            )

            if success:
                self._current_video_hash = video_hash
                if progress_callback:
                    progress_callback("Face classifier trained successfully")
                app_state.add_log("Face classifier trained for video")
                return True
            else:
                if progress_callback:
                    progress_callback("Classifier training skipped (not enough data)")
                return False

        except Exception as e:
            app_state.add_log(f"Classifier training failed: {e}", "warning")
            if progress_callback:
                progress_callback(f"Training failed: {e}")
            return False

    async def export_clips(
        self,
        video_path: str,
        clips: List[Dict[str, Any]],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> List[str]:
        """
        Export multiple clips

        Args:
            video_path: Source video path
            clips: List of clip dictionaries with start, end, title
            output_dir: Output directory
            progress_callback: Callback(current, total, clip_title, status)

        Returns:
            List of exported file paths
        """
        self._loop = asyncio.get_event_loop()
        self._cancel_flag = False  # Reset cancel flag at start

        os.makedirs(output_dir, exist_ok=True)

        # Get settings
        s = app_state.settings

        # STEP 1: Get processor (classifier training is DISABLED - using motion detection)
        # Motion-based detection is now built into prescan, no need for separate classifier
        app_state.add_log("Using motion-based face detection (no classifier training needed)")

        processor = self._get_processor()
        processor.reset_cancel()  # Reset processor cancel flag

        # Classifier is disabled - motion detection handles human vs poster
        processor.set_classifier(None)

        # Extract word timestamps from transcription for subtitle sync
        word_timestamps = []
        if app_state.processing.transcription:
            transcription = app_state.processing.transcription
            for segment in transcription.get('segments', []):
                for word in segment.get('words', []):
                    word_timestamps.append({
                        'word': word.get('word', '').strip(),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0)
                    })
            if word_timestamps:
                app_state.add_log(f"Subtitle: Loaded {len(word_timestamps)} words from transcription")

        # Generate speaker timeline for dynamic focus if enabled
        speaker_timeline = []
        if s.get('dynamic_focus', False) and app_state.processing.transcription:
            try:
                from backend.modules.transcriber import WhisperTranscriber
                temp_transcriber = WhisperTranscriber()
                speaker_timeline = temp_transcriber.analyze_speaker_timeline(
                    video_path,
                    app_state.processing.transcription
                )
                if speaker_timeline:
                    app_state.add_log(f"Dynamic Focus: Analyzed {len(speaker_timeline)} speaker segments")
            except Exception as e:
                app_state.add_log(f"Dynamic Focus: Speaker analysis failed - {e}", "warning")

        exported_files = []
        total = len(clips)

        for i, clip in enumerate(clips):
            if self._cancel_flag:
                app_state.add_log("Export cancelled by user", "warning")
                break

            current = i + 1
            clip_title = clip.get('title', f'Clip {current}')

            # Clean filename
            safe_title = "".join(c for c in clip_title if c.isalnum() or c in (' ', '-', '_')).strip()
            output_path = os.path.join(output_dir, f"{safe_title}_{current}.mp4")

            # Capture current values for closure
            current_idx = current
            total_clips = total
            title = clip_title
            clip_idx = i

            def make_sync_callback(idx, tot, ttl, cidx):
                def sync_callback(percent: float, status: str):
                    if progress_callback:
                        progress_callback(idx, tot, ttl, status)
                    app_state.update_progress('export', (cidx * 100 + percent) / tot, f"{ttl}: {status}")
                    self._schedule_broadcast('export_progress', {
                        'current': idx,
                        'total': tot,
                        'clip_title': ttl,
                        'status': status,
                        'percent': percent
                    })
                return sync_callback

            sync_callback = make_sync_callback(current_idx, total_clips, title, clip_idx)

            try:
                app_state.add_log(f"Exporting clip {current}/{total}: {clip_title}")

                # Set word timestamps for subtitle sync (relative to this clip's start time)
                if word_timestamps:
                    processor.set_word_timestamps(word_timestamps, clip['start'])

                # Set speaker timeline for dynamic focus (relative to this clip's start time)
                if speaker_timeline:
                    processor.set_speaker_timeline(speaker_timeline, clip['start'])

                await self._loop.run_in_executor(
                    None,
                    lambda op=output_path, sc=sync_callback: processor.process_clip_with_audio(
                        video_path,
                        op,
                        clip['start'],
                        clip['end'],
                        sc
                    )
                )

                exported_files.append(output_path)
                app_state.add_log(f"Exported: {os.path.basename(output_path)}")

                # Generate SEO content and write companion .txt file
                try:
                    sync_callback(95, "Generating SEO content...")
                    app_state.add_log(f"Generating SEO for: {output_path}")
                    clip_transcript = self._get_clip_transcript(clip['start'], clip['end'])
                    app_state.add_log(f"Transcript length: {len(clip_transcript)} chars")
                    seo_content = await self._loop.run_in_executor(
                        None,
                        lambda: self._generate_seo_content(
                            clip_title,
                            clip.get('reason', 'Interesting content'),
                            clip_transcript
                        )
                    )
                    app_state.add_log(f"SEO content generated: {seo_content.get('title', 'NO TITLE')[:50]}")
                    self._write_seo_file(output_path, seo_content)
                except Exception as e:
                    app_state.add_log(f"SEO generation error: {e}", "warning")

            except Exception as e:
                app_state.add_log(f"Error exporting {clip_title}: {str(e)}", "error")
                self._schedule_broadcast('error', {
                    'message': f"Error exporting {clip_title}: {str(e)}"
                })

        # === MEMORY CLEANUP AFTER ALL CLIPS EXPORTED ===
        app_state.add_log(f"Export complete: {len(exported_files)}/{total} clips exported")

        # Clear processor reference to free VRAM/RAM
        if self.processor:
            self.processor = None

        # Clear face classifier cache
        self.face_classifier = None
        self._current_video_hash = None

        # Full app state cleanup (clears transcription, clips, forces gc.collect())
        app_state.full_cleanup()

        return exported_files
