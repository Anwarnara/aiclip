"""
Global State Management
"""

import json
import os
import gc
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime


# Settings file path
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "settings.json")

# Default settings
DEFAULT_SETTINGS = {
    "tracking_method": "dlib",
    "yolo_model": "yolov8n-face.pt",  # Selected YOLO model file
    "use_prescan": True,  # True = pre-scan video for stable mode, False = real-time detection
    "face_classifier": False,  # DISABLED - not needed
    "cinematic_mode": False,  # AE-style tracking
    "dynamic_tracking": True, # ON = deep scan for small faces
    "dynamic_focus": False,  # Auto-zoom to active speaker
    "tracking_analyzer": False, # DISABLED - simplified pipeline
    "auto_process": False,  # Auto export after analysis
    "auto_clip_count": False,  # AI decides how many clips
    "smoothing": 0.20,  # Position smoothing (0.05-0.5)
    "tracking_speed": 0.5,  # How fast camera follows face (0.1-1.0)
    "deadzone": 40,
    "confidence": 0.40,
    "single_zoom": 1.0,
    "split_zoom": 1.0,
    "split_screen": True,
    "min_clip_duration": 15,
    "max_clip_duration": 60,
    "clips_to_find": 5,
    # Subtitle settings
    "subtitle_enabled": True,
    "subtitle_font_size": 48,
    "subtitle_font_path": "",  # Custom font path (empty = auto)
    "subtitle_max_words": 5,  # Max words per line
    "subtitle_position": 85,  # 0-100 percentage from top (85 = near bottom)
    "subtitle_style": "uppercase",  # "uppercase" = highlight with CAPS, "bold" = bold highlight
    "subtitle_color": "#FFFFFF",  # Normal text color
    "subtitle_highlight_color": "#FFFF00",  # Highlighted word color (yellow)
    "subtitle_bg_enabled": True,  # Background behind text
    "subtitle_bg_color": "#000000",  # Background color
    "subtitle_bg_opacity": 0.5,  # Background opacity (0-1)
    # AI API settings
    "ai_selected": "A",  # "A" = Anthropic API (Gemini 3 Pro), "B" = Raw Response (Gemini 2.5 Flash)
    "ai_chunk_tokens": 0,  # Max tokens per chunk (0 = send all as one)
    "ai_chunk_cooldown": 2,  # Cooldown in seconds between chunk requests
    "ai_auto_chunk": True,  # Auto chunking enabled
    # Debug settings
    "debug_mode": False,  # Enable basic tracking logs in terminal (minimalist)
    "debug_mode_advanced": False  # Enable detailed/verbose tracking logs in terminal
}


def load_settings() -> Dict:
    """Load settings from file or return defaults"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                return {**DEFAULT_SETTINGS, **saved}
    except Exception:
        pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


@dataclass
class ProcessingState:
    """Processing state for a video"""

    # Video info
    video_path: Optional[str] = None
    video_title: Optional[str] = None
    video_duration: float = 0
    is_local_file: bool = False
    source_url: Optional[str] = None

    # Processing flags
    is_processing: bool = False
    cancel_requested: bool = False
    current_stage: str = "idle"  # idle, downloading, transcribing, analyzing, exporting

    # Progress (0-100 for each stage)
    progress: Dict[str, float] = field(default_factory=lambda: {
        'download': 0,
        'transcribe': 0,
        'analyze': 0,
        'export': 0
    })
    progress_status: Dict[str, str] = field(default_factory=lambda: {
        'download': 'Waiting...',
        'transcribe': 'Waiting...',
        'analyze': 'Waiting...',
        'export': 'Waiting...'
    })

    # Clips
    clips: List[Dict] = field(default_factory=list)

    # Transcription data
    transcription: Optional[Dict] = None

    # Logs
    logs: List[Dict] = field(default_factory=list)


class AppState:
    """Global application state with WebSocket broadcast"""

    def __init__(self):
        self.processing = ProcessingState()
        self.settings = load_settings()
        self.websocket_clients: Set = set()
        self._lock = asyncio.Lock()

    def add_log(self, message: str, level: str = "info", broadcast: bool = True):
        """Add a log message and optionally broadcast to WebSocket clients"""
        # Prevent duplicate consecutive logs
        if self.processing.logs and self.processing.logs[-1]["message"] == message:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        self.processing.logs.append(log_entry)

        # Also print to console for debugging
        print(f"[LOG] {timestamp} [{level}] {message}")

        # Keep only last 200 logs (increased from 100)
        if len(self.processing.logs) > 200:
            self.processing.logs = self.processing.logs[-200:]

        # Schedule broadcast to WebSocket clients (non-blocking)
        if broadcast and self.websocket_clients:
            self._schedule_log_broadcast(log_entry)

    def _schedule_log_broadcast(self, log_entry: dict):
        """Schedule a log broadcast from sync context"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self.broadcast('log', log_entry),
                loop
            )
        except RuntimeError:
            # No running loop - try to get event loop from any thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast('log', log_entry),
                        loop
                    )
            except:
                # Still no luck, skip broadcast (log is still saved)
                pass

    def clear_logs(self):
        """Clear all logs"""
        self.processing.logs = []
        self.add_log("Log cleared.")

    def update_progress(self, stage: str, value: float, status: str):
        """Update progress for a stage"""
        self.processing.progress[stage] = value
        self.processing.progress_status[stage] = status

    def reset_progress(self):
        """Reset all progress"""
        for key in self.processing.progress:
            self.processing.progress[key] = 0
            self.processing.progress_status[key] = 'Waiting...'

    def set_clips(self, clips: List[Dict]):
        """Set discovered clips"""
        self.processing.clips = clips

    def clear_clips(self):
        """Clear all clips"""
        self.processing.clips = []

    def update_setting(self, key: str, value: Any):
        """Update a setting and save"""
        self.settings[key] = value
        save_settings(self.settings)

    def update_settings(self, new_settings: Dict):
        """Update multiple settings"""
        self.settings.update(new_settings)
        save_settings(self.settings)

    def reset(self):
        """Reset processing state"""
        self.processing = ProcessingState()

    def full_cleanup(self):
        """
        Full memory cleanup after all clips are processed.
        Call this after export is complete (e.g., 6/6 clips done).
        Clears all cached data and forces garbage collection.
        """
        # Clear processing state
        self.processing.video_path = None
        self.processing.video_title = None
        self.processing.video_duration = 0
        self.processing.source_url = None
        self.processing.is_processing = False
        self.processing.cancel_requested = False
        self.processing.current_stage = "idle"

        # Clear transcription data (can be large)
        self.processing.transcription = None

        # Clear clips
        self.processing.clips = []

        # Reset progress
        self.reset_progress()

        # Keep only last 50 logs after cleanup
        if len(self.processing.logs) > 50:
            self.processing.logs = self.processing.logs[-50:]

        # Force garbage collection
        gc.collect()

        self.add_log("Memory cleanup completed", "info")

    def get_status(self) -> Dict:
        """Get current status for API"""
        return {
            "is_processing": self.processing.is_processing,
            "current_stage": self.processing.current_stage,
            "video_title": self.processing.video_title,
            "video_duration": self.processing.video_duration,
            "progress": self.processing.progress,
            "progress_status": self.processing.progress_status,
            "clips_count": len(self.processing.clips)
        }

    async def broadcast(self, event: str, data: Any):
        """Broadcast event to all WebSocket clients"""
        if not self.websocket_clients:
            return

        message = json.dumps({"event": event, "data": data})
        disconnected = set()

        for ws in self.websocket_clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)

        # Remove disconnected clients
        self.websocket_clients -= disconnected


# Global state instance
app_state = AppState()
