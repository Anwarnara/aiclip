"""
Auto Process Service
Manages auto-processing settings and tracks processed videos
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Data paths
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
SETTINGS_PATH = os.path.join(DATA_DIR, "auto_process_settings.json")
PROCESSED_VIDEOS_PATH = os.path.join(DATA_DIR, "processed_videos.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# Default settings
DEFAULT_SETTINGS = {
    "enabled": False,
    "resolution": "1080",  # 720, 1080, 1440, 2160 (4K)
    "check_interval_hours": 6,  # Check every X hours
    "active_hours_start": 8,  # Start checking at 8 AM
    "active_hours_end": 22,  # Stop checking at 10 PM
    "processing_priority": "ai_recommendation",  # ai_recommendation, engagement, viral_potential, most_views, random
    "genres": ["0"],  # List of category IDs, ["0"] = All
    "duration_filter": "medium",  # any, short, medium, long
    "region_code": "ID",  # Country code
    "max_videos_per_run": 5,  # Max videos to process per run
    "max_downloads_per_scan": 10,  # Max videos to download per scan
    "search_query": "",  # Search query for auto scraping
    "skip_processed": True,  # Skip already processed videos
    "show_completed_label": True,  # Show "completed" label on processed video thumbnails
}


class AutoProcessService:
    """Service for managing auto-processing configuration"""

    def __init__(self):
        self.settings = self._load_settings()
        self.processed_videos = self._load_processed_videos()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**DEFAULT_SETTINGS, **saved}
        except Exception as e:
            print(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS.copy()

    def _save_settings(self) -> bool:
        """Save settings to file"""
        try:
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def _load_processed_videos(self) -> Dict[str, Dict[str, Any]]:
        """Load processed videos list from file"""
        try:
            if os.path.exists(PROCESSED_VIDEOS_PATH):
                with open(PROCESSED_VIDEOS_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading processed videos: {e}")
        return {}

    def _save_processed_videos(self) -> bool:
        """Save processed videos to file"""
        try:
            with open(PROCESSED_VIDEOS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.processed_videos, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving processed videos: {e}")
            return False

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self.settings.copy()

    def update_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings"""
        for key, value in updates.items():
            if key in DEFAULT_SETTINGS:
                self.settings[key] = value
        self._save_settings()
        return self.settings.copy()

    def reset_settings(self) -> Dict[str, Any]:
        """Reset settings to defaults"""
        self.settings = DEFAULT_SETTINGS.copy()
        self._save_settings()
        return self.settings.copy()

    def mark_video_processed(
        self,
        video_id: str,
        title: str,
        channel: str,
        output_path: Optional[str] = None,
        clips_count: int = 0
    ) -> bool:
        """Mark a video as processed"""
        self.processed_videos[video_id] = {
            "title": title,
            "channel": channel,
            "processed_at": datetime.now().isoformat(),
            "output_path": output_path,
            "clips_count": clips_count
        }
        return self._save_processed_videos()

    def unmark_video_processed(self, video_id: str) -> bool:
        """Remove a video from processed list"""
        if video_id in self.processed_videos:
            del self.processed_videos[video_id]
            return self._save_processed_videos()
        return False

    def is_video_processed(self, video_id: str) -> bool:
        """Check if a video has been processed"""
        return video_id in self.processed_videos

    def get_processed_videos(self) -> Dict[str, Dict[str, Any]]:
        """Get all processed videos"""
        return self.processed_videos.copy()

    def get_processed_video_ids(self) -> List[str]:
        """Get list of processed video IDs"""
        return list(self.processed_videos.keys())

    def clear_processed_videos(self) -> bool:
        """Clear all processed videos"""
        self.processed_videos = {}
        return self._save_processed_videos()

    def get_processed_count(self) -> int:
        """Get count of processed videos"""
        return len(self.processed_videos)


# Global instance
auto_process_service = AutoProcessService()
