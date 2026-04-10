"""
Queue Service
Manages download and processing queues for videos
"""

import os
import json
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from queue import Queue
import uuid

# Data paths
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
QUEUE_STATE_PATH = os.path.join(DATA_DIR, "queue_state.json")

os.makedirs(DATA_DIR, exist_ok=True)


class QueueItemStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"  # New status for pause/resume


@dataclass
class QueueItem:
    id: str
    video_id: str
    video_url: str
    title: str
    channel: str
    thumbnail: str
    resolution: str
    status: QueueItemStatus
    created_at: str
    download_progress: float = 0
    process_progress: float = 0
    downloaded_path: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    clips_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "video_url": self.video_url,
            "title": self.title,
            "channel": self.channel,
            "thumbnail": self.thumbnail,
            "resolution": self.resolution,
            "status": self.status.value if isinstance(self.status, QueueItemStatus) else self.status,
            "created_at": self.created_at,
            "download_progress": self.download_progress,
            "process_progress": self.process_progress,
            "downloaded_path": self.downloaded_path,
            "output_path": self.output_path,
            "error": self.error,
            "clips_count": self.clips_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueItem':
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = QueueItemStatus(status)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            video_id=data.get("video_id", ""),
            video_url=data.get("video_url", ""),
            title=data.get("title", ""),
            channel=data.get("channel", ""),
            thumbnail=data.get("thumbnail", ""),
            resolution=data.get("resolution", "1080"),
            status=status,
            created_at=data.get("created_at", datetime.now().isoformat()),
            download_progress=data.get("download_progress", 0),
            process_progress=data.get("process_progress", 0),
            downloaded_path=data.get("downloaded_path"),
            output_path=data.get("output_path"),
            error=data.get("error"),
            clips_count=data.get("clips_count", 0)
        )


class QueueService:
    """Service for managing download and processing queues"""

    def __init__(self):
        self.items: Dict[str, QueueItem] = {}
        self.download_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_paused = False  # Global pause flag
        self.lock = threading.Lock()

        # Callbacks
        self.on_status_change: Optional[Callable[[QueueItem], None]] = None
        self.on_download_progress: Optional[Callable[[str, float], None]] = None
        self.on_process_progress: Optional[Callable[[str, float], None]] = None

        # Load saved state
        self._load_state()

    def _load_state(self):
        """Load queue state from file"""
        has_pending = False
        try:
            if os.path.exists(QUEUE_STATE_PATH):
                with open(QUEUE_STATE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data.get("items", []):
                        item = QueueItem.from_dict(item_data)
                        # Reset in-progress items to pending on restart
                        if item.status in [QueueItemStatus.DOWNLOADING, QueueItemStatus.PROCESSING]:
                            if item.downloaded_path and os.path.exists(item.downloaded_path):
                                item.status = QueueItemStatus.DOWNLOADED
                                has_pending = True
                            else:
                                item.status = QueueItemStatus.PENDING
                                has_pending = True
                        elif item.status in [QueueItemStatus.PENDING, QueueItemStatus.DOWNLOADED]:
                            has_pending = True
                        self.items[item.id] = item

                    # Save reset states
                    if has_pending:
                        self._save_state()

        except Exception as e:
            print(f"Error loading queue state: {e}")

        # Auto-start workers if there are pending items
        if has_pending:
            print(f"[QUEUE] Found pending items, starting workers...")
            self._ensure_running()

    def _save_state(self):
        """Save queue state to file"""
        try:
            with open(QUEUE_STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    "items": [item.to_dict() for item in self.items.values()]
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving queue state: {e}")

    def add_to_queue(
        self,
        video_id: str,
        video_url: str,
        title: str,
        channel: str,
        thumbnail: str = "",
        resolution: str = "1080"
    ) -> QueueItem:
        """Add a video to the download queue"""
        with self.lock:
            # Check if already in queue
            for item in self.items.values():
                if item.video_id == video_id and item.status not in [QueueItemStatus.COMPLETED, QueueItemStatus.FAILED]:
                    return item

            item = QueueItem(
                id=str(uuid.uuid4()),
                video_id=video_id,
                video_url=video_url,
                title=title,
                channel=channel,
                thumbnail=thumbnail,
                resolution=resolution,
                status=QueueItemStatus.PENDING,
                created_at=datetime.now().isoformat()
            )
            self.items[item.id] = item
            self._save_state()

            # Start queue processing if not running
            self._ensure_running()

            return item

    def remove_from_queue(self, item_id: str, force: bool = False) -> bool:
        """Remove an item from the queue. Use force=True to cancel active downloads."""
        with self.lock:
            if item_id in self.items:
                item = self.items[item_id]
                # Only remove if not currently processing, unless force is True
                if force or item.status not in [QueueItemStatus.DOWNLOADING, QueueItemStatus.PROCESSING]:
                    # Mark for cancellation if currently active
                    if item.status in [QueueItemStatus.DOWNLOADING, QueueItemStatus.PROCESSING]:
                        item.status = QueueItemStatus.FAILED
                        item.error = "Cancelled by user"
                        print(f"[QUEUE] Cancelled: {item.title}")
                    del self.items[item_id]
                    self._save_state()
                    return True
            return False

    def clear_completed(self) -> int:
        """Clear all completed and failed items"""
        with self.lock:
            to_remove = [
                item_id for item_id, item in self.items.items()
                if item.status in [QueueItemStatus.COMPLETED, QueueItemStatus.FAILED]
            ]
            for item_id in to_remove:
                del self.items[item_id]
            self._save_state()
            return len(to_remove)

    def get_queue(self) -> List[Dict[str, Any]]:
        """Get all queue items"""
        with self.lock:
            return [item.to_dict() for item in sorted(
                self.items.values(),
                key=lambda x: x.created_at
            )]

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific queue item"""
        with self.lock:
            if item_id in self.items:
                return self.items[item_id].to_dict()
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self.lock:
            stats = {
                "pending": 0,
                "downloading": 0,
                "downloaded": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "total": len(self.items)
            }
            for item in self.items.values():
                status_key = item.status.value if isinstance(item.status, QueueItemStatus) else item.status
                if status_key in stats:
                    stats[status_key] += 1
            return stats

    def _ensure_running(self):
        """Ensure download and process threads are running"""
        if not self.is_running:
            self.is_running = True

            if self.download_thread is None or not self.download_thread.is_alive():
                self.download_thread = threading.Thread(target=self._download_worker, daemon=True)
                self.download_thread.start()

            if self.process_thread is None or not self.process_thread.is_alive():
                self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
                self.process_thread.start()

    def _download_worker(self):
        """Worker thread for downloading videos one at a time"""
        print("[QUEUE] Download worker started")

        # Import at top of worker to catch import errors
        try:
            from backend.services.youtube_service import youtube_service
            from backend.core.config import settings
        except Exception as e:
            print(f"[QUEUE] Import error in download worker: {e}")
            return

        while self.is_running:
            try:
                # Check if paused
                if self.is_paused:
                    time.sleep(2)
                    continue

                # Find next pending item (skip paused items)
                next_item = None
                with self.lock:
                    for item in sorted(self.items.values(), key=lambda x: x.created_at):
                        if item.status == QueueItemStatus.PENDING:
                            next_item = item
                            break

                if next_item is None:
                    # No pending items, wait and check again
                    time.sleep(1)  # Reduced from 2s for faster response
                    continue

                print(f"[QUEUE] Starting download: {next_item.title}")

                # Start downloading
                with self.lock:
                    next_item.status = QueueItemStatus.DOWNLOADING
                    next_item.download_progress = 0
                    self._save_state()

                if self.on_status_change:
                    self.on_status_change(next_item)

                try:
                    # Download progress callback
                    def progress_callback(percent: float, status: str):
                        with self.lock:
                            next_item.download_progress = percent
                            self._save_state()
                        if self.on_download_progress:
                            self.on_download_progress(next_item.id, percent)

                    # Get format ID based on resolution
                    format_id = f"bestvideo[height<={next_item.resolution}]+bestaudio/best[height<={next_item.resolution}]/best"

                    print(f"[QUEUE] Calling youtube_service.download_video...")
                    result = youtube_service.download_video(
                        video_url=next_item.video_url,
                        output_dir=settings.DOWNLOAD_DIR,
                        format_id=format_id,
                        progress_callback=progress_callback
                    )
                    print(f"[QUEUE] Download result: {result}")

                    with self.lock:
                        if result["success"]:
                            next_item.status = QueueItemStatus.DOWNLOADED
                            next_item.downloaded_path = result["path"]
                            next_item.download_progress = 100
                            print(f"[QUEUE] ✓ Download completed: {next_item.title}")
                            print(f"[QUEUE] → Ready for processing (status: DOWNLOADED)")
                        else:
                            next_item.status = QueueItemStatus.FAILED
                            next_item.error = result.get("error", "Download failed")
                            print(f"[QUEUE] ✗ Download failed: {next_item.error}")
                        self._save_state()

                    if self.on_status_change:
                        self.on_status_change(next_item)

                except Exception as e:
                    print(f"[QUEUE] Download exception: {e}")
                    import traceback
                    traceback.print_exc()
                    with self.lock:
                        next_item.status = QueueItemStatus.FAILED
                        next_item.error = str(e)
                        self._save_state()

                    if self.on_status_change:
                        self.on_status_change(next_item)

                # Delay between downloads to avoid YouTube rate limit
                print(f"[QUEUE] Waiting 5s before next download (avoid rate limit)...")
                time.sleep(5)

            except Exception as e:
                print(f"[QUEUE] Worker loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def _process_worker(self):
        """Worker thread for processing downloaded videos via main panel"""
        print("[QUEUE] Process worker started")

        try:
            from backend.services.auto_process_service import auto_process_service
            from backend.core.state import app_state
            import requests
        except Exception as e:
            print(f"[QUEUE] Import error in process worker: {e}")
            return

        while self.is_running:
            try:
                # Check if paused
                if self.is_paused:
                    time.sleep(2)
                    continue

                # Find next downloaded item to process (skip paused items)
                next_item = None
                with self.lock:
                    for item in sorted(self.items.values(), key=lambda x: x.created_at):
                        if item.status == QueueItemStatus.DOWNLOADED:
                            next_item = item
                            break

                if next_item is None:
                    # No items to process, wait and check again
                    time.sleep(1)  # Reduced from 2s for faster pickup
                    continue

                # Check if main panel is already processing
                if app_state.processing.is_processing:
                    # Wait for main panel to finish
                    time.sleep(2)  # Reduced from 5s
                    continue

                print(f"[QUEUE] Starting processing: {next_item.title}")

                # Start processing via main panel
                with self.lock:
                    next_item.status = QueueItemStatus.PROCESSING
                    next_item.process_progress = 0
                    self._save_state()

                if self.on_status_change:
                    self.on_status_change(next_item)

                try:
                    app_state.add_log(f"Queue: Sending to main panel for processing: {next_item.title}")

                    # Call the local API to process the video via main panel
                    response = requests.post(
                        "http://127.0.0.1:8000/api/video/local",
                        json={"url": next_item.downloaded_path},
                        timeout=10
                    )

                    if response.status_code == 200:
                        app_state.add_log(f"Queue: Video sent to main panel successfully")

                        # Wait for processing to complete by monitoring app_state
                        max_wait = 3600  # Max 1 hour
                        wait_count = 0
                        last_stage = ""

                        while wait_count < max_wait:
                            time.sleep(2)
                            wait_count += 2

                            # Update progress based on main panel state
                            current_stage = app_state.processing.current_stage
                            if current_stage != last_stage:
                                last_stage = current_stage
                                if current_stage == "transcribing":
                                    with self.lock:
                                        next_item.process_progress = 20
                                        self._save_state()
                                elif current_stage == "analyzing":
                                    with self.lock:
                                        next_item.process_progress = 50
                                        self._save_state()
                                elif current_stage == "exporting":
                                    with self.lock:
                                        next_item.process_progress = 70
                                        self._save_state()

                            # Check if processing is done (OUTSIDE the stage change block)
                            if not app_state.processing.is_processing and current_stage == "idle":
                                # Check if we have clips (success) or error
                                clips = app_state.processing.clips
                                print(f"[QUEUE] Processing done. Clips found: {len(clips) if clips else 0}")
                                if clips and len(clips) > 0:
                                    with self.lock:
                                        next_item.status = QueueItemStatus.COMPLETED
                                        next_item.clips_count = len(clips)
                                        next_item.process_progress = 100

                                        # Mark as processed
                                        auto_process_service.mark_video_processed(
                                            video_id=next_item.video_id,
                                            title=next_item.title,
                                            channel=next_item.channel,
                                            output_path=next_item.output_path,
                                            clips_count=next_item.clips_count
                                        )
                                        self._save_state()

                                    app_state.add_log(f"Queue: Completed {next_item.title} - {len(clips)} clips found")
                                    print(f"[QUEUE] ✓ Completed: {next_item.title} - {len(clips)} clips")
                                    print(f"[QUEUE] → Looking for next video to process...")
                                else:
                                    # Check logs for what happened
                                    recent_logs = app_state.processing.logs[-5:] if app_state.processing.logs else []
                                    log_summary = "; ".join([l.get("message", "") for l in recent_logs])
                                    error_detail = f"No clips found. Recent logs: {log_summary[:200]}"
                                    with self.lock:
                                        next_item.status = QueueItemStatus.FAILED
                                        next_item.error = error_detail
                                        self._save_state()
                                    print(f"[QUEUE] ✗ Failed (no clips): {next_item.title}")
                                    print(f"[QUEUE] Recent logs: {log_summary}")
                                break

                        else:
                            # Timeout
                            with self.lock:
                                next_item.status = QueueItemStatus.FAILED
                                next_item.error = "Processing timeout (exceeded 1 hour)"
                                self._save_state()
                            print(f"[QUEUE] Timeout: {next_item.title}")

                    elif response.status_code == 409:
                        # Main panel is busy, retry later
                        app_state.add_log(f"Queue: Main panel busy, will retry later")
                        with self.lock:
                            next_item.status = QueueItemStatus.DOWNLOADED  # Reset to downloaded
                            self._save_state()
                        time.sleep(10)
                        continue

                    else:
                        raise Exception(f"API error: {response.status_code} - {response.text[:200]}")

                    if self.on_status_change:
                        self.on_status_change(next_item)

                except Exception as e:
                    print(f"[QUEUE] Process exception: {e}")
                    import traceback
                    traceback.print_exc()
                    with self.lock:
                        next_item.status = QueueItemStatus.FAILED
                        next_item.error = str(e)
                        self._save_state()

                    if self.on_status_change:
                        self.on_status_change(next_item)

                # Small delay between processing
                time.sleep(1)

            except Exception as e:
                print(f"[QUEUE] Process worker loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def stop(self):
        """Stop the queue workers"""
        self.is_running = False

    def pause(self) -> bool:
        """Pause all queue operations"""
        self.is_paused = True
        return True

    def resume(self) -> bool:
        """Resume queue operations"""
        self.is_paused = False
        self._ensure_running()
        return True

    def is_queue_paused(self) -> bool:
        """Check if queue is paused"""
        return self.is_paused

    def pause_item(self, item_id: str) -> bool:
        """Pause a specific item"""
        with self.lock:
            if item_id in self.items:
                item = self.items[item_id]
                if item.status in [QueueItemStatus.PENDING, QueueItemStatus.DOWNLOADED]:
                    item.status = QueueItemStatus.PAUSED
                    self._save_state()
                    return True
            return False

    def resume_item(self, item_id: str) -> bool:
        """Resume a paused item"""
        with self.lock:
            if item_id in self.items:
                item = self.items[item_id]
                if item.status == QueueItemStatus.PAUSED:
                    # Resume to appropriate state
                    if item.downloaded_path and os.path.exists(item.downloaded_path):
                        item.status = QueueItemStatus.DOWNLOADED
                    else:
                        item.status = QueueItemStatus.PENDING
                    self._save_state()
                    self._ensure_running()
                    return True
            return False

    def retry_failed(self, item_id: str) -> bool:
        """Retry a failed item"""
        with self.lock:
            if item_id in self.items:
                item = self.items[item_id]
                if item.status == QueueItemStatus.FAILED:
                    item.status = QueueItemStatus.PENDING
                    item.error = None
                    item.download_progress = 0
                    item.process_progress = 0
                    self._save_state()
                    self._ensure_running()
                    return True
            return False

    def save_transcription(self, video_id: str, transcription: Dict[str, Any]) -> str:
        """Save transcription to JSON file for retry later"""
        transcription_path = os.path.join(DATA_DIR, f"transcription_{video_id}.json")
        try:
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            return transcription_path
        except Exception as e:
            print(f"Error saving transcription: {e}")
            return ""

    def load_transcription(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load saved transcription from JSON file"""
        transcription_path = os.path.join(DATA_DIR, f"transcription_{video_id}.json")
        try:
            if os.path.exists(transcription_path):
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading transcription: {e}")
        return None

    def delete_transcription(self, video_id: str) -> bool:
        """Delete saved transcription JSON file after successful processing"""
        transcription_path = os.path.join(DATA_DIR, f"transcription_{video_id}.json")
        try:
            if os.path.exists(transcription_path):
                os.remove(transcription_path)
                return True
        except Exception as e:
            print(f"Error deleting transcription: {e}")
        return False

    def has_saved_transcription(self, video_id: str) -> bool:
        """Check if there's a saved transcription for this video"""
        transcription_path = os.path.join(DATA_DIR, f"transcription_{video_id}.json")
        return os.path.exists(transcription_path)


# Global instance
queue_service = QueueService()
