"""
YouTube Auto Clip Maker - Modules Package
"""

from .downloader import YouTubeDownloader
from .transcriber import WhisperTranscriber
from .analyzer import ClipAnalyzer
from .face_tracker import FaceTracker
from .face_classifier import FaceClassifier
from .face_embedder import FaceEmbedder, DLIB_EMBEDDER_AVAILABLE
from .video_processor import VideoProcessor
from .tracking_analyzer import TrackingAnalyzer
from .subtitle_renderer import SubtitleRenderer

__all__ = [
    'YouTubeDownloader',
    'WhisperTranscriber',
    'ClipAnalyzer',
    'FaceTracker',
    'FaceClassifier',
    'FaceEmbedder',
    'DLIB_EMBEDDER_AVAILABLE',
    'VideoProcessor',
    'TrackingAnalyzer',
    'SubtitleRenderer'
]
