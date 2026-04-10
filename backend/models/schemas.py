"""
Pydantic Schemas for API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class VideoProcessRequest(BaseModel):
    """Request to process a video"""
    url: str = Field(..., description="YouTube URL or local file path")


class VideoInfoResponse(BaseModel):
    """Response with video information"""
    title: str
    duration: float
    thumbnail: Optional[str] = None
    uploader: Optional[str] = None
    is_local: bool = False


class ClipResponse(BaseModel):
    """A discovered clip"""
    id: int
    start: float
    end: float
    start_formatted: str
    end_formatted: str
    duration: float
    title: str
    reason: str


class ExportRequest(BaseModel):
    """Request to export clips"""
    clip_ids: List[int] = Field(..., description="List of clip indices to export")


class ExportProgress(BaseModel):
    """Export progress update"""
    current: int
    total: int
    clip_title: str
    status: str


class SettingsModel(BaseModel):
    """Application settings"""
    tracking_method: str = "dlib"
    yolo_model: str = "yolov8n-face.pt"  # Selected YOLO model file
    use_prescan: bool = True
    face_classifier: bool = True
    cinematic_mode: bool = False
    dynamic_tracking: bool = True  # Deep scan for faces
    dynamic_focus: bool = False  # Auto-zoom to active speaker
    tracking_analyzer: bool = True  # AI stability analyzer
    auto_process: bool = False  # Auto export after analysis
    auto_clip_count: bool = False  # AI decides how many clips
    smoothing: float = 0.05
    tracking_speed: float = 0.5  # How fast camera follows face (0.1-1.0)
    deadzone: int = 40
    confidence: float = 0.50
    single_zoom: float = 1.0
    split_zoom: float = 1.0
    split_screen: bool = True
    min_clip_duration: int = 15
    max_clip_duration: int = 60
    clips_to_find: int = 5
    # Subtitle settings
    subtitle_enabled: bool = True
    subtitle_font_size: int = 48
    subtitle_font_path: str = ""
    subtitle_max_words: int = 5
    subtitle_position: int = 85  # 0-100 percentage from top
    subtitle_style: str = "uppercase"
    subtitle_color: str = "#FFFFFF"
    subtitle_highlight_color: str = "#FFFF00"
    subtitle_bg_enabled: bool = True
    subtitle_bg_color: str = "#000000"
    subtitle_bg_opacity: float = 0.5
    # AI API settings
    ai_selected: str = "A"  # "A" = Anthropic API, "B" = Raw Response
    ai_auto_chunk: bool = True  # Auto calculate optimal chunk size
    ai_chunk_tokens: int = 0  # 0 = send all as one, otherwise max tokens per chunk (used when auto_chunk is off)
    ai_chunk_cooldown: int = 2  # Cooldown in seconds between chunk requests
    # Debug settings
    debug_mode: bool = False  # Enable basic tracking logs in terminal (minimalist)
    debug_mode_advanced: bool = False  # Enable detailed/verbose tracking logs in terminal


class SettingsUpdateModel(BaseModel):
    """Partial settings update - all fields optional"""
    tracking_method: Optional[str] = None
    yolo_model: Optional[str] = None  # Selected YOLO model file
    use_prescan: Optional[bool] = None
    face_classifier: Optional[bool] = None
    cinematic_mode: Optional[bool] = None
    dynamic_tracking: Optional[bool] = None
    dynamic_focus: Optional[bool] = None  # Auto-zoom to active speaker
    tracking_analyzer: Optional[bool] = None
    auto_process: Optional[bool] = None
    auto_clip_count: Optional[bool] = None
    smoothing: Optional[float] = None
    tracking_speed: Optional[float] = None
    deadzone: Optional[int] = None
    confidence: Optional[float] = None
    single_zoom: Optional[float] = None
    split_zoom: Optional[float] = None
    split_screen: Optional[bool] = None
    min_clip_duration: Optional[int] = None
    max_clip_duration: Optional[int] = None
    clips_to_find: Optional[int] = None
    # Subtitle settings
    subtitle_enabled: Optional[bool] = None
    subtitle_font_size: Optional[int] = None
    subtitle_font_path: Optional[str] = None
    subtitle_max_words: Optional[int] = None
    subtitle_position: Optional[int] = None
    subtitle_style: Optional[str] = None
    subtitle_color: Optional[str] = None
    subtitle_highlight_color: Optional[str] = None
    subtitle_bg_enabled: Optional[bool] = None
    subtitle_bg_color: Optional[str] = None
    subtitle_bg_opacity: Optional[float] = None
    # AI API settings
    ai_selected: Optional[str] = None  # "A" = Anthropic API, "B" = Raw Response
    ai_auto_chunk: Optional[bool] = None  # Auto calculate optimal chunk size
    ai_chunk_tokens: Optional[int] = None  # 0 = send all as one, otherwise max tokens per chunk (used when auto_chunk is off)
    ai_chunk_cooldown: Optional[int] = None  # Cooldown in seconds between chunk requests
    # Debug settings
    debug_mode: Optional[bool] = None  # Enable basic tracking logs in terminal (minimalist)
    debug_mode_advanced: Optional[bool] = None  # Enable detailed/verbose tracking logs in terminal


class ProgressEvent(BaseModel):
    """Progress update event"""
    stage: str
    value: float
    status: str


class LogEvent(BaseModel):
    """Log message event"""
    timestamp: str
    message: str
    level: str = "info"


class StatusResponse(BaseModel):
    """Current processing status"""
    is_processing: bool
    current_stage: str
    video_title: Optional[str]
    video_duration: float
    progress: Dict[str, float]
    progress_status: Dict[str, str]
    clips_count: int


class GPUStatusResponse(BaseModel):
    """GPU status information"""
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_memory: Optional[str] = None
    device: str = "cpu"


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
