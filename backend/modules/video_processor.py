"""
Video Processor Module
Crops video to vertical format with face tracking
GPU acceleration with CUDA when available
"""

import os
import cv2
import subprocess
import numpy as np
import torch
from typing import Optional, Callable, Tuple, List, Dict, Any
from .face_tracker import FaceTracker
from .face_classifier import FaceClassifier
from .face_embedder import FaceEmbedder, DLIB_EMBEDDER_AVAILABLE
from .optical_flow_filter import OpticalFlowFilter

# Check for CUDA support
USE_CUDA = torch.cuda.is_available()
USE_CV_CUDA = False

try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_CV_CUDA = True
        print(f"OpenCV CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
except:
    pass

if USE_CUDA:
    print(f"PyTorch CUDA enabled: {torch.cuda.get_device_name(0)}")
    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class VideoProcessor:
    """Processes video clips with face-centered vertical cropping"""

    def __init__(
        self,
        output_width: int = 1080,
        output_height: int = 1920,
        output_fps: int = 30,
        yolo_model_path: str = None,
        classifier_model_path: str = None,
        # Face tracking settings
        confidence: float = 0.5,
        smoothing: float = 0.3,
        tracking_speed: float = 0.5,
        detection_interval: int = 5,
        # Split screen
        enable_split_screen: bool = True,
        # GPU optimization
        batch_size: int = 4,
        use_gpu_resize: bool = True,
        # Face padding (zoom control)
        face_padding_single: float = 1.0,  # 1.0 = normal, 0.5 = closer/zoomed, 1.5 = further
        face_padding_split: float = 1.0,
        # Pre-scan mode
        use_prescan: bool = True,  # True = scan first for stable mode, False = real-time
        cinematic_mode: bool = False,
        tracking_method: str = "yolo",  # yolo or dlib - used for both prescan and tracking
        deadzone: float = 40,
        dynamic_tracking: bool = True,
        tracking_analyzer: bool = True, # Enable Tracking Analyzer by default
        dynamic_focus: bool = False,  # Auto-zoom to active speaker
        # Subtitle settings
        subtitle_enabled: bool = True,
        subtitle_font_size: int = 48,
        subtitle_font_path: str = "",
        subtitle_max_words: int = 5,
        subtitle_position: int = 85,  # 0-100 percentage from top
        subtitle_style: str = "uppercase",
        subtitle_color: str = "#FFFFFF",
        subtitle_highlight_color: str = "#FFFF00",
        subtitle_bg_enabled: bool = True,
        subtitle_bg_color: str = "#000000",
        subtitle_bg_opacity: float = 0.5,
        # Cancel callback
        cancel_checker: Optional[Callable[[], bool]] = None,
        # Logging callback
        log_callback: Optional[Callable[[str], None]] = None,
        # Debug mode - basic terminal logging (minimalist)
        debug_mode: bool = False,
        # Debug mode advanced - detailed/verbose terminal logging
        debug_mode_advanced: bool = False,
        # Optical Flow Poster Filter
        optical_flow_enabled: bool = True,
        optical_flow_threshold: float = 2.0,
        optical_flow_min_samples: int = 5,
        optical_flow_consistency: float = 0.7,
        optical_flow_dense: bool = False
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.output_fps = output_fps
        self.enable_split_screen = enable_split_screen
        self.batch_size = batch_size
        self.use_gpu_resize = use_gpu_resize and USE_CV_CUDA
        self.use_prescan = use_prescan  # Pre-scan mode toggle
        self.cinematic_mode = cinematic_mode
        self.dynamic_tracking = dynamic_tracking
        self.enable_tracking_analyzer = tracking_analyzer
        self.dynamic_focus = dynamic_focus  # Auto-zoom to active speaker
        self.debug_mode = debug_mode  # Basic terminal logging
        self.debug_mode_advanced = debug_mode_advanced  # Detailed terminal logging

        # Optical Flow Poster Filter
        self.optical_flow_enabled = optical_flow_enabled
        self.optical_flow_filter: Optional[OpticalFlowFilter] = None
        if optical_flow_enabled:
            self.optical_flow_filter = OpticalFlowFilter(
                flow_threshold=optical_flow_threshold,
                min_samples=optical_flow_min_samples,
                consistency_ratio=optical_flow_consistency,
                use_dense_flow=optical_flow_dense
            )

        # Speaker timeline for dynamic focus (set by caller)
        # Format: [{'start': float, 'end': float, 'speaker': 'left'|'right'|'both'}]
        self.speaker_timeline: List[Dict] = []

        # Subtitle settings
        self.subtitle_enabled = subtitle_enabled
        self.subtitle_renderer = None
        if subtitle_enabled:
            from .subtitle_renderer import SubtitleRenderer
            self.subtitle_renderer = SubtitleRenderer(
                frame_width=output_width,
                frame_height=output_height,
                font_size=subtitle_font_size,
                font_path=subtitle_font_path if subtitle_font_path else None,
                position=subtitle_position,
                style=subtitle_style,
                text_color=subtitle_color,
                highlight_color=subtitle_highlight_color,
                bg_enabled=subtitle_bg_enabled,
                bg_color=subtitle_bg_color,
                bg_opacity=subtitle_bg_opacity,
                max_words_per_line=subtitle_max_words
            )

        # Word timestamps for subtitle sync (set by caller before processing)
        self.word_timestamps: List[Dict] = []
        self.clip_start_time: float = 0.0

        # Callbacks
        self._cancel_checker = cancel_checker
        self._log_callback = log_callback
        self._cancelled = False

        # Initialize Tracking Analyzer (MOVED HERE to avoid AttributeError)
        self.tracking_monitor = None
        if self.enable_tracking_analyzer:
            from .tracking_analyzer import TrackingAnalyzer
            self.tracking_monitor = TrackingAnalyzer()
            self._log("Tracking Analyzer: Active and monitoring...") # Now safe to call log

        # Face padding controls (zoom level)
        self.face_padding_single = face_padding_single
        self.face_padding_split = face_padding_split

        # Tracking smoothing settings
        # smoothing = how smooth the camera moves (higher = smoother but slower response)
        # tracking_speed = how fast camera follows face (higher = faster response)
        # Combined: lerp factor = tracking_speed * (1 - smoothing * 0.9)
        # Example: speed=0.5, smoothing=0.2 → lerp = 0.5 * (1 - 0.18) = 0.41
        # Example: speed=0.5, smoothing=0.5 → lerp = 0.5 * (1 - 0.45) = 0.275
        self.tracking_smoothing = smoothing  # Use setting from UI (0.05-0.5)
        self.tracking_speed = tracking_speed  # Use setting from UI (0.1-1.0)
        self.tracking_deadzone = deadzone  # Ignore movements smaller than this (pixels)

        # Tracking method: True = dlib embedding, False = YOLO + classifier
        # Same method used for both prescan and tracking
        self.use_dlib_tracking = (tracking_method == "dlib") and DLIB_EMBEDDER_AVAILABLE

        # Face classifier (human vs poster) - DISABLED, kept for reference
        # self.face_classifier = FaceClassifier(classifier_model_path)
        # self.use_classifier = self.face_classifier.is_loaded
        self.face_classifier = None
        self.use_classifier = False

        # Face embedder (pre-trained, understands general face characteristics)
        self.face_embedder = FaceEmbedder() if DLIB_EMBEDDER_AVAILABLE else None
        self.use_embedder = DLIB_EMBEDDER_AVAILABLE

        self.face_tracker = FaceTracker(
            confidence_threshold=confidence,
            smoothing_factor=smoothing,
            tracking_speed=tracking_speed,
            detection_interval=detection_interval,
            model_path=yolo_model_path,
            cinematic_mode=cinematic_mode,
            deadzone=deadzone
        )
        # Second face tracker for split screen
        self.face_tracker_2 = FaceTracker(
            confidence_threshold=confidence,
            smoothing_factor=smoothing,
            tracking_speed=tracking_speed,
            detection_interval=detection_interval,
            model_path=yolo_model_path,
            cinematic_mode=cinematic_mode,
            deadzone=deadzone
        )

        # Pre-allocate GPU memory for resize operations
        if self.use_gpu_resize:
            self._init_gpu_buffers()

    def _init_gpu_buffers(self):
        """Pre-allocate GPU buffers for faster processing"""
        try:
            # Create GPU mats for resize operations
            self.gpu_frame = cv2.cuda_GpuMat()
            self.gpu_resized = cv2.cuda_GpuMat()
        except Exception as e:
            print(f"GPU buffer init failed: {e}")
            self.use_gpu_resize = False

    def _gpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame using GPU if available, always returns contiguous numpy array"""
        result = None

        if self.use_gpu_resize:
            try:
                self.gpu_frame.upload(frame)
                self.gpu_resized = cv2.cuda.resize(self.gpu_frame, target_size)
                result = self.gpu_resized.download()
            except Exception as e:
                # Fallback to CPU
                result = cv2.resize(frame, target_size)
        else:
            result = cv2.resize(frame, target_size)

        # Ensure result is contiguous numpy array
        if result is not None and not result.flags['C_CONTIGUOUS']:
            result = np.ascontiguousarray(result)

        return result

    def _match_faces_by_proximity(
        self,
        faces: List[Dict],
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> Tuple[Dict, Dict]:
        """
        Match detected faces to tracker positions by proximity.
        This maintains identity consistency when embedding matching fails.

        Args:
            faces: List of detected faces (at least 2)
            pos1: Last known position of person 1 (x, y)
            pos2: Last known position of person 2 (x, y)

        Returns:
            Tuple of (face_1, face_2) matched to positions
        """
        if len(faces) < 2:
            return faces[0], faces[0] if faces else (None, None)

        # Calculate distances from each face to each position
        best_match = None
        best_total_dist = float('inf')

        # Try all pairings and pick the one with minimum total distance
        for i, face_a in enumerate(faces):
            for j, face_b in enumerate(faces):
                if i == j:
                    continue

                ax, ay = face_a['center']
                bx, by = face_b['center']

                dist_a_to_1 = ((ax - pos1[0])**2 + (ay - pos1[1])**2)**0.5
                dist_b_to_2 = ((bx - pos2[0])**2 + (by - pos2[1])**2)**0.5

                total_dist = dist_a_to_1 + dist_b_to_2

                if total_dist < best_total_dist:
                    best_total_dist = total_dist
                    best_match = (face_a, face_b)

        return best_match if best_match else (faces[0], faces[1])

    def _calculate_split_crop_region(
        self,
        frame_width: int,
        frame_height: int,
        face_center: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop region for split screen (9:8 aspect ratio for each half)
        This prevents squashing when resizing to half height

        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            face_center: Face center position (x, y)

        Returns:
            Crop region as (x1, y1, x2, y2)
        """
        # Split screen uses 9:8 aspect ratio (half of 9:16)
        aspect_w, aspect_h = 9, 8

        # Calculate base crop dimensions
        if frame_height / frame_width > aspect_h / aspect_w:
            # Width is limiting factor
            crop_width = frame_width
            crop_height = int(frame_width * aspect_h / aspect_w)
        else:
            # Height is limiting factor
            crop_height = frame_height
            crop_width = int(frame_height * aspect_w / aspect_h)

        # Apply padding (zoom control)
        # padding < 1.0 = zoom in (closer to face)
        # padding > 1.0 = zoom out (more background)
        crop_width = int(crop_width * self.face_padding_split)
        crop_height = int(crop_height * self.face_padding_split)

        # Ensure we don't exceed frame dimensions
        crop_width = min(crop_width, frame_width)
        crop_height = min(crop_height, frame_height)

        center_x, center_y = face_center

        # Center crop on face
        x1 = center_x - crop_width // 2
        x2 = center_x + crop_width // 2
        y1 = center_y - crop_height // 2
        y2 = center_y + crop_height // 2

        # Clamp to frame boundaries
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > frame_width:
            x1 -= (x2 - frame_width)
            x2 = frame_width
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > frame_height:
            y1 -= (y2 - frame_height)
            y2 = frame_height

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        return (int(x1), int(y1), int(x2), int(y2))

    def _calculate_single_crop_region(
        self,
        frame_width: int,
        frame_height: int,
        face_center: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop region for single screen (9:16 aspect ratio) with padding control

        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            face_center: Face center position (x, y)

        Returns:
            Crop region as (x1, y1, x2, y2)
        """
        aspect_w, aspect_h = 9, 16

        # Calculate base crop dimensions
        if frame_height / frame_width > aspect_h / aspect_w:
            crop_width = frame_width
            crop_height = int(frame_width * aspect_h / aspect_w)
        else:
            crop_height = frame_height
            crop_width = int(frame_height * aspect_w / aspect_h)

        # Apply padding (zoom control)
        crop_width = int(crop_width * self.face_padding_single)
        crop_height = int(crop_height * self.face_padding_single)

        # Ensure we don't exceed frame dimensions
        crop_width = min(crop_width, frame_width)
        crop_height = min(crop_height, frame_height)

        center_x, center_y = face_center

        # Center crop on face X, start from top for Y
        x1 = center_x - crop_width // 2
        x2 = center_x + crop_width // 2
        y1 = 0
        y2 = crop_height

        # Clamp X to frame boundaries
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > frame_width:
            x1 -= (x2 - frame_width)
            x2 = frame_width

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        return (int(x1), int(y1), int(x2), int(y2))
    
    def set_classifier(self, classifier: FaceClassifier):
        """Set face classifier for human/poster detection"""
        self.face_classifier = classifier
        self.use_classifier = classifier.is_loaded if classifier else False
    
    def set_split_screen(self, enabled: bool):
        """Enable or disable split screen mode"""
        self.enable_split_screen = enabled

    def set_prescan(self, enabled: bool):
        """Enable or disable pre-scan tracking mode"""
        self.use_prescan = enabled

    def set_cinematic_mode(self, enabled: bool):
        """Enable or disable cinematic motion tracking"""
        self.cinematic_mode = enabled
        self.face_tracker.update_settings(cinematic_mode=enabled)
        self.face_tracker_2.update_settings(cinematic_mode=enabled)

    def set_dynamic_tracking(self, enabled: bool):
        """Enable or disable dynamic tracking (deep scan)"""
        self.dynamic_tracking = enabled

    def set_tracking_analyzer(self, enabled: bool):
        """Enable or disable AI tracking analyzer"""
        self.enable_tracking_analyzer = enabled
        if enabled and self.tracking_monitor is None:
            from .tracking_analyzer import TrackingAnalyzer
            self.tracking_monitor = TrackingAnalyzer()

    def set_word_timestamps(self, words: List[Dict], clip_start: float):
        """
        Set word timestamps for subtitle rendering

        Args:
            words: List of word dicts with 'word', 'start', 'end' keys
            clip_start: Start time of clip in original video (seconds)
        """
        self.word_timestamps = words
        self.clip_start_time = clip_start
        if self.subtitle_enabled and self.subtitle_renderer and words:
            # Build segments for TikTok/CapCut style display
            self.subtitle_renderer.build_segments_from_words(words)
            self._log(f"Subtitle: {len(words)} words → {len(self.subtitle_renderer.segments)} segments")

    def set_speaker_timeline(self, timeline: List[Dict], clip_start: float):
        """
        Set speaker timeline for dynamic focus

        Args:
            timeline: List of dicts with 'start', 'end', 'speaker' keys
                      speaker: 'left', 'right', or 'both'
            clip_start: Start time of clip in original video (seconds)
        """
        # Adjust timeline to be relative to clip start
        self.speaker_timeline = []
        for segment in timeline:
            self.speaker_timeline.append({
                'start': segment['start'] - clip_start,
                'end': segment['end'] - clip_start,
                'speaker': segment['speaker']
            })
        if self.speaker_timeline:
            self._log(f"Dynamic Focus: {len(self.speaker_timeline)} speaker segments loaded")

    def _get_active_speaker(self, current_time: float) -> str:
        """
        Get the active speaker at a given time

        Args:
            current_time: Time in seconds (relative to clip start)

        Returns:
            'left', 'right', 'both', or None if no speaker info
        """
        if not self.speaker_timeline:
            return None

        for segment in self.speaker_timeline:
            if segment['start'] <= current_time <= segment['end']:
                return segment['speaker']

        return None

    def update_tracking_settings(self, confidence: float = None, smoothing: float = None,
                                  speed: float = None, interval: int = None, cinematic_mode: bool = None,
                                  deadzone: float = None, tracking_method: str = None,
                                  dynamic_tracking: bool = None, tracking_analyzer: bool = None):
        """Update face tracking settings"""
        if cinematic_mode is not None:
            self.cinematic_mode = cinematic_mode

        if dynamic_tracking is not None:
            self.dynamic_tracking = dynamic_tracking

        if tracking_analyzer is not None:
            self.set_tracking_analyzer(tracking_analyzer)

        if deadzone is not None:
            self.tracking_deadzone = deadzone

        if tracking_method is not None:
            self.use_dlib_tracking = (tracking_method == "dlib") and DLIB_EMBEDDER_AVAILABLE

        self.face_tracker.update_settings(confidence, smoothing, speed, interval,
                                          cinematic_mode=self.cinematic_mode, deadzone=self.tracking_deadzone)
        self.face_tracker_2.update_settings(confidence, smoothing, speed, interval,
                                            cinematic_mode=self.cinematic_mode, deadzone=self.tracking_deadzone)

    def update_padding_settings(self, single: float = None, split: float = None):
        """
        Update face padding (zoom) settings

        Args:
            single: Padding for single screen mode (0.5 = zoomed in, 1.0 = normal, 1.5 = zoomed out)
            split: Padding for split screen mode
        """
        if single is not None:
            self.face_padding_single = max(0.3, min(2.0, single))
        if split is not None:
            self.face_padding_split = max(0.3, min(2.0, split))

    def _get_lerp_factor(self) -> float:
        """
        Calculate interpolation factor based on smoothing and tracking_speed settings.

        - smoothing (0-1): Higher = smoother/more gradual movement
        - tracking_speed (0-1): Higher = faster camera follow

        The lerp factor determines how quickly the camera catches up to the face.

        HIGH SMOOTHING (0.9) + LOW SPEED (0.1) = Very smooth, almost no movement
        LOW SMOOTHING (0.1) + HIGH SPEED (0.9) = Responsive, can be jerky
        """
        smoothing = getattr(self, 'tracking_smoothing', 0.5)
        speed = getattr(self, 'tracking_speed', 0.5)

        # NEW FORMULA: Simpler and more intuitive
        # Base lerp from speed: 0.02 (slow) to 0.3 (fast)
        base_lerp = 0.02 + (speed * 0.28)

        # Smoothing reduces the lerp exponentially
        # smoothing=0 → no reduction, smoothing=1 → reduce to 10%
        smooth_reduction = 1.0 - (smoothing * 0.9)

        lerp = base_lerp * smooth_reduction

        # Clamp between 0.01 (very smooth) and 0.3 (responsive)
        return max(0.01, min(0.3, lerp))

    def _get_effective_deadzone(self) -> float:
        """
        Calculate effective deadzone based on smoothing.
        Higher smoothing = smaller deadzone for fluid motion.

        Deadzone prevents micro-jitter but can cause jerky motion if too high.
        """
        smoothing = getattr(self, 'tracking_smoothing', 0.2)
        base_deadzone = getattr(self, 'tracking_deadzone', 40)

        # High smoothing = very small deadzone (almost continuous motion)
        # smoothing=0.85 → deadzone = 40 * 0.15 = 6px, then reduce further
        # We want smoothing=0.85 to give deadzone ~2-3px
        effective = base_deadzone * ((1.0 - smoothing) ** 1.5)
        return max(1, min(base_deadzone, effective))  # Min 1px, max base

    def _log(self, message: str):
        """Send log message if callback is available"""
        if self._log_callback:
            self._log_callback(message)

    def _debug(self, message: str):
        """Print debug message to terminal if advanced debug mode is enabled"""
        if self.debug_mode_advanced:
            import time
            timestamp = time.strftime("%H:%M:%S")
            print(f"[DEBUG {timestamp}] {message}")

    def _debug_basic(self, message: str):
        """Print basic debug message to terminal if normal debug mode is enabled"""
        if self.debug_mode or self.debug_mode_advanced:
            import time
            timestamp = time.strftime("%H:%M:%S")
            print(f"[INFO {timestamp}] {message}")

    def _debug_frame(
        self,
        frame_idx: int,
        total_frames: int,
        fps: float,
        faces: List[Dict],
        mode: str,
        detection_time_ms: float = 0
    ):
        """Log detailed per-frame debug info"""
        if not self.debug_mode_advanced:
            return

        time_sec = frame_idx / fps if fps > 0 else 0
        face_count = len(faces)

        # Build face info string
        face_info = []
        for i, f in enumerate(faces):
            cx, cy = f.get('center', (0, 0))
            w, h = f.get('width', 0), f.get('height', 0)
            conf = f.get('confidence', 0)
            face_info.append(f"F{i+1}:({cx:.0f},{cy:.0f}) {w:.0f}x{h:.0f} conf={conf:.2f}")

        faces_str = " | ".join(face_info) if face_info else "none"

        print(f"[FRAME {frame_idx}/{total_frames}] @{time_sec:.2f}s | {face_count} faces | {mode.upper()} | det={detection_time_ms:.1f}ms | {faces_str}")

    def _debug_mode_switch(
        self,
        frame_idx: int,
        fps: float,
        old_mode: str,
        new_mode: str,
        face_count: int,
        face_positions: List[Tuple[float, float]],
        reason: str = ""
    ):
        """Log mode switch event"""
        if not self.debug_mode_advanced:
            return

        time_sec = frame_idx / fps if fps > 0 else 0
        pos_str = " | ".join([f"({x:.0f},{y:.0f})" for x, y in face_positions])

        print(f"")
        print(f"{'='*60}")
        print(f"MODE SWITCH: {old_mode.upper()} → {new_mode.upper()} @ frame {frame_idx} ({time_sec:.2f}s)")
        print(f"  Faces: {face_count} | Positions: {pos_str}")
        if reason:
            print(f"  Reason: {reason}")
        print(f"{'='*60}")
        print(f"")

    def _debug_tracking(
        self,
        frame_idx: int,
        fps: float,
        raw_pos: Tuple[float, float],
        smooth_pos: Tuple[float, float],
        delta: Tuple[float, float],
        person_id: int = 1
    ):
        """Log tracking quality info"""
        if not self.debug_mode_advanced:
            return

        # Only log every 30 frames to avoid spam
        if frame_idx % 30 != 0:
            return

        time_sec = frame_idx / fps if fps > 0 else 0
        dx, dy = delta
        movement = (dx**2 + dy**2)**0.5

        print(f"[TRACK P{person_id}] @{time_sec:.1f}s | raw=({raw_pos[0]:.0f},{raw_pos[1]:.0f}) → smooth=({smooth_pos[0]:.0f},{smooth_pos[1]:.0f}) | delta=({dx:.1f},{dy:.1f}) move={movement:.1f}px")

    def _debug_prescan_segment(
        self,
        segment_idx: int,
        start_frame: int,
        end_frame: int,
        fps: float,
        mode: str,
        face_count_stats: Dict[int, int]
    ):
        """Log prescan segment info"""
        if not self.debug_mode_advanced:
            return

        start_sec = start_frame / fps if fps > 0 else 0
        end_sec = end_frame / fps if fps > 0 else 0
        duration = end_sec - start_sec

        stats_str = ", ".join([f"{k}f:{v}x" for k, v in sorted(face_count_stats.items())])

        print(f"[SEGMENT {segment_idx}] frames {start_frame}-{end_frame} ({start_sec:.1f}s-{end_sec:.1f}s, {duration:.1f}s) = {mode.upper()} | face counts: {stats_str}")

    def set_cancel_checker(self, checker: Optional[Callable[[], bool]]):
        """Set cancel checker callback"""
        self._cancel_checker = checker
        self._cancelled = False

    def set_log_callback(self, callback: Optional[Callable[[str], None]]):
        """Set logging callback"""
        self._log_callback = callback

    def is_cancelled(self) -> bool:
        """Check if processing should be cancelled"""
        if self._cancelled:
            return True
        if self._cancel_checker and self._cancel_checker():
            self._cancelled = True
            return True
        return False

    def cancel(self):
        """Request cancellation"""
        self._cancelled = True

    def reset_cancel(self):
        """Reset cancellation flag"""
        self._cancelled = False
        
    def extract_clip(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Extract and process a clip from video with face tracking
        
        Args:
            input_path: Path to source video
            output_path: Path for output video
            start_time: Start time in seconds
            end_time: End time in seconds
            progress_callback: Callback function(progress_percent, status_message)
            
        Returns:
            Path to processed video
        """
        if progress_callback:
            progress_callback(0, "Preparing video extraction...")

        # Check for cancellation before starting
        if self.is_cancelled():
            if progress_callback:
                progress_callback(0, "Cancelled")
            return output_path

        # First, extract the clip without processing using FFmpeg (faster)
        temp_clip_path = output_path.replace('.mp4', '_temp.mp4')

        self._extract_segment_ffmpeg(input_path, temp_clip_path, start_time, end_time)

        # Check for cancellation after extraction
        if self.is_cancelled():
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
            if progress_callback:
                progress_callback(0, "Cancelled")
            return output_path

        if progress_callback:
            progress_callback(20, "Analyzing face positions...")
        
        # Process with face tracking
        self._process_with_face_tracking(
            temp_clip_path,
            output_path,
            progress_callback
        )

        # Clean up temp file
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

        # Check if cancelled - don't report complete
        if self.is_cancelled():
            if progress_callback:
                progress_callback(0, "Cancelled")
            return output_path

        if progress_callback:
            progress_callback(100, "Clip processing complete!")

        return output_path
    
    def _extract_segment_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> None:
        """Extract video segment using FFmpeg with accurate seeking"""
        duration = end_time - start_time

        # Use -ss AFTER -i for accurate frame-level seeking (decode mode)
        # This is slower but more accurate for A/V sync
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-ss', str(start_time),          # Seek AFTER input for accuracy
            '-t', str(duration),
            '-c:v', 'libx264',               # Re-encode for frame accuracy
            '-preset', 'fast',
            '-crf', '18',
            '-an',                           # No audio (will add later)
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='ignore', check=True)

    def _deep_scan_for_faces(self, frame: np.ndarray, force_deep: bool = False) -> List[Dict]:
        """
        Perform a deeper scan by dividing the frame into regions to detect smaller faces.
        Useful for wide shots where faces are small (Deep Scan).

        Args:
            frame: Video frame to scan
            force_deep: If True, always perform deep scan regardless of dynamic_tracking setting.
                        Used by prescan for accuracy.
        """
        # 1. Standard scan first
        if self.use_dlib_tracking and self.face_embedder and self.face_embedder.is_loaded:
            faces = self.face_embedder.get_face_embeddings(frame)
        else:
            faces = self.face_tracker.detect_faces(frame)

        # If we found 2+ faces, return immediately (no need for deep scan)
        # Only skip deep scan based on dynamic_tracking when NOT forced
        if len(faces) >= 2:
            return faces
        if not force_deep and not self.dynamic_tracking:
            return faces

        # 2. Deep scan: split into regions if simple scan failed to find 2 people
        height, width = frame.shape[:2]

        # Define overlapping regions (Quadrants with overlap)
        # Overlap is important to detect faces at boundaries
        overlap = 150
        half_w = width // 2
        half_h = height // 2

        regions = [
            (0, 0, half_w + overlap, half_h + overlap),          # TL
            (half_w - overlap, 0, width, half_h + overlap),      # TR
            (0, half_h - overlap, half_w + overlap, height),     # BL
            (half_w - overlap, half_h - overlap, width, height)  # BR
        ]

        detected_in_regions = []

        for (x1, y1, x2, y2) in regions:
            # Ensure within bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 50:
                continue

            # Detect in crop - use dlib if available for embeddings
            if self.use_dlib_tracking and self.face_embedder and self.face_embedder.is_loaded:
                region_faces = self.face_embedder.get_face_embeddings(crop)
            else:
                region_faces = self.face_tracker.detect_faces(crop)

            # Map back to original coordinates
            for face in region_faces:
                fx1, fy1, fx2, fy2 = face['bbox']
                # Adjust bbox
                original_bbox = (fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1)
                # Adjust center
                cx, cy = face['center']
                original_center = (cx + x1, cy + y1)

                face['bbox'] = original_bbox
                face['center'] = original_center

                detected_in_regions.append(face)

        # 3. Merge results (NMS-like deduplication with embedding similarity)
        all_faces = faces + detected_in_regions
        unique_faces = []
        EMBEDDING_THRESHOLD = 0.72  # Threshold for same person (higher = stricter)

        for face in all_faces:
            is_new = True
            face_emb = face.get('embedding')

            for known in unique_faces:
                known_emb = known.get('embedding')

                # PRIMARY: Use embedding similarity if both have embeddings
                if face_emb is not None and known_emb is not None:
                    similarity = FaceEmbedder.cosine_similarity(face_emb, known_emb)
                    if similarity >= EMBEDDING_THRESHOLD:
                        # Keep the one with higher confidence
                        if face['confidence'] > known['confidence']:
                            known.update(face)
                        is_new = False
                        break

                # FALLBACK: Distance-based check (if either doesn't have embedding)
                dist = ((face['center'][0] - known['center'][0])**2 + (face['center'][1] - known['center'][1])**2)**0.5
                x_diff = abs(face['center'][0] - known['center'][0])

                # Same face if: close distance OR similar X position (vertical duplicate)
                if dist < 100 or x_diff < 80:
                    # Keep the one with higher confidence or area
                    if face['confidence'] > known['confidence']:
                        known.update(face) # Update existing
                    is_new = False
                    break

            if is_new:
                unique_faces.append(face)

        # Sort by area
        unique_faces.sort(key=lambda x: x['area'], reverse=True)
        return unique_faces

    def _scan_video_for_modes(
        self,
        input_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict]:
        """
        Pre-scan video to determine split/single mode for ENTIRE video.

        VERY SIMPLE LOGIC:
        1. Sample frames throughout the video
        2. For each frame, count unique faces (after deduplication)
        3. Calculate the MODE (most frequent face count)
        4. MODE >= 2 → SPLIT for entire video
        5. MODE == 1 → SINGLE for entire video

        This is simple, reliable, and doesn't over-complicate things.
        """
        cap = None
        try:
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Sample every N frames (10 samples/sec is enough)
            sample_interval = max(1, int(fps / 10))

            if progress_callback:
                progress_callback(20, "Pre-scanning video...")

            self._log(f"╔══════════════════════════════════════════")
            self._log(f"║ PRESCAN START (Simple Mode)")
            self._log(f"║ Video: {total_frames} frames, {fps:.1f} fps, {total_frames/fps:.1f}s")
            self._log(f"║ Sample interval: every {sample_interval} frames")
            self._log(f"║")
            self._log(f"║ SIMPLE LOGIC:")
            self._log(f"║   Count faces per frame")
            self._log(f"║   Most frequent count (MODE) decides:")
            self._log(f"║   MODE >= 2 → SPLIT for entire video")
            self._log(f"║   MODE == 1 → SINGLE for entire video")
            self._log(f"╚══════════════════════════════════════════")

            frame_idx = 0
            face_counts = []  # List of face counts per sampled frame
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # === NEW APPROACH: Identity Clustering ===
            # Track face "identities" by spatial position over time
            # Each identity = a cluster of detections at similar positions
            # Real person: large, present in most frames, position varies (head movement)
            # Poster: may be large, present in some frames, position NEVER varies independently
            
            poster_positions = set()  # Confirmed poster grid positions
            
            # Identity tracker: cluster faces by position across frames
            # Each identity has: positions list, areas list, frame_indices list
            identities = []  # List of identity dicts
            IDENTITY_MERGE_DIST = 150  # Max distance to consider same identity
            
            total_sampled_frames = 0

            # Simple deduplication by distance
            def simple_dedupe(faces, min_distance=100):
                if len(faces) <= 1:
                    return faces
                sorted_faces = sorted(faces, key=lambda f: f.get('confidence', 0), reverse=True)
                unique = []
                for face in sorted_faces:
                    cx, cy = face['center']
                    is_dup = False
                    for known in unique:
                        kx, ky = known['center']
                        if ((cx - kx)**2 + (cy - ky)**2)**0.5 < min_distance:
                            is_dup = True
                            break
                    if not is_dup:
                        unique.append(face)
                return unique

            def assign_to_identity(face, frame_num):
                """Assign a detected face to an existing identity or create new one"""
                cx, cy = face['center']
                area = face.get('area', 0)
                
                # Find closest existing identity
                best_identity = None
                best_dist = IDENTITY_MERGE_DIST
                
                for identity in identities:
                    # Use average position of last 10 detections
                    recent = identity['positions'][-10:]
                    avg_x = sum(p[0] for p in recent) / len(recent)
                    avg_y = sum(p[1] for p in recent) / len(recent)
                    dist = ((cx - avg_x)**2 + (cy - avg_y)**2)**0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_identity = identity
                
                if best_identity:
                    best_identity['positions'].append((cx, cy))
                    best_identity['areas'].append(area)
                    best_identity['frames'].append(frame_num)
                else:
                    # New identity
                    identities.append({
                        'positions': [(cx, cy)],
                        'areas': [area],
                        'frames': [frame_num]
                    })

            self._log(f"")
            self._log(f"Scanning frames...")

            while True:
                if self.is_cancelled():
                    return [{'start': 0, 'end': total_frames, 'mode': 'single'}], []

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    total_sampled_frames += 1
                    
                    # Detect faces
                    faces = self.face_tracker.detect_faces(frame)
                    
                    # Basic filters: skip tiny faces and extreme positions
                    min_area = 2000
                    faces = [f for f in faces if f.get('area', 0) >= min_area]
                    
                    # Deduplicate
                    faces = simple_dedupe(faces, min_distance=100)
                    
                    # Assign each face to an identity
                    for face in faces:
                        assign_to_identity(face, frame_idx)
                    
                    # Record raw face count
                    face_counts.append(len(faces))

                    # Log every 5 seconds
                    time_sec = frame_idx / fps
                    if time_sec % 5 < (sample_interval / fps):
                        self._log(f"  @{time_sec:.1f}s → {len(faces)} faces, {len(identities)} identities")

                    if progress_callback and frame_idx % (sample_interval * 10) == 0:
                        progress_callback(20 + (frame_idx / total_frames) * 10, f"Scanning {time_sec:.0f}s...")

                frame_idx += 1

            if not face_counts or total_sampled_frames == 0:
                self._log("No samples - defaulting to SINGLE")
                return [{'start': 0, 'end': total_frames, 'mode': 'single'}], []

            # === ANALYZE IDENTITIES ===
            self._log(f"")
            self._log(f"╔══════════════════════════════════════════")
            self._log(f"║ IDENTITY ANALYSIS")
            self._log(f"╠══════════════════════════════════════════")
            self._log(f"║ Total sampled frames: {total_sampled_frames}")
            self._log(f"║ Identities found: {len(identities)}")
            
            # Score each identity
            scored_identities = []
            for i, identity in enumerate(identities):
                num_detections = len(identity['frames'])
                presence_ratio = num_detections / total_sampled_frames  # 0-1
                avg_area = sum(identity['areas']) / len(identity['areas'])
                
                # Position variance (how much this face moves)
                positions = identity['positions']
                if len(positions) >= 3:
                    avg_x = sum(p[0] for p in positions) / len(positions)
                    avg_y = sum(p[1] for p in positions) / len(positions)
                    var_x = sum((p[0] - avg_x)**2 for p in positions) / len(positions)
                    var_y = sum((p[1] - avg_y)**2 for p in positions) / len(positions)
                    pos_variance = (var_x + var_y) ** 0.5
                else:
                    pos_variance = 0
                
                # Average position
                avg_pos = (
                    sum(p[0] for p in positions) / len(positions),
                    sum(p[1] for p in positions) / len(positions)
                )
                
                scored_identities.append({
                    'index': i,
                    'detections': num_detections,
                    'presence': presence_ratio,
                    'avg_area': avg_area,
                    'pos_variance': pos_variance,
                    'avg_pos': avg_pos,
                    'identity': identity
                })
                
                self._log(f"║ Identity {i}: detections={num_detections}, presence={presence_ratio*100:.0f}%, area={avg_area:.0f}, var={pos_variance:.1f}px, pos=({avg_pos[0]:.0f},{avg_pos[1]:.0f})")
            
            self._log(f"╠══════════════════════════════════════════")
            
            # === DETERMINE REAL PEOPLE vs POSTERS ===
            # Criteria for REAL PERSON:
            # 1. Present in at least 50% of frames (consistent presence)
            # 2. Area is significant (not tiny background face)
            # 3. Position variance > 5px (shows SOME independent movement)
            #    Note: even with camera shake, a real person adds head movement on top
            #
            # Criteria for POSTER:
            # 1. May be present in many frames BUT
            # 2. Position variance is VERY LOW relative to real people
            #    (poster only moves from camera shake, real person moves MORE)
            
            # Find the identity with HIGHEST position variance = most likely real person
            if scored_identities:
                max_variance = max(s['pos_variance'] for s in scored_identities)
            else:
                max_variance = 0
            
            real_people = []
            posters = []
            
            for scored in scored_identities:
                presence = scored['presence']
                variance = scored['pos_variance']
                area = scored['avg_area']
                
                # Must be present in at least 30% of frames to matter
                if presence < 0.30:
                    self._log(f"║ Identity {scored['index']}: IGNORED (presence {presence*100:.0f}% < 30%)")
                    continue
                
                # KEY INSIGHT: Compare variance to the most-moving face
                # Real person has variance that's at least 50% of the max
                # Poster has much lower variance (only camera shake)
                if max_variance > 0:
                    variance_ratio = variance / max_variance
                else:
                    variance_ratio = 1.0
                
                # Decision logic:
                # - If this face moves at least 40% as much as the most-moving face → REAL
                # - If this face moves less than 40% of the most-moving face → POSTER
                # - Exception: if ALL faces have very low variance (< 10px), use area-based logic
                
                if max_variance < 10:
                    # Very static video (everyone sitting still, minimal camera movement)
                    # Fall back to: largest + most present = real
                    if presence >= 0.50 and area >= 5000:
                        real_people.append(scored)
                        self._log(f"║ Identity {scored['index']}: REAL (static video, presence={presence*100:.0f}%, area={area:.0f})")
                    else:
                        posters.append(scored)
                        self._log(f"║ Identity {scored['index']}: POSTER (static video, low presence/area)")
                else:
                    # Normal video with some movement
                    if variance_ratio >= 0.40:
                        real_people.append(scored)
                        self._log(f"║ Identity {scored['index']}: REAL (var_ratio={variance_ratio:.2f}, var={variance:.1f}px)")
                    else:
                        posters.append(scored)
                        # Mark as poster position
                        pos = scored['avg_pos']
                        grid_key = (int(pos[0] // 30) * 30, int(pos[1] // 30) * 30)
                        poster_positions.add(grid_key)
                        self._log(f"║ Identity {scored['index']}: POSTER (var_ratio={variance_ratio:.2f}, var={variance:.1f}px vs max={max_variance:.1f}px)")
            
            self._log(f"╠══════════════════════════════════════════")
            self._log(f"║ Real people: {len(real_people)}")
            self._log(f"║ Posters: {len(posters)}")
            
            # === FINAL DECISION ===
            if len(real_people) >= 2:
                video_mode = 'split'
                self._log(f"║ ★ DECISION: SPLIT MODE ★")
                self._log(f"║   Reason: {len(real_people)} real people detected")
            else:
                video_mode = 'single'
                self._log(f"║ ★ DECISION: SINGLE MODE ★")
                if len(real_people) == 1:
                    self._log(f"║   Reason: Only 1 real person (others are posters/inserts)")
                else:
                    self._log(f"║   Reason: No consistent real people detected")
            
            self._log(f"╚══════════════════════════════════════════")

            # Log detected poster positions
            if poster_positions:
                self._log(f"")
                self._log(f"[POSTER] Detected {len(poster_positions)} static poster positions:")
                for pos in list(poster_positions)[:5]:  # Show max 5
                    self._log(f"  - Grid position: ({pos[0]}, {pos[1]})")

            # Log optical flow results
            if self.optical_flow_enabled and self.optical_flow_filter:
                of_stats = self.optical_flow_filter.get_stats()
                if of_stats['confirmed_posters'] > 0 or of_stats['confirmed_real'] > 0:
                    self._log(f"")
                    self._log(f"[OPTICAL FLOW] Results:")
                    self._log(f"  - Confirmed REAL faces: {of_stats['confirmed_real']}")
                    self._log(f"  - Confirmed POSTER faces: {of_stats['confirmed_posters']}")
                    self._log(f"  - Pending (undecided): {of_stats['pending']}")
                    self._log(f"  - Total samples analyzed: {of_stats['total_samples']}")
                    for pos in of_stats['poster_positions'][:5]:
                        self._log(f"  - Poster at grid: ({pos[0]}, {pos[1]})")

                # Merge optical flow posters into poster_positions
                of_poster_positions = self.optical_flow_filter.get_poster_positions()
                for pos in of_poster_positions:
                    poster_positions.add(pos)

            if progress_callback:
                progress_callback(35, f"Prescan: {video_mode.upper()} mode")

            # Return single segment for entire video + poster positions
            segments = [{'start': 0, 'end': total_frames, 'mode': video_mode}]

            return segments, list(poster_positions)

        finally:
            # Ensure cap is always released
            if cap is not None:
                cap.release()

    def _get_mode_for_frame(self, frame_idx: int, segments: List[Dict]) -> str:
        """Get the mode (split/single) for a specific frame based on pre-scanned segments."""
        for seg in segments:
            if seg['start'] <= frame_idx < seg['end']:
                return seg['mode']
        return 'single'

    def _process_with_face_tracking(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Process video with face-centered cropping. Uses pre-scan or real-time based on settings."""

        if self.use_prescan:
            # Pre-scan mode: scan first, then process with stable segments
            self._process_with_prescan(input_path, output_path, progress_callback)
        else:
            # Real-time mode: decide mode on-the-fly (faster but less stable)
            self._process_realtime(input_path, output_path, progress_callback)

    def _process_with_prescan(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Process video with pre-scanned mode segments for stability"""

        self._log("━━━ STARTING PRESCAN MODE ━━━")

        # PHASE 1: Pre-scan video to determine mode segments and static faces
        result = self._scan_video_for_modes(input_path, progress_callback)

        # Handle both old format (just segments) and new format (segments, static_positions)
        if isinstance(result, tuple):
            segments, prescan_poster_positions = result
        else:
            segments = result
            prescan_poster_positions = []

        # Convert prescan poster list to set for fast lookup
        known_poster_positions = set()
        for pos in prescan_poster_positions:
            if isinstance(pos, tuple) and len(pos) >= 2:
                known_poster_positions.add((pos[0], pos[1]))

        if self.is_cancelled() or not segments:
            self._log("Prescan cancelled or no segments")
            return

        self._log(f"Prescan complete: {len(segments)} segment(s), {len(known_poster_positions)} poster position(s) to filter")

        # PHASE 2: Process video using pre-determined segments
        cap = cv2.VideoCapture(input_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if progress_callback:
            progress_callback(36, "Processing video with face tracking...")

        # Determine which tracking method to use
        use_dlib = self.use_dlib_tracking and self.use_embedder and self.face_embedder

        # Create video writer
        temp_output = output_path.replace('.mp4', '_raw.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (self.output_width, self.output_height))

        if not out.isOpened():
            temp_output = output_path.replace('.mp4', '_raw.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (self.output_width, self.output_height))

        self.face_tracker.reset_smoothing()
        self.face_tracker_2.reset_smoothing()
        frame_idx = 0
        default_center = (frame_width // 2, frame_height // 2)
        half_height = self.output_height // 2

        # Track current segment for logging
        current_segment_idx = 0

        # Smoothed positions - will be initialized on first frame with actual face positions
        smooth_x_1, smooth_y_1 = None, None
        smooth_x_2, smooth_y_2 = None, None
        smooth_x_single, smooth_y_single = None, None
        positions_initialized = False

        # Reference embeddings for identity tracking (set on first detection in split mode)
        ref_embedding_1 = None  # Person 1 (top/left)
        ref_embedding_2 = None  # Person 2 (bottom/right)

        # Track previous mode for transitions
        prev_mode = None

        # Use prescan poster positions ONLY - no runtime detection
        # Runtime detection was too aggressive and marked real faces as posters
        # known_poster_positions is from prescan (set of (grid_x, grid_y) tuples)

        def is_poster_face(face):
            """
            Check if this face is a known poster from prescan.
            Uses proximity matching (not exact grid) to handle slight position drift.
            """
            if not known_poster_positions:
                return False  # No posters detected in prescan

            cx, cy = face['center']

            # Check multiple grid sizes to handle mismatch between prescan (20px) and processing
            for grid_size in [20, 30, 40]:
                grid_x = int(cx // grid_size) * grid_size
                grid_y = int(cy // grid_size) * grid_size
                grid_key = (grid_x, grid_y)

                if grid_key in known_poster_positions:
                    if self.debug_mode_advanced:
                        self._debug(f"    [POSTER] Skipping face at ({cx:.0f},{cy:.0f}) - matches prescan poster at grid ({grid_x},{grid_y})")
                    return True

            # Also check proximity: if face center is within 50px of any known poster position
            for pos in known_poster_positions:
                dist = ((cx - pos[0]) ** 2 + (cy - pos[1]) ** 2) ** 0.5
                if dist < 50:
                    if self.debug_mode_advanced:
                        self._debug(f"    [POSTER] Skipping face at ({cx:.0f},{cy:.0f}) - within 50px of poster at ({pos[0]},{pos[1]})")
                    return True

            return False

        def filter_static_faces(faces):
            """Remove faces that were detected as posters during prescan"""
            if not known_poster_positions:
                return faces  # No filtering if no posters detected
            return [f for f in faces if not is_poster_face(f)]

        # Runtime poster detection during processing
        # Uses relative motion: compares how each face moves vs others
        runtime_face_positions: Dict[str, List[Tuple[float, float, int]]] = {}  # key -> [(cx, cy, frame_idx)]
        runtime_confirmed_posters: set = set()

        def runtime_poster_check(faces, current_frame_idx):
            """
            Runtime detection using RELATIVE MOTION.
            If a face always moves identically to other faces (no independent motion),
            it's a poster riding on camera shake.
            """
            nonlocal runtime_confirmed_posters

            # Track all faces
            for face in faces:
                cx, cy = face['center']
                grid_key = f"{int(cx // 40) * 40}_{int(cy // 40) * 40}"
                
                if grid_key not in runtime_face_positions:
                    runtime_face_positions[grid_key] = []
                runtime_face_positions[grid_key].append((cx, cy, current_frame_idx))
                
                # Keep last 90 positions
                if len(runtime_face_positions[grid_key]) > 90:
                    runtime_face_positions[grid_key] = runtime_face_positions[grid_key][-90:]

            # Need at least 2 tracked faces with enough history
            tracked_keys = [k for k, v in runtime_face_positions.items() 
                          if len(v) >= 15 and k not in runtime_confirmed_posters]
            
            if len(tracked_keys) >= 2 and current_frame_idx % 30 == 0:  # Check every 30 frames
                for key in tracked_keys:
                    if key in runtime_confirmed_posters:
                        continue
                    
                    my_positions = runtime_face_positions[key]
                    other_keys = [k for k in tracked_keys if k != key and k not in runtime_confirmed_posters]
                    if not other_keys:
                        continue
                    
                    # Calculate relative displacements
                    relative_motions = []
                    for i in range(1, min(len(my_positions), 30)):
                        my_dx = my_positions[i][0] - my_positions[i-1][0]
                        my_dy = my_positions[i][1] - my_positions[i-1][1]
                        
                        other_dxs = []
                        other_dys = []
                        for other_key in other_keys:
                            other_pos = runtime_face_positions[other_key]
                            if i < len(other_pos):
                                other_dxs.append(other_pos[i][0] - other_pos[i-1][0])
                                other_dys.append(other_pos[i][1] - other_pos[i-1][1])
                        
                        if other_dxs:
                            avg_other_dx = sum(other_dxs) / len(other_dxs)
                            avg_other_dy = sum(other_dys) / len(other_dys)
                            rel_mag = ((my_dx - avg_other_dx)**2 + (my_dy - avg_other_dy)**2)**0.5
                            relative_motions.append(rel_mag)
                    
                    if len(relative_motions) >= 10:
                        avg_rel = sum(relative_motions) / len(relative_motions)
                        independent_ratio = sum(1 for r in relative_motions if r > 1.5) / len(relative_motions)
                        
                        if avg_rel < 1.0 and independent_ratio < 0.15:
                            runtime_confirmed_posters.add(key)
                            self._log(f"[RUNTIME POSTER] Detected: {key} avg_rel={avg_rel:.2f}px indep={independent_ratio*100:.0f}%")

            # Filter out confirmed posters
            result = []
            for face in faces:
                cx, cy = face['center']
                grid_key = f"{int(cx // 40) * 40}_{int(cy // 40) * 40}"
                if grid_key not in runtime_confirmed_posters:
                    result.append(face)
            return result

        # Mode override DISABLED in prescan mode
        # Prescan segments are authoritative - no runtime override needed

        # Helper function to filter out likely posters/banners
        def filter_posters(faces, frame_height, frame_width=None):
            """
            Filter out faces that are likely posters/banners/static images.
            LESS AGGRESSIVE than prescan - we want to keep real faces.
            Only filter:
            - Faces in top 10% (definitely poster/banner area)
            - Faces in bottom 10% (definitely ad/overlay area)
            - Very small faces (area < 2000)
            """
            min_y = frame_height * 0.10  # Only top 10% is poster area
            max_y = frame_height * 0.90  # Only bottom 10% is ad area
            min_area = 2000  # Lower threshold - keep more faces

            filtered = []
            for face in faces:
                cx, cy = face['center']
                area = face.get('area', 10000)

                # Skip faces in top 10% of frame (definitely posters/banners)
                if cy < min_y:
                    if self.debug_mode_advanced:
                        self._debug(f"    [FILTER] Skip y={cy:.0f} (top 10%, min_y={min_y:.0f})")
                    continue

                # Skip faces in bottom 10% of frame (definitely ads/overlays)
                if cy > max_y:
                    if self.debug_mode_advanced:
                        self._debug(f"    [FILTER] Skip y={cy:.0f} (bottom 10%, max_y={max_y:.0f})")
                    continue

                # Skip very small faces
                if area < min_area:
                    if self.debug_mode_advanced:
                        self._debug(f"    [FILTER] Skip area={area:.0f} (too small, min={min_area})")
                    continue

                filtered.append(face)

            return filtered

        # Helper function to deduplicate faces - SIMPLE distance-based
        def deduplicate_faces(faces, min_distance=100):
            """
            Simple deduplication: remove faces that are too close to each other.
            Keeps the one with highest confidence.
            """
            if len(faces) <= 1:
                return faces

            # Sort by confidence (highest first)
            sorted_faces = sorted(faces, key=lambda f: f.get('confidence', 0), reverse=True)
            unique = []

            for face in sorted_faces:
                cx, cy = face['center']
                is_duplicate = False

                for known in unique:
                    kx, ky = known['center']
                    dist = ((cx - kx)**2 + (cy - ky)**2)**0.5
                    if dist < min_distance:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique.append(face)

            return unique

        def filter_by_area_ratio_processing(faces):
            """
            If there are 2 faces and one is much smaller, the smaller is likely a poster.
            Used during processing phase.
            """
            if len(faces) != 2:
                return faces

            face_a, face_b = faces
            area_a = face_a.get('area', 10000)
            area_b = face_b.get('area', 10000)

            # Calculate ratio (bigger / smaller)
            if area_a > area_b:
                ratio = area_a / area_b if area_b > 0 else 999
                larger, smaller = face_a, face_b
            else:
                ratio = area_b / area_a if area_a > 0 else 999
                larger, smaller = face_b, face_a

            # If ratio > 2.0:1, the smaller one is likely a poster
            if ratio > 2.0:
                smaller_cx = smaller['center'][0]
                # If smaller face is near edge (outer 25%), definitely poster
                if smaller_cx < frame_width * 0.25 or smaller_cx > frame_width * 0.75:
                    self._log(f"[POSTER] Filtered: ratio={ratio:.1f}x, edge x={smaller_cx:.0f}")
                    return [larger]

                # If ratio > 2.5:1, filter regardless of position
                if ratio > 2.5:
                    self._log(f"[POSTER] Filtered: ratio={ratio:.1f}x (area diff too large)")
                    return [larger]

            return faces

        # Log prescan segments
        self._log(f"")
        self._log(f"╔══════════════════════════════════════════")
        self._log(f"║ PROCESSING START")
        self._log(f"║ Total frames: {total_frames}, FPS: {fps:.1f}")
        self._log(f"║ Segments from prescan: {len(segments)}")
        if known_poster_positions:
            self._log(f"║ Poster positions to filter: {len(known_poster_positions)}")
            for pos in list(known_poster_positions)[:3]:  # Show max 3
                self._log(f"║   → ({pos[0]:.0f},{pos[1]:.0f})")
        for seg in segments:
            duration = (seg['end'] - seg['start']) / fps
            self._log(f"║   → Frame {seg['start']}-{seg['end']} ({duration:.1f}s): {seg['mode'].upper()}")
        self._log(f"╚══════════════════════════════════════════")

        # Basic debug: Print minimal settings at start
        if self.debug_mode or self.debug_mode_advanced:
            self._debug_basic(f"Processing: {total_frames} frames @ {fps:.1f} FPS, {len(segments)} segments")
            if known_poster_positions:
                self._debug_basic(f"Filtering {len(known_poster_positions)} poster positions")

        # Debug: Print detailed settings at start
        if self.debug_mode_advanced:
            self._debug("="*60)
            self._debug("DEBUG MODE ENABLED - Detailed tracking logs active")
            self._debug("="*60)
            self._debug(f"Video: {total_frames} frames @ {fps:.1f} FPS")
            self._debug(f"Tracking settings:")
            self._debug(f"  - smoothing: {self.tracking_smoothing}")
            self._debug(f"  - tracking_speed: {self.tracking_speed}")
            self._debug(f"  - deadzone: {self.tracking_deadzone}")
            self._debug(f"  - lerp_factor: {self._get_lerp_factor():.3f}")
            self._debug(f"  - effective_deadzone: {self._get_effective_deadzone():.1f}px")
            self._debug(f"  - split_screen: {self.enable_split_screen}")
            self._debug(f"  - prescan: {self.use_prescan}")
            self._debug(f"Segments: {len(segments)}")
            for i, seg in enumerate(segments):
                duration = (seg['end'] - seg['start']) / fps
                self._debug(f"  [{i}] frames {seg['start']}-{seg['end']} ({duration:.1f}s) = {seg['mode'].upper()}")
            if known_poster_positions:
                self._debug(f"Poster positions from prescan: {len(known_poster_positions)}")
                for pos in list(known_poster_positions)[:5]:
                    self._debug(f"  - ({pos[0]:.0f}, {pos[1]:.0f})")
            self._debug("="*60)

        while True:
            # Check for cancellation
            if self.is_cancelled():
                if progress_callback:
                    progress_callback(0, "Processing cancelled")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Simple face detection - just use YOLO directly
            faces = self.face_tracker.detect_faces(frame)
            raw_count = len(faces)

            # DEBUG: Log raw detections every 3 seconds
            log_every_n_frames = max(1, int(fps * 3))
            if frame_idx % log_every_n_frames == 0:
                time_sec = frame_idx / fps
                if self.debug_mode or self.debug_mode_advanced:
                    raw_str = " | ".join([f"({f['center'][0]:.0f},{f['center'][1]:.0f}) area={f.get('area',0):.0f}" for f in faces]) if faces else "none"
                    self._debug_basic(f"@{time_sec:.1f}s RAW: {raw_count} faces: {raw_str}")

            # Filter out posters/banners (faces too high up, too low, at edges, or too small)
            faces = filter_posters(faces, frame_height, frame_width)
            poster_filtered_count = len(faces)

            # DEBUG: Log after poster filter
            if frame_idx % log_every_n_frames == 0 and self.debug_mode_advanced:
                if poster_filtered_count != raw_count:
                    self._debug(f"  → After filter_posters: {raw_count} → {poster_filtered_count}")

            # Filter out static faces (posters that don't move) - uses prescan + runtime detection
            faces = filter_static_faces(faces)
            static_filtered_count = len(faces)

            # DEBUG: Log after static filter
            if frame_idx % log_every_n_frames == 0 and self.debug_mode_advanced:
                if static_filtered_count != poster_filtered_count:
                    self._debug(f"  → After filter_static: {poster_filtered_count} → {static_filtered_count}")

            # Filter by area ratio - if 2 faces and one is much smaller, it's likely a poster
            faces = filter_by_area_ratio_processing(faces)

            # Simple deduplication - remove faces too close together
            faces = deduplicate_faces(faces, min_distance=100)

            # Optical Flow filtering during processing (uses prescan data)
            if self.optical_flow_enabled and self.optical_flow_filter:
                # Continue analyzing for faces not yet confirmed
                self.optical_flow_filter.analyze_frame(frame, faces)
                # Filter out confirmed posters
                faces = self.optical_flow_filter.filter_faces(faces)

            # Runtime poster detection (catches posters that prescan missed)
            faces = runtime_poster_check(faces, frame_idx)

            # RESPECT PRESCAN DECISION: Use the mode determined by prescan
            # Prescan already analyzed identities and determined real people vs posters
            # Only use split if prescan said split AND we actually see 2+ faces
            prescan_mode = segments[0]['mode'] if segments else 'single'
            
            if prescan_mode == 'split' and len(faces) >= 2 and self.enable_split_screen:
                render_mode = 'split'
            else:
                render_mode = 'single'
                # Keep only largest face if multiple detected
                if len(faces) > 1:
                    faces = [max(faces, key=lambda f: f['area'])]

            # Log every 3 seconds (now includes raw count for debugging)
            if frame_idx % log_every_n_frames == 0:
                time_sec = frame_idx / fps
                face_str = " | ".join([f"({f['center'][0]:.0f},{f['center'][1]:.0f})" for f in faces]) if faces else "none"
                # Show raw vs filtered count if different
                if raw_count != len(faces):
                    self._log(f"@{time_sec:.1f}s → raw={raw_count} → {len(faces)} faces → {render_mode.upper()}: {face_str}")
                else:
                    self._log(f"@{time_sec:.1f}s → {len(faces)} faces → {render_mode.upper()}: {face_str}")

            # Initialize positions on first frame with faces
            if not positions_initialized and len(faces) >= 1:
                if len(faces) >= 2:
                    # Sort by X position: LEFT person = face_1 (top), RIGHT person = face_2 (bottom)
                    sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                    smooth_x_1 = float(sorted_faces[0]['center'][0])
                    smooth_y_1 = float(sorted_faces[0]['center'][1])
                    smooth_x_2 = float(sorted_faces[1]['center'][0])
                    smooth_y_2 = float(sorted_faces[1]['center'][1])
                    self._log(f"Init SPLIT: LEFT=({smooth_x_1:.0f},{smooth_y_1:.0f}) → TOP, RIGHT=({smooth_x_2:.0f},{smooth_y_2:.0f}) → BOTTOM")
                else:
                    smooth_x_single = float(faces[0]['center'][0])
                    smooth_y_single = float(faces[0]['center'][1])
                positions_initialized = True

            prev_mode = render_mode

            # === SIMPLE RENDERING ===
            if render_mode == 'split' and len(faces) >= 2:
                # SPLIT MODE: 2 faces detected
                # Use PROXIMITY MATCHING to maintain identity (not re-sorting every frame)
                if smooth_x_1 is not None and smooth_x_2 is not None:
                    # Match faces to closest previous position
                    face_a, face_b = faces[0], faces[1]

                    # Calculate distances to previous positions
                    dist_a_to_1 = ((face_a['center'][0] - smooth_x_1)**2 + (face_a['center'][1] - smooth_y_1)**2)**0.5
                    dist_a_to_2 = ((face_a['center'][0] - smooth_x_2)**2 + (face_a['center'][1] - smooth_y_2)**2)**0.5
                    dist_b_to_1 = ((face_b['center'][0] - smooth_x_1)**2 + (face_b['center'][1] - smooth_y_1)**2)**0.5
                    dist_b_to_2 = ((face_b['center'][0] - smooth_x_2)**2 + (face_b['center'][1] - smooth_y_2)**2)**0.5

                    # Assign based on minimum total distance
                    if dist_a_to_1 + dist_b_to_2 <= dist_a_to_2 + dist_b_to_1:
                        face_1, face_2 = face_a, face_b
                    else:
                        face_1, face_2 = face_b, face_a
                else:
                    # First time: sort by X (LEFT = top, RIGHT = bottom)
                    sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                    face_1, face_2 = sorted_faces[0], sorted_faces[1]

                # Smooth tracking
                lerp = self._get_lerp_factor()
                target_x1, target_y1 = float(face_1['center'][0]), float(face_1['center'][1])
                target_x2, target_y2 = float(face_2['center'][0]), float(face_2['center'][1])

                if smooth_x_1 is None:
                    smooth_x_1, smooth_y_1 = target_x1, target_y1
                    smooth_x_2, smooth_y_2 = target_x2, target_y2
                else:
                    smooth_x_1 += (target_x1 - smooth_x_1) * lerp
                    smooth_y_1 += (target_y1 - smooth_y_1) * lerp
                    smooth_x_2 += (target_x2 - smooth_x_2) * lerp
                    smooth_y_2 += (target_y2 - smooth_y_2) * lerp

                # Render split screen
                half_height = self.output_height // 2
                center_1 = (int(smooth_x_1), int(smooth_y_1))
                center_2 = (int(smooth_x_2), int(smooth_y_2))

                crop1 = self._calculate_split_crop_region(frame_width, frame_height, center_1)
                x1, y1, x2, y2 = crop1
                cropped1 = frame[y1:y2, x1:x2]
                resized1 = self._gpu_resize(cropped1, (self.output_width, half_height))

                crop2 = self._calculate_split_crop_region(frame_width, frame_height, center_2)
                x1, y1, x2, y2 = crop2
                cropped2 = frame[y1:y2, x1:x2]
                resized2 = self._gpu_resize(cropped2, (self.output_width, half_height))

                output_frame = np.vstack([resized1, resized2])

            else:
                # SINGLE MODE: 0 or 1 face detected
                if len(faces) >= 1:
                    face = faces[0]
                    target_x = float(face['center'][0])
                    target_y = float(face['center'][1])

                    # Check for large position jump (person switch) - INSTANT JUMP
                    if smooth_x_single is not None:
                        distance = ((target_x - smooth_x_single)**2 + (target_y - smooth_y_single)**2)**0.5
                        if distance > 150:  # Large jump = instant switch, no drifting
                            smooth_x_single, smooth_y_single = target_x, target_y
                        else:
                            # Smooth tracking for small movements
                            lerp = self._get_lerp_factor()
                            smooth_x_single += (target_x - smooth_x_single) * lerp
                            smooth_y_single += (target_y - smooth_y_single) * lerp
                    else:
                        smooth_x_single, smooth_y_single = target_x, target_y

                    center = (int(smooth_x_single), int(smooth_y_single))
                else:
                    # No face - use center or last known position
                    if smooth_x_single is not None:
                        center = (int(smooth_x_single), int(smooth_y_single))
                    else:
                        center = (frame_width // 2, frame_height // 2)

                crop_region = self._calculate_single_crop_region(frame_width, frame_height, center)
                x1, y1, x2, y2 = crop_region
                cropped = frame[y1:y2, x1:x2]
                output_frame = self._gpu_resize(cropped, (self.output_width, self.output_height))

            # Write frame
            if output_frame is not None and output_frame.size > 0:
                if not output_frame.flags['C_CONTIGUOUS']:
                    output_frame = np.ascontiguousarray(output_frame)
                if output_frame.dtype != np.uint8:
                    output_frame = output_frame.astype(np.uint8)

                # Render subtitle if enabled (TikTok/CapCut style)
                if self.subtitle_enabled and self.subtitle_renderer and self.subtitle_renderer.segments:
                    current_time = frame_idx / fps
                    output_frame = self.subtitle_renderer.render_subtitle(
                        output_frame,
                        current_time,
                        self.clip_start_time
                    )

                out.write(output_frame)

            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress = 36 + (frame_idx / total_frames) * 49  # 36% to 85%
                progress_callback(progress, f"Frame {frame_idx}/{total_frames} ({render_mode})")

        # === PROCESSING SUMMARY LOG ===
        self._log(f"")
        self._log(f"╔══════════════════════════════════════════")
        self._log(f"║ PROCESSING COMPLETE")
        self._log(f"║ Total frames processed: {frame_idx}")
        self._log(f"║ Segments used: {len(segments)}")
        for seg in segments:
            seg_frames = seg['end'] - seg['start']
            seg_duration = seg_frames / fps
            self._log(f"║   {seg['mode'].upper()}: frame {seg['start']}-{seg['end']} ({seg_duration:.1f}s)")
        self._log(f"╚══════════════════════════════════════════")

        # Basic debug: completion summary
        if self.debug_mode or self.debug_mode_advanced:
            self._debug_basic(f"Complete: {frame_idx} frames processed, {len(segments)} segments")

        # Release resources BEFORE re-encoding (to unlock the file)
        cap.release()
        out.release()

        if progress_callback:
            progress_callback(85, "Encoding final video...")

        # Re-encode with FFmpeg for better compression
        self._reencode_video(temp_output, output_path, fps)

        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)

    def _process_realtime(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Process video with real-time face tracking (no prescan)"""

        cap = cv2.VideoCapture(input_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._log(f"=== REALTIME MODE (No Prescan) ===")
        self._log(f"Video: {total_frames} frames, {fps:.1f} fps")
        self._log(f"Split screen enabled: {self.enable_split_screen}")

        # Log tracking mode
        if not self.dynamic_tracking and not self.enable_tracking_analyzer:
            self._log(f"Mode: CLASSIFIER ONLY (face_classifier.pt)")
        elif not self.dynamic_tracking:
            self._log(f"Mode: Standard + Tracking Analyzer")
        elif not self.enable_tracking_analyzer:
            self._log(f"Mode: Dynamic Tracking (no AI analyzer)")
        else:
            self._log(f"Mode: Full AI (Dynamic + Analyzer)")

        if progress_callback:
            progress_callback(20, "Loading face recognition model...")

        # Determine which tracking method to use
        use_dlib = self.use_dlib_tracking and self.use_embedder and self.face_embedder

        # Load face embedder if using dlib method
        if use_dlib:
            def embedder_log(msg):
                if progress_callback:
                    progress_callback(21, msg)
            self.face_embedder.load_model(log_callback=embedder_log)
            if progress_callback:
                progress_callback(22, "Face recognition ready (dlib CUDA)")

        # Create video writer
        temp_output = output_path.replace('.mp4', '_raw.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (self.output_width, self.output_height))

        if not out.isOpened():
            temp_output = output_path.replace('.mp4', '_raw.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (self.output_width, self.output_height))

        self.face_tracker.reset_smoothing()
        self.face_tracker_2.reset_smoothing()
        frame_idx = 0
        default_center = (frame_width // 2, frame_height // 2)
        half_height = self.output_height // 2

        # Split screen state
        split_screen_active = False
        prev_split_screen_active = False

        # Smoothed positions - will be initialized on first frame
        smooth_x_1, smooth_y_1 = None, None
        smooth_x_2, smooth_y_2 = None, None
        smooth_x_single, smooth_y_single = None, None
        positions_initialized = False

        # Reference embeddings for identity tracking
        ref_embedding_1 = None
        ref_embedding_2 = None

        # Hysteresis counter - FASTER response for realtime mode
        frames_with_2_faces = 0
        frames_with_1_face = 0
        # Enter split quickly (1s) when we see 2 faces
        ENTER_SPLIT_THRESHOLD = 30
        # Exit split quickly (0.5s) when faces drop
        EXIT_SPLIT_THRESHOLD = 15

        self._log(f"Thresholds: enter_split={ENTER_SPLIT_THRESHOLD} frames, exit_split={EXIT_SPLIT_THRESHOLD} frames")

        # Mode override (from AI Analyzer)
        override_mode = None
        override_until_frame = 0

        while True:
            if self.is_cancelled():
                if progress_callback:
                    progress_callback(0, "Processing cancelled")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Apply AI Override if active
            if override_mode and frame_idx < override_until_frame:
                if override_mode == 'split':
                    # Sanity check: Even if overridden, if we really don't have 2 faces, don't force it blindly
                    # unless it's a very short drop
                    split_screen_active = True
                    frames_with_2_faces = ENTER_SPLIT_THRESHOLD + 10
                    frames_with_1_face = 0
                else:
                    split_screen_active = False
                    frames_with_1_face = EXIT_SPLIT_THRESHOLD + 10
                    frames_with_2_faces = 0
            elif frame_idx >= override_until_frame:
                override_mode = None

            # Use Deep Scan (which handles simple scan internally)
            if self.dynamic_tracking:
                faces = self._deep_scan_for_faces(frame)
            else:
                # Detect faces (Standard) - rely on classifier for filtering
                if use_dlib and self.face_embedder.is_loaded:
                    faces = self.face_embedder.get_face_embeddings(frame)
                else:
                    faces = self.face_tracker.detect_faces(frame)

            # Deduplicate faces with similar X positions (same person detected twice)
            # Two faces with X within 100px are likely the same person
            pre_dedup_count = len(faces)
            if len(faces) > 1:
                # Sort by X position
                sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                deduplicated = []
                i = 0
                x_threshold = 100
                while i < len(sorted_faces):
                    current = sorted_faces[i]
                    cx = current['center'][0]
                    group = [current]
                    j = i + 1
                    while j < len(sorted_faces):
                        nx = sorted_faces[j]['center'][0]
                        if abs(nx - cx) < x_threshold:
                            group.append(sorted_faces[j])
                            j += 1
                        else:
                            break
                    best = max(group, key=lambda f: f.get('confidence', 0))
                    deduplicated.append(best)
                    i = j
                faces = deduplicated
            dedup_count = pre_dedup_count - len(faces)

            # Classifier DISABLED - motion detection is used in prescan mode
            # In realtime mode, we skip filtering (use raw detections)

            # Log face count periodically (every 2 seconds)
            if frame_idx % int(fps * 2) == 0:
                dedup_note = f" (-{dedup_count} dup)" if dedup_count > 0 else ""
                self._log(f"Frame {frame_idx}: {len(faces)} faces{dedup_note}, mode={'SPLIT' if split_screen_active else 'SINGLE'}")

            # Tracking Analyzer DISABLED for simpler pipeline

            # Initialize positions on first frame with actual face positions (prevents initial drifting)
            if not positions_initialized:
                if len(faces) >= 2:
                    sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                    smooth_x_1, smooth_y_1 = float(sorted_faces[0]['center'][0]), float(sorted_faces[0]['center'][1])
                    smooth_x_2, smooth_y_2 = float(sorted_faces[1]['center'][0]), float(sorted_faces[1]['center'][1])
                    largest = max(faces, key=lambda f: f['area'])
                    smooth_x_single, smooth_y_single = float(largest['center'][0]), float(largest['center'][1])
                elif len(faces) == 1:
                    smooth_x_single, smooth_y_single = float(faces[0]['center'][0]), float(faces[0]['center'][1])
                    smooth_x_1, smooth_y_1 = smooth_x_single, smooth_y_single
                    smooth_x_2, smooth_y_2 = smooth_x_single, smooth_y_single
                else:
                    smooth_x_single, smooth_y_single = float(default_center[0]), float(default_center[1])
                    smooth_x_1, smooth_y_1 = smooth_x_single, smooth_y_single
                    smooth_x_2, smooth_y_2 = smooth_x_single, smooth_y_single
                positions_initialized = True

            # Count faces for split/single decision
            if len(faces) >= 2 and self.enable_split_screen:
                frames_with_2_faces += 1
                frames_with_1_face = 0
                if frames_with_2_faces >= ENTER_SPLIT_THRESHOLD:
                    if not split_screen_active:
                        self._log(f"Frame {frame_idx}: Detected 2+ faces for {ENTER_SPLIT_THRESHOLD} frames → ENTERING SPLIT MODE")
                    split_screen_active = True
            else:
                frames_with_1_face += 1
                frames_with_2_faces = 0
                if frames_with_1_face >= EXIT_SPLIT_THRESHOLD:
                    if split_screen_active:
                        self._log(f"Frame {frame_idx}: Less than 2 faces for {EXIT_SPLIT_THRESHOLD} frames → EXITING SPLIT MODE")
                    split_screen_active = False

            # Detect mode change - initialize positions to actual face positions to prevent drifting/glitch
            if split_screen_active != prev_split_screen_active:
                if split_screen_active:
                    # Entering split mode - initialize to actual face positions
                    if len(faces) >= 2:
                        sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                        smooth_x_1, smooth_y_1 = float(sorted_faces[0]['center'][0]), float(sorted_faces[0]['center'][1])
                        smooth_x_2, smooth_y_2 = float(sorted_faces[1]['center'][0]), float(sorted_faces[1]['center'][1])
                    else:
                        smooth_x_1, smooth_y_1 = smooth_x_single, smooth_y_single
                        smooth_x_2, smooth_y_2 = smooth_x_single, smooth_y_single
                    # Reset embeddings for new split segment
                    ref_embedding_1 = None
                    ref_embedding_2 = None
                else:
                    # Exiting split mode - jump to largest face position directly
                    if len(faces) >= 1:
                        largest = max(faces, key=lambda f: f['area'])
                        smooth_x_single = float(largest['center'][0])
                        smooth_y_single = float(largest['center'][1])
                    else:
                        smooth_x_single = (smooth_x_1 + smooth_x_2) / 2
                        smooth_y_single = (smooth_y_1 + smooth_y_2) / 2
                prev_split_screen_active = split_screen_active

            # Render based on current mode
            if split_screen_active:
                # SPLIT MODE (Robust: render split if active, even if faces drop momentarily)
                if len(faces) >= 2:
                    # === SMART FACE SELECTION FOR 3+ PEOPLE ===
                    if len(faces) >= 3 and self.dynamic_focus:
                        # 3+ faces: Pick speaker + largest face
                        current_time = frame_idx / fps
                        active_speaker = self._get_active_speaker(current_time)

                        # Sort by area (largest first)
                        sorted_by_size = sorted(faces, key=lambda f: f.get('area', 0), reverse=True)
                        largest_face = sorted_by_size[0]

                        # Sort by X position (left to right)
                        sorted_by_x = sorted(faces, key=lambda f: f['center'][0])
                        left_face = sorted_by_x[0]
                        right_face = sorted_by_x[-1]

                        # Determine speaker face based on audio timeline
                        if active_speaker == 'left':
                            speaker_face = left_face
                        elif active_speaker == 'right':
                            speaker_face = right_face
                        else:
                            # 'both' or None - use second largest as speaker
                            speaker_face = sorted_by_size[1] if len(sorted_by_size) > 1 else largest_face

                        # Avoid duplicates: if speaker is same as largest, pick next largest
                        if speaker_face == largest_face and len(sorted_by_size) > 1:
                            face_1 = speaker_face  # Speaker on top
                            face_2 = sorted_by_size[1]  # Second largest on bottom
                        else:
                            face_1 = speaker_face  # Speaker on top
                            face_2 = largest_face  # Largest on bottom

                        # Log periodically
                        if frame_idx % int(fps * 3) == 0:
                            self._log(f"3+ faces: speaker={active_speaker}, using speaker + largest")

                    elif use_dlib and self.face_embedder.is_loaded:
                        # Handle 3+ faces without dynamic focus: use 2 largest
                        if len(faces) >= 3:
                            sorted_by_size = sorted(faces, key=lambda f: f.get('area', 0), reverse=True)
                            face_1 = sorted_by_size[0]
                            face_2 = sorted_by_size[1]
                            ref_embedding_1 = face_1.get('embedding')
                            ref_embedding_2 = face_2.get('embedding')
                            if frame_idx % int(fps * 3) == 0:
                                self._log(f"3+ faces (no dynamic focus): using 2 largest faces")
                        elif ref_embedding_1 is None or ref_embedding_2 is None:
                            sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                            face_1 = sorted_faces[0]
                            face_2 = sorted_faces[1]
                            ref_embedding_1 = face_1.get('embedding')
                            ref_embedding_2 = face_2.get('embedding')
                        else:
                            face_1, face_2 = self.face_embedder.match_faces_by_embedding(
                                faces, ref_embedding_1, ref_embedding_2
                            )
                            if face_1 is None or face_2 is None:
                                sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                                face_1 = sorted_faces[0]
                                face_2 = sorted_faces[1]
                            else:
                                if face_1.get('embedding') is not None:
                                    ref_embedding_1 = 0.95 * ref_embedding_1 + 0.05 * face_1['embedding']
                                if face_2.get('embedding') is not None:
                                    ref_embedding_2 = 0.95 * ref_embedding_2 + 0.05 * face_2['embedding']
                    else:
                        # Non-dlib path: for 3+ faces, use 2 largest; otherwise use leftmost 2
                        if len(faces) >= 3:
                            sorted_by_size = sorted(faces, key=lambda f: f.get('area', 0), reverse=True)
                            face_1 = sorted_by_size[0]
                            face_2 = sorted_by_size[1]
                            if frame_idx % int(fps * 3) == 0:
                                self._log(f"3+ faces (YOLO): using 2 largest faces")
                        else:
                            sorted_faces = sorted(faces, key=lambda f: f['center'][0])
                            face_1 = sorted_faces[0]
                            face_2 = sorted_faces[1]

                    lerp = self._get_lerp_factor()
                    deadzone = self.tracking_deadzone

                    target_x1, target_y1 = float(face_1['center'][0]), float(face_1['center'][1])
                    dx1 = target_x1 - smooth_x_1
                    dy1 = target_y1 - smooth_y_1
                    if abs(dx1) > deadzone:
                        smooth_x_1 += dx1 * lerp
                    if abs(dy1) > deadzone:
                        smooth_y_1 += dy1 * lerp

                    target_x2, target_y2 = float(face_2['center'][0]), float(face_2['center'][1])
                    dx2 = target_x2 - smooth_x_2
                    dy2 = target_y2 - smooth_y_2
                    if abs(dx2) > deadzone:
                        smooth_x_2 += dx2 * lerp
                    if abs(dy2) > deadzone:
                        smooth_y_2 += dy2 * lerp

                elif len(faces) == 1:
                     # Only 1 face found in split mode.
                     # Update the tracker closest to this face, hold the other one.
                     found_face = faces[0]
                     fx, fy = found_face['center']

                     # Calculate distance to current smooth positions
                     dist1 = ((fx - smooth_x_1)**2 + (fy - smooth_y_1)**2)**0.5
                     dist2 = ((fx - smooth_x_2)**2 + (fy - smooth_y_2)**2)**0.5

                     lerp = self._get_lerp_factor()
                     deadzone = self._get_effective_deadzone()

                     if dist1 < dist2:
                         # Update Tracker 1
                         dx1 = float(fx) - smooth_x_1
                         dy1 = float(fy) - smooth_y_1
                         if abs(dx1) > deadzone: smooth_x_1 += dx1 * lerp
                         if abs(dy1) > deadzone: smooth_y_1 += dy1 * lerp
                     else:
                         # Update Tracker 2
                         dx2 = float(fx) - smooth_x_2
                         dy2 = float(fy) - smooth_y_2
                         if abs(dx2) > deadzone: smooth_x_2 += dx2 * lerp
                         if abs(dy2) > deadzone: smooth_y_2 += dy2 * lerp

                # If 0 faces, we just hold last positions

                smooth_center_1 = (int(smooth_x_1), int(smooth_y_1))
                smooth_center_2 = (int(smooth_x_2), int(smooth_y_2))

                # Dynamic Focus: Determine which speaker goes on top
                # Default: left face (person 1) on top
                speaker_1_on_top = True

                if self.dynamic_focus and self.speaker_timeline:
                    current_time = frame_idx / fps
                    active_speaker = self._get_active_speaker(current_time)

                    if active_speaker == 'right':
                        # Right speaker is active - put them on top
                        speaker_1_on_top = False
                    elif active_speaker == 'left':
                        speaker_1_on_top = True
                    # 'both' or None - keep default (left on top)

                if speaker_1_on_top:
                    top_center = smooth_center_1
                    bottom_center = smooth_center_2
                else:
                    top_center = smooth_center_2
                    bottom_center = smooth_center_1

                crop1 = self._calculate_split_crop_region(frame_width, frame_height, top_center)
                x1, y1, x2, y2 = crop1
                cropped1 = frame[y1:y2, x1:x2]
                resized1 = self._gpu_resize(cropped1, (self.output_width, half_height))

                crop2 = self._calculate_split_crop_region(frame_width, frame_height, bottom_center)
                x1, y1, x2, y2 = crop2
                cropped2 = frame[y1:y2, x1:x2]
                resized2 = self._gpu_resize(cropped2, (self.output_width, half_height))

                output_frame = np.vstack([resized1, resized2])
                if not output_frame.flags['C_CONTIGUOUS']:
                    output_frame = np.ascontiguousarray(output_frame)
                cv2.line(output_frame, (0, half_height), (self.output_width, half_height), (30, 30, 30), 2)

            else:
                # SINGLE MODE
                if len(faces) >= 1:
                    largest = max(faces, key=lambda f: f['area'])
                    target_x, target_y = float(largest['center'][0]), float(largest['center'][1])
                else:
                    target_x, target_y = smooth_x_single, smooth_y_single

                # Check for large position jump (person switch) - INSTANT JUMP
                distance = ((target_x - smooth_x_single)**2 + (target_y - smooth_y_single)**2)**0.5
                if distance > 150:  # Large jump = instant switch, no drifting
                    smooth_x_single, smooth_y_single = target_x, target_y
                else:
                    # Smooth tracking for small movements
                    lerp = self._get_lerp_factor()
                    deadzone = self._get_effective_deadzone()

                    dx = target_x - smooth_x_single
                    dy = target_y - smooth_y_single
                    if abs(dx) > deadzone:
                        smooth_x_single += dx * lerp
                    if abs(dy) > deadzone:
                        smooth_y_single += dy * lerp

                smooth_center = (int(smooth_x_single), int(smooth_y_single))

                # --- TRACKING ANALYZER MONITORING ---
                if self.enable_tracking_analyzer and self.tracking_monitor:
                    # Collect stats for this frame
                    frame_stats = {
                        "frame": frame_idx,
                        "timestamp": frame_idx / fps,
                        "mode": "split" if split_screen_active else "single",
                        "faces_count": len(faces),
                        "avg_confidence": sum(f.get('confidence', 0) for f in faces) / len(faces) if faces else 0
                    }

                    # Analyze stability
                    report = self.tracking_monitor.analyze_frame_stability(frame_stats)

                    if report["status"] == "ACTION":
                        self._log(f"⚠ Analyzer Action (RT): {report.get('issue', 'unknown').upper()} -> {report['suggestion']}")

                        # Corrective Action
                        if report["suggestion"] == "force_split":
                            if not split_screen_active:
                                 self._log(f"  -> Forcing SPLIT mode to stabilize")
                                 split_screen_active = True
                                 # Reset hysteresis to keep it here for a while
                                 frames_with_2_faces = ENTER_SPLIT_THRESHOLD + 10
                                 frames_with_1_face = 0

                    elif report["status"] != "OK":
                        if report.get("severity") == "high":
                            self._log(f"⚠ Analyzer Warning (RT): {report.get('issue', 'unknown')}")

                crop_region = self._calculate_single_crop_region(frame_width, frame_height, smooth_center)
                x1, y1, x2, y2 = crop_region
                cropped = frame[y1:y2, x1:x2]
                output_frame = self._gpu_resize(cropped, (self.output_width, self.output_height))

            if output_frame is not None and output_frame.size > 0:
                if not output_frame.flags['C_CONTIGUOUS']:
                    output_frame = np.ascontiguousarray(output_frame)
                if output_frame.dtype != np.uint8:
                    output_frame = output_frame.astype(np.uint8)

                # Render subtitle if enabled (TikTok/CapCut style)
                if self.subtitle_enabled and self.subtitle_renderer and self.subtitle_renderer.segments:
                    current_time = frame_idx / fps
                    output_frame = self.subtitle_renderer.render_subtitle(
                        output_frame,
                        current_time,
                        self.clip_start_time
                    )

                out.write(output_frame)

            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                mode = "split" if split_screen_active else "single"
                progress = 20 + (frame_idx / total_frames) * 65
                progress_callback(progress, f"Frame {frame_idx}/{total_frames} ({mode})")

        # Release resources BEFORE re-encoding (to unlock the file)
        cap.release()
        out.release()

        if progress_callback:
            progress_callback(85, "Encoding final video...")

        self._reencode_video(temp_output, output_path, fps)

        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)

    def _reencode_video(self, input_path: str, output_path: str, fps: float) -> None:
        """Re-encode video with FFmpeg using stream copy or re-encode as fallback"""
        # First try stream copy (fastest, no re-encoding)
        cmd_copy = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c', 'copy',
            '-movflags', '+faststart',
            output_path
        ]

        try:
            result = subprocess.run(cmd_copy, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=300)
            if result.returncode == 0:
                return
        except Exception:
            pass

        # Fallback: re-encode with libx264
        cmd_encode = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]

        try:
            result = subprocess.run(cmd_encode, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=300)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            raise Exception("FFmpeg timeout")
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg.")
    
    def add_audio_from_source(
        self,
        video_path: str,
        source_video: str,
        start_time: float,
        end_time: float,
        output_path: str
    ) -> str:
        """
        Add audio from source video to processed video

        Args:
            video_path: Processed video without audio
            source_video: Original video with audio
            start_time: Audio start time
            end_time: Audio end time
            output_path: Final output path

        Returns:
            Path to final video with audio
        """
        duration = end_time - start_time

        # Two-pass approach for better audio sync:
        # 1. Extract audio segment from source with precise seeking
        # 2. Merge extracted audio with processed video

        import tempfile
        import os as os_module

        # Create temp file for extracted audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.aac', delete=False).name

        try:
            # Step 1: Extract audio with precise seeking
            # -ss AFTER -i for frame-accurate audio extraction (slower but accurate)
            extract_cmd = [
                'ffmpeg', '-y',
                '-i', source_video,
                '-ss', str(start_time),              # Seek AFTER input for accuracy
                '-t', str(duration),
                '-vn',                               # No video
                '-acodec', 'aac',
                '-b:a', '192k',
                temp_audio
            ]
            extract_result = subprocess.run(extract_cmd, capture_output=True, encoding='utf-8', errors='ignore', timeout=120)

            # Check if audio extraction succeeded
            if extract_result.returncode != 0:
                self._log(f"Audio extraction failed: {extract_result.stderr[:200] if extract_result.stderr else 'Unknown error'}")
                import shutil
                shutil.copy(video_path, output_path)
                return output_path

            # Step 2: Merge video + extracted audio
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,                    # Processed video (no audio)
                '-i', temp_audio,                    # Extracted audio
                '-map', '0:v:0',                     # Video from input 0
                '-map', '1:a:0',                     # Audio from input 1
                '-c:v', 'copy',                      # Copy video (no re-encode)
                '-c:a', 'copy',                      # Copy audio (already AAC)
                '-shortest',
                output_path
            ]
            result = subprocess.run(merge_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=300)

            if result.returncode != 0:
                self._log(f"Audio merge error: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                import shutil
                shutil.copy(video_path, output_path)
            else:
                self._log(f"Audio merged successfully")

        except Exception as e:
            self._log(f"Audio merge failed: {e}")
            import shutil
            shutil.copy(video_path, output_path)
        finally:
            # Clean up temp file
            try:
                os_module.unlink(temp_audio)
            except:
                pass

        return output_path
    
    def process_clip_with_audio(
        self,
        source_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Full pipeline: extract, crop with face tracking, and add audio

        Args:
            source_video: Source video path
            output_path: Final output path
            start_time: Clip start time
            end_time: Clip end time
            progress_callback: Progress callback

        Returns:
            Path to final processed video
        """
        # Log settings at start of each clip
        tracking_mode = "Classifier Only" if (not self.dynamic_tracking and not self.enable_tracking_analyzer) else "Full AI"
        lerp_factor = self._get_lerp_factor()
        effective_deadzone = self._get_effective_deadzone()
        self._log(f"╔══════════════════════════════════════════")
        self._log(f"║ Processing clip: {start_time:.1f}s - {end_time:.1f}s")
        self._log(f"║ Prescan: {self.use_prescan}")
        self._log(f"║ Split Screen: {self.enable_split_screen}")
        self._log(f"║ Dynamic Tracking: {self.dynamic_tracking}")
        self._log(f"║ Tracking Analyzer: {self.enable_tracking_analyzer}")
        self._log(f"║ Smoothing: {self.tracking_smoothing:.0%} | Speed: {self.tracking_speed:.0%}")
        self._log(f"║ → Lerp: {lerp_factor:.3f} | Deadzone: {effective_deadzone:.0f}px")
        self._log(f"║ Mode: {tracking_mode}")
        self._log(f"╚══════════════════════════════════════════")

        # Temporary paths
        temp_video = output_path.replace('.mp4', '_noaudio.mp4')
        
        if progress_callback:
            progress_callback(0, "Starting clip processing...")
        
        # Extract and process with face tracking
        self.extract_clip(
            source_video,
            temp_video,
            start_time,
            end_time,
            lambda p, s: progress_callback(p * 0.8, s) if progress_callback else None
        )
        
        if progress_callback:
            progress_callback(80, "Adding audio track...")
        
        # Add audio from source
        self.add_audio_from_source(
            temp_video,
            source_video,
            start_time,
            end_time,
            output_path
        )
        
        # Clean up
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        if progress_callback:
            progress_callback(100, "Clip complete!")
        
        return output_path


if __name__ == "__main__":
    # Test video processor
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "models")

    processor = VideoProcessor(
        yolo_model_path=os.path.join(models_dir, "yolo26m.pt"),
        classifier_model_path=os.path.join(models_dir, "face_classifier.pt")
    )
    
    def on_progress(percent, status):
        print(f"\r{status} - {percent:.1f}%", end="")
    
    # Test with a video file
    # processor.process_clip_with_audio(
    #     "test_video.mp4",
    #     "output_clip.mp4",
    #     10.0,
    #     30.0,
    #     on_progress
    # )
