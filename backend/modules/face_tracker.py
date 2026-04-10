"""
Face Tracker Module
Uses YOLO for face detection and tracking in video
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# Optimize CUDA if available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class FaceTracker:
    """Detects and tracks faces in video frames using YOLO"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        smoothing_factor: float = 0.3,
        tracking_speed: float = 0.5,
        detection_interval: int = 5,
        model_path: str = None,
        instant_switch: bool = True,
        instant_threshold: float = 100.0,
        cinematic_mode: bool = False,
        deadzone: float = 30.0
    ):
        """
        Initialize the face tracker

        Args:
            confidence_threshold: Minimum confidence for face detection (0.1-1.0)
            smoothing_factor: Position smoothing (0.1-1.0, higher = smoother)
            tracking_speed: How fast camera follows face (0.1-1.0)
            detection_interval: Detect face every N frames (1-30)
            model_path: Optional path to YOLO model file (.pt)
            instant_switch: If True, instantly jump to new position when distance > instant_threshold
            instant_threshold: Distance threshold for instant switch (in pixels)
            cinematic_mode: If True, use AE-style smooth damping with momentum
            deadzone: Deadzone radius in pixels for cinematic mode (camera stays still if face moves within this radius)
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_factor = smoothing_factor
        self.tracking_speed = tracking_speed
        self.detection_interval = detection_interval
        self.model_path = model_path
        self.model = None
        self.instant_switch = instant_switch
        self.instant_threshold = instant_threshold
        self.cinematic_mode = cinematic_mode
        self.deadzone = deadzone
        self.position_history = deque(maxlen=max(5, int(20 * (1 - tracking_speed))))

        # Cinematic state
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def update_settings(self, confidence: float = None, smoothing: float = None,
                       speed: float = None, interval: int = None,
                       instant_switch: bool = None, instant_threshold: float = None,
                       cinematic_mode: bool = None, deadzone: float = None):
        """Update tracking settings dynamically"""
        if confidence is not None:
            self.confidence_threshold = max(0.1, min(1.0, confidence))
        if smoothing is not None:
            self.smoothing_factor = max(0.1, min(1.0, smoothing))
        if speed is not None:
            self.tracking_speed = max(0.1, min(1.0, speed))
            self.position_history = deque(maxlen=max(5, int(20 * (1 - speed))))
        if interval is not None:
            self.detection_interval = max(1, min(30, interval))
        if instant_switch is not None:
            self.instant_switch = instant_switch
        if instant_threshold is not None:
            self.instant_threshold = max(10.0, instant_threshold)
        if cinematic_mode is not None:
            self.cinematic_mode = cinematic_mode
            # Reset velocity when mode changes
            self.velocity_x = 0.0
            self.velocity_y = 0.0
        if deadzone is not None:
            self.deadzone = max(0.0, deadzone)
        
    def load_model(self) -> None:
        """Load the YOLO face detection model from local path only"""
        import os

        # Determine model path - prioritize explicit path, then local models folder
        if self.model_path and os.path.exists(self.model_path):
            model_file = self.model_path
        else:
            # Use local model from models directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_file = os.path.join(base_dir, "models", "yolo26m.pt")

        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"YOLO model not found at: {model_file}\n"
                "Please ensure the trained model 'yolo26m.pt' exists in the 'models/' directory."
            )

        print(f"Loading YOLO model from local: {model_file}")
        self.model = YOLO(model_file)

        # Force GPU inference if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("YOLO model moved to GPU")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detected faces with bounding boxes and confidence
        """
        if self.model is None:
            self.load_model()

        # Run inference with GPU optimization
        results = self.model(
            frame,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=torch.cuda.is_available()  # Use FP16 on GPU for speed
        )[0]

        faces = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                })

        # Sort by area (largest first)
        faces.sort(key=lambda x: x['area'], reverse=True)

        return faces

    def detect_faces_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect faces in multiple frames at once (batch processing)

        Args:
            frames: List of BGR images as numpy arrays

        Returns:
            List of face detection results for each frame
        """
        if self.model is None:
            self.load_model()

        if not frames:
            return []

        # Batch inference
        results = self.model(
            frames,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=torch.cuda.is_available()
        )

        all_faces = []
        for result in results:
            faces = []
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1)
                    })
            faces.sort(key=lambda x: x['area'], reverse=True)
            all_faces.append(faces)

        return all_faces
    
    def get_primary_face(self, faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the primary (largest) face from detected faces"""
        if not faces:
            return None
        return faces[0]
    
    def get_smoothed_center(self, current_center: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get smoothed center - INSTANT switch for large movements, smooth for small

        Args:
            current_center: Current face center (x, y)

        Returns:
            Smoothed center position
        """
        self.position_history.append(current_center)

        # Initialize last smoothed position
        if not hasattr(self, 'last_smoothed_center'):
            self.last_smoothed_center = current_center
            return current_center

        last_x, last_y = self.last_smoothed_center
        curr_x, curr_y = current_center

        # Calculate distance
        dist = ((curr_x - last_x) ** 2 + (curr_y - last_y) ** 2) ** 0.5

        # CINEMATIC MODE: Physics-based damping (spring/mass system simulation)
        if self.cinematic_mode:
            # If movement is HUGE (scene cut), just cut
            if self.instant_switch and dist > self.instant_threshold * 1.5:
                self.last_smoothed_center = current_center
                self.velocity_x = 0.0
                self.velocity_y = 0.0
                return current_center

            # Deadzone: if target is close, don't move (steady cam)
            # Dynamic deadzone based on face size could be better, but fixed is stable
            deadzone = self.deadzone  # pixels
            if dist < deadzone:
                # Slight drift towards center to avoid getting stuck at edge of deadzone
                target_x = last_x + (curr_x - last_x) * 0.01
                target_y = last_y + (curr_y - last_y) * 0.01
                self.last_smoothed_center = (target_x, target_y)
                # Decay velocity
                self.velocity_x *= 0.9
                self.velocity_y *= 0.9
                return (int(target_x), int(target_y))

            # Physics parameters
            # stiffness: how hard it pulls towards target (low = loose/laggy)
            # damping: friction (high = no overshoot)
            # mass: heaviness (high = slow to start/stop)

            # Use tracking_speed settings to modulate physics
            # Low speed = heavy mass, loose spring
            # High speed = light mass, stiff spring

            # TWEAKED FOR SMOOTHER MOTION (Heavier feel)
            # Reduced stiffness slightly for less jitter
            stiffness = 0.02 + (self.tracking_speed * 0.05)  # Lowered stiffness
            # Increased damping slightly to prevent rubber-banding
            damping = 0.85 + (self.smoothing_factor * 0.10)  # Increased damping

            # Force vector
            force_x = (curr_x - last_x) * stiffness
            force_y = (curr_y - last_y) * stiffness

            # Update velocity (v = v + a)
            self.velocity_x = (self.velocity_x + force_x) * damping
            self.velocity_y = (self.velocity_y + force_y) * damping

            # Update position (p = p + v)
            new_x = last_x + self.velocity_x
            new_y = last_y + self.velocity_y

            self.last_smoothed_center = (new_x, new_y)
            return (int(new_x), int(new_y))

        # STANDARD MODE: Linear Interpolation (Lerp)
        # INSTANT switch mode: immediately jump when distance exceeds threshold
        if self.instant_switch and dist > self.instant_threshold:
            self.last_smoothed_center = current_center
            self.position_history.clear()
            self.position_history.append(current_center)
            return current_center

        # Small movements: apply smoothing
        if dist > 50:
            lerp_factor = 0.2 + (self.tracking_speed * 0.1)  # Slower response
        elif dist > 20:
            lerp_factor = 0.1 + (self.tracking_speed * 0.08) # Slower response
        else:
            lerp_factor = 0.05 + (self.tracking_speed * 0.05) # Much slower for micro-movements

        # Apply smoothing factor
        lerp_factor = lerp_factor * (1.0 - self.smoothing_factor * 0.5) # Stronger smoothing effect
        lerp_factor = max(0.02, min(0.5, lerp_factor)) # Cap max speed to avoid robot-jerkiness

        # Interpolate
        new_x = last_x + (curr_x - last_x) * lerp_factor
        new_y = last_y + (curr_y - last_y) * lerp_factor

        self.last_smoothed_center = (new_x, new_y)

        return (int(new_x), int(new_y))
    
    def calculate_crop_region(
        self,
        frame_width: int,
        frame_height: int,
        face_center: Tuple[int, int],
        output_aspect_ratio: Tuple[int, int] = (9, 16)
    ) -> Tuple[int, int, int, int]:
        """
        Calculate the crop region for vertical video centered on face
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height  
            face_center: Face center position (x, y)
            output_aspect_ratio: Target aspect ratio (width, height)
            
        Returns:
            Crop region as (x1, y1, x2, y2)
        """
        aspect_w, aspect_h = output_aspect_ratio
        
        # Calculate crop dimensions maintaining aspect ratio
        if frame_height / frame_width > aspect_h / aspect_w:
            # Width is the limiting factor
            crop_width = frame_width
            crop_height = int(frame_width * aspect_h / aspect_w)
        else:
            # Height is the limiting factor
            crop_height = frame_height
            crop_width = int(frame_height * aspect_w / aspect_h)
        
        # Center crop on face X position
        center_x, center_y = face_center
        
        # Calculate crop boundaries
        x1 = center_x - crop_width // 2
        x2 = center_x + crop_width // 2
        y1 = 0  # Always start from top for vertical video
        y2 = crop_height
        
        # Clamp to frame boundaries
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
    
    def process_video_for_tracking(
        self,
        video_path: str,
        sample_interval: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Process entire video to get face positions at regular intervals
        
        Args:
            video_path: Path to video file
            sample_interval: Sample every N frames
            
        Returns:
            List of frame data with face positions
        """
        if self.model is None:
            self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        tracking_data = []
        frame_idx = 0
        last_known_center = (frame_width // 2, frame_height // 2)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                faces = self.detect_faces(frame)
                primary_face = self.get_primary_face(faces)
                
                if primary_face:
                    center = primary_face['center']
                    last_known_center = center
                else:
                    center = last_known_center
                
                tracking_data.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'center': center,
                    'has_face': primary_face is not None
                })
            
            frame_idx += 1
        
        cap.release()
        
        return tracking_data
    
    def reset_smoothing(self) -> None:
        """Reset position history for new video"""
        self.position_history.clear()
        if hasattr(self, 'last_smoothed_center'):
            del self.last_smoothed_center


if __name__ == "__main__":
    import os
    # Use local trained model
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "yolo26m.pt")

    # Test face tracker with local model
    tracker = FaceTracker(model_path=model_path)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = tracker.detect_faces(frame)
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, face['center'], 5, (0, 0, 255), -1)
        
        cv2.imshow('Face Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
