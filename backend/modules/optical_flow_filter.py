"""
Optical Flow Poster Filter Module
Detects poster/static faces by comparing face movement vs background movement.

Logic:
- Real human faces move INDEPENDENTLY from the background (head turns, nods, etc.)
- Poster faces move EXACTLY with the background (camera shake, pan, etc.)

Method:
- Calculate optical flow (dense or sparse) around each face region
- Compare face region flow vs surrounding background flow
- If face flow ≈ background flow → it's a poster (moves with camera)
- If face flow ≠ background flow → it's a real person (independent movement)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class OpticalFlowFilter:
    """
    Filters out poster/static faces using optical flow analysis.
    
    Compares motion of face region vs surrounding background.
    Posters move identically to background (camera motion only).
    Real faces have independent motion (head movement, expressions).
    """

    def __init__(
        self,
        flow_threshold: float = 2.0,
        min_samples: int = 5,
        consistency_ratio: float = 0.7,
        bg_sample_margin: float = 1.5,
        use_dense_flow: bool = False
    ):
        """
        Initialize optical flow filter.

        Args:
            flow_threshold: Minimum difference between face flow and bg flow 
                           to consider face as "real" (pixels). Lower = more sensitive.
            min_samples: Minimum number of frame pairs to analyze before making decision.
            consistency_ratio: Ratio of frames where face must show independent motion
                              to be considered real (0-1). Higher = stricter.
            bg_sample_margin: How much larger the background sample area is vs face bbox.
                            1.5 = 50% larger on each side.
            use_dense_flow: If True, use Farneback dense flow (slower but more accurate).
                           If False, use Lucas-Kanade sparse flow (faster).
        """
        self.flow_threshold = flow_threshold
        self.min_samples = min_samples
        self.consistency_ratio = consistency_ratio
        self.bg_sample_margin = bg_sample_margin
        self.use_dense_flow = use_dense_flow

        # State
        self.prev_gray: Optional[np.ndarray] = None
        self.face_history: Dict[str, List[float]] = defaultdict(list)
        # key = grid position string, value = list of flow differences

        # Confirmed posters (faces that consistently match background motion)
        self.confirmed_posters: set = set()
        # Confirmed real faces (faces with independent motion)
        self.confirmed_real: set = set()

        # Farneback parameters for dense flow
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Lucas-Kanade parameters for sparse flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def _get_face_key(self, face: Dict[str, Any], grid_size: int = 40) -> str:
        """Get a grid-based key for a face position (for tracking over time)"""
        cx, cy = face['center']
        grid_x = int(cx // grid_size) * grid_size
        grid_y = int(cy // grid_size) * grid_size
        return f"{grid_x}_{grid_y}"

    def _get_face_region(self, face: Dict[str, Any], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get face bounding box clamped to frame"""
        x1, y1, x2, y2 = face['bbox']
        h, w = frame_shape[:2]
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    def _get_bg_region(self, face: Dict[str, Any], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get background sample region (larger area around face)"""
        x1, y1, x2, y2 = face['bbox']
        h, w = frame_shape[:2]

        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * (self.bg_sample_margin - 1.0))
        margin_y = int(face_h * (self.bg_sample_margin - 1.0))

        bg_x1 = max(0, x1 - margin_x)
        bg_y1 = max(0, y1 - margin_y)
        bg_x2 = min(w, x2 + margin_x)
        bg_y2 = min(h, y2 + margin_y)

        return (bg_x1, bg_y1, bg_x2, bg_y2)

    def _compute_region_flow_dense(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Compute average optical flow in a region using Farneback (dense)"""
        x1, y1, x2, y2 = region

        if x2 - x1 < 10 or y2 - y1 < 10:
            return (0.0, 0.0)

        prev_roi = prev_gray[y1:y2, x1:x2]
        curr_roi = curr_gray[y1:y2, x1:x2]

        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None, **self.farneback_params
        )

        # Average flow in region
        avg_flow_x = float(np.mean(flow[:, :, 0]))
        avg_flow_y = float(np.mean(flow[:, :, 1]))

        return (avg_flow_x, avg_flow_y)

    def _compute_region_flow_sparse(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        region: Tuple[int, int, int, int],
        num_points: int = 20
    ) -> Tuple[float, float]:
        """Compute average optical flow in a region using Lucas-Kanade (sparse)"""
        x1, y1, x2, y2 = region

        if x2 - x1 < 10 or y2 - y1 < 10:
            return (0.0, 0.0)

        # Generate grid points in the region
        step_x = max(1, (x2 - x1) // int(num_points ** 0.5))
        step_y = max(1, (y2 - y1) // int(num_points ** 0.5))

        points = []
        for y in range(y1 + step_y, y2 - step_y, step_y):
            for x in range(x1 + step_x, x2 - step_x, step_x):
                points.append([[float(x), float(y)]])

        if not points:
            return (0.0, 0.0)

        prev_pts = np.array(points, dtype=np.float32)

        # Calculate optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        if next_pts is None or status is None:
            return (0.0, 0.0)

        # Filter good points
        good_mask = status.flatten() == 1
        if not np.any(good_mask):
            return (0.0, 0.0)

        good_prev = prev_pts[good_mask]
        good_next = next_pts[good_mask]

        # Calculate average displacement
        displacement = good_next - good_prev
        avg_flow_x = float(np.mean(displacement[:, 0, 0]))
        avg_flow_y = float(np.mean(displacement[:, 0, 1]))

        return (avg_flow_x, avg_flow_y)

    def analyze_frame(
        self,
        frame: np.ndarray,
        faces: List[Dict[str, Any]]
    ) -> None:
        """
        Analyze a frame to accumulate optical flow data for each face.
        Call this for multiple frames before calling filter_faces().

        Args:
            frame: Current BGR frame
            faces: List of detected faces with 'bbox' and 'center' keys
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return

        # For each face, compute flow difference between face region and background
        for face in faces:
            face_key = self._get_face_key(face)

            # Skip already confirmed faces
            if face_key in self.confirmed_real or face_key in self.confirmed_posters:
                continue

            face_region = self._get_face_region(face, frame.shape)
            bg_region = self._get_bg_region(face, frame.shape)

            # Compute flow
            if self.use_dense_flow:
                face_flow = self._compute_region_flow_dense(self.prev_gray, curr_gray, face_region)
                bg_flow = self._compute_region_flow_dense(self.prev_gray, curr_gray, bg_region)
            else:
                face_flow = self._compute_region_flow_sparse(self.prev_gray, curr_gray, face_region)
                bg_flow = self._compute_region_flow_sparse(self.prev_gray, curr_gray, bg_region)

            # Calculate difference (independent motion)
            diff_x = abs(face_flow[0] - bg_flow[0])
            diff_y = abs(face_flow[1] - bg_flow[1])
            flow_diff = (diff_x ** 2 + diff_y ** 2) ** 0.5

            self.face_history[face_key].append(flow_diff)

            # Keep history manageable
            if len(self.face_history[face_key]) > 50:
                self.face_history[face_key] = self.face_history[face_key][-50:]

            # Check if we have enough samples to make a decision
            history = self.face_history[face_key]
            if len(history) >= self.min_samples:
                # Count how many frames show independent motion
                independent_count = sum(1 for d in history if d >= self.flow_threshold)
                ratio = independent_count / len(history)

                if ratio >= self.consistency_ratio:
                    # Face has consistent independent motion → REAL
                    self.confirmed_real.add(face_key)
                elif ratio <= (1.0 - self.consistency_ratio):
                    # Face consistently matches background → POSTER
                    self.confirmed_posters.add(face_key)

        self.prev_gray = curr_gray

    def is_poster(self, face: Dict[str, Any]) -> bool:
        """
        Check if a face is a confirmed poster.

        Args:
            face: Face dict with 'center' key

        Returns:
            True if face is confirmed as poster
        """
        face_key = self._get_face_key(face)
        return face_key in self.confirmed_posters

    def is_real(self, face: Dict[str, Any]) -> bool:
        """
        Check if a face is confirmed as real.

        Args:
            face: Face dict with 'center' key

        Returns:
            True if face is confirmed as real person
        """
        face_key = self._get_face_key(face)
        return face_key in self.confirmed_real

    def filter_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out confirmed poster faces from a list.

        Args:
            faces: List of detected faces

        Returns:
            Filtered list with posters removed
        """
        return [f for f in faces if not self.is_poster(f)]

    def get_poster_positions(self) -> List[Tuple[int, int]]:
        """Get list of confirmed poster grid positions"""
        positions = []
        for key in self.confirmed_posters:
            parts = key.split('_')
            if len(parts) == 2:
                positions.append((int(parts[0]), int(parts[1])))
        return positions

    def get_stats(self) -> Dict[str, Any]:
        """Get current filter statistics"""
        return {
            'confirmed_posters': len(self.confirmed_posters),
            'confirmed_real': len(self.confirmed_real),
            'pending': len(self.face_history) - len(self.confirmed_posters) - len(self.confirmed_real),
            'poster_positions': self.get_poster_positions(),
            'total_samples': sum(len(h) for h in self.face_history.values())
        }

    def reset(self):
        """Reset all state for a new video"""
        self.prev_gray = None
        self.face_history.clear()
        self.confirmed_posters.clear()
        self.confirmed_real.clear()
