"""
Face Embedder Module
Uses dlib with CUDA (pre-trained) to generate face embeddings for identity tracking.
This model understands general human face characteristics, not memorizing specific individuals.

No new libraries needed - uses dlib which is already installed with CUDA support.
Models are auto-downloaded if not present.
"""

import os
import bz2
import numpy as np
import cv2
import requests
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Try to import dlib
try:
    import dlib
    DLIB_AVAILABLE = True
    # Check if CUDA is available in dlib
    DLIB_CUDA = dlib.DLIB_USE_CUDA
    if DLIB_CUDA:
        print(f"dlib CUDA enabled: {dlib.cuda.get_num_devices()} device(s)")
    else:
        print("dlib CUDA not available, using CPU")
except ImportError:
    DLIB_AVAILABLE = False
    DLIB_CUDA = False
    print("dlib not available for face embedding")


# Model URLs for auto-download
DLIB_MODELS = {
    "mmod_human_face_detector.dat": {
        "url": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
        "description": "CNN Face Detector (CUDA)"
    },
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "description": "Face Landmarks Predictor"
    },
    "dlib_face_recognition_resnet_model_v1.dat": {
        "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        "description": "Face Recognition Model"
    }
}


def download_and_extract_bz2(url: str, output_path: str, description: str = "", log_callback=None) -> bool:
    """
    Download a .bz2 file and extract it.

    Args:
        url: URL to download from
        output_path: Path to save extracted file
        description: Description for progress bar
        log_callback: Optional callback for logging (receives string)

    Returns:
        True if successful
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    try:
        log(f"Downloading {description}...")
        log(f"  URL: {url}")

        # Download with progress bar
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Download to memory
        compressed_data = b""
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                compressed_data += chunk
                pbar.update(len(chunk))

        log(f"  Extracting to: {output_path}")

        # Extract bz2
        decompressed_data = bz2.decompress(compressed_data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write extracted file
        with open(output_path, 'wb') as f:
            f.write(decompressed_data)

        log(f"  Done! Size: {len(decompressed_data) / (1024*1024):.1f} MB")
        return True

    except Exception as e:
        log(f"  Error downloading {description}: {e}")
        return False


def ensure_models_exist(models_dir: str, log_callback=None) -> Dict[str, bool]:
    """
    Check and download missing dlib models.
    Also extracts .bz2 files if they exist but haven't been extracted.

    Args:
        models_dir: Directory to store models
        log_callback: Optional callback for logging

    Returns:
        Dict of model_name -> exists status
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    os.makedirs(models_dir, exist_ok=True)

    status = {}

    for model_name, model_info in DLIB_MODELS.items():
        model_path = os.path.join(models_dir, model_name)
        bz2_path = model_path + ".bz2"  # Check for compressed file

        # Debug: show what we're checking
        log(f"Checking: {model_path}")

        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            log(f"[OK] {model_info['description']}: {model_name} ({file_size:.1f} MB)")
            status[model_name] = True
        elif os.path.exists(bz2_path):
            # .bz2 file exists but not extracted - extract it
            log(f"[EXTRACTING] Found {model_name}.bz2, extracting...")
            try:
                with open(bz2_path, 'rb') as f:
                    compressed_data = f.read()
                decompressed_data = bz2.decompress(compressed_data)
                with open(model_path, 'wb') as f:
                    f.write(decompressed_data)
                log(f"  Done! Size: {len(decompressed_data) / (1024*1024):.1f} MB")
                # Optionally remove .bz2 file after extraction
                # os.remove(bz2_path)
                status[model_name] = True
            except Exception as e:
                log(f"  Error extracting: {e}")
                status[model_name] = False
        else:
            log(f"[MISSING] {model_info['description']}: {model_name}")
            log(f"  Expected at: {model_path}")
            log(f"  Or compressed: {bz2_path}")
            success = download_and_extract_bz2(
                model_info['url'],
                model_path,
                model_info['description'],
                log_callback
            )
            status[model_name] = success

    return status


class FaceEmbedder:
    """
    Generate face embeddings using pre-trained dlib model with CUDA acceleration.

    This model was trained on millions of faces and understands:
    - General human face characteristics (skin texture, depth, features)
    - NOT memorizing specific individuals
    - Can distinguish Person A from Person B based on facial features
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize face embedder.

        Args:
            models_dir: Directory containing dlib model files
        """
        if models_dir is None:
            # Default to models folder in project root
            self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        else:
            self.models_dir = os.path.abspath(models_dir)

        self.detector = None
        self.cnn_detector = None  # CNN detector for CUDA
        self.shape_predictor = None
        self.face_rec_model = None
        self.is_loaded = False
        self.use_cuda = DLIB_CUDA

        if not DLIB_AVAILABLE:
            print("Warning: dlib not available. Face embedding disabled.")
            return

    def load_model(self, log_callback=None) -> bool:
        """Load the dlib models with CUDA support. Auto-download if missing."""
        if not DLIB_AVAILABLE:
            return False

        def log(msg):
            if log_callback:
                log_callback(msg)
            print(msg)

        try:
            # Ensure all models are downloaded
            log("=== Checking dlib face recognition models ===")
            model_status = ensure_models_exist(self.models_dir, log_callback)

            # Check if critical models are available
            shape_ok = model_status.get("shape_predictor_68_face_landmarks.dat", False)
            rec_ok = model_status.get("dlib_face_recognition_resnet_model_v1.dat", False)

            if not shape_ok or not rec_ok:
                log("Critical models missing. Face embedding disabled.")
                return False

            # CNN Face detector (CUDA accelerated) - more accurate than HOG
            cnn_detector_path = os.path.join(self.models_dir, "mmod_human_face_detector.dat")
            if os.path.exists(cnn_detector_path) and self.use_cuda:
                self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
                log(f"dlib CNN face detector loaded (CUDA: {self.use_cuda})")
            else:
                # Fallback to HOG detector (CPU)
                self.detector = dlib.get_frontal_face_detector()
                log("Using HOG detector (CPU) as fallback")

            # Shape predictor for face landmarks
            shape_predictor_path = os.path.join(self.models_dir, "shape_predictor_68_face_landmarks.dat")
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

            # Face recognition model (generates 128-dim embedding)
            face_rec_path = os.path.join(self.models_dir, "dlib_face_recognition_resnet_model_v1.dat")
            self.face_rec_model = dlib.face_recognition_model_v1(face_rec_path)

            self.is_loaded = True
            detector_type = "CNN (CUDA)" if self.cnn_detector else "HOG (CPU)"
            log(f"=== dlib face recognition ready - Detector: {detector_type} ===")
            return True

        except Exception as e:
            log(f"Failed to load dlib models: {e}")
            return False

    def get_face_embeddings(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces and get embeddings for each using CUDA if available.

        Args:
            frame: BGR numpy array

        Returns:
            List of dicts with 'bbox', 'embedding', 'center', 'area'
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        try:
            # Convert BGR to RGB for dlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using CNN (CUDA) or HOG (CPU)
            if self.cnn_detector is not None:
                # CNN detector returns mmod_rectangles
                dets = self.cnn_detector(rgb_frame, 0)  # 0 = no upsampling
                # Extract rectangles from mmod_rectangles
                rects = [d.rect for d in dets]
            else:
                # HOG detector returns rectangles directly
                rects = self.detector(rgb_frame, 0)

            results = []
            for rect in rects:
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()

                # Clamp to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Get face landmarks
                shape = self.shape_predictor(rgb_frame, rect)

                # Get 128-dimensional face embedding (uses CUDA internally if available)
                embedding = np.array(self.face_rec_model.compute_face_descriptor(rgb_frame, shape))

                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'embedding': embedding,  # 128-dim vector
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'area': (x2 - x1) * (y2 - y1),
                    'confidence': 1.0
                })

            # Sort by area (largest first)
            results.sort(key=lambda x: x['area'], reverse=True)

            return results

        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        if emb1 is None or emb2 is None:
            return 0.0

        # Normalize
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, (similarity + 1) / 2))

    def match_faces_by_embedding(
        self,
        current_faces: List[Dict],
        ref_embedding_1: Optional[np.ndarray],
        ref_embedding_2: Optional[np.ndarray],
        similarity_threshold: float = 0.6
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Match detected faces to reference identities using embeddings.
        Uses Hungarian-style assignment to prevent swapping.

        Args:
            current_faces: List of detected faces with embeddings
            ref_embedding_1: Reference embedding for Person 1 (top)
            ref_embedding_2: Reference embedding for Person 2 (bottom)
            similarity_threshold: Minimum similarity to consider a match

        Returns:
            (face_for_slot_1, face_for_slot_2) or (None, None) if not enough faces
        """
        if len(current_faces) < 2:
            return None, None

        # First frame - no reference yet, use X position
        if ref_embedding_1 is None or ref_embedding_2 is None:
            sorted_faces = sorted(current_faces, key=lambda f: f['center'][0])
            return sorted_faces[0], sorted_faces[1]

        # Calculate similarity matrix for all faces vs both references
        scores = []
        for face in current_faces:
            emb = face.get('embedding')
            if emb is None:
                scores.append((0.0, 0.0, face))
            else:
                sim_1 = self.cosine_similarity(emb, ref_embedding_1)
                sim_2 = self.cosine_similarity(emb, ref_embedding_2)
                scores.append((sim_1, sim_2, face))

        # Try both assignments and pick the one with higher total similarity
        # Assignment A: face[i] -> slot1, face[j] -> slot2
        # Assignment B: face[i] -> slot2, face[j] -> slot1
        best_total = -1
        best_match_1 = None
        best_match_2 = None

        for i, (sim1_i, sim2_i, face_i) in enumerate(scores):
            for j, (sim1_j, sim2_j, face_j) in enumerate(scores):
                if i == j:
                    continue
                # Assignment: face_i -> slot1, face_j -> slot2
                total = sim1_i + sim2_j
                if total > best_total:
                    best_total = total
                    best_match_1 = face_i
                    best_match_2 = face_j

        # Verify matches meet threshold
        if best_match_1 and best_match_2:
            emb1 = best_match_1.get('embedding')
            emb2 = best_match_2.get('embedding')
            if emb1 is not None and emb2 is not None:
                final_sim_1 = self.cosine_similarity(emb1, ref_embedding_1)
                final_sim_2 = self.cosine_similarity(emb2, ref_embedding_2)
                # If both matches are poor, fall back to X position
                if final_sim_1 < similarity_threshold and final_sim_2 < similarity_threshold:
                    sorted_faces = sorted(current_faces, key=lambda f: f['center'][0])
                    return sorted_faces[0], sorted_faces[1]

        return best_match_1, best_match_2


# For compatibility
INSIGHTFACE_AVAILABLE = False  # We're using dlib instead
DLIB_EMBEDDER_AVAILABLE = DLIB_AVAILABLE


# Singleton instance
_embedder_instance = None

def get_embedder() -> FaceEmbedder:
    """Get or create the global FaceEmbedder instance"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = FaceEmbedder()
    return _embedder_instance


if __name__ == "__main__":
    # Test
    print(f"dlib available: {DLIB_AVAILABLE}")
    print(f"dlib CUDA: {DLIB_CUDA}")

    embedder = FaceEmbedder()
    if embedder.load_model():
        print("dlib face recognition ready!")

        # Test with webcam
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = embedder.get_face_embeddings(frame)

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i+1}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Face Embedder Test (CUDA)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to load dlib models")
