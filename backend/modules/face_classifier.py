"""
Face Classifier Module
Trains and uses a local model to distinguish human faces from posters/printed faces

KEY CONCEPT: This classifier learns TEXTURE CHARACTERISTICS, not face identities.
- Human faces: natural skin texture, micro-pores, organic color gradients, 3D lighting
- Poster faces: flat texture, print patterns, artificial colors, uniform lighting

The model should generalize across different people, not memorize specific individuals.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from typing import Optional, Callable, List, Tuple

# Optimize CUDA if available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class TextureFeatureExtractor(nn.Module):
    """
    Extract texture-based features that distinguish human skin from printed material.
    Focuses on:
    - Local Binary Patterns (texture)
    - Frequency components (print vs natural)
    - Color distribution patterns
    """

    def __init__(self):
        super().__init__()

        # Multi-scale texture extraction
        # Use different kernel sizes to capture various texture scales
        self.texture_branch = nn.Sequential(
            # Fine texture (pores, micro-details)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Medium texture
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            # Larger patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )

        # Frequency analysis branch (detects print patterns like halftone)
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # Larger kernel for frequency
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Combine branches
        self.combine = nn.Sequential(
            nn.Conv2d(128 + 64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classifier head with strong regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),  # High dropout to prevent memorizing
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)  # 0: poster, 1: human
        )

    def forward(self, x):
        # Extract texture features
        tex = self.texture_branch(x)

        # Extract frequency features
        freq = self.frequency_branch(x)

        # Combine (resize freq to match tex)
        freq = nn.functional.interpolate(freq, size=tex.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([tex, freq], dim=1)

        # Final features
        features = self.combine(combined)
        output = self.classifier(features)

        return output


# Keep old class name for compatibility but use new architecture
class FaceClassifierCNN(TextureFeatureExtractor):
    """Alias for backward compatibility"""
    pass


class FaceDataset(Dataset):
    """
    Dataset for face classification training.

    IMPORTANT: This dataset uses AGGRESSIVE augmentation to prevent the model
    from memorizing specific faces. The goal is to learn TEXTURE patterns
    (human skin vs printed material), not individual identities.
    """

    def __init__(self, human_images: List[np.ndarray], poster_images: List[np.ndarray]):
        self.images = []
        self.labels = []

        # AGGRESSIVE augmentation to prevent memorizing specific faces
        # Forces model to learn texture patterns, not identities
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((80, 80)),  # Larger for more aggressive crop
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Unusual angle
            transforms.RandomRotation(degrees=20),  # More rotation
            transforms.ColorJitter(
                brightness=0.4,  # Strong brightness variation
                contrast=0.4,    # Strong contrast variation
                saturation=0.3,
                hue=0.15         # More hue variation to break color memorization
            ),
            transforms.RandomGrayscale(p=0.1),  # Sometimes grayscale
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Perspective change
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Random blur
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Occlude parts
        ])

        # Simpler transform for validation (no random augmentation)
        self.eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Add human faces (label = 1)
        for img in human_images:
            self.images.append(img)
            self.labels.append(1)

        # Add poster faces (label = 0)
        for img in poster_images:
            self.images.append(img)
            self.labels.append(0)

        self.training = True

    def set_eval(self):
        """Switch to evaluation mode (no augmentation)"""
        self.training = False

    def set_train(self):
        """Switch to training mode (with augmentation)"""
        self.training = True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.training:
            img = self.transform(img)
        else:
            img = self.eval_transform(img)

        label = self.labels[idx]

        return img, label


class FaceClassifier:
    """Classifier to distinguish human faces from posters"""
    
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FaceClassifierCNN().to(self.device)
        self.model_path = model_path
        self.is_loaded = False
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Try to load existing model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, path: str) -> bool:
        """Load trained model"""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print(f"Face classifier loaded from: {path}")
            return True
        except Exception as e:
            print(f"Failed to load face classifier: {e}")
            return False
    
    def save_model(self, path: str):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Face classifier saved to: {path}")
    
    def train(
        self,
        human_images: List[np.ndarray],
        poster_images: List[np.ndarray],
        epochs: int = 80,
        patience: int = 10,
        min_delta: float = 0.0005,
        val_split: float = 0.2,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> float:
        """
        Train the classifier with proper early stopping using validation set

        Args:
            human_images: List of face crops (numpy BGR) of human faces
            poster_images: List of face crops (numpy BGR) of poster/fake faces
            epochs: Maximum number of training epochs
            patience: Stop if no improvement for this many epochs
            min_delta: Minimum improvement to reset patience counter
            val_split: Fraction of data to use for validation (0.1-0.3)
            progress_callback: Callback(progress_percent, status)

        Returns:
            Best validation accuracy
        """
        if len(human_images) < 5 or len(poster_images) < 5:
            raise ValueError("Need at least 5 samples of each class")

        # Create dataset
        full_dataset = FaceDataset(human_images, poster_images)

        # Split into train and validation
        val_size = max(4, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Use larger batch size for better gradient estimates
        batch_size = min(16, train_size // 4) if train_size > 16 else 8
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training setup with regularization to prevent memorizing
        # Label smoothing (0.1) prevents overconfidence and improves generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.02)  # Higher weight decay

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        self.model.train()

        if progress_callback:
            progress_callback(0, f"Training: {train_size} train, {val_size} val samples")

        # Early stopping variables
        best_val_loss = float('inf')
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # === Training phase ===
            self.model.train()
            full_dataset.training = True  # Enable augmentation
            train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total * 100

            # === Validation phase ===
            self.model.eval()
            full_dataset.training = False  # Disable augmentation for validation
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = val_correct / val_total * 100 if val_total > 0 else 0

            # Update learning rate scheduler
            scheduler.step(avg_val_loss)

            # Check for improvement (using validation loss)
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if progress_callback:
                progress = (epoch + 1) / epochs * 100
                lr = optimizer.param_groups[0]['lr']
                progress_callback(
                    progress,
                    f"Epoch {epoch+1}: Train {train_acc:.0f}% Val {val_acc:.0f}% LR={lr:.1e} (p {patience_counter}/{patience})"
                )

            # Early stopping check
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(100, f"Early stop @ epoch {epoch+1}: Val acc {best_val_acc:.1f}%")
                break

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.model.eval()
        self.is_loaded = True

        if self.model_path:
            self.save_model(self.model_path)

        return best_val_acc
    
    def auto_train_from_video(
        self,
        video_path: str,
        face_detector,  # FaceTracker instance
        max_samples: int = 100,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> bool:
        """
        Automatically train classifier from video frames using ADVANCED MOVEMENT detection.
        Uses multiple metrics to distinguish human faces from static images:
        - Position movement (humans move around)
        - Size variation (humans move closer/further from camera)
        - Texture variation (human skin has micro-expressions)
        - Edge consistency (posters have sharp consistent edges)
        - Temporal patterns (humans have natural movement rhythms)

        Args:
            video_path: Path to video file
            face_detector: FaceTracker instance for detection
            max_samples: Max samples per class (increased to 100 for better training)
            progress_callback: Progress callback

        Returns:
            True if training successful
        """
        import cv2

        if progress_callback:
            progress_callback(0, "Analyzing face movements (improved algorithm)...")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_duration_sec = total_frames / fps if fps > 0 else 0
        video_duration_min = int(video_duration_sec / 60) + 1

        # Track face positions over time
        # Key: approximate position zone, Value: list of detections with rich data
        face_tracks = {}  # zone_key -> list of detections

        # Improved sampling: scan more densely for better movement detection
        # At least 3 samples per second for accurate movement analysis
        target_scan_count = max(int(max_samples * 4), int(video_duration_sec * 3))
        sample_interval = max(1, total_frames // target_scan_count)

        if progress_callback:
            progress_callback(2, f"Video: {video_duration_min} min, scanning ~{target_scan_count} frames...")

        frame_idx = 0

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                faces = face_detector.detect_faces(frame)

                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    cx, cy = face['center']

                    # Zone key based on rough position (larger zones for better grouping)
                    frame_w = frame.shape[1]
                    frame_h = frame.shape[0]
                    zone_x = int(cx / frame_w * 4)  # 4x4 grid
                    zone_y = int(cy / frame_h * 4)
                    zone_key = f"{zone_x}_{zone_y}"

                    face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                    if face_crop.size == 0 or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                        continue

                    # Calculate texture metrics for this face crop
                    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                    # Laplacian variance (focus/sharpness) - posters are sharper
                    laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

                    # Edge density using Canny
                    edges = cv2.Canny(gray_crop, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size

                    # Color variance (humans have more natural color variation)
                    hsv_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                    color_variance = np.std(hsv_crop[:,:,0])  # Hue variance

                    if zone_key not in face_tracks:
                        face_tracks[zone_key] = []

                    face_tracks[zone_key].append({
                        'frame': frame_idx,
                        'bbox': (x1, y1, x2, y2),
                        'center': (cx, cy),
                        'crop': face_crop.copy(),
                        'laplacian': laplacian_var,
                        'edge_density': edge_density,
                        'color_var': color_variance,
                        'size': (x2 - x1) * (y2 - y1)
                    })

                if progress_callback:
                    progress = min(40, (frame_idx / total_frames) * 40)
                    progress_callback(progress, f"Tracking faces: {len(face_tracks)} regions")

            frame_idx += 1

        cap.release()

        if progress_callback:
            progress_callback(42, f"Analyzing {len(face_tracks)} face regions with multi-metric scoring...")

        human_samples = []
        poster_samples = []

        # Track scores for debugging
        all_tracks_info = []

        # Analyze each tracked zone with MULTIPLE METRICS
        for zone_key, detections in face_tracks.items():
            if len(detections) < 5:
                continue  # Need enough detections for reliable analysis

            # ===== METRIC 1: Position Movement =====
            centers = [d['center'] for d in detections]
            movements = []
            for i in range(1, len(centers)):
                dx = centers[i][0] - centers[i-1][0]
                dy = centers[i][1] - centers[i-1][1]
                movements.append((dx**2 + dy**2) ** 0.5)

            avg_movement = np.mean(movements) if movements else 0
            movement_variance = np.std(movements) if movements else 0

            # ===== METRIC 2: Size Variation =====
            sizes = [d['size'] for d in detections]
            size_mean = np.mean(sizes)
            size_std = np.std(sizes)
            size_cv = (size_std / (size_mean + 1)) * 100  # Coefficient of variation

            # ===== METRIC 3: Texture Consistency =====
            laplacians = [d['laplacian'] for d in detections]
            laplacian_mean = np.mean(laplacians)
            laplacian_std = np.std(laplacians)

            # Posters have VERY consistent sharpness, humans vary due to expressions
            texture_consistency = laplacian_std / (laplacian_mean + 1)

            # ===== METRIC 4: Edge Pattern =====
            edge_densities = [d['edge_density'] for d in detections]
            edge_mean = np.mean(edge_densities)
            edge_std = np.std(edge_densities)

            # Posters have high, consistent edges; humans have variable edges
            edge_consistency = edge_std / (edge_mean + 0.01)

            # ===== METRIC 5: Color Variation =====
            color_vars = [d['color_var'] for d in detections]
            color_var_mean = np.mean(color_vars)
            color_var_std = np.std(color_vars)

            # ===== METRIC 6: Temporal Pattern =====
            # Real humans have organic movement patterns, not perfectly still or perfectly regular
            frame_gaps = [detections[i]['frame'] - detections[i-1]['frame'] for i in range(1, len(detections))]
            temporal_spread = max(detections[-1]['frame'] - detections[0]['frame'], 1)

            # Calculate "liveness score" - higher = more likely human
            liveness_score = 0.0

            # Movement contribution (humans move 3-50 pixels typically)
            if avg_movement > 3:
                liveness_score += min(1.0, avg_movement / 20) * 25  # Max 25 points

            # Movement variability (humans don't move at constant speed)
            if movement_variance > 2:
                liveness_score += min(1.0, movement_variance / 10) * 15  # Max 15 points

            # Size variation (humans move in 3D space)
            if size_cv > 3:
                liveness_score += min(1.0, size_cv / 15) * 20  # Max 20 points

            # Texture variation (micro-expressions)
            if texture_consistency > 0.05:
                liveness_score += min(1.0, texture_consistency / 0.3) * 15  # Max 15 points

            # Edge variation (facial expressions change edges)
            if edge_consistency > 0.03:
                liveness_score += min(1.0, edge_consistency / 0.15) * 15  # Max 15 points

            # Color variation bonus
            if color_var_std > 5:
                liveness_score += min(1.0, color_var_std / 20) * 10  # Max 10 points

            # Track info for logging
            track_info = {
                'zone': zone_key,
                'samples': len(detections),
                'movement': avg_movement,
                'size_cv': size_cv,
                'texture_var': texture_consistency,
                'edge_var': edge_consistency,
                'liveness': liveness_score
            }
            all_tracks_info.append(track_info)

            # Classification threshold: 40 out of 100 for human
            is_human = liveness_score >= 40

            # Get samples from this track (spread across time)
            samples_per_track = min(len(detections), max_samples // max(len(face_tracks), 1) + 3)
            sample_step = max(1, len(detections) // samples_per_track)

            for i in range(0, len(detections), sample_step):
                crop = detections[i]['crop']

                if is_human and len(human_samples) < max_samples:
                    human_samples.append(crop)
                elif not is_human and len(poster_samples) < max_samples:
                    poster_samples.append(crop)

        # Log analysis results
        print(f"\n=== Face Classifier Training Analysis ===")
        for info in sorted(all_tracks_info, key=lambda x: -x['liveness']):
            label = "HUMAN" if info['liveness'] >= 40 else "POSTER"
            print(f"  Zone {info['zone']}: {label} (score={info['liveness']:.1f}, "
                  f"mov={info['movement']:.1f}, size_cv={info['size_cv']:.1f}, "
                  f"tex={info['texture_var']:.3f}, edge={info['edge_var']:.3f})")

        if progress_callback:
            progress_callback(50, f"Found {len(human_samples)} human, {len(poster_samples)} poster candidates")

        # Need minimum human samples
        if len(human_samples) < 10:
            if progress_callback:
                progress_callback(100, f"Not enough human faces detected ({len(human_samples)} < 10)")
            print(f"Training failed: Only {len(human_samples)} human samples detected")
            return False

        # Generate DIVERSE synthetic poster samples if needed
        # These represent various types of non-human face displays
        if len(poster_samples) < 10:
            if progress_callback:
                progress_callback(55, "Generating diverse synthetic static samples...")

            # Multiple techniques to create realistic poster/screen samples
            for img in human_samples[:min(50, len(human_samples))]:
                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 1: Printed Material (smooth, reduced texture) ===
                # Bilateral filter removes micro-texture while keeping edges
                bilateral = cv2.bilateralFilter(img, 9, 75, 75)
                poster_samples.append(bilateral)

                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 2: Low Quality Print (posterized colors) ===
                # Reduce color depth like cheap printing
                posterized = (img // 32) * 32
                poster_samples.append(posterized)

                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 3: Screen Display (slight blur + color shift) ===
                # Simulates face on a monitor/TV screen
                screen = cv2.GaussianBlur(img, (3, 3), 0)
                # Add slight color cast (screens have different white balance)
                screen = screen.astype(np.float32)
                screen[:,:,0] *= 0.95  # Less blue
                screen[:,:,2] *= 1.05  # More red
                screen = np.clip(screen, 0, 255).astype(np.uint8)
                poster_samples.append(screen)

                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 4: Halftone Pattern (magazine/newspaper print) ===
                # Add subtle dot pattern like printed photos
                halftone = img.copy()
                h, w = halftone.shape[:2]
                # Create dot pattern overlay
                for y in range(0, h, 4):
                    for x in range(0, w, 4):
                        if (x + y) % 8 < 4:
                            halftone[y:min(y+2,h), x:min(x+2,w)] = halftone[y:min(y+2,h), x:min(x+2,w)] * 0.9
                poster_samples.append(halftone)

                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 5: Flat Lighting (removes 3D depth cues) ===
                # Real faces have shadows from 3D structure, posters are flat
                flat = cv2.GaussianBlur(img, (15, 15), 0)  # Heavy blur
                # Blend original with blurred to flatten lighting
                flattened = cv2.addWeighted(img, 0.5, flat, 0.5, 0)
                poster_samples.append(flattened)

                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 6: Paper Texture ===
                # Add uniform noise like paper grain
                paper = img.copy().astype(np.float32)
                noise = np.random.uniform(-10, 10, paper.shape)
                paper = np.clip(paper + noise, 0, 255).astype(np.uint8)
                poster_samples.append(paper)

                if len(poster_samples) >= max_samples:
                    break

                # === METHOD 7: Saturated Colors (poster/banner style) ===
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,1] *= 1.3  # Increase saturation
                hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                poster_samples.append(saturated)

        print(f"Training with {len(human_samples)} human samples, {len(poster_samples)} poster samples")

        # Train the model with improved parameters
        if progress_callback:
            progress_callback(60, "Training classifier with improved parameters...")

        try:
            def train_progress(p, msg):
                if progress_callback:
                    progress_callback(60 + p * 0.4, msg)

            accuracy = self.train(
                human_samples,
                poster_samples,
                epochs=80,  # More epochs for better convergence
                patience=10,  # More patience for better model
                min_delta=0.0005,  # Smaller delta for finer tuning
                val_split=0.2,
                progress_callback=train_progress
            )

            if progress_callback:
                progress_callback(100, f"Trained! Accuracy: {accuracy:.1f}% ({len(human_samples)} human, {len(poster_samples)} static)")

            print(f"Training complete! Validation accuracy: {accuracy:.1f}%")
            return True

        except Exception as e:
            if progress_callback:
                progress_callback(100, f"Training failed: {e}")
            print(f"Training exception: {e}")
            return False
    
    def is_human_face(self, face_crop: np.ndarray, threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Classify if face is human or poster

        Args:
            face_crop: BGR numpy array of face region
            threshold: Confidence threshold for human classification

        Returns:
            (is_human, confidence)
        """
        if not self.is_loaded:
            # Default to True if no model loaded
            return True, 1.0

        try:
            # Preprocess
            if len(face_crop.shape) == 2:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
            elif face_crop.shape[2] == 4:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGBA2RGB)
            else:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            img = self.transform(face_crop).unsqueeze(0).to(self.device, non_blocking=True)

            # Inference with mixed precision on GPU
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(img)
                else:
                    outputs = self.model(img)
                probs = torch.softmax(outputs, dim=1)
                human_prob = probs[0][1].item()

            is_human = human_prob >= threshold
            return is_human, human_prob

        except Exception as e:
            print(f"Classification error: {e}")
            return True, 1.0

    def classify_faces_batch(
        self,
        face_crops: List[np.ndarray],
        threshold: float = 0.6
    ) -> List[Tuple[bool, float]]:
        """
        Classify multiple faces at once (batch processing for GPU efficiency)

        Args:
            face_crops: List of BGR numpy arrays of face regions
            threshold: Confidence threshold for human classification

        Returns:
            List of (is_human, confidence) tuples
        """
        if not self.is_loaded or not face_crops:
            return [(True, 1.0)] * len(face_crops)

        try:
            # Preprocess all faces
            batch_tensors = []
            for face_crop in face_crops:
                if len(face_crop.shape) == 2:
                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
                elif face_crop.shape[2] == 4:
                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGBA2RGB)
                else:
                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                batch_tensors.append(self.transform(face_crop))

            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device, non_blocking=True)

            # Batch inference with mixed precision
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                human_probs = probs[:, 1].cpu().numpy()

            results = []
            for prob in human_probs:
                results.append((prob >= threshold, float(prob)))

            return results

        except Exception as e:
            print(f"Batch classification error: {e}")
            return [(True, 1.0)] * len(face_crops)
    
    def filter_human_faces(
        self,
        frame: np.ndarray,
        faces: List[dict],
        threshold: float = 0.6
    ) -> List[dict]:
        """
        Filter faces to only include human faces (uses batch processing for GPU efficiency)

        Args:
            frame: Full frame (BGR numpy)
            faces: List of face dicts with 'bbox' key
            threshold: Confidence threshold

        Returns:
            Filtered list of human faces only
        """
        if not self.is_loaded or not faces:
            return faces  # Return all if no model

        # Extract all face crops
        face_crops = []
        valid_indices = []
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['bbox']
            face_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size > 0 and face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
                face_crops.append(face_crop)
                valid_indices.append(i)

        if not face_crops:
            return []

        # Batch classify all faces at once
        results = self.classify_faces_batch(face_crops, threshold)

        # Filter based on results - also store confidence on ALL faces for debugging
        human_faces = []
        for idx, (is_human, conf) in zip(valid_indices, results):
            faces[idx]['human_confidence'] = conf
            faces[idx]['is_human'] = is_human
            if is_human:
                human_faces.append(faces[idx])

        return human_faces


if __name__ == "__main__":
    # Test
    classifier = FaceClassifier()
    print(f"Device: {classifier.device}")
    print(f"Model loaded: {classifier.is_loaded}")
