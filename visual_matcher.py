"""
Visual Feature Matcher for Object Recognition
Supports both ORB features and deep learning embeddings
"""
import collections
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

try:
    from deep_feature_extractor import DeepFeatureExtractor
    DEEP_FEATURES_AVAILABLE = True
except Exception as e:
    DEEP_FEATURES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Deep features not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center with replicate border."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale image: zoom in (center crop) if scale > 1, shrink and pad if scale < 1."""
    h, w = img.shape[:2]
    if scale == 1.0 or h <= 2 or w <= 2:
        return img
    if scale > 1.0:
        new_w = max(1, int(w / scale))
        new_h = max(1, int(h / scale))
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = img[y1:y1 + new_h, x1:x1 + new_w]
        return cv2.resize(cropped, (w, h))
    else:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        shrunk = cv2.resize(img, (new_w, new_h))
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        return cv2.copyMakeBorder(shrunk, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)


def _adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness by factor with uint8 clipping."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _generate_crop_augmentations(crop: np.ndarray, max_aug: int = None) -> List[np.ndarray]:
    """
    Generate augmented copies for an object crop:
    - Base crop
    - Rotations, scales, brightness, and horizontal flip variations
    """
    if crop.size == 0:
        return []
    if max_aug is None:
        try:
            import config
            max_aug = getattr(config, 'ENROLL_AUGMENT_PER_FRAME', 2)
        except Exception:
            max_aug = 2
            
    variations = [crop]
    angles = [-15.0, 15.0, -30.0, 30.0, -10.0, 10.0, -20.0, 20.0, -5.0, 5.0, -25.0, 25.0]
    scales = [0.85, 1.15, 0.90, 1.10, 0.95, 1.05]
    brightnesses = [0.8, 1.2, 0.85, 1.15]
    
    # Add angles
    for a in angles:
        if len(variations) > max_aug:
            break
        variations.append(_rotate_image(crop, a))
        
    # Add scales
    for s in scales:
        if len(variations) > max_aug:
            break
        variations.append(_scale_image(crop, s))
        
    # Add brightnesses
    for b in brightnesses:
        if len(variations) > max_aug:
            break
        variations.append(_adjust_brightness(crop, b))

    # Add flipped variation if still needed
    if len(variations) <= max_aug:
        flipped = cv2.flip(crop, 1)
        variations.append(flipped)
        for a in [-15.0, 15.0]:
            if len(variations) > max_aug:
                break
            variations.append(_rotate_image(flipped, a))

    return variations[:max_aug + 1]


class VisualMatcher:
    """Extracts and matches visual features for object recognition"""
    
    def __init__(self, feature_type: str = 'ORB', match_threshold: float = 0.7, use_deep_features: bool = False, smoothing_window: int = 5):
        """
        Initialize the visual matcher
        
        Args:
            feature_type: 'ORB', 'SIFT', 'AKAZE', or 'DEEP'
            match_threshold: Minimum ratio of good matches (0-1)
            use_deep_features: Use deep learning embeddings (more robust)
            smoothing_window: Number of frames to average confidence over (temporal smoothing)
        """
        self.feature_type = feature_type
        self.match_threshold = match_threshold
        self.use_deep_features = use_deep_features or (feature_type == 'DEEP')
        self.smoothing_window = smoothing_window
        self._confidence_history: Dict[str, collections.deque] = {}
        
        # Initialize deep feature extractor if requested
        self.deep_extractor = None
        if self.use_deep_features:
            if DEEP_FEATURES_AVAILABLE:
                try:
                    self.deep_extractor = DeepFeatureExtractor(model_name='mobilenet_v2')
                    logger.info(f"Visual matcher initialized with DEEP features (MobileNetV2)")
                    return  # Skip traditional feature initialization
                except Exception as e:
                    logger.error(f"Failed to initialize deep features: {e}")
                    logger.info("Falling back to ORB features")
                    self.use_deep_features = False
            else:
                logger.warning("Deep features requested but not available, using ORB")
                self.use_deep_features = False
        
        # Initialize traditional feature detector
        if feature_type == 'ORB' or not self.use_deep_features:
            self.detector = cv2.ORB_create(nfeatures=500)
        elif feature_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=500)
        elif feature_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            logger.warning(f"Unknown feature type {feature_type}, using ORB")
            self.detector = cv2.ORB_create(nfeatures=500)
            self.feature_type = 'ORB'
        
        # Initialize matcher
        if self.feature_type == 'ORB' or self.feature_type == 'AKAZE':
            # Binary descriptors use Hamming distance
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Float descriptors use L2 distance
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        logger.info(f"Visual matcher initialized with {self.feature_type} features")
    
    def extract_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> Dict:
        """
        Extract visual features from an image region
        
        Args:
            image: Input image (BGR)
            bbox: Optional bounding box (x1, y1, x2, y2) to extract from
            
        Returns:
            Dictionary containing keypoints, descriptors, and color histogram
        """
        # Use deep learning embeddings if enabled
        if self.use_deep_features and self.deep_extractor:
            embedding = self.deep_extractor.extract_embedding(image, bbox)
            logger.info("Deep embedding: 1280D vector")
            return {
                'deep_embedding': embedding.tolist(),
                'embedding_size': len(embedding),
                'feature_type': 'deep'
            }
        
        # Traditional feature extraction (ORB/SIFT/AKAZE)
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        if roi.size == 0:
            logger.warning("Empty ROI for feature extraction")
            return {'keypoints': [], 'descriptors': None, 'color_hist': []}
        
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        # Convert keypoints to serializable format
        kp_coords = []
        kp_sizes = []
        kp_angles = []
        
        if keypoints:
            for kp in keypoints:
                kp_coords.append([float(kp.pt[0]), float(kp.pt[1])])
                kp_sizes.append(float(kp.size))
                kp_angles.append(float(kp.angle))
        
        # Extract color histogram (3 channels, 32 bins each)
        color_hist = []
        for i in range(3):
            hist = cv2.calcHist([roi], [i], None, [32], [0, 256])
            color_hist.extend(hist.flatten().tolist())
        
        # Convert descriptors to list for JSON serialization
        desc_list = descriptors.tolist() if descriptors is not None else None
        
        features = {
            'keypoints': kp_coords,
            'keypoint_sizes': kp_sizes,
            'keypoint_angles': kp_angles,
            'descriptors': desc_list,
            'color_hist': color_hist,
            'image_shape': roi.shape[:2]  # (height, width)
        }
        
        logger.debug(f"Extracted {len(kp_coords)} keypoints from object")
        return features
    
    def extract_multi_frame_features(self, images: List[np.ndarray], 
                                    bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """
        Extract features from multiple frames and aggregate them.
        For deep features, adds augmented copies (rotations +-15, +-30 deg, scale 0.85-1.15, brightness +-20%)
        and stores exactly 24 embeddings in 'deep_embeddings' as well as the normalized mean in 'deep_embedding'.
        
        Args:
            images: List of images
            bboxes: List of bounding boxes for each image
            
        Returns:
            Aggregated features dictionary
        """
        if self.use_deep_features:
            all_embeddings = []
            valid_crops = []
            import config
            for img, bbox in zip(images, bboxes):
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    h, w = img.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    crop = img[y1:y2, x1:x2]
                else:
                    crop = img

                if crop.size == 0:
                    continue
                valid_crops.append(crop)

            all_crops = []
            if valid_crops:
                if self.deep_extractor:
                    # Generate augmentations so total crops = 24
                    m = len(valid_crops)
                    augs_per_crop = max(2, (24 + m - 1) // m)
                    for c in valid_crops:
                        all_crops.extend(_generate_crop_augmentations(c, max_aug=augs_per_crop))

                    if len(all_crops) > 24:
                        all_crops = all_crops[:24]
                    elif len(all_crops) < 24:
                        orig_len = len(all_crops)
                        idx = 0
                        while len(all_crops) < 24:
                            all_crops.append(all_crops[idx % orig_len].copy())
                            idx += 1
                else:
                    for img, bbox in zip(images, bboxes):
                        feat = self.extract_features(img, bbox)
                        if feat.get('deep_embedding'):
                            emb = np.array(feat['deep_embedding'])
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb = emb / norm
                            all_embeddings.append(emb)

            if self.deep_extractor and all_crops:
                if hasattr(self.deep_extractor, 'extract_embeddings_batch'):
                    batch_embs = self.deep_extractor.extract_embeddings_batch(all_crops)
                    for emb in batch_embs:
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        all_embeddings.append(emb)
                else:
                    for c in all_crops:
                        try:
                            emb = self.deep_extractor.extract_embedding(c, bbox=None)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb = emb / norm
                            all_embeddings.append(emb)
                        except Exception as e:
                            logger.debug(f"Error extracting embedding: {e}")

            if all_embeddings and len(all_embeddings) != 24:
                if len(all_embeddings) > 24:
                    all_embeddings = all_embeddings[:24]
                else:
                    orig_n = len(all_embeddings)
                    i = 0
                    while len(all_embeddings) < 24:
                        all_embeddings.append(all_embeddings[i % orig_n].copy())
                        i += 1

            if not all_embeddings:
                logger.warning("No valid deep features extracted from frames")
                return {'deep_embedding': None, 'deep_embeddings': [], 'feature_type': 'deep'}

            avg_embedding = np.mean(all_embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            logger.info("Deep embedding: 1280D vector")
            logger.info(f"Aggregated {len(all_embeddings)} deep embeddings from {len(images)} frames (each enrollment stores 24 vectors of 1280D)")

            return {
                'deep_embedding': avg_embedding.tolist(),
                'deep_embeddings': [e.tolist() for e in all_embeddings],
                'embedding_size': len(avg_embedding),
                'feature_type': 'deep',
                'num_frames': len(images),
                'num_embeddings': len(all_embeddings)
            }

        # Traditional feature aggregation (ORB/SIFT/AKAZE)
        all_features = []
        for img, bbox in zip(images, bboxes):
            features = self.extract_features(img, bbox)
            if features.get('descriptors') is not None:
                all_features.append(features)

        if not all_features:
            logger.warning("No valid features extracted from frames")
            return {'keypoints': [], 'descriptors': None, 'color_hist': []}

        aggregated = {
            'keypoints': [],
            'keypoint_sizes': [],
            'keypoint_angles': [],
            'descriptors': [],
            'color_hist': [],
            'num_frames': len(all_features)
        }
        
        for feat in all_features:
            aggregated['keypoints'].extend(feat['keypoints'])
            aggregated['keypoint_sizes'].extend(feat['keypoint_sizes'])
            aggregated['keypoint_angles'].extend(feat['keypoint_angles'])
            if feat['descriptors']:
                aggregated['descriptors'].extend(feat['descriptors'])
            aggregated['color_hist'].extend(feat['color_hist'])
        
        # Check if we actually got descriptors
        if not aggregated['descriptors']:
            logger.warning("No descriptors in aggregated features")
            return {'keypoints': [], 'descriptors': None, 'color_hist': []}
        
        # Average the color histograms
        if aggregated['color_hist']:
            hist_array = np.array(aggregated['color_hist']).reshape(len(all_features), -1)
            aggregated['color_hist'] = np.mean(hist_array, axis=0).tolist()
        
        logger.info(f"Aggregated features from {len(all_features)} frames: "
                   f"{len(aggregated['keypoints'])} keypoints, "
                   f"{len(aggregated['descriptors'])} descriptors")
        
        return aggregated
    
    def match_vectorized(self, query_emb: np.ndarray, matrix: np.ndarray, row_labels: List[str]) -> Tuple[Optional[str], float]:
        """
        Match query embedding against a matrix of stored embeddings in a single vectorized cosine dot product.
        Matching = one vectorized matrix product (cosine), top-3 mean per object.
        
        Args:
            query_emb: 1D query embedding (or 2D 1xD)
            matrix: 2D numpy array of shape (N, D), where each row is L2-normalized
            row_labels: List of object names (length N) corresponding to rows in matrix
            
        Returns:
            (best_name, best_score)
        """
        if matrix is None or matrix.size == 0 or not row_labels:
            return None, 0.0
        
        q = np.array(query_emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        else:
            return None, 0.0

        # One vectorized matrix product!
        sims = np.dot(matrix, q)
        sims = np.clip(sims, 0.0, 1.0)
        
        # Group similarities by object name and calculate top-3 mean
        label_scores = collections.defaultdict(list)
        for idx, sim in enumerate(sims):
            label_scores[row_labels[idx]].append(float(sim))
        
        best_name = None
        best_score = -1.0
        for name, scores in label_scores.items():
            if len(scores) >= 3:
                # Top-3 mean
                top3 = sorted(scores, reverse=True)[:3]
                mean_score = sum(top3) / 3.0
            else:
                mean_score = sum(scores) / float(len(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                
        return best_name, max(0.0, float(best_score))

    def match_features(self, query_features: Dict, reference_features: Dict) -> Tuple[float, int]:
        """
        Match query features against reference features
        
        Args:
            query_features: Features from detected object
            reference_features: Features from enrolled object
            
        Returns:
            (confidence_score, num_good_matches)
        """
        # Handle deep learning embeddings
        if query_features.get('feature_type') == 'deep' and reference_features.get('feature_type') == 'deep':
            if query_features.get('deep_embedding') is None:
                return 0.0, 0
            query_emb = np.array(query_features['deep_embedding']).flatten()
            q_norm = np.linalg.norm(query_emb)
            if q_norm > 0:
                query_emb = query_emb / q_norm

            stored_set = reference_features.get('deep_embeddings')
            if stored_set and isinstance(stored_set, list) and len(stored_set) > 0:
                similarities = []
                for ref_e in stored_set:
                    ref_arr = np.array(ref_e).flatten()
                    r_norm = np.linalg.norm(ref_arr)
                    if r_norm > 0:
                        ref_arr = ref_arr / r_norm
                    sim = float(np.dot(query_emb, ref_arr))
                    sim = float(np.clip(sim, 0.0, 1.0))
                    similarities.append(sim)

                # Match = mean of top-3 cosine similarities against the stored set
                similarities.sort(reverse=True)
                top_k = similarities[:3]
                similarity = float(np.mean(top_k)) if top_k else 0.0
            elif reference_features.get('deep_embedding') is not None:
                # Backward compatible: single mean embedding
                ref_emb = np.array(reference_features['deep_embedding']).flatten()
                r_norm = np.linalg.norm(ref_emb)
                if r_norm > 0:
                    ref_emb = ref_emb / r_norm
                similarity = float(np.clip(np.dot(query_emb, ref_emb), 0.0, 1.0))
            else:
                return 0.0, 0

            logger.debug(f"Deep embedding similarity: {similarity:.3f}")
            return similarity, 1  # Return 1 as "match count" for consistency
        
        # Traditional feature matching (ORB/SIFT/AKAZE)
        # Check if descriptors exist
        if (query_features.get('descriptors') is None or 
            reference_features.get('descriptors') is None):
            logger.debug("No descriptors to match")
            return 0.0, 0
        
        # Convert descriptors back to numpy arrays
        query_desc = np.array(query_features['descriptors'], dtype=np.uint8 
                             if self.feature_type in ['ORB', 'AKAZE'] else np.float32)
        ref_desc = np.array(reference_features['descriptors'], dtype=np.uint8
                           if self.feature_type in ['ORB', 'AKAZE'] else np.float32)
        
        if len(query_desc) < 2 or len(ref_desc) < 2:
            logger.debug("Not enough descriptors for matching")
            return 0.0, 0
        
        # Match descriptors using KNN (k=2 for ratio test)
        try:
            matches = self.matcher.knnMatch(query_desc, ref_desc, k=2)
        except Exception as e:
            logger.error(f"Matching error: {e}")
            return 0.0, 0
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Lowe's ratio
                    good_matches.append(m)
        
        # Calculate confidence score
        if len(query_desc) == 0:
            return 0.0, 0
        
        match_ratio = len(good_matches) / len(query_desc)
        
        # Additional scoring: color histogram similarity
        if (query_features.get('color_hist') and 
            reference_features.get('color_hist')):
            query_hist = np.array(query_features['color_hist'])
            ref_hist = np.array(reference_features['color_hist'])
            
            # Normalize histograms
            query_hist = query_hist / (np.sum(query_hist) + 1e-6)
            ref_hist = ref_hist[:len(query_hist)] / (np.sum(ref_hist[:len(query_hist)]) + 1e-6)
            
            # Correlation coefficient
            hist_similarity = cv2.compareHist(
                query_hist.astype(np.float32), 
                ref_hist.astype(np.float32), 
                cv2.HISTCMP_CORREL
            )
            
            # Combine feature matching and color similarity
            confidence = 0.7 * match_ratio + 0.3 * max(0, hist_similarity)
        else:
            confidence = match_ratio
        
        logger.debug(f"Match: {len(good_matches)}/{len(query_desc)} good matches, "
                    f"confidence: {confidence:.3f}")
        
        return float(confidence), len(good_matches)
    
    def find_best_match(self, query_features: Dict, 
                       candidates: List[Tuple[int, str, Dict]]) -> Optional[Tuple[int, str, float]]:
        """
        Find the best matching object from a list of candidates
        
        Args:
            query_features: Features from detected object
            candidates: List of (object_id, custom_name, features) tuples
            
        Returns:
            (object_id, custom_name, confidence) of best match, or None
        """
        if not candidates:
            return None
        
        best_match = None
        best_confidence = 0.0
        best_id = None
        best_name = None
        
        for item in candidates:
            if len(item) == 3:
                obj_id, custom_name, ref_features = item
            elif len(item) >= 4:
                obj_id = item[0]
                custom_name = item[1]
                ref_features = item[3]
            else:
                continue
            confidence, num_matches = self.match_features(query_features, ref_features)
            
            if confidence > best_confidence and confidence >= self.match_threshold:
                best_confidence = confidence
                best_id = obj_id
                best_name = custom_name
        
        if best_id is not None:
            # Temporal smoothing: update history only for the winning candidate
            history = self._confidence_history.setdefault(
                best_name, collections.deque(maxlen=self.smoothing_window)
            )
            history.append(best_confidence)
            smoothed_confidence = sum(history) / len(history)
            logger.debug(f"Best match: '{best_name}' with confidence {smoothed_confidence:.3f} (raw {best_confidence:.3f}, smoothed over {len(history)} frames)")
            return (best_id, best_name, smoothed_confidence)
        
        logger.debug(f"No matches above threshold {self.match_threshold}")
        return None
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                         features1: Dict, features2: Dict) -> np.ndarray:
        """
        Visualize feature matches between two images (for debugging)
        
        Args:
            img1: Query image
            img2: Reference image
            features1: Features from img1
            features2: Features from img2
            
        Returns:
            Visualization image with matches drawn
        """
        # Reconstruct keypoints from features
        kp1 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=20) 
               for kp in features1['keypoints'][:100]]  # Limit for visualization
        kp2 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=20) 
               for kp in features2['keypoints'][:100]]
        
        # Match and draw
        if features1['descriptors'] and features2['descriptors']:
            desc1 = np.array(features1['descriptors'][:100], 
                           dtype=np.uint8 if self.feature_type in ['ORB', 'AKAZE'] else np.float32)
            desc2 = np.array(features2['descriptors'][:100],
                           dtype=np.uint8 if self.feature_type in ['ORB', 'AKAZE'] else np.float32)
            
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Draw matches
            match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return match_img
        
        return np.hstack([img1, img2])


if __name__ == "__main__":
    # Quick test with two similar images
    print("Visual Matcher Test")
    print("=" * 60)
    
    matcher = VisualMatcher(feature_type='ORB', match_threshold=0.5)
    print(f"[OK] Matcher initialized")
    
    # Create test images
    img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img2 = img1.copy()
    
    # Add some noise to img2
    noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Extract features
    features1 = matcher.extract_features(img1)
    features2 = matcher.extract_features(img2)
    
    print(f"[OK] Extracted {len(features1['keypoints'])} keypoints from image 1")
    print(f"[OK] Extracted {len(features2['keypoints'])} keypoints from image 2")
    
    # Match features
    confidence, num_matches = matcher.match_features(features1, features2)
    print(f"[OK] Match confidence: {confidence:.3f} ({num_matches} good matches)")
    
    print("=" * 60)
    print("Visual matcher is ready!")
