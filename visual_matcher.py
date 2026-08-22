"""
Visual Feature Matcher for Object Recognition
Supports both ORB features and deep learning embeddings
"""
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


class VisualMatcher:
    """Extracts and matches visual features for object recognition"""
    
    def __init__(self, feature_type: str = 'ORB', match_threshold: float = 0.7, use_deep_features: bool = False):
        """
        Initialize the visual matcher
        
        Args:
            feature_type: 'ORB', 'SIFT', 'AKAZE', or 'DEEP'
            match_threshold: Minimum ratio of good matches (0-1)
            use_deep_features: Use deep learning embeddings (more robust)
        """
        self.feature_type = feature_type
        self.match_threshold = match_threshold
        self.use_deep_features = use_deep_features or (feature_type == 'DEEP')
        
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
        Extract features from multiple frames and aggregate them
        
        Args:
            images: List of images
            bboxes: List of bounding boxes for each image
            
        Returns:
            Aggregated features dictionary
        """
        all_features = []
        
        for img, bbox in zip(images, bboxes):
            features = self.extract_features(img, bbox)
            if self.use_deep_features:
                # For deep features, check if embedding exists
                if features.get('deep_embedding'):
                    all_features.append(features)
            else:
                # For traditional features, check if descriptors exist
                if features['descriptors'] is not None:
                    all_features.append(features)
        
        if not all_features:
            logger.warning("No valid features extracted from frames")
            return {'keypoints': [], 'descriptors': None, 'color_hist': []}
        
        # Handle deep learning embeddings
        if self.use_deep_features and all_features[0].get('feature_type') == 'deep':
            # Average embeddings across frames
            embeddings = [np.array(f['deep_embedding']) for f in all_features]
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Normalize
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            logger.info(f"Aggregated deep embeddings from {len(all_features)} frames: "
                       f"{len(avg_embedding)}D vector")
            
            return {
                'deep_embedding': avg_embedding.tolist(),
                'embedding_size': len(avg_embedding),
                'feature_type': 'deep',
                'num_frames': len(all_features)
            }
        
        # Traditional feature aggregation (ORB/SIFT/AKAZE)
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
            query_emb = np.array(query_features['deep_embedding']).flatten()
            ref_emb = np.array(reference_features['deep_embedding']).flatten()
            
            # Compute cosine similarity
            similarity = np.dot(query_emb, ref_emb).item()  # .item() converts to Python scalar
            similarity = float(np.clip(similarity, 0, 1))
            
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
        
        for obj_id, custom_name, ref_features in candidates:
            confidence, num_matches = self.match_features(query_features, ref_features)
            
            if confidence > best_confidence and confidence >= self.match_threshold:
                best_confidence = confidence
                best_id = obj_id
                best_name = custom_name
        
        if best_id is not None:
            logger.info(f"Best match: '{best_name}' with confidence {best_confidence:.3f}")
            return (best_id, best_name, best_confidence)
        
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
