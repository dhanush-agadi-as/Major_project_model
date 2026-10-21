"""
Deep Learning Feature Extractor for Object Recognition
Uses pre-trained CNN (MobileNetV2) to extract robust embeddings
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScopedTorchThreads:
    """Context manager for scoping torch threads to prevent CPU hogging during background extraction
    without permanently altering the global thread count used by YOLO/NMS."""
    def __init__(self, threads: int = 2):
        self.threads = threads
        self.prev_threads = None

    def __enter__(self):
        try:
            self.prev_threads = torch.get_num_threads()
            torch.set_num_threads(self.threads)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.prev_threads is not None:
                torch.set_num_threads(self.prev_threads)
        except Exception:
            pass


class DeepFeatureExtractor:
    """Extract deep learning embeddings using pre-trained CNN"""
    
    def __init__(self, model_name: str = 'mobilenet_v2'):
        """
        Initialize deep feature extractor
        
        Args:
            model_name: 'mobilenet_v2', 'resnet18', or 'efficientnet_b0'
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # Remove classifier, apply global average pooling
            self.model = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
            self.embedding_size = 1280
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove final FC layer
            self.model = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
            self.embedding_size = 512
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
            self.embedding_size = 1280
        else:
            logger.warning(f"Unknown model {model_name}, using mobilenet_v2")
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
            self.embedding_size = 1280
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Deep feature extractor initialized: {model_name} ({self.embedding_size}D embeddings) on {self.device}")
    
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> np.ndarray:
        """
        Extract deep learning embedding from image region
        
        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box (x1, y1, x2, y2)
            
        Returns:
            Embedding vector (1D numpy array)
        """
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        if roi.size == 0:
            logger.warning("Empty ROI for embedding extraction")
            return np.zeros(self.embedding_size, dtype=np.float32)
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Preprocess and convert to tensor
        try:
            input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return np.zeros(self.embedding_size, dtype=np.float32)
        
        # Extract features
        with torch.no_grad():
            with ScopedTorchThreads(2):
                feat = self.model(input_tensor)
                # Apply global average pooling (mean over H and W)
                if feat.dim() == 4:
                    feat = feat.mean(dim=[-2, -1])
                elif feat.dim() == 3:
                    feat = feat.mean(dim=[-2, -1])
        
        # Flatten and convert to numpy
        embedding = feat.squeeze().cpu().numpy().astype(np.float32).flatten()
        if embedding.size != self.embedding_size:
            if embedding.size > self.embedding_size:
                embedding = embedding[:self.embedding_size]
            else:
                embedding = np.pad(embedding, (0, self.embedding_size - embedding.size))
        
        # Normalize to unit vector (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = np.zeros(self.embedding_size, dtype=np.float32)
        
        logger.info(f"Deep embedding: {len(embedding)}D vector")
        return embedding
    
    def extract_embeddings_batch(self, crops: list) -> list:
        """
        Extract deep learning embeddings from a list of image crops in a single batched forward pass.
        
        Args:
            crops: List of crop images (BGR numpy arrays)
            
        Returns:
            List of 1D normalized numpy array embeddings
        """
        if not crops:
            return []
        
        tensors = []
        valid_indices = []
        for idx, roi in enumerate(crops):
            if roi is None or roi.size == 0:
                continue
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                t = self.transform(roi_rgb)
                tensors.append(t)
                valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"Error transforming crop {idx}: {e}")
        
        results = [np.zeros(self.embedding_size, dtype=np.float32) for _ in range(len(crops))]
        if not tensors:
            return results
        
        try:
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                with ScopedTorchThreads(2):
                    out = self.model(batch_tensor)
                    # Apply global average pooling (mean over H and W)
                    if out.dim() == 4:
                        out = out.mean(dim=[-2, -1])
                    elif out.dim() == 3:
                        out = out.mean(dim=[-2, -1])
            
            # Handle shape: (B, C) -> (B, 1280)
            embs = out.view(out.size(0), -1).cpu().numpy().astype(np.float32)
            if embs.shape[1] != self.embedding_size:
                if embs.shape[1] > self.embedding_size:
                    embs = embs[:, :self.embedding_size]
                else:
                    embs = np.pad(embs, ((0, 0), (0, self.embedding_size - embs.shape[1])))
            # L2 normalize each row
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
            
            for i, v_idx in enumerate(valid_indices):
                results[v_idx] = embs[i]
                logger.info(f"Deep embedding: {self.embedding_size}D vector")
        except Exception as e:
            logger.error(f"Error in batch embedding extraction: {e}")
            # Fallback to single crop extraction
            for v_idx in valid_indices:
                results[v_idx] = self.extract_embedding(crops[v_idx])
                
        return results

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0-1, higher is more similar)
        """
        # Cosine similarity (already L2-normalized, so just dot product)
        similarity = np.dot(embedding1, embedding2)
        
        # Clip to [0, 1] range
        similarity = np.clip(similarity, 0, 1)
        
        return float(similarity)


if __name__ == "__main__":
    # Quick test
    print("Deep Feature Extractor Test")
    print("=" * 60)
    
    extractor = DeepFeatureExtractor(model_name='mobilenet_v2')
    print(f"[OK] Extractor initialized")
    print(f"  Embedding size: {extractor.embedding_size}D")
    print(f"  Device: {extractor.device}")
    
    # Create test images
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = img1.copy()
    
    # Add noise
    noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    emb1 = extractor.extract_embedding(img1)
    emb2 = extractor.extract_embedding(img2)
    
    print(f"[OK] Embedding 1 shape: {emb1.shape}")
    print(f"[OK] Embedding 2 shape: {emb2.shape}")
    
    # Compute similarity
    similarity = extractor.compute_similarity(emb1, emb2)
    print(f"[OK] Similarity: {similarity:.3f}")
    
    print("\n" + "=" * 60)
    print("Deep feature extractor is ready!")
