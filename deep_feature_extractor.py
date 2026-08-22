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
            # Remove classifier, keep feature extractor
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.embedding_size = 1280
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove final FC layer
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.embedding_size = 512
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.embedding_size = 1280
        else:
            logger.warning(f"Unknown model {model_name}, using mobilenet_v2")
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(model.children())[:-1])
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
            return np.zeros(self.embedding_size)
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Preprocess and convert to tensor
        try:
            input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return np.zeros(self.embedding_size)
        
        # Extract features
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Flatten and convert to numpy
        embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize to unit vector (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
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
