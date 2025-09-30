"""
Embedding Generation Module

Converts face images to vector representations using FaceNet, DeepFace, or other models.
Includes embedding normalization and consistency validation.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
import face_recognition
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate face embeddings using various models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config.get('embedding', {})
        self.model_name = self.config.get('model', 'facenet')
        self.embedding_size = self.config.get('embedding_size', 128)
        self.normalization = self.config.get('normalization', True)
        
        # Initialize model based on configuration
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 config.get('performance', {}).get('use_gpu', False) 
                                 else 'cpu')
        
        self._initialize_model()
        logger.info(f"Embedding generator initialized with model: {self.model_name}")
    
    def _initialize_model(self):
        """Initialize the selected embedding model."""
        try:
            if self.model_name == 'facenet':
                self._initialize_facenet()
            elif self.model_name == 'face_recognition':
                # face_recognition library uses dlib's ResNet model
                logger.info("Using face_recognition library (dlib ResNet)")
            else:
                raise ValueError(f"Unsupported embedding model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to face_recognition
            self.model_name = 'face_recognition'
            logger.info("Falling back to face_recognition library")
    
    def _initialize_facenet(self):
        """Initialize FaceNet model."""
        try:
            # Load pre-trained FaceNet model
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.mtcnn_resnet = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=self.device
            )
            logger.info("FaceNet model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FaceNet model: {e}")
            raise
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding from face image.
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Embedding vector or None if generation fails
        """
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            if self.model_name == 'facenet':
                return self._generate_facenet_embedding(face_image)
            elif self.model_name == 'face_recognition':
                return self._generate_face_recognition_embedding(face_image)
            else:
                logger.error(f"Unknown model: {self.model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _generate_facenet_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using FaceNet."""
        try:
            # Convert numpy array to PIL Image
            if face_image.dtype != np.uint8:
                # If normalized [0,1], scale back to [0,255]
                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
            
            # Convert BGR to RGB if necessary
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(face_image)
            
            # Preprocess with MTCNN (resize and normalize)
            face_tensor = self.mtcnn_resnet(pil_image)
            
            if face_tensor is None:
                logger.warning("MTCNN preprocessing failed")
                return None
            
            # Ensure correct dimensions
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            face_tensor = face_tensor.to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            # Normalize if requested
            if self.normalization:
                embedding = self.normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"FaceNet embedding generation failed: {e}")
            return None
    
    def _generate_face_recognition_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using face_recognition library."""
        try:
            # Ensure image is in correct format
            if face_image.dtype != np.uint8:
                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
            
            # Convert BGR to RGB
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Generate embedding
            encodings = face_recognition.face_encodings(rgb_image)
            
            if len(encodings) == 0:
                logger.warning("No face encoding generated")
                return None
            
            # Use the first encoding if multiple faces detected
            embedding = encodings[0]
            
            # Normalize if requested
            if self.normalization:
                embedding = self.normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"face_recognition embedding generation failed: {e}")
            return None
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector using L2 normalization.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        if embedding is None or len(embedding) == 0:
            return embedding
        
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        
        return embedding / norm
    
    def batch_generate_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple face images.
        
        Args:
            face_images: List of preprocessed face images
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.generate_embedding(face_image)
            embeddings.append(embedding)
        
        return embeddings
    
    def validate_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Validate embedding quality and consistency.
        
        Args:
            embedding: Generated embedding vector
            
        Returns:
            Validation results
        """
        if embedding is None:
            return {'valid': False, 'reason': 'None embedding'}
        
        if len(embedding) == 0:
            return {'valid': False, 'reason': 'Empty embedding'}
        
        # Check for expected dimensions
        expected_sizes = [128, 512]  # Common embedding sizes
        if len(embedding) not in expected_sizes:
            logger.warning(f"Unexpected embedding size: {len(embedding)}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return {'valid': False, 'reason': 'NaN or infinite values'}
        
        # Check embedding magnitude
        magnitude = np.linalg.norm(embedding)
        if magnitude == 0:
            return {'valid': False, 'reason': 'Zero magnitude'}
        
        # Check if normalized (for normalized embeddings)
        is_normalized = abs(magnitude - 1.0) < 0.01 if self.normalization else True
        
        return {
            'valid': True,
            'size': len(embedding),
            'magnitude': magnitude,
            'is_normalized': is_normalized,
            'mean': np.mean(embedding),
            'std': np.std(embedding)
        }
    
    def compare_embeddings(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray, 
                          metric: str = 'cosine') -> float:
        """
        Compare two embeddings using specified metric.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            logger.error("Embedding size mismatch")
            return 0.0
        
        try:
            if metric == 'cosine':
                # Cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return max(0.0, similarity)  # Ensure non-negative
                
            elif metric == 'euclidean':
                # Euclidean distance (convert to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                # Convert distance to similarity (0-1 range)
                similarity = 1.0 / (1.0 + distance)
                return similarity
                
            else:
                logger.error(f"Unknown metric: {metric}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Embedding comparison failed: {e}")
            return 0.0
    
    def get_embedding_stats(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate statistics for a collection of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Statistical information
        """
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if not valid_embeddings:
            return {'count': 0, 'valid': False}
        
        embeddings_array = np.array(valid_embeddings)
        
        stats = {
            'count': len(valid_embeddings),
            'total_count': len(embeddings),
            'valid': True,
            'embedding_size': len(valid_embeddings[0]),
            'mean_magnitude': np.mean([np.linalg.norm(emb) for emb in valid_embeddings]),
            'std_magnitude': np.std([np.linalg.norm(emb) for emb in valid_embeddings]),
            'mean_values': np.mean(embeddings_array, axis=0),
            'std_values': np.std(embeddings_array, axis=0)
        }
        
        return stats