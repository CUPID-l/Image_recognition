"""
Embedding Generation Module

Converts face images to vector representations using DeepFace models.
Includes embedding normalization and consistency validation.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from deepface import DeepFace
from PIL import Image
import cv2
import os

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
        self.model_name = self.config.get('model', 'Facenet')
        self.embedding_size = self.config.get('embedding_size', 128)
        self.normalization = self.config.get('normalization', True)
        
        # DeepFace model options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
        self.detector_backend = self.config.get('detector_backend', 'opencv')
        
        self._initialize_model()
        logger.info(f"Embedding generator initialized with model: {self.model_name}")
    
    def _initialize_model(self):
        """Initialize the selected embedding model."""
        try:
            # Validate model name for DeepFace
            valid_models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']
            if self.model_name not in valid_models:
                logger.warning(f"Model {self.model_name} not in valid models {valid_models}, using Facenet")
                self.model_name = 'Facenet'
            
            # Test model by creating a dummy embedding
            logger.info(f"Testing DeepFace model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to Facenet
            self.model_name = 'Facenet'
            logger.info("Falling back to Facenet model")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding from face image using DeepFace.
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Embedding vector or None if generation fails
        """
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            return self._generate_deepface_embedding(face_image)
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _generate_deepface_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using DeepFace."""
        try:
            # Convert BGR to RGB if necessary
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Generate embedding using DeepFace
            embedding_obj = DeepFace.represent(
                img_path=rgb_image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
                normalization='base'
            )
            
            # Extract embedding from result
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embedding = np.array(embedding_obj[0]['embedding'])
            else:
                embedding = np.array(embedding_obj['embedding'])
            
            # Apply normalization if requested
            if self.normalization:
                embedding = self.normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"DeepFace embedding generation failed: {e}")
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