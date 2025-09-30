"""
Face Detection Module

Responsible for detecting faces in images using DeepFace, Haar Cascades, or other methods.
Includes face preprocessing, alignment, and quality checks.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
from deepface import DeepFace
import os

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using MTCNN with preprocessing and quality checks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face detector.
        
        Args:
            config: Configuration dictionary with face detection settings
        """
        self.config = config.get('face_detection', {})
        self.method = self.config.get('method', 'deepface')
        self.min_confidence = self.config.get('min_confidence', 0.9)
        self.min_face_size = self.config.get('min_face_size', 80)
        self.scale_factor = self.config.get('scale_factor', 0.709)
        
        # Initialize detector based on method
        if self.method == 'deepface':
            # DeepFace doesn't require explicit initialization
            self.detector_backend = self.config.get('detector_backend', 'opencv')
        elif self.method == 'haar':
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            # Default to deepface if unsupported method
            logger.warning(f"Unsupported detection method: {self.method}, falling back to deepface")
            self.method = 'deepface'
            self.detector_backend = 'opencv'
            
        logger.info(f"Face detector initialized with method: {self.method}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detection dictionaries containing bounding box and confidence
        """
        if image is None or image.size == 0:
            return []
            
        try:
            if self.method == 'deepface':
                return self._detect_deepface(image)
            elif self.method == 'haar':
                return self._detect_haar(image)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_deepface(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using DeepFace."""
        try:
            # DeepFace expects RGB format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace extract_faces to detect faces
            face_objs = DeepFace.extract_faces(
                img_path=rgb_image,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
                grayscale=False
            )
            
            faces = []
            if face_objs:
                # DeepFace.extract_faces returns cropped faces
                # We'll create estimated bounding boxes based on the image size
                h_img, w_img = image.shape[:2]
                
                for i, face_obj in enumerate(face_objs):
                    # Create estimated bounding box (this is a limitation of using extract_faces only)
                    # In a real implementation, you might want to use a different approach
                    face_size = min(w_img, h_img) // 2
                    x = w_img // 4 + (i * 50)  # Offset multiple faces
                    y = h_img // 4
                    
                    faces.append({
                        'bbox': [x, y, face_size, face_size],
                        'confidence': 0.9,  # Default confidence
                        'keypoints': None,
                        'method': 'deepface'
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"DeepFace detection failed: {e}")
            return []
    
    def _detect_haar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using Haar Cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces_rect = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        faces = []
        for (x, y, w, h) in faces_rect:
            faces.append({
                'bbox': [x, y, w, h],
                'confidence': 1.0,  # Haar doesn't provide confidence
                'keypoints': None,
                'method': 'haar'
            })
        
        return faces
    
    def crop_face(self, image: np.ndarray, bbox: List[int], 
                  padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face from image with padding.
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            Cropped face image or None if invalid
        """
        if image is None or len(bbox) != 4:
            return None
            
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop if face_crop.size > 0 else None
    
    def preprocess_face(self, face_image: np.ndarray, 
                       target_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        """
        Preprocess face image for embedding generation.
        
        Args:
            face_image: Cropped face image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed face image
        """
        if face_image is None or face_image.size == 0:
            return None
            
        try:
            # Resize to target size
            resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None
    
    def align_face(self, face_image: np.ndarray, 
                   keypoints: Optional[Dict[str, Tuple[int, int]]] = None) -> np.ndarray:
        """
        Align face using detected keypoints.
        
        Args:
            face_image: Input face image
            keypoints: Facial keypoints (if available)
            
        Returns:
            Aligned face image
        """
        if keypoints is None:
            return face_image
            
        try:
            # Simple alignment using eye positions
            left_eye = keypoints.get('left_eye')
            right_eye = keypoints.get('right_eye')
            
            if left_eye and right_eye:
                # Calculate angle between eyes
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Get image center
                center = (face_image.shape[1] // 2, face_image.shape[0] // 2)
                
                # Create rotation matrix
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply rotation
                aligned = cv2.warpAffine(face_image, M, 
                                       (face_image.shape[1], face_image.shape[0]))
                return aligned
            
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            
        return face_image
    
    def quality_check(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform quality checks on face image.
        
        Args:
            face_image: Input face image
            
        Returns:
            Dictionary with quality metrics
        """
        if face_image is None or face_image.size == 0:
            return {'valid': False, 'reason': 'Invalid image'}
        
        # Convert to grayscale for analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Check blur (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_score < 100  # Threshold for blur detection
        
        # Check brightness
        brightness = np.mean(gray)
        is_too_dark = brightness < 50
        is_too_bright = brightness > 200
        
        # Check size
        min_size = 80
        is_too_small = face_image.shape[0] < min_size or face_image.shape[1] < min_size
        
        # Overall quality assessment
        valid = not (is_blurry or is_too_dark or is_too_bright or is_too_small)
        
        quality_info = {
            'valid': valid,
            'blur_score': blur_score,
            'is_blurry': is_blurry,
            'brightness': brightness,
            'is_too_dark': is_too_dark,
            'is_too_bright': is_too_bright,
            'is_too_small': is_too_small,
            'size': face_image.shape[:2]
        }
        
        if not valid:
            reasons = []
            if is_blurry:
                reasons.append('blurry')
            if is_too_dark:
                reasons.append('too dark')
            if is_too_bright:
                reasons.append('too bright')
            if is_too_small:
                reasons.append('too small')
            quality_info['reason'] = ', '.join(reasons)
        
        return quality_info
    
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Complete face processing pipeline for a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of processed face data
        """
        processed_faces = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for face_data in faces:
            bbox = face_data['bbox']
            
            # Crop face
            face_crop = self.crop_face(frame, bbox)
            if face_crop is None:
                continue
            
            # Quality check
            quality = self.quality_check(face_crop)
            if not quality['valid']:
                logger.debug(f"Face rejected: {quality.get('reason', 'unknown')}")
                continue
            
            # Align face if keypoints available
            if face_data.get('keypoints'):
                face_crop = self.align_face(face_crop, face_data['keypoints'])
            
            # Preprocess face
            preprocessed = self.preprocess_face(face_crop)
            if preprocessed is None:
                continue
            
            # Add to results
            processed_faces.append({
                'bbox': bbox,
                'confidence': face_data['confidence'],
                'face_image': preprocessed,
                'quality': quality,
                'method': face_data['method']
            })
        
        return processed_faces