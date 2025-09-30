"""
Test cases for Face Detection Module
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.face_detector import FaceDetector


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'face_detection': {
                'method': 'haar',  # Use Haar for testing (faster)
                'min_confidence': 0.5,
                'min_face_size': 30
            }
        }
    
    @pytest.fixture
    def detector(self, config):
        """Create face detector instance."""
        return FaceDetector(config)
    
    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        # Create a simple test image with basic face-like features
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Draw a simple face-like pattern
        cv2.circle(image, (100, 100), 80, (128, 128, 128), -1)  # Face
        cv2.circle(image, (80, 80), 10, (255, 255, 255), -1)    # Left eye
        cv2.circle(image, (120, 80), 10, (255, 255, 255), -1)   # Right eye
        cv2.ellipse(image, (100, 120), (20, 10), 0, 0, 180, (255, 255, 255), -1)  # Mouth
        
        return image
    
    def test_detector_initialization(self, config):
        """Test detector initialization."""
        detector = FaceDetector(config)
        assert detector.method == 'haar'
        assert detector.min_confidence == 0.5
        assert detector.detector is not None
    
    def test_detect_faces_empty_image(self, detector):
        """Test face detection on empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect_faces(empty_image)
        assert isinstance(faces, list)
    
    def test_detect_faces_none_image(self, detector):
        """Test face detection on None image."""
        faces = detector.detect_faces(None)
        assert faces == []
    
    def test_crop_face_valid_bbox(self, detector, test_image):
        """Test face cropping with valid bounding box."""
        bbox = [50, 50, 100, 100]  # [x, y, width, height]
        cropped = detector.crop_face(test_image, bbox)
        assert cropped is not None
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0
    
    def test_crop_face_invalid_bbox(self, detector, test_image):
        """Test face cropping with invalid bounding box."""
        bbox = [300, 300, 100, 100]  # Outside image bounds
        cropped = detector.crop_face(test_image, bbox)
        assert cropped is None or cropped.size == 0
    
    def test_preprocess_face(self, detector, test_image):
        """Test face preprocessing."""
        bbox = [50, 50, 100, 100]
        cropped = detector.crop_face(test_image, bbox)
        preprocessed = detector.preprocess_face(cropped)
        
        if preprocessed is not None:
            assert preprocessed.shape == (160, 160, 3)  # Default size
            assert preprocessed.dtype == np.float32
            assert 0 <= preprocessed.max() <= 1  # Should be normalized
    
    def test_quality_check_valid_image(self, detector, test_image):
        """Test quality check on valid image."""
        quality = detector.quality_check(test_image)
        assert isinstance(quality, dict)
        assert 'valid' in quality
        assert 'blur_score' in quality
        assert 'brightness' in quality
    
    def test_quality_check_none_image(self, detector):
        """Test quality check on None image."""
        quality = detector.quality_check(None)
        assert quality['valid'] is False
        assert 'reason' in quality
    
    def test_process_frame(self, detector, test_image):
        """Test complete frame processing pipeline."""
        processed_faces = detector.process_frame(test_image)
        assert isinstance(processed_faces, list)
        
        # Check structure of processed faces
        for face_data in processed_faces:
            assert 'bbox' in face_data
            assert 'confidence' in face_data
            assert 'face_image' in face_data
            assert 'quality' in face_data