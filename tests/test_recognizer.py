"""
Unit tests for FaceRecognizer module.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.recognizer import FaceRecognizer


class TestFaceRecognizer(unittest.TestCase):
    """Test cases for FaceRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create minimal test configuration
        self.config = {
            'face_detection': {
                'method': 'deepface',
                'detector_backend': 'opencv',
                'min_confidence': 0.7,
                'min_face_size': 80
            },
            'embedding': {
                'model': 'Facenet',
                'detector_backend': 'opencv',
                'normalization': True
            },
            'vector_store': {
                'backend': 'sklearn',  # Use sklearn for testing (faster)
                'similarity_metric': 'cosine'
            },
            'recognition': {
                'similarity_threshold': 0.7,
                'auto_enroll': True,
                'confidence_threshold': 0.8
            },
            'storage': {
                'embeddings_path': os.path.join(self.test_dir, 'embeddings'),
                'database_file': os.path.join(self.test_dir, 'vectors.pkl')
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_recognizer_initialization(self):
        """Test that recognizer initializes without errors."""
        try:
            recognizer = FaceRecognizer(self.config)
            self.assertIsNotNone(recognizer)
            self.assertEqual(recognizer.similarity_threshold, 0.7)
            self.assertTrue(recognizer.auto_enroll)
        except Exception as e:
            self.fail(f"Recognizer initialization failed: {e}")
    
    def test_update_person_name(self):
        """Test updating person name."""
        try:
            recognizer = FaceRecognizer(self.config)
            
            # First, we need to manually add a person to test renaming
            # Since we don't have actual faces, we'll add a synthetic embedding
            embedding = np.random.rand(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            face_data = {
                'face_image': None,
                'confidence': 0.95,
                'quality': {'blur_score': 0.8, 'valid': True},
                'bbox': [0, 0, 100, 100]
            }
            
            # Enroll a person
            person_id = recognizer._enroll_new_person(embedding, face_data, "TestPerson")
            self.assertIsNotNone(person_id)
            
            # Update the name
            success = recognizer.update_person_name(person_id, "NewName")
            self.assertTrue(success)
            
            # Verify the name was updated
            person_info = recognizer.get_person_info(person_id)
            self.assertEqual(person_info['name'], "NewName")
            
        except Exception as e:
            self.fail(f"Update person name test failed: {e}")
    
    def test_list_all_people(self):
        """Test listing all enrolled people."""
        try:
            recognizer = FaceRecognizer(self.config)
            
            # Initially, there should be no people
            people = recognizer.list_all_people()
            self.assertEqual(len(people), 0)
            
            # Add a synthetic person
            embedding = np.random.rand(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            face_data = {
                'face_image': None,
                'confidence': 0.95,
                'quality': {'blur_score': 0.8, 'valid': True},
                'bbox': [0, 0, 100, 100]
            }
            
            person_id = recognizer._enroll_new_person(embedding, face_data, "TestPerson")
            
            # Now there should be one person
            people = recognizer.list_all_people()
            self.assertEqual(len(people), 1)
            self.assertEqual(people[0]['name'], "TestPerson")
            self.assertEqual(people[0]['person_id'], person_id)
            
        except Exception as e:
            self.fail(f"List all people test failed: {e}")
    
    def test_get_recognition_statistics(self):
        """Test getting recognition statistics."""
        try:
            recognizer = FaceRecognizer(self.config)
            stats = recognizer.get_recognition_statistics()
            
            self.assertIsNotNone(stats)
            self.assertIn('total_detections', stats)
            self.assertIn('successful_recognitions', stats)
            self.assertIn('new_enrollments', stats)
            self.assertEqual(stats['total_detections'], 0)
            
        except Exception as e:
            self.fail(f"Get recognition statistics test failed: {e}")


if __name__ == '__main__':
    unittest.main()
