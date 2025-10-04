#!/usr/bin/env python3
"""
Validation script for face recognition auto-enrollment and labeling.

This script simulates the workflow without requiring a camera.
"""

import os
import sys
import tempfile
import shutil
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.recognizer import FaceRecognizer


def create_test_config(test_dir):
    """Create a test configuration."""
    return {
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
            'backend': 'sklearn',  # Use sklearn for faster testing
            'similarity_metric': 'cosine'
        },
        'recognition': {
            'similarity_threshold': 0.7,
            'auto_enroll': True,
            'confidence_threshold': 0.8
        },
        'storage': {
            'embeddings_path': os.path.join(test_dir, 'embeddings'),
            'database_file': os.path.join(test_dir, 'vectors.pkl'),
            'backup_interval': 100
        }
    }


def test_auto_enrollment_workflow():
    """Test the complete auto-enrollment and labeling workflow."""
    print("=" * 60)
    print("Face Recognition Auto-Enrollment & Labeling Validation")
    print("=" * 60)
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize recognizer
        print("\n1. Initializing face recognizer...")
        config = create_test_config(test_dir)
        recognizer = FaceRecognizer(config)
        print("   ✓ Recognizer initialized")
        
        # Simulate auto-enrollment of 3 different people
        print("\n2. Simulating auto-enrollment of 3 people...")
        
        person_ids = []
        for i in range(3):
            # Create a random embedding (simulating a face)
            embedding = np.random.rand(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            face_data = {
                'face_image': None,
                'confidence': 0.95,
                'quality': {'blur_score': 0.8, 'valid': True},
                'bbox': [0, 0, 100, 100]
            }
            
            # Auto-enroll (no name provided)
            person_id = recognizer._enroll_new_person(embedding, face_data)
            person_ids.append(person_id)
            print(f"   ✓ Auto-enrolled Person {person_id}")
        
        # List all people
        print("\n3. Listing all enrolled people...")
        people = recognizer.list_all_people()
        for person in people:
            print(f"   - ID {person['person_id']}: {person['name']} "
                  f"({person['total_embeddings']} embeddings)")
        
        # Verify auto-generated names
        print("\n4. Verifying auto-generated names...")
        success = True
        for person_id in person_ids:
            person_info = recognizer.get_person_info(person_id)
            expected_name = f"Person_{person_id}"
            if person_info['name'] == expected_name:
                print(f"   ✓ Person {person_id} has correct name: {expected_name}")
            else:
                print(f"   ✗ Person {person_id} has incorrect name: {person_info['name']} "
                      f"(expected: {expected_name})")
                success = False
        
        if not success:
            print("\n⚠ Auto-generated names verification FAILED!")
            return False
        
        # Rename people
        print("\n5. Renaming people with proper names...")
        new_names = ["Alice Johnson", "Bob Smith", "Charlie Brown"]
        for person_id, new_name in zip(person_ids, new_names):
            success = recognizer.update_person_name(person_id, new_name)
            if success:
                print(f"   ✓ Renamed Person {person_id} to '{new_name}'")
            else:
                print(f"   ✗ Failed to rename Person {person_id}")
                return False
        
        # Verify renamed people
        print("\n6. Verifying renamed people...")
        people = recognizer.list_all_people()
        for person, expected_name in zip(people, new_names):
            if person['name'] == expected_name:
                print(f"   ✓ ID {person['person_id']}: {person['name']}")
            else:
                print(f"   ✗ ID {person['person_id']}: {person['name']} "
                      f"(expected: {expected_name})")
                return False
        
        # Save and verify persistence
        print("\n7. Testing persistence...")
        recognizer.save_system_state()
        print("   ✓ System state saved")
        
        # Create new recognizer and verify data persists
        recognizer2 = FaceRecognizer(config)
        people2 = recognizer2.list_all_people()
        if len(people2) == len(people):
            print(f"   ✓ Loaded {len(people2)} people from saved state")
            for person in people2:
                print(f"     - ID {person['person_id']}: {person['name']}")
        else:
            print(f"   ✗ Expected {len(people)} people, got {len(people2)}")
            return False
        
        # Get statistics
        print("\n8. System statistics:")
        stats = recognizer2.get_recognition_statistics()
        print(f"   - Total people enrolled: {stats['total_people']}")
        print(f"   - Total embeddings: {stats['total_embeddings']}")
        print(f"   - Total detections: {stats['total_detections']}")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe system correctly:")
        print("  1. Auto-enrolls new faces with Person_X names")
        print("  2. Stores names properly in metadata")
        print("  3. Allows renaming of enrolled people")
        print("  4. Persists changes to disk")
        print("  5. Loads data correctly on restart")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    success = test_auto_enrollment_workflow()
    sys.exit(0 if success else 1)
