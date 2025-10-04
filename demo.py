#!/usr/bin/env python3
"""
Face Recognition System Demo

This demo shows how to use the face recognition system programmatically.
Note: Requires opencv-python, mtcnn, face-recognition, faiss-cpu and other dependencies.
"""

import os
import sys
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_demo_config():
    """Create a minimal configuration for demo."""
    return {
        'face_detection': {
            'method': 'deepface',  # Use DeepFace for detection
            'detector_backend': 'opencv',  # Use OpenCV backend (no compilation needed)
            'min_confidence': 0.7,
            'min_face_size': 80
        },
        'embedding': {
            'model': 'Facenet',  # Use DeepFace Facenet model
            'detector_backend': 'opencv',
            'normalization': True
        },
        'vector_store': {
            'backend': 'chromadb',  # Use ChromaDB (pure Python)
            'similarity_metric': 'cosine'
        },
        'recognition': {
            'similarity_threshold': 0.7,
            'auto_enroll': True,
            'confidence_threshold': 0.8
        },
        'storage': {
            'embeddings_path': 'data/embeddings',
            'database_file': 'data/demo_vectors.pkl'
        }
    }

def demo_face_recognition():
    """Demonstrate face recognition system usage."""
    print("Face Recognition System Demo")
    print("=" * 40)
    
    try:
        # Import modules (will fail if dependencies not installed)
        from src.recognizer import FaceRecognizer
        import cv2
        
        print("✓ All modules imported successfully")
        
        # Create configuration
        config = create_demo_config()
        print("✓ Configuration created")
        
        # Initialize recognizer
        recognizer = FaceRecognizer(config)
        print("✓ Face recognizer initialized")
        
        # Demo with a synthetic image (since we don't have a camera)
        print("\nCreating synthetic test image...")
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Draw a simple face pattern for testing
        cv2.circle(test_image, (150, 150), 100, (128, 128, 128), -1)  # Face
        cv2.circle(test_image, (120, 120), 15, (255, 255, 255), -1)   # Left eye
        cv2.circle(test_image, (180, 120), 15, (255, 255, 255), -1)   # Right eye
        cv2.ellipse(test_image, (150, 180), (30, 15), 0, 0, 180, (255, 255, 255), -1)  # Mouth
        
        print("✓ Test image created")
        
        # Process the image
        print("\nProcessing image for face recognition...")
        results = recognizer.recognize_faces(test_image)
        
        print(f"Recognition results:")
        print(f"- Total faces detected: {results['total_faces']}")
        print(f"- Recognized faces: {results['recognized_faces']}")
        print(f"- New enrollments: {results['new_enrollments']}")
        
        # Show statistics
        stats = recognizer.get_recognition_statistics()
        print(f"\nSystem statistics:")
        print(f"- Total detections: {stats['total_detections']}")
        print(f"- Total people enrolled: {stats['total_people']}")
        
        # List enrolled people
        people = recognizer.list_all_people()
        print(f"\nEnrolled people: {len(people)}")
        for person in people:
            print(f"- ID {person['person_id']}: {person['name']}")
        
        print("\n✓ Demo completed successfully!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nTo run this demo, install the required dependencies:")
        print("pip install opencv-python numpy pillow deepface chromadb scikit-learn pyyaml")
        
    except Exception as e:
        print(f"✗ Demo error: {e}")

def show_usage():
    """Show how to use the system."""
    print("\nFace Recognition System Usage:")
    print("=" * 40)
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Run the system:")
    print("   python main.py                    # Default camera")
    print("   python main.py --camera 1         # Specific camera")
    print("   python main.py --video video.mp4  # Process video file")
    print("   python main.py --interactive      # Interactive mode")
    print("\n3. Interactive commands:")
    print("   q - Quit")
    print("   s - Save system state")
    print("   r - Reset statistics")
    print("   p - Print statistics")
    print("   l - List enrolled people")
    print("   e - Enroll person manually")
    print("   n - Rename/label a person by ID")
    print("\n4. Configuration:")
    print("   Edit config/config.yaml to customize behavior")
    print("\n5. Workflow:")
    print("   - System auto-enrolls unknown faces as 'Person_X'")
    print("   - Use 'l' command to list all enrolled people")
    print("   - Use 'n' command to rename them with proper names")
    print("   - System will recognize renamed people in future frames")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--usage':
        show_usage()
    else:
        demo_face_recognition()
        print("\nRun 'python demo.py --usage' for usage instructions")