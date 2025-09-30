"""
Main Application Module

Handles real-time video capture, frame processing, and user interface.
Implements the continuous processing loop with performance optimization.
"""

import cv2
import numpy as np
import logging
import yaml
import argparse
import time
from typing import Dict, Any, Optional, List
import os
import sys
from datetime import datetime

from .recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class FaceRecognitionApp:
    """Main face recognition application."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the face recognition application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.video_config = self.config.get('video', {})
        
        # Video processing settings
        self.fps_limit = self.video_config.get('fps_limit', 30)
        self.frame_skip = self.video_config.get('frame_skip', 1)
        self.display_annotations = self.video_config.get('display_annotations', True)
        self.save_crops = self.video_config.get('save_crops', False)
        
        # Initialize components
        self.recognizer = FaceRecognizer(self.config)
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        logger.info("Face recognition application initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'face_detection': {
                'method': 'mtcnn',
                'min_confidence': 0.9,
                'min_face_size': 80
            },
            'embedding': {
                'model': 'face_recognition',
                'normalization': True
            },
            'vector_store': {
                'backend': 'faiss',
                'similarity_metric': 'cosine'
            },
            'recognition': {
                'similarity_threshold': 0.7,
                'auto_enroll': True
            },
            'video': {
                'fps_limit': 30,
                'frame_skip': 1,
                'display_annotations': True
            }
        }
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize video camera.
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            True if camera initialized successfully
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera {camera_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def process_video_file(self, video_path: str):
        """
        Process a video file instead of live camera.
        
        Args:
            video_path: Path to video file
        """
        try:
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return
            
            logger.info(f"Processing video file: {video_path}")
            self.run_recognition_loop()
            
        except Exception as e:
            logger.error(f"Video file processing error: {e}")
    
    def annotate_frame(self, frame: np.ndarray, 
                      recognition_results: Dict[str, Any]) -> np.ndarray:
        """
        Annotate frame with recognition results.
        
        Args:
            frame: Input frame
            recognition_results: Recognition results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw faces and labels
        for face_result in recognition_results['faces']:
            bbox = face_result['bbox']
            x, y, w, h = bbox
            
            # Determine colors based on recognition status
            if face_result.get('is_new_enrollment'):
                color = (0, 255, 255)  # Yellow for new enrollments
                status = "NEW"
            elif face_result.get('person_id') is not None:
                color = (0, 255, 0)  # Green for recognized faces
                status = "RECOGNIZED"
            else:
                color = (0, 0, 255)  # Red for unrecognized faces
                status = "UNKNOWN"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label_parts = []
            if face_result.get('person_name'):
                label_parts.append(face_result['person_name'])
            elif face_result.get('person_id'):
                label_parts.append(f"ID: {face_result['person_id']}")
            
            if face_result.get('similarity'):
                label_parts.append(f"{face_result['similarity']:.2f}")
            
            label_text = " | ".join(label_parts) if label_parts else status
            
            # Draw label background
            label_size = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)[0]
            cv2.rectangle(annotated_frame, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text, (x, y - 5), 
                       self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Draw statistics
        self._draw_statistics(annotated_frame, recognition_results)
        
        return annotated_frame
    
    def _draw_statistics(self, frame: np.ndarray, 
                        recognition_results: Dict[str, Any]):
        """Draw system statistics on frame."""
        stats_y = 30
        stats_color = (255, 255, 255)
        stats_bg_color = (0, 0, 0)
        
        # FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        self._draw_text_with_background(frame, fps_text, (10, stats_y), stats_color, stats_bg_color)
        
        # Face count
        face_count_text = f"Faces: {recognition_results['total_faces']}"
        self._draw_text_with_background(frame, face_count_text, (150, stats_y), stats_color, stats_bg_color)
        
        # Recognition count
        recognized_text = f"Recognized: {recognition_results['recognized_faces']}"
        self._draw_text_with_background(frame, recognized_text, (250, stats_y), stats_color, stats_bg_color)
        
        # New enrollments
        if recognition_results['new_enrollments'] > 0:
            new_text = f"New: {recognition_results['new_enrollments']}"
            self._draw_text_with_background(frame, new_text, (400, stats_y), (0, 255, 255), stats_bg_color)
    
    def _draw_text_with_background(self, frame: np.ndarray, text: str, 
                                  position: tuple, text_color: tuple, bg_color: tuple):
        """Draw text with background rectangle."""
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        x, y = position
        
        # Draw background
        cv2.rectangle(frame, (x - 2, y - text_size[1] - 2), 
                     (x + text_size[0] + 2, y + 2), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, position, self.font, self.font_scale, text_color, self.font_thickness)
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_recognition_loop(self):
        """Run the main recognition loop."""
        if self.cap is None:
            logger.error("Camera not initialized")
            return
        
        self.is_running = True
        logger.info("Starting face recognition loop")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Skip frames if configured
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Process frame for face recognition
                start_time = time.time()
                recognition_results = self.recognizer.recognize_faces(frame)
                processing_time = time.time() - start_time
                
                # Annotate frame if enabled
                if self.display_annotations:
                    annotated_frame = self.annotate_frame(frame, recognition_results)
                else:
                    annotated_frame = frame
                
                # Display frame
                cv2.imshow('Face Recognition System', annotated_frame)
                
                # Update FPS
                self._update_fps()
                
                # Log processing time for performance monitoring
                if processing_time > 0.1:  # Log if processing takes more than 100ms
                    logger.debug(f"Frame processing time: {processing_time:.3f}s")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    # Save current system state
                    self.recognizer.save_system_state()
                    logger.info("System state saved")
                elif key == ord('r'):
                    # Reset statistics
                    self.recognizer.reset_statistics()
                    logger.info("Statistics reset")
                elif key == ord('p'):
                    # Print statistics
                    stats = self.recognizer.get_recognition_statistics()
                    logger.info(f"Recognition Statistics: {stats}")
                
                # FPS limiting
                if self.fps_limit > 0:
                    time.sleep(max(0, 1.0/self.fps_limit - processing_time))
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Recognition loop error: {e}")
        finally:
            self._cleanup()
    
    def run_interactive_mode(self):
        """Run in interactive mode with additional commands."""
        if not self.initialize_camera():
            return
        
        print("\n=== Face Recognition System ===")
        print("Commands:")
        print("  'q' - Quit")
        print("  's' - Save system state")
        print("  'r' - Reset statistics")
        print("  'p' - Print statistics")
        print("  'l' - List all enrolled people")
        print("  'e' - Enroll person manually")
        print("Starting recognition loop...\n")
        
        # Start background recognition
        import threading
        recognition_thread = threading.Thread(target=self.run_recognition_loop)
        recognition_thread.daemon = True
        recognition_thread.start()
        
        # Handle interactive commands
        while self.is_running:
            try:
                cmd = input().strip().lower()
                
                if cmd == 'q':
                    self.is_running = False
                    break
                elif cmd == 's':
                    self.recognizer.save_system_state()
                    print("System state saved")
                elif cmd == 'r':
                    self.recognizer.reset_statistics()
                    print("Statistics reset")
                elif cmd == 'p':
                    stats = self.recognizer.get_recognition_statistics()
                    print(f"Recognition Statistics: {stats}")
                elif cmd == 'l':
                    people = self.recognizer.list_all_people()
                    print(f"Enrolled people ({len(people)}):")
                    for person in people:
                        print(f"  ID {person['person_id']}: {person['name']} "
                              f"({person['total_embeddings']} embeddings)")
                elif cmd == 'e':
                    self._manual_enrollment()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Command error: {e}")
        
        self.is_running = False
    
    def _manual_enrollment(self):
        """Handle manual person enrollment."""
        try:
            name = input("Enter person name: ").strip()
            if not name:
                print("Name cannot be empty")
                return
            
            print("Point the camera at the person and press Enter...")
            input()
            
            # Capture current frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                return
            
            # Enroll person
            result = self.recognizer.enroll_person_manually(frame, name)
            
            if result['success']:
                print(f"Successfully enrolled {name} with ID {result['person_id']}")
            else:
                print(f"Enrollment failed: {result['error']}")
                
        except Exception as e:
            print(f"Manual enrollment error: {e}")
    
    def _cleanup(self):
        """Clean up resources."""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Save final system state
        self.recognizer.save_system_state()
        
        logger.info("Application cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--video', '-v', type=str,
                       help='Video file path (instead of camera)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize application
        app = FaceRecognitionApp(args.config)
        
        if args.video:
            # Process video file
            app.process_video_file(args.video)
        elif args.interactive:
            # Interactive mode
            app.run_interactive_mode()
        else:
            # Standard camera mode
            if app.initialize_camera(args.camera):
                app.run_recognition_loop()
            else:
                logger.error("Failed to initialize camera")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())