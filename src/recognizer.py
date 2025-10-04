"""
Face Recognition Module

Implements face matching logic, new person enrollment, and confidence scoring.
Combines face detection, embedding generation, and vector database operations.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import cv2
from datetime import datetime
import uuid

from .face_detector import FaceDetector
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Main face recognition system combining all components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face recognizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.recognition_config = config.get('recognition', {})
        
        # Recognition settings
        self.similarity_threshold = self.recognition_config.get('similarity_threshold', 0.7)
        self.max_distance = self.recognition_config.get('max_distance', 0.6)
        self.auto_enroll = self.recognition_config.get('auto_enroll', True)
        self.confidence_threshold = self.recognition_config.get('confidence_threshold', 0.8)
        
        # Initialize components
        self.face_detector = FaceDetector(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_store = VectorStore(config)
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_recognitions': 0,
            'new_enrollments': 0,
            'failed_embeddings': 0,
            'session_start': datetime.now().isoformat()
        }
        
        logger.info("Face recognizer initialized successfully")
    
    def recognize_faces(self, frame: np.ndarray, 
                       enroll_unknown: bool = None) -> Dict[str, Any]:
        """
        Complete face recognition pipeline for a single frame.
        
        Args:
            frame: Input frame/image
            enroll_unknown: Whether to enroll unknown faces (overrides config)
            
        Returns:
            Recognition results with detected faces and identities
        """
        if enroll_unknown is None:
            enroll_unknown = self.auto_enroll
        
        # Process frame to detect faces
        processed_faces = self.face_detector.process_frame(frame)
        self.stats['total_detections'] += len(processed_faces)
        
        recognition_results = {
            'frame_shape': frame.shape,
            'timestamp': datetime.now().isoformat(),
            'faces': [],
            'total_faces': len(processed_faces),
            'recognized_faces': 0,
            'new_enrollments': 0
        }
        
        for face_data in processed_faces:
            face_result = self._process_single_face(
                face_data, 
                enroll_unknown=enroll_unknown
            )
            recognition_results['faces'].append(face_result)
            
            # Update counters
            if face_result.get('person_id') is not None:
                if face_result.get('is_new_enrollment'):
                    recognition_results['new_enrollments'] += 1
                    self.stats['new_enrollments'] += 1
                else:
                    recognition_results['recognized_faces'] += 1
                    self.stats['successful_recognitions'] += 1
        
        return recognition_results
    
    def _process_single_face(self, face_data: Dict[str, Any], 
                           enroll_unknown: bool = True) -> Dict[str, Any]:
        """
        Process a single detected face.
        
        Args:
            face_data: Face detection data
            enroll_unknown: Whether to enroll if no match found
            
        Returns:
            Face recognition result
        """
        result = {
            'bbox': face_data['bbox'],
            'confidence': face_data['confidence'],
            'quality': face_data['quality'],
            'person_id': None,
            'person_name': None,
            'similarity': 0.0,
            'is_new_enrollment': False,
            'embedding_generated': False,
            'error': None
        }
        
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(
                face_data['face_image']
            )
            
            if embedding is None:
                result['error'] = 'Failed to generate embedding'
                self.stats['failed_embeddings'] += 1
                return result
            
            result['embedding_generated'] = True
            
            # Validate embedding
            validation = self.embedding_generator.validate_embedding(embedding)
            if not validation['valid']:
                result['error'] = f"Invalid embedding: {validation['reason']}"
                return result
            
            # Search for matches
            matches = self.vector_store.search_similar(
                embedding, 
                k=3, 
                threshold=self.similarity_threshold
            )
            
            if matches:
                # Use best match
                best_match = matches[0]
                result.update({
                    'person_id': best_match['person_id'],
                    'similarity': best_match['similarity'],
                    'person_name': best_match['metadata'].get('name', f"Person_{best_match['person_id']}")
                })
                
                # Add embedding to existing person for continuous learning
                self.vector_store.add_embedding(
                    person_id=best_match['person_id'],
                    embedding=embedding,
                    metadata={
                        'detection_confidence': face_data['confidence'],
                        'quality_score': face_data['quality'].get('blur_score', 0),
                        'match_similarity': best_match['similarity']
                    }
                )
                
            elif enroll_unknown:
                # Enroll new person
                person_id = self._enroll_new_person(embedding, face_data)
                result.update({
                    'person_id': person_id,
                    'person_name': f"Person_{person_id}",
                    'similarity': 1.0,  # Perfect match with self
                    'is_new_enrollment': True
                })
            
        except Exception as e:
            logger.error(f"Face processing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _enroll_new_person(self, embedding: np.ndarray, 
                          face_data: Dict[str, Any], 
                          person_name: Optional[str] = None) -> int:
        """
        Enroll a new person in the database.
        
        Args:
            embedding: Face embedding
            face_data: Face detection data
            person_name: Optional person name
            
        Returns:
            New person ID
        """
        metadata = {
            'name': person_name,
            'enrollment_timestamp': datetime.now().isoformat(),
            'detection_confidence': face_data['confidence'],
            'quality_score': face_data['quality'].get('blur_score', 0),
            'enrollment_id': str(uuid.uuid4())
        }
        
        person_id = self.vector_store.add_embedding(
            person_id=None,  # Auto-assign new ID
            embedding=embedding,
            metadata=metadata
        )
        
        logger.info(f"Enrolled new person with ID: {person_id}")
        return person_id
    
    def enroll_person_manually(self, frame: np.ndarray, person_name: str, 
                              bbox: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Manually enroll a person with a given name.
        
        Args:
            frame: Input frame
            person_name: Name for the person
            bbox: Optional bounding box (if not provided, uses best face)
            
        Returns:
            Enrollment result
        """
        result = {
            'success': False,
            'person_id': None,
            'person_name': person_name,
            'error': None
        }
        
        try:
            if bbox is not None:
                # Use provided bounding box
                face_crop = self.face_detector.crop_face(frame, bbox)
                if face_crop is None:
                    result['error'] = 'Failed to crop face from provided bbox'
                    return result
                
                # Quality check
                quality = self.face_detector.quality_check(face_crop)
                if not quality['valid']:
                    result['error'] = f"Poor face quality: {quality.get('reason')}"
                    return result
                
                # Preprocess
                preprocessed = self.face_detector.preprocess_face(face_crop)
                if preprocessed is None:
                    result['error'] = 'Face preprocessing failed'
                    return result
                
                face_data = {
                    'face_image': preprocessed,
                    'confidence': 1.0,
                    'quality': quality
                }
            else:
                # Detect best face in frame
                processed_faces = self.face_detector.process_frame(frame)
                if not processed_faces:
                    result['error'] = 'No faces detected in frame'
                    return result
                
                # Use highest confidence face
                face_data = max(processed_faces, key=lambda x: x['confidence'])
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(
                face_data['face_image']
            )
            
            if embedding is None:
                result['error'] = 'Failed to generate embedding'
                return result
            
            # Check for existing similar faces
            matches = self.vector_store.search_similar(
                embedding, 
                k=1, 
                threshold=self.similarity_threshold
            )
            
            if matches:
                logger.warning(f"Similar face already exists (Person {matches[0]['person_id']}), enrolling anyway")
            
            # Enroll person
            person_id = self._enroll_new_person(embedding, face_data, person_name)
            
            result.update({
                'success': True,
                'person_id': person_id,
                'similarity_check': matches[0]['similarity'] if matches else 0.0
            })
            
        except Exception as e:
            logger.error(f"Manual enrollment error: {e}")
            result['error'] = str(e)
        
        return result
    
    def update_person_name(self, person_id: int, new_name: str) -> bool:
        """
        Update the name of an existing person.
        
        Args:
            person_id: Person identifier
            new_name: New name for the person
            
        Returns:
            True if updated successfully
        """
        try:
            # Use the vector store's update method
            success = self.vector_store.update_person_metadata(
                person_id, 
                {'name': new_name}
            )
            
            if success:
                logger.info(f"Updated name for person {person_id} to '{new_name}'")
            else:
                logger.error(f"Person {person_id} not found")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update person name: {e}")
            return False
    
    def remove_person(self, person_id: int) -> bool:
        """
        Remove a person from the database.
        
        Args:
            person_id: Person identifier
            
        Returns:
            True if removed successfully
        """
        return self.vector_store.remove_person(person_id)
    
    def get_person_info(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Person information or None if not found
        """
        embeddings_metadata = self.vector_store.get_person_embeddings(person_id)
        if not embeddings_metadata:
            return None
        
        # Get information from most recent embedding
        latest_metadata = max(
            [metadata for _, metadata in embeddings_metadata],
            key=lambda x: x.get('timestamp', '')
        )
        
        return {
            'person_id': person_id,
            'name': latest_metadata.get('name', f'Person_{person_id}'),
            'total_embeddings': len(embeddings_metadata),
            'enrollment_date': latest_metadata.get('enrollment_timestamp'),
            'latest_seen': latest_metadata.get('timestamp'),
            'metadata': latest_metadata
        }
    
    def list_all_people(self) -> List[Dict[str, Any]]:
        """
        Get a list of all enrolled people.
        
        Returns:
            List of person information dictionaries
        """
        people = []
        stats = self.vector_store.get_statistics()
        
        for person_id in range(1, stats['next_person_id']):
            person_info = self.get_person_info(person_id)
            if person_info:
                people.append(person_info)
        
        return people
    
    def get_recognition_statistics(self) -> Dict[str, Any]:
        """Get recognition system statistics."""
        vector_stats = self.vector_store.get_statistics()
        
        stats = {
            **self.stats,
            **vector_stats,
            'recognition_rate': (
                self.stats['successful_recognitions'] / 
                max(1, self.stats['total_detections'])
            ),
            'enrollment_rate': (
                self.stats['new_enrollments'] / 
                max(1, self.stats['total_detections'])
            ),
            'embedding_success_rate': (
                (self.stats['total_detections'] - self.stats['failed_embeddings']) / 
                max(1, self.stats['total_detections'])
            )
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset recognition statistics."""
        self.stats = {
            'total_detections': 0,
            'successful_recognitions': 0,
            'new_enrollments': 0,
            'failed_embeddings': 0,
            'session_start': datetime.now().isoformat()
        }
    
    def save_system_state(self):
        """Save the current system state."""
        self.vector_store.save_database()
        logger.info("System state saved")
    
    def cleanup_old_data(self, max_embeddings_per_person: int = 10):
        """
        Clean up old embeddings to manage storage.
        
        Args:
            max_embeddings_per_person: Maximum embeddings to keep per person
        """
        removed = self.vector_store.cleanup_old_embeddings(max_embeddings_per_person)
        logger.info(f"Cleaned up {removed} old embeddings")
        return removed