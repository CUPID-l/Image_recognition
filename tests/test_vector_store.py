"""
Unit tests for VectorStore module.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create minimal test configuration
        self.config = {
            'vector_store': {
                'backend': 'sklearn',  # Use sklearn for testing (faster)
                'similarity_metric': 'cosine'
            },
            'storage': {
                'embeddings_path': os.path.join(self.test_dir, 'embeddings'),
                'database_file': os.path.join(self.test_dir, 'vectors.pkl'),
                'backup_interval': 100
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_vector_store_initialization(self):
        """Test that vector store initializes without errors."""
        try:
            store = VectorStore(self.config)
            self.assertIsNotNone(store)
            self.assertEqual(store.backend, 'sklearn')
            self.assertEqual(store.similarity_metric, 'cosine')
        except Exception as e:
            self.fail(f"Vector store initialization failed: {e}")
    
    def test_add_embedding(self):
        """Test adding an embedding."""
        try:
            store = VectorStore(self.config)
            
            # Create a random embedding
            embedding = np.random.rand(128).astype(np.float32)
            metadata = {'name': 'TestPerson', 'test_field': 'test_value'}
            
            # Add embedding
            person_id = store.add_embedding(None, embedding, metadata)
            self.assertIsNotNone(person_id)
            self.assertEqual(person_id, 1)
            
            # Verify it was added
            embeddings = store.get_person_embeddings(person_id)
            self.assertEqual(len(embeddings), 1)
            
        except Exception as e:
            self.fail(f"Add embedding test failed: {e}")
    
    def test_update_person_metadata(self):
        """Test updating person metadata."""
        try:
            store = VectorStore(self.config)
            
            # Add an embedding
            embedding = np.random.rand(128).astype(np.float32)
            metadata = {'name': 'OriginalName', 'test_field': 'test_value'}
            person_id = store.add_embedding(None, embedding, metadata)
            
            # Update metadata
            success = store.update_person_metadata(person_id, {'name': 'NewName'})
            self.assertTrue(success)
            
            # Verify metadata was updated
            embeddings = store.get_person_embeddings(person_id)
            self.assertEqual(len(embeddings), 1)
            _, updated_metadata = embeddings[0]
            self.assertEqual(updated_metadata['name'], 'NewName')
            self.assertEqual(updated_metadata['test_field'], 'test_value')  # Other fields should remain
            self.assertIn('metadata_updated', updated_metadata)  # Should have timestamp
            
        except Exception as e:
            self.fail(f"Update person metadata test failed: {e}")
    
    def test_update_nonexistent_person(self):
        """Test updating metadata for a non-existent person."""
        try:
            store = VectorStore(self.config)
            
            # Try to update a person that doesn't exist
            success = store.update_person_metadata(999, {'name': 'NewName'})
            self.assertFalse(success)
            
        except Exception as e:
            self.fail(f"Update nonexistent person test failed: {e}")
    
    def test_search_similar(self):
        """Test similarity search."""
        try:
            store = VectorStore(self.config)
            
            # Add some embeddings
            embedding1 = np.random.rand(128).astype(np.float32)
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            person_id1 = store.add_embedding(None, embedding1, {'name': 'Person1'})
            
            embedding2 = np.random.rand(128).astype(np.float32)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            person_id2 = store.add_embedding(None, embedding2, {'name': 'Person2'})
            
            # Search for similar embeddings
            results = store.search_similar(embedding1, k=2, threshold=0.0)
            self.assertGreater(len(results), 0)
            
            # The first result should be person_id1 with high similarity
            self.assertEqual(results[0]['person_id'], person_id1)
            self.assertGreater(results[0]['similarity'], 0.9)  # Should be very similar to itself
            
        except Exception as e:
            self.fail(f"Search similar test failed: {e}")
    
    def test_get_statistics(self):
        """Test getting statistics."""
        try:
            store = VectorStore(self.config)
            
            # Initial statistics
            stats = store.get_statistics()
            self.assertIsNotNone(stats)
            self.assertIn('total_embeddings', stats)
            self.assertIn('total_people', stats)
            self.assertEqual(stats['total_embeddings'], 0)
            
            # Add an embedding
            embedding = np.random.rand(128).astype(np.float32)
            store.add_embedding(None, embedding, {'name': 'Test'})
            
            # Check updated statistics
            stats = store.get_statistics()
            self.assertEqual(stats['total_embeddings'], 1)
            self.assertEqual(stats['total_people'], 1)
            
        except Exception as e:
            self.fail(f"Get statistics test failed: {e}")


if __name__ == '__main__':
    unittest.main()
