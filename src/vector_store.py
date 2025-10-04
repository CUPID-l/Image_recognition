"""
Vector Database Module

Manages storage and retrieval of face embeddings using ChromaDB or scikit-learn.
Includes indexing for fast similarity search and persistence functionality.
"""

import numpy as np
import logging
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import chromadb
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for storing and searching face embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector store.
        
        Args:
            config: Configuration dictionary with vector store settings
        """
        self.config = config
        self.vector_config = config.get('vector_store', {})
        self.storage_config = config.get('storage', {})
        
        self.backend = self.vector_config.get('backend', 'chromadb')
        self.similarity_metric = self.vector_config.get('similarity_metric', 'cosine')
        self.index_type = self.vector_config.get('index_type', 'flat')
        
        # Storage paths
        self.embeddings_path = self.storage_config.get('embeddings_path', 'data/embeddings')
        self.database_file = self.storage_config.get('database_file', 'data/vectors.pkl')
        self.backup_interval = self.storage_config.get('backup_interval', 100)
        
        # Initialize storage
        self._ensure_directories()
        
        # Database components
        if self.backend == 'chromadb':
            # ChromaDB setup
            self.chroma_client = chromadb.PersistentClient(path=self.embeddings_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine" if self.similarity_metric == "cosine" else "l2"}
            )
        else:
            # Fallback to sklearn
            self.nn_model = None
            
        self.embeddings = []  # List of embedding vectors
        self.metadata = []    # List of metadata dictionaries
        self.person_ids = []  # List of person IDs
        self.id_to_index = {}  # Mapping from person_id to index positions
        self.next_person_id = 1
        self.embedding_dim = None
        self.save_counter = 0
        
        # Initialize index
        self._initialize_index()
        
        # Load existing data
        self.load_database()
        
        logger.info(f"Vector store initialized with backend: {self.backend}")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.embeddings_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
    
    def _initialize_index(self, embedding_dim: int = 128):
        """Initialize the vector index."""
        self.embedding_dim = embedding_dim
        
        if self.backend == 'chromadb':
            # ChromaDB handles indexing internally
            logger.info(f"ChromaDB collection ready with dimension: {embedding_dim}")
        else:
            # Use sklearn NearestNeighbors as fallback
            metric = 'cosine' if self.similarity_metric == 'cosine' else 'euclidean'
            self.nn_model = NearestNeighbors(
                n_neighbors=5,
                metric=metric,
                algorithm='auto'
            )
            logger.info(f"sklearn NearestNeighbors initialized with dimension: {embedding_dim}")
    
    
    def add_embedding(self, person_id: Optional[int], embedding: np.ndarray, 
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add embedding to the vector store.
        
        Args:
            person_id: Person identifier (None for new person)
            embedding: Face embedding vector
            metadata: Additional metadata
            
        Returns:
            Person ID of the added embedding
        """
        if embedding is None or len(embedding) == 0:
            raise ValueError("Invalid embedding")
        
        # Initialize index if needed
        if self.backend == 'chromadb':
            if self.embedding_dim is None:
                self._initialize_index(len(embedding))
        else:
            if self.nn_model is None or self.embedding_dim != len(embedding):
                self._initialize_index(len(embedding))
        
        # Ensure embedding is normalized for cosine similarity
        if self.similarity_metric == 'cosine':
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # Assign person ID if not provided
        if person_id is None:
            person_id = self.next_person_id
            self.next_person_id += 1
        else:
            self.next_person_id = max(self.next_person_id, person_id + 1)
        
        # Prepare metadata - ensure all values are proper types for ChromaDB
        if metadata is None:
            metadata = {}
        
        # Clean metadata to ensure ChromaDB compatibility
        clean_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
        
        clean_metadata.update({
            'person_id': int(person_id),
            'timestamp': datetime.now().isoformat(),
            'embedding_size': int(len(embedding))
        })
        
        # Add to storage
        self.embeddings.append(embedding.copy())
        self.metadata.append(clean_metadata)
        self.person_ids.append(person_id)
        
        # Update ID mapping
        if person_id not in self.id_to_index:
            self.id_to_index[person_id] = []
        self.id_to_index[person_id].append(len(self.embeddings) - 1)
        
        # Add to backend
        if self.backend == 'chromadb':
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[clean_metadata],
                ids=[f"person_{person_id}_{len(self.id_to_index[person_id])}"]
            )
        else:
            # Rebuild sklearn model if needed
            if len(self.embeddings) > 0:
                self.nn_model.fit(np.array(self.embeddings))
        
        # Auto-save periodically
        self.save_counter += 1
        if self.save_counter >= self.backup_interval:
            self.save_database()
            self.save_counter = 0
        
        logger.debug(f"Added embedding for person {person_id}")
        return person_id
    
    def search_similar(self, query_embedding: np.ndarray, 
                      k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            threshold: Similarity threshold
            
        Returns:
            List of matching results with scores and metadata
        """
        if len(self.embeddings) == 0:
            return []
        
        if query_embedding is None or len(query_embedding) == 0:
            return []
        
        # Ensure embedding dimension matches
        if self.embedding_dim and len(query_embedding) != self.embedding_dim:
            logger.error(f"Query embedding dimension mismatch: {len(query_embedding)} vs {self.embedding_dim}")
            return []
        
        # Normalize query embedding for cosine similarity
        if self.similarity_metric == 'cosine':
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        results = []
        
        if self.backend == 'chromadb':
            # Search using ChromaDB
            try:
                chroma_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k
                )
                
                if chroma_results['distances'] and chroma_results['metadatas']:
                    for i, (distance, metadata) in enumerate(zip(chroma_results['distances'][0], chroma_results['metadatas'][0])):
                        if self.similarity_metric == 'cosine':
                            similarity = 1.0 - distance  # ChromaDB returns distance, convert to similarity
                        else:
                            similarity = 1.0 / (1.0 + distance)
                        
                        if similarity >= threshold:
                            results.append({
                                'person_id': metadata['person_id'],
                                'similarity': similarity,
                                'distance': distance,
                                'metadata': metadata,
                                'index': i
                            })
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}")
                return []
                
        else:
            # Use sklearn for search
            if self.nn_model is None or len(self.embeddings) == 0:
                return []
            
            # Calculate similarities
            embeddings_array = np.array(self.embeddings)
            
            if self.similarity_metric == 'cosine':
                similarities = cosine_similarity([query_embedding], embeddings_array)[0]
                # Sort by similarity (descending)
                sorted_indices = np.argsort(similarities)[::-1]
            else:
                distances = euclidean_distances([query_embedding], embeddings_array)[0]
                similarities = 1.0 / (1.0 + distances)
                # Sort by similarity (descending)
                sorted_indices = np.argsort(similarities)[::-1]
            
            # Get top k results above threshold
            for idx in sorted_indices[:k]:
                similarity = similarities[idx]
                if similarity >= threshold:
                    distance = 1.0 - similarity if self.similarity_metric == 'cosine' else euclidean_distances([query_embedding], [embeddings_array[idx]])[0][0]
                    
                    results.append({
                        'person_id': self.person_ids[idx],
                        'similarity': similarity,
                        'distance': distance,
                        'metadata': self.metadata[idx].copy(),
                        'index': idx
                    })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def get_best_match(self, query_embedding: np.ndarray, 
                      threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """
        Get the best matching person for a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            threshold: Similarity threshold
            
        Returns:
            Best match result or None
        """
        results = self.search_similar(query_embedding, k=1, threshold=threshold)
        return results[0] if results else None
    
    def get_person_embeddings(self, person_id: int) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get all embeddings for a specific person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            List of (embedding, metadata) tuples
        """
        if person_id not in self.id_to_index:
            return []
        
        embeddings_metadata = []
        for idx in self.id_to_index[person_id]:
            embedding = self.embeddings[idx]
            metadata = self.metadata[idx]
            embeddings_metadata.append((embedding, metadata))
        
        return embeddings_metadata
    
    def update_person_metadata(self, person_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for all embeddings of a specific person.
        
        Args:
            person_id: Person identifier
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            True if updated successfully
        """
        if person_id not in self.id_to_index:
            logger.error(f"Person {person_id} not found")
            return False
        
        try:
            # Update metadata in memory
            for idx in self.id_to_index[person_id]:
                self.metadata[idx].update(metadata_updates)
                # Add timestamp for the update
                self.metadata[idx]['metadata_updated'] = datetime.now().isoformat()
            
            # Rebuild ChromaDB to apply changes
            if self.backend == 'chromadb':
                self._rebuild_index()
            
            # Save to persist changes
            self.save_database()
            
            logger.info(f"Updated metadata for person {person_id}: {metadata_updates}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update person metadata: {e}")
            return False
    
    def remove_person(self, person_id: int) -> bool:
        """
        Remove all embeddings for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            True if person was removed, False if not found
        """
        if person_id not in self.id_to_index:
            return False
        
        # Get indices to remove (in reverse order to maintain index validity)
        indices_to_remove = sorted(self.id_to_index[person_id], reverse=True)
        
        # Remove from lists
        for idx in indices_to_remove:
            del self.embeddings[idx]
            del self.metadata[idx]
            del self.person_ids[idx]
        
        # Update id_to_index mapping
        del self.id_to_index[person_id]
        
        # Update other indices
        for pid in self.id_to_index:
            updated_indices = []
            for idx in self.id_to_index[pid]:
                # Count how many removed indices were before this one
                offset = sum(1 for removed_idx in indices_to_remove if removed_idx < idx)
                updated_indices.append(idx - offset)
            self.id_to_index[pid] = updated_indices
        
        # Rebuild index
        self._rebuild_index()
        
        logger.info(f"Removed person {person_id}")
        return True
    
    def _rebuild_index(self):
        """Rebuild index from current embeddings."""
        if not self.embeddings:
            if self.embedding_dim:
                self._initialize_index(self.embedding_dim)
            return
        
        if self.backend == 'chromadb':
            # Clear and rebuild ChromaDB collection
            try:
                self.chroma_client.delete_collection("face_embeddings")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="face_embeddings",
                    metadata={"hnsw:space": "cosine" if self.similarity_metric == "cosine" else "l2"}
                )
                
                # Re-add all embeddings
                if self.embeddings:
                    embeddings_list = [emb.tolist() for emb in self.embeddings]
                    ids_list = [f"person_{pid}_{i}" for i, pid in enumerate(self.person_ids)]
                    # Clean metadata for ChromaDB
                    clean_metadatas = []
                    for meta in self.metadata:
                        clean_meta = {}
                        for key, value in meta.items():
                            if value is not None:
                                if isinstance(value, (str, int, float, bool)):
                                    clean_meta[key] = value
                                else:
                                    clean_meta[key] = str(value)
                        clean_metadatas.append(clean_meta)
                    
                    self.collection.add(
                        embeddings=embeddings_list,
                        metadatas=clean_metadatas,
                        ids=ids_list
                    )
                    
                logger.info("ChromaDB collection rebuilt")
            except Exception as e:
                logger.error(f"Failed to rebuild ChromaDB collection: {e}")
        else:
            # Rebuild sklearn model
            if self.embeddings:
                self.nn_model.fit(np.array(self.embeddings))
                logger.info("sklearn NearestNeighbors model rebuilt")
    
    
    def save_database(self, filepath: Optional[str] = None) -> bool:
        """
        Save vector database to file.
        
        Args:
            filepath: Custom file path (optional)
            
        Returns:
            True if saved successfully
        """
        if filepath is None:
            filepath = self.database_file
        
        try:
            # Prepare data for saving
            data = {
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'person_ids': self.person_ids,
                'id_to_index': self.id_to_index,
                'next_person_id': self.next_person_id,
                'embedding_dim': self.embedding_dim,
                'config': self.vector_config,
                'save_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save to pickle file
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Database saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
    
    def load_database(self, filepath: Optional[str] = None) -> bool:
        """
        Load vector database from file.
        
        Args:
            filepath: Custom file path (optional)
            
        Returns:
            True if loaded successfully
        """
        if filepath is None:
            filepath = self.database_file
        
        if not os.path.exists(filepath):
            logger.info("No existing database found, starting fresh")
            return False
        
        try:
            # Load from pickle file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Restore data
            self.embeddings = data.get('embeddings', [])
            self.metadata = data.get('metadata', [])
            self.person_ids = data.get('person_ids', [])
            self.id_to_index = data.get('id_to_index', {})
            self.next_person_id = data.get('next_person_id', 1)
            self.embedding_dim = data.get('embedding_dim')
            
            # Rebuild index for the current backend
            if self.embeddings:
                self._rebuild_index()
            
            logger.info(f"Database loaded from {filepath}")
            logger.info(f"Loaded {len(self.embeddings)} embeddings for {len(self.id_to_index)} people")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_embeddings': len(self.embeddings),
            'total_people': len(self.id_to_index),
            'embedding_dimension': self.embedding_dim,
            'backend': self.backend,
            'similarity_metric': self.similarity_metric,
            'index_type': self.index_type,
            'next_person_id': self.next_person_id
        }
        
        if self.embeddings:
            # Calculate per-person statistics
            embeddings_per_person = [len(indices) for indices in self.id_to_index.values()]
            stats.update({
                'avg_embeddings_per_person': np.mean(embeddings_per_person),
                'max_embeddings_per_person': max(embeddings_per_person),
                'min_embeddings_per_person': min(embeddings_per_person)
            })
        
        return stats
    
    def cleanup_old_embeddings(self, max_per_person: int = 10):
        """
        Remove old embeddings to limit storage per person.
        
        Args:
            max_per_person: Maximum embeddings to keep per person
        """
        removed_count = 0
        
        for person_id in list(self.id_to_index.keys()):
            indices = self.id_to_index[person_id]
            
            if len(indices) > max_per_person:
                # Sort by timestamp (keep newest)
                indexed_metadata = [(idx, self.metadata[idx]) for idx in indices]
                indexed_metadata.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
                
                # Keep only the newest embeddings
                to_keep = [idx for idx, _ in indexed_metadata[:max_per_person]]
                to_remove = [idx for idx, _ in indexed_metadata[max_per_person:]]
                
                # Remove old embeddings
                for idx in sorted(to_remove, reverse=True):
                    del self.embeddings[idx]
                    del self.metadata[idx]
                    del self.person_ids[idx]
                    removed_count += 1
                
                # Update mapping
                self.id_to_index[person_id] = to_keep
        
        if removed_count > 0:
            # Update all indices after removal
            self._rebuild_index()
            logger.info(f"Cleaned up {removed_count} old embeddings")
        
        return removed_count