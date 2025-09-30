"""
Face Recognition System

A real-time face recognition system that detects faces in images,
generates vector embeddings for unique identification, and continuously
recognizes previously detected individuals.
"""

__version__ = "1.0.0"
__author__ = "Face Recognition System Team"

from .face_detector import FaceDetector
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .recognizer import FaceRecognizer

__all__ = [
    "FaceDetector",
    "EmbeddingGenerator", 
    "VectorStore",
    "FaceRecognizer"
]