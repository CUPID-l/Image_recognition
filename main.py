#!/usr/bin/env python3
"""
Face Recognition System - Main Entry Point

Run this file to start the face recognition system.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main

if __name__ == '__main__':
    sys.exit(main())