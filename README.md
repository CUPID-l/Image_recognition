# Face Recognition System

A real-time face recognition system that detects faces in images, generates vector embeddings for unique identification, and continuously recognizes previously detected individuals.

**✨ Now requires only Python packages - no compilation needed!**

## Features

- **Real-time Face Detection**: Uses DeepFace or Haar Cascades for accurate face detection
- **Face Embedding Generation**: Converts faces to vector representations using DeepFace models
- **Vector Database**: Efficient storage and retrieval using ChromaDB with similarity search
- **Continuous Recognition**: Real-time identification of known individuals with auto-enrollment
- **Quality Checks**: Blur detection, lighting validation, and face quality assessment
- **Interactive Mode**: Manual enrollment, statistics, and system management

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CUPID-l/Image_recognition.git
cd Image_recognition
```

2. Install dependencies (Python-only, no compilation required):
```bash
pip install opencv-python numpy pillow deepface chromadb scikit-learn tf-keras pyyaml
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

3. Run the system:
```bash
python main.py
```

### Basic Usage

```bash
# Start with default camera
python main.py

# Use specific camera
python main.py --camera 1

# Process video file
python main.py --video path/to/video.mp4

# Interactive mode with commands
python main.py --interactive

# Custom configuration
python main.py --config path/to/config.yaml
```

## Python-Only Dependencies

This system now uses only Python packages that don't require compilation:

- **opencv-python**: Pre-compiled OpenCV for Python
- **numpy**: Numerical computing
- **pillow**: Image processing
- **deepface**: Face detection and recognition (replaces MTCNN, face_recognition, facenet-pytorch)
- **chromadb**: Vector database (replaces FAISS)
- **scikit-learn**: Machine learning utilities
- **tf-keras**: Deep learning backend for DeepFace

## System Architecture

### Core Components

1. **Face Detection Module** (`src/face_detector.py`)
   - DeepFace and Haar Cascade support
   - Face preprocessing and alignment
   - Quality validation

2. **Embedding Generator** (`src/embedding_generator.py`)
   - DeepFace models (Facenet, VGG-Face, ArcFace, etc.)
   - Vector normalization
   - Embedding validation

3. **Vector Store** (`src/vector_store.py`)
   - ChromaDB-based similarity search
   - Persistent storage
   - Efficient indexing

4. **Face Recognizer** (`src/recognizer.py`)
   - Complete recognition pipeline
   - Auto-enrollment of new faces
   - Person management

5. **Main Application** (`src/main.py`)
   - Real-time video processing
   - Interactive interface
   - Performance monitoring

## Configuration

Edit `config/config.yaml` to customize system behavior:

```yaml
# Face Detection Settings
face_detection:
  method: "deepface"  # Options: deepface, haar
  detector_backend: "opencv"  # DeepFace backend: opencv, ssd, dlib, mtcnn, retinaface
  min_confidence: 0.9
  min_face_size: 80

# Embedding Settings
embedding:
  model: "Facenet"  # DeepFace models: VGG-Face, Facenet, OpenFace, DeepFace, etc.
  detector_backend: "opencv"
  normalization: true

# Vector Database Settings
vector_store:
  backend: "chromadb"  # Options: chromadb, sklearn
  similarity_metric: "cosine"

# Recognition Settings
recognition:
  similarity_threshold: 0.7
  auto_enroll: true
  confidence_threshold: 0.8

# Video Processing
video:
  fps_limit: 30
  display_annotations: true
```

## Interactive Commands

When running in interactive mode (`--interactive`):

- `q` - Quit application
- `s` - Save system state
- `r` - Reset statistics
- `p` - Print recognition statistics
- `l` - List all enrolled people
- `e` - Enroll person manually
- `n` - Rename/label a person by ID (useful for renaming auto-enrolled faces)

## Auto-Enrollment & Labeling Workflow

The system now supports automatic enrollment with post-facto labeling:

1. **Auto-Enrollment**: When a new face is detected, it's automatically enrolled as "Person_1", "Person_2", etc.
2. **Recognition**: When that face appears again, the system recognizes it
3. **Manual Labeling**: Use the `l` command to list all enrolled people with their IDs, then use `n` command to rename them with proper names
4. **Persistence**: All changes are automatically saved to disk

Example workflow:
```bash
# Start in interactive mode
python main.py --interactive

# System detects and auto-enrolls faces
# Face 1 detected -> "Person_1" 
# Face 2 detected -> "Person_2"

# List all enrolled people
> l
Enrolled people (2):
  ID 1: Person_1 (3 embeddings)
  ID 2: Person_2 (2 embeddings)

# Rename Person_1 to "John Doe"
> n
Enter Person ID to rename: 1
Enter new name: John Doe
Successfully renamed Person 1 to 'John Doe'

# Now the system will recognize this person as "John Doe"
```

## API Usage

```python
from src.recognizer import FaceRecognizer
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize recognizer
recognizer = FaceRecognizer(config)

# Process frame
import cv2
frame = cv2.imread('image.jpg')
results = recognizer.recognize_faces(frame)

# Manual enrollment
result = recognizer.enroll_person_manually(frame, "John Doe")

# Get statistics
stats = recognizer.get_recognition_statistics()
```

## Performance

### Recommended Hardware

- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: SSD recommended for better I/O performance

### Optimization Tips

1. **Model Selection**: Use faster DeepFace models for better performance
2. **Frame Skipping**: Increase `frame_skip` for better performance
3. **Resolution**: Lower camera resolution for better FPS
4. **Backend Selection**: Use OpenCV detector backend for fastest detection

## Directory Structure

```
face_recognition_system/
├── src/                    # Source code
│   ├── face_detector.py    # Face detection logic
│   ├── embedding_generator.py # Embedding generation
│   ├── vector_store.py     # Vector database
│   ├── recognizer.py       # Recognition logic
│   └── main.py            # Main application
├── data/                   # Data storage (auto-created)
│   ├── embeddings/        # ChromaDB files
│   └── faces/             # Face images
├── config/                 # Configuration
│   └── config.yaml        # System settings
├── tests/                  # Unit tests
├── requirements.txt        # Python-only dependencies
└── main.py                # Entry point
```

## Testing

Run the demo to test functionality:

```bash
python demo.py
```

Run tests with pytest:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Troubleshooting

### Common Issues

1. **Camera not found**: Check camera permissions and device ID
2. **Poor recognition**: Adjust `similarity_threshold` in config
3. **Slow performance**: Reduce resolution or increase `frame_skip`
4. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### DeepFace Models

Available models in DeepFace:
- **Facenet**: Good balance of speed and accuracy
- **VGG-Face**: High accuracy, slower
- **ArcFace**: State-of-the-art accuracy
- **OpenFace**: Fast, lower accuracy
- **DeepFace**: Original Facebook model

### Logging

Check `face_recognition.log` for detailed error messages and performance metrics.

## Migration from Compilation Dependencies

This version replaces the following compilation-heavy dependencies with Python-only alternatives:

- ❌ `mtcnn` → ✅ `deepface` (detection)
- ❌ `face_recognition` → ✅ `deepface` (embeddings)  
- ❌ `facenet-pytorch` → ✅ `deepface` (embeddings)
- ❌ `faiss-cpu` → ✅ `chromadb` (vector database)
- ❌ `dlib` → ✅ Not needed (DeepFace handles everything)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- DeepFace for comprehensive face analysis
- ChromaDB for efficient vector storage
- OpenCV for computer vision operations