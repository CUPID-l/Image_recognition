# Face Recognition System

A real-time face recognition system that detects faces in images, generates vector embeddings for unique identification, and continuously recognizes previously detected individuals.

## Features

- **Real-time Face Detection**: Uses MTCNN or Haar Cascades for accurate face detection
- **Face Embedding Generation**: Converts faces to vector representations using FaceNet or face_recognition library
- **Vector Database**: Efficient storage and retrieval using FAISS with similarity search
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

2. Install dependencies:
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

## System Architecture

### Core Components

1. **Face Detection Module** (`src/face_detector.py`)
   - MTCNN and Haar Cascade support
   - Face preprocessing and alignment
   - Quality validation

2. **Embedding Generator** (`src/embedding_generator.py`)
   - FaceNet and face_recognition models
   - Vector normalization
   - Embedding validation

3. **Vector Store** (`src/vector_store.py`)
   - FAISS-based similarity search
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
  method: "mtcnn"  # Options: mtcnn, haar
  min_confidence: 0.9
  min_face_size: 80

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
- **GPU**: NVIDIA GTX 1660+ for GPU acceleration (optional)
- **Storage**: SSD recommended for better I/O performance

### Optimization Tips

1. **GPU Acceleration**: Install `faiss-gpu` and set `use_gpu: true` in config
2. **Frame Skipping**: Increase `frame_skip` for better performance
3. **Model Selection**: Use `face_recognition` for faster processing, `facenet` for accuracy
4. **Resolution**: Lower camera resolution for better FPS

## Directory Structure

```
face_recognition_system/
├── src/                    # Source code
│   ├── face_detector.py    # Face detection logic
│   ├── embedding_generator.py # Embedding generation
│   ├── vector_store.py     # Vector database
│   ├── recognizer.py       # Recognition logic
│   └── main.py            # Main application
├── data/                   # Data storage
│   ├── embeddings/        # Face embeddings
│   └── faces/             # Face images
├── config/                 # Configuration
│   └── config.yaml        # System settings
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── main.py                # Entry point
```

## Testing

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
4. **Import errors**: Ensure all dependencies are installed

### Logging

Check `face_recognition.log` for detailed error messages and performance metrics.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MTCNN for face detection
- FaceNet for face embeddings
- FAISS for efficient similarity search
- OpenCV for computer vision operations