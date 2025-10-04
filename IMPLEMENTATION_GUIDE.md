# Face Recognition System - Auto-Enrollment & Manual Labeling

## Problem Solved

The system now properly supports:
1. **Automatic face enrollment** - Unknown faces are automatically enrolled with ID-based names
2. **Face recognition** - Auto-enrolled faces are recognized in subsequent frames
3. **Manual labeling** - Users can rename auto-enrolled faces with proper names

## Changes Made

### Core Functionality

#### 1. Vector Store Metadata Update (`src/vector_store.py`)
- Added `update_person_metadata()` method
- Properly updates metadata for all embeddings of a person
- Rebuilds ChromaDB index to persist changes
- Handles both ChromaDB and sklearn backends

#### 2. Face Recognizer (`src/recognizer.py`)
- Fixed `_enroll_new_person()` to auto-generate "Person_X" names when no name is provided
- Simplified `update_person_name()` to use vector store's update method
- Ensures proper name storage in metadata for auto-enrolled faces

#### 3. User Interface (`src/main.py`)
- Added 'n' command in interactive mode to rename people
- Implemented `_rename_person()` method with user-friendly prompts
- Added helpful command reference at bottom of video display

### Documentation
- Updated README.md with new workflow and command
- Updated demo.py with usage instructions
- Added complete workflow example

### Testing
- Added unit tests for VectorStore (`tests/test_vector_store.py`)
- Added unit tests for FaceRecognizer (`tests/test_recognizer.py`)
- Created validation script (`validate_fix.py`)

## Usage

### Running the System

#### Basic Mode (with camera)
```bash
python main.py
```

#### Interactive Mode (recommended)
```bash
python main.py --interactive
```

### Workflow

1. **Start the system in interactive mode:**
   ```bash
   python main.py --interactive
   ```

2. **The system will automatically enroll new faces:**
   - When a new face is detected, it's enrolled as "Person_1", "Person_2", etc.
   - You'll see "NEW" in yellow on the video display
   - The counter shows "New: X" for new enrollments

3. **When faces are recognized:**
   - Previously enrolled faces are recognized automatically
   - You'll see "RECOGNIZED" in green on the video display
   - The person's current name (e.g., "Person_1") is displayed

4. **List enrolled people:**
   ```
   > l
   Enrolled people (3):
     ID 1: Person_1 (5 embeddings)
     ID 2: Person_2 (3 embeddings)
     ID 3: Person_3 (2 embeddings)
   ```

5. **Rename people with proper names:**
   ```
   > n
   
   Enrolled people:
     ID 1: Person_1 (5 embeddings)
     ID 2: Person_2 (3 embeddings)
     ID 3: Person_3 (2 embeddings)
   
   Enter Person ID to rename: 1
   Enter new name: John Doe
   Successfully renamed Person 1 to 'John Doe'
   ```

6. **The system now recognizes them by their proper names:**
   - Future recognitions will show "John Doe" instead of "Person_1"
   - All embeddings for that person are updated with the new name

### Interactive Commands

- `q` - Quit application
- `s` - Save system state
- `r` - Reset statistics
- `p` - Print recognition statistics
- `l` - List all enrolled people
- `e` - Enroll person manually (with known name)
- `n` - Rename/label a person by ID

## Technical Details

### Auto-Enrollment Process

1. Face is detected by the detector
2. Embedding is generated
3. Vector store searches for similar embeddings
4. If no match found (and auto_enroll=true):
   - New person ID is assigned
   - Embedding is stored with metadata
   - Name is set to "Person_{ID}" automatically
   - Person is marked as "new enrollment"

### Recognition Process

1. Face is detected
2. Embedding is generated
3. Vector store searches for similar embeddings
4. If match found:
   - Person ID is retrieved
   - Name is retrieved from metadata
   - Additional embedding is added for continuous learning
   - Person is marked as "recognized"

### Renaming Process

1. User lists all enrolled people (command 'l')
2. User selects person ID to rename (command 'n')
3. System updates all embeddings' metadata for that person
4. ChromaDB index is rebuilt to persist changes
5. Database is saved to disk
6. Future recognitions use the new name

## Configuration

Edit `config/config.yaml` to customize:

```yaml
recognition:
  similarity_threshold: 0.7    # Threshold for matching faces
  auto_enroll: true            # Enable automatic enrollment
  confidence_threshold: 0.8    # Minimum confidence for detection
```

## Validation

Run the validation script to test the workflow:

```bash
python validate_fix.py
```

This will:
1. Create a test environment
2. Simulate auto-enrollment of 3 people
3. Verify auto-generated names
4. Rename all people
5. Verify renamed names
6. Test persistence across restarts
7. Clean up test data

## Benefits

✅ **No manual intervention required** - System auto-enrolls faces automatically
✅ **Recognize returning faces** - Previously seen faces are recognized immediately
✅ **Easy labeling** - Simple command to assign proper names later
✅ **Persistent storage** - All data is saved automatically
✅ **Continuous learning** - System adds more embeddings each time a person is seen
✅ **User-friendly interface** - Clear visual feedback and simple commands

## Notes

- Auto-enrolled faces are stored with "Person_X" names in the metadata
- The system can recognize these faces before they are renamed
- Renaming updates all embeddings for that person
- Changes are persisted to both ChromaDB and pickle file
- The system saves automatically at regular intervals
