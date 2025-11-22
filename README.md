# 🚨 Intrusion Detection System

A real-time intrusion detection system using YOLO object detection and DeepSORT tracking to monitor restricted zones in video surveillance.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![YOLO](https://img.shields.io/badge/yolo-v8-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Features

- ✅ **Real-time Person Detection** - Powered by YOLOv8
- ✅ **Multi-Object Tracking** - DeepSORT algorithm for persistent IDs
- ✅ **Interactive Zone Drawing** - Define restricted areas with mouse clicks
- ✅ **Alarm System** - Visual alerts when intrusion detected
- ✅ **3-Second Cooldown** - Smart alarm management
- ✅ **FPS Display** - Real-time performance monitoring
- ✅ **Zone Persistence** - Zones saved to JSON and auto-loaded



## 🏗️ Architecture
```
FIRST_PROJECT/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration management
│   ├── detector/
│   │   ├── __init__.py
│   │   └── yolo_detector.py       # YOLO-based person detection
│   ├── tracker/
│   │   ├── __init__.py
│   │   └── deepsort_tracker.py    # DeepSORT tracking implementation
│   ├── zones/
│   │   ├── __init__.py
│   │   └── zone_manager.py        # Zone drawing and management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geometry.py            # Geometric utilities
│   │   ├── timer.py               # Timer for alarm cooldown
│   │   └── drawing.py             # Visualization utilities
│   └── pipeline.py                # Main processing pipeline
├── main.py                         # Entry point
├── test.mp4                        # Test video file
├── restricted_zones.json           # Zone configuration (auto-generated)
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/nurmuhammad1160/intrusion-detection-yolo.git
cd intrusion-detection-yolo
```

### Step 2: Create Virtual Environment
```bash
# Linux/Mac
python3 -m venv env
source env/bin/activate

# Windows
python -m venv env
env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The first run will automatically download YOLOv8 model weights (~11MB).

## 📖 Usage

### Quick Start
```bash
python main.py
```

### First-Time Setup: Drawing Zones

When you run the system for the first time, you'll be prompted to draw restricted zones:

#### Zone Drawing Instructions:

1. **Left Click** - Add a point to the zone polygon
2. **Right Click** - Remove the last point
3. Press **'c'** - Complete and save the current zone
4. Press **'r'** - Reset current zone (start over)
5. Press **'q'** - Quit zone drawing mode

**Requirements:**
- Minimum 3 points to create a valid zone
- Zones are automatically saved to `restricted_zones.json`

#### Example Zone Drawing:
```
1. Click on first corner of restricted area
2. Click on second corner
3. Click on third corner
4. Click on fourth corner
5. Press 'c' to save
6. Press 'q' to exit drawing mode
```

### Using Existing Zones

If `restricted_zones.json` exists, you'll see:
```
Options:
  1. Use existing zones
  2. Redraw zones

Enter choice (1/2):
```

- Choose **1** to use existing zones
- Choose **2** to redraw from scratch

### Runtime Controls

While the system is running:

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `p` | Pause/Resume video processing |
| `r` | Restart video from beginning |

### Understanding the Display

**Color Codes:**
- 🟢 **Green Box** - Person detected, not in restricted zone
- 🔴 **Red Box** - Person in restricted zone (INTRUSION!)
- 🔴 **Red Polygon** - Restricted zone boundary
- ⚪ **White Circle** - Tracking point (person's feet position)

**Statistics Panel (Top Right):**
- `FPS` - Current processing speed
- `Persons` - Total tracked individuals
- `Intruders` - Number of people in restricted zones
- `Zones` - Number of configured zones

**Alarm Banner:**
- Appears when intrusion detected
- Shows "ПРОНИКНОВЕНИЕ!" (INTRUSION!)
- Displays intruder track IDs
- Remains active for 3 seconds after person leaves zone

## ⚙️ Configuration

Edit `src/core/config.py` to customize system behavior:

### Detection Settings
```python
YOLO_MODEL = "yolov8s.pt"        # Model: yolov8n, yolov8s, yolov8m
CONFIDENCE_THRESHOLD = 0.35      # Detection confidence (0.0-1.0)
PROCESS_EVERY_N_FRAMES = 2       # Process every N frames (speed vs accuracy)
```

### Tracking Settings
```python
MAX_AGE = 40                     # Frames to keep track without detection
MIN_HITS = 2                     # Detections needed to confirm track
IOU_THRESHOLD = 0.25             # Matching threshold
```

### Alarm Settings
```python
ALARM_COOLDOWN_SECONDS = 3       # Seconds before alarm deactivates
```

## 🔧 Troubleshooting

### Issue: Video not opening

**Solution:** Ensure `test.mp4` is in the project root directory.
```bash
ls -la test.mp4  # Should show the file
```

### Issue: Slow performance

**Solutions:**

1. **Use lighter model:**
```python
   YOLO_MODEL = "yolov8n.pt"  # Fastest
```

2. **Process fewer frames:**
```python
   PROCESS_EVERY_N_FRAMES = 3  # Skip more frames
```

3. **Lower resolution:**
```python
   PROCESS_WIDTH = 480  # Smaller resolution
```

### Issue: Poor detection accuracy

**Solutions:**

1. **Use larger model:**
```python
   YOLO_MODEL = "yolov8m.pt"  # More accurate
```

2. **Lower confidence threshold:**
```python
   CONFIDENCE_THRESHOLD = 0.25  # More detections
```

3. **Process every frame:**
```python
   PROCESS_EVERY_N_FRAMES = 1  # No skipping
```

### Issue: False alarms

**Solution:** Adjust minimum tracking hits:
```python
MIN_HITS = 3  # Require more confirmations
```

## 🎯 Technical Details

### YOLO Detection

- **Model:** YOLOv8 (nano/small/medium variants)
- **Classes:** Person detection only (COCO class 0)
- **Input Size:** 640x640 pixels
- **Confidence Filtering:** Configurable threshold
- **Aspect Ratio Filtering:** Removes non-person shapes

### DeepSORT Tracking

- **Algorithm:** Custom implementation
- **Matching:** IoU-based association
- **Features:** 
  - Persistent track IDs
  - Occlusion handling
  - Track lifecycle management

### Zone Detection

- **Algorithm:** Point-in-polygon (ray casting)
- **Input:** Polygon vertices (minimum 3 points)
- **Storage:** JSON format with coordinates
- **Visualization:** Semi-transparent overlay

### Alarm Logic
```
1. Person enters zone → Alarm ACTIVE
2. Person stays in zone → Alarm REMAINS ACTIVE
3. Person exits zone → Cooldown timer STARTS (3s)
4. Timer expires → Alarm DEACTIVATES
5. Person re-enters → Alarm RE-ACTIVATES immediately
```

## 📊 Performance

**Tested on:**
- CPU: Intel Core i5 (4 cores)
- RAM: 8GB
- Video: 1920x1080, 30 FPS

**Results:**
- YOLOv8n: ~25 FPS (real-time capable)
- YOLOv8s: ~15 FPS (good balance)
- YOLOv8m: ~8 FPS (high accuracy)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

