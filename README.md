# AI-Powered Depth Estimation via Flask API

This project implements a **Flask-based API** for real-time **depth estimation** using the [MiDaS](https://github.com/isl-org/MiDaS) model. It captures frames from an IP camera stream, computes a depth map, and analyzes the scene to determine whether a container/truck is **FULL** or **NOT FULL** based on depth statistics.

---

## Features

- Real-time depth estimation using Intel ISL’s MiDaS DPT Hybrid model
- Flask REST API endpoint (`/trigger`) for triggering depth capture and analysis
- ROI (Region of Interest) processing for focused analysis
- First trigger sets FULL reference; subsequent triggers compare mean depth
- Saves color-mapped depth visualization with ROI overlay

---

## Technology Stack

- **Backend**: Python 3, Flask  
- **AI Model**: MiDaS DPT Hybrid (PyTorch Hub)  
- **Libraries**: OpenCV, NumPy, TorchVision  
- **Camera Input**: IP Webcam stream  

---

## Project Structure

volume_detection/
│
├── app.py # Flask server with depth estimation logic
├── requirements.txt # Python dependencies
├── .gitignore # Ignore venv, cache, large files
└── output/ # Saved depth visualizations


---

## Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
# source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run Flask app
python app.py
```

## API Usage
Endpoint: /trigger
Method: POST

First call → Captures and stores FULL reference.

Subsequent calls → Compares new frame against FULL reference and returns decision.
Example
{
  "frame": "test",
  "mean": 0.234,
  "median": 0.220,
  "min": 0.001,
  "max": 0.650,
  "std": 0.050,
  "depth_map_image": "output/test_depth_roi.png",
  "decision": "⚠️ NOT full",
  "mean_diff": 0.015
}


 ## Configuration
IP Camera URL: Change VIDEO_STREAM_URL in app.py

ROI Ratio: Adjustable in analyze_depth() (default = 1.0)

## Future Improvements
Multi-ROI support

Frontend dashboard for visualization

Cloud deployment with GPU acceleration
