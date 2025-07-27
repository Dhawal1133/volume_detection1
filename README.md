name: AI-Powered Depth Estimation via Flask API
description: |
  Flask-based API for real-time depth estimation using MiDaS DPT Hybrid model.
  Captures frames from an IP camera, computes depth maps, and determines
  whether a container/truck is FULL or NOT FULL using depth statistics.
features:
  - Real-time depth estimation via Intel ISL MiDaS DPT Hybrid
  - REST API endpoint `/trigger` to capture and analyze frames
  - ROI (Region of Interest) based focused analysis
  - Comparison logic with FULL reference frame
  - Depth visualization with color-mapped ROI overlays
tech_stack:
  backend: Python 3, Flask
  ai_model: MiDaS DPT Hybrid (PyTorch Hub)
  libraries: [OpenCV, NumPy, TorchVision, Torch]
  input: IP Webcam stream
project_structure: |
  volume_detection/
    ├── app.py                # Flask server with depth estimation logic
    ├── requirements.txt      # Python dependencies
    ├── .gitignore            # Ignore venv, cache, large files
    └── output/               # Saved depth visualizations
setup:
  steps:
    - Clone repository:
      command: git clone https://github.com/<your-username>/<repo-name>.git
    - Enter directory:
      command: cd <repo-name>
    - Create virtual environment:
      command: python -m venv venv
    - Activate environment:
      linux_mac: source venv/bin/activate
      windows: venv\Scripts\activate
    - Install dependencies:
      command: pip install -r requirements.txt
    - Run Flask app:
      command: python app.py
api:
  endpoint: /trigger
  method: POST
  behavior: |
    - First call stores FULL reference depth stats
    - Subsequent calls compare mean depth with FULL reference
    - Returns decision: "FULL" or "NOT FULL"
  example_response: |
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
configuration:
  ip_camera_url: Set `VIDEO_STREAM_URL` in app.py
  roi_ratio: Adjustable in analyze_depth() (default 1.0)
best_practices:
  - Ignore venv and large model weights in .gitignore
  - Use `pip freeze > requirements.txt` to share environment
  - Consider Git LFS for `.pt` or `.h5` model files
future_improvements:
  - Multi-ROI support
  - Frontend dashboard visualization
  - Cloud deployment with GPU acceleration
license: MIT
author:
  name: Dhawal Phalak
  bio: Final Year Electronics & Telecommunication (Data Science & ML)
