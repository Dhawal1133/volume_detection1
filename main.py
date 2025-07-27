from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import time
import os
from torchvision.transforms import Compose

# Load DPT_Hybrid Model
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Depth Estimation Function
def estimate_depth_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    with torch.inference_mode():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    return prediction, frame

# ROI & Analysis Utilities
def get_roi(depth_map, roi_ratio=1.0):
    h, w = depth_map.shape
    if roi_ratio >= 1.0:
        return depth_map, (0, h, 0, w)
    dh = int(h * (1 - roi_ratio) / 2)
    dw = int(w * (1 - roi_ratio) / 2)
    return depth_map[dh:h-dh, dw:w-dw], (dh, h-dh, dw, w-dw)

def analyze_depth(depth_map, roi_ratio=1.0):
    roi, (dh, hdh, dw, wdw) = get_roi(depth_map, roi_ratio)
    stats = {
        "mean": float(np.mean(roi)),
        "median": float(np.median(roi)),
        "min": float(np.min(roi)),
        "max": float(np.max(roi)),
        "std": float(np.std(roi)),
        "roi_coords": (dh, hdh, dw, wdw)
    }
    return stats, roi

def save_depth_visualization(depth_map, img, roi_coords, out_path):
    vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    vis = (vis * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
    dh, hdh, dw, wdw = roi_coords
    cv2.rectangle(vis_color, (dw, dh), (wdw-1, hdh-1), (0,255,0), 2)
    cv2.imwrite(out_path, vis_color)

# Flask App
app = Flask(__name__)
VIDEO_STREAM_URL = "http://192.168.1.62/stream"  # ✅ correct


frame_count = 0
full_stats = None

@app.route('/trigger', methods=['POST'])
def trigger():
    global frame_count, full_stats
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Could not open video stream"}), 500

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None or frame.size == 0:
        return jsonify({"status": "error", "message": "Invalid frame"}), 500

    frame_count += 1
    os.makedirs("output", exist_ok=True)

    depth_map, img = estimate_depth_from_frame(frame)
    stats, roi = analyze_depth(depth_map, roi_ratio=1.0)
    label = "full" if frame_count == 1 else "test"
    save_path = f"output/{label}_depth_roi.png"
    save_depth_visualization(depth_map, img, stats["roi_coords"], save_path)

    result = {
        "frame": label,
        "mean": stats['mean'],
        "median": stats['median'],
        "min": stats['min'],
        "max": stats['max'],
        "std": stats['std'],
        "depth_map_image": save_path
    }

    if label == "full":
        full_stats = stats
        result["decision"] = "Stored as FULL reference"
    else:
        if full_stats is None:
            result["decision"] = "FULL reference not set. Press trigger again."
        else:
            full_mean = full_stats["mean"]
            test_mean = stats["mean"]
            if test_mean > full_mean * 1.05:
                result["decision"] = "⚠️ NOT full"
            else:
                result["decision"] = "✅ FULL"
            result["mean_diff"] = test_mean - full_mean
            
    print("\n=== AI Trigger Result ===")
    print(f"Frame: {result['frame']}")
    print(f"Mean Depth: {result['mean']:.3f}")
    print(f"Decision: {result['decision']}")
    if 'mean_diff' in result:
        print(f"Mean Depth Difference: {result['mean_diff']:.3f}")
    print("=========================\n")

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
