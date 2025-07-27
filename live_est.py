import torch
import cv2
import numpy as np
import time
import os
from torchvision.transforms import Compose

# ---- Load MiDaS DPT_Hybrid Model ----
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# ---- Utility Functions ----
def estimate_depth(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    with torch.inference_mode():
        start = time.time()
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
        end = time.time()

    inference_time = end - start
    return prediction, inference_time

def get_roi(depth_map, ratio=1.0):
    h, w = depth_map.shape
    if ratio >= 1.0:
        return depth_map, (0, h, 0, w)
    dh = int(h * (1 - ratio) / 2)
    dw = int(w * (1 - ratio) / 2)
    return depth_map[dh:h - dh, dw:w - dw], (dh, h - dh, dw, w - dw)

def analyze_depth(depth_map, ratio=1.0):
    roi, coords = get_roi(depth_map, ratio)
    return {
        "mean": float(np.mean(roi)),
        "median": float(np.median(roi)),
        "min": float(np.min(roi)),
        "max": float(np.max(roi)),
        "std": float(np.std(roi)),
        "roi_coords": coords,
        "roi_shape": roi.shape
    }

def save_depth_image(depth_map, coords, out_path):
    vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
    dh, hdh, dw, wdw = coords
    cv2.rectangle(vis, (dw, dh), (wdw - 1, hdh - 1), (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)

# ---- Settings ----
VIDEO_STREAM_URL = "http://192.168.1.62/stream"
ROI_RATIO = 1.0  # Change to e.g. 0.8 to crop edges
SAVE_RAW = False
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

print("📷 Press 'c' to capture (1st = reference, 2nd = test)")
print("❌ Press 'q' to quit")

cap = cv2.VideoCapture(VIDEO_STREAM_URL)
if not cap.isOpened():
    print("❌ Could not open stream. Check ESP32 IP.")
    exit()

depth_stats = {}
frame_count = 0

# ---- Main Loop ----
test_counter = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("⚠️  Frame grab failed. Retrying...")
        time.sleep(0.5)
        cap.release()
        cap = cv2.VideoCapture(VIDEO_STREAM_URL)
        continue

    cv2.imshow("Live ESP32-CAM Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    if key == ord('c'):
        frame_count += 1
        label = "full" if frame_count == 1 else "test"
        print(f"\n📸 Capturing frame {frame_count} ({label.upper()})...")

        depth_map, inference_time = estimate_depth(frame)
        stats = analyze_depth(depth_map, ROI_RATIO)
        depth_stats[label] = stats
        
        
        if frame_count == 1:
            save_depth = f"{OUT_DIR}/full_depth.png"
        else:
            test_counter += 1
            save_depth = f"{OUT_DIR}/test{test_counter}_depth.png"
        save_depth_image(depth_map, stats["roi_coords"], save_depth)

        if SAVE_RAW:
            cv2.imwrite(f"{OUT_DIR}/{label}_raw.jpg", frame)

        print(f"📈 Inference time: {inference_time:.2f}s")
        print(f"ROI Mean Depth:   {stats['mean']:.3f}")
        print(f"ROI Median:       {stats['median']:.3f}")
        print(f"ROI Min–Max:      {stats['min']:.3f} – {stats['max']:.3f}")
        print(f"ROI Std Dev:      {stats['std']:.3f}")
        print(f"ROI Shape:        {stats['roi_shape']}")

        if frame_count == 1:
            print("📌 Stored as REFERENCE (FULL).")
        else:
            print("\n🧠 Comparing with reference...")
            mean_diff = stats["mean"] - depth_stats["full"]["mean"]
            print(f"📏 Mean Difference: {mean_diff:.3f}")
            if mean_diff > depth_stats["full"]["mean"] * 0.05:
                print("⚠️  Container is NOT full (lower fill level detected).")
            else:
                print("✅ Container is FULL.")

    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
