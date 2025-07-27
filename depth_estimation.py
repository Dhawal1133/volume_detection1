import torch
import cv2
import numpy as np
import os
from torchvision.transforms import Compose

# --- Model Setup ---
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

def estimate_depth(img_path: str): 
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    with torch.inference_mode():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    return prediction, img

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
    # Draw ROI rectangle
    dh, hdh, dw, wdw = roi_coords
    cv2.rectangle(vis_color, (dw, dh), (wdw-1, hdh-1), (0,255,0), 2)
    cv2.imwrite(out_path, vis_color)

# --- Input Images ---
image_paths = {
    "full": r"D:/volume-detection-main/images/box_full.jpeg",
    "half": r"D:/volume-detection-main/images/box_half.jpeg"
}

os.makedirs("output", exist_ok=True)
depth_stats = {}

# --- Main Processing ---
for label, path in image_paths.items():
    depth_map, img = estimate_depth(path)
    stats, roi = analyze_depth(depth_map, roi_ratio=1.0)  # Use full image
    depth_stats[label] = stats

    # Save visualization
    save_depth_visualization(depth_map, img, stats["roi_coords"], f"output/{label}_depth_roi.png")

    print(f"\n--- {label.upper()} ---")
    print(f"ROI mean depth:   {stats['mean']:.3f}")
    print(f"ROI median depth: {stats['median']:.3f}")
    print(f"ROI min-max:      {stats['min']:.3f} - {stats['max']:.3f}")
    print(f"ROI std dev:      {stats['std']:.3f}")
    print(f"ROI shape: {roi.shape}")

# --- Decision Logic ---
full_mean = depth_stats["full"]["mean"]
half_mean = depth_stats["half"]["mean"]

print("\n=== RESULT ===")
if half_mean > full_mean * 1.05:
    print("⚠️  Container is NOT full (detected lower fill level).")
else:
    print("✅ Container appears FULL.")

# Optional: print the difference for debugging
print(f"\nMean depth difference: {half_mean - full_mean:.3f}")

