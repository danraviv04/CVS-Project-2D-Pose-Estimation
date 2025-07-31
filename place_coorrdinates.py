import os
import json
import cv2
import matplotlib.pyplot as plt

# === CONFIG ===
image_dir = "/home/student/project/output/coco_data/images/"
json_dir = "/home/student/project/output/keypoints/"
annotated_dir = "/home/student/project/output/annotated/"
os.makedirs(annotated_dir, exist_ok=True)

# === Color map by part name ===
COLOR_MAP = {
    "tip_left": (255, 0, 0),      # Red
    "tip_right": (200, 0, 0),     # Darker red
    "stem_left": (0, 255, 0),     # Green
    "stem_right": (0, 200, 0),    # Darker green
    "base": (0, 0, 255),          # Blue
    "shaft_left": (255, 255, 0),  # Yellow
    "shaft_right": (200, 200, 0), # Dark yellow
    "ring_left": (0, 255, 255),   # Cyan
    "ring_right": (0, 200, 200),  # Dark cyan
}

# === Process all image/JSON pairs ===
for filename in os.listdir(image_dir):
    if not filename.endswith(".png"):
        continue

    name = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)
    json_path = os.path.join(json_dir, f"{name}.json")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(json_path, "r") as f:
        annotations = json.load(f)

    for ann in annotations:
        kp_dict = ann["keypoints"]
        for part_name, (x, y, v) in kp_dict.items():
            if v > 0:
                color = COLOR_MAP.get(part_name, (255, 255, 255))  # default white
                cv2.circle(image_rgb, (x, y), radius=6, color=color, thickness=-1)
                cv2.putText(image_rgb, part_name, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Save visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(f"Annotated: {name}")
    out_path = os.path.join(annotated_dir, f"annotated_{name}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✅ Saved: {out_path}")