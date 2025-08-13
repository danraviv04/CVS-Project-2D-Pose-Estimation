import os
import cv2
import matplotlib.pyplot as plt

# === CONFIG ===
base_dir = "/home/student/project/output/yolo_data"
splits = ["train", "val"]
txt_dir = os.path.join(base_dir, "labels")
img_dir = os.path.join(base_dir, "images")
annot_dir = os.path.join(base_dir, "annotated")
os.makedirs(annot_dir, exist_ok=True)

# === Keypoint order from YAML ===
ALL_KEYPOINTS = [
    "tip_right", "tip_left",
    "shaft_right", "shaft_left",
    "ring_right", "ring_left",
    "base"
]

# === Color map by part name ===
COLOR_MAP = {
    "tip_left": (255, 0, 0),
    "tip_right": (200, 0, 0),
    "base": (0, 0, 255),
    "shaft_left": (255, 255, 0),
    "shaft_right": (200, 200, 0),
    "ring_left": (0, 255, 255),
    "ring_right": (0, 200, 200)
}

img_width = 960
img_height = 544

for split in splits:
    image_dir = os.path.join(img_dir, split)
    label_dir = os.path.join(txt_dir, split)
    out_dir = os.path.join(annot_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.endswith(".png"):
            continue

        name = os.path.splitext(filename)[0]
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, f"{name}.txt")
        if not os.path.exists(label_path):
            print(f"❌ Missing label: {label_path}")
            continue

        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse each object in .txt
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2 + len(ALL_KEYPOINTS) * 3:
                    print(f"⚠️ Skipping malformed line in {label_path}")
                    continue

                class_id = int(parts[0])
                cx = float(parts[1]) * img_width
                cy = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height

                # Compute bounding box corners
                x_min = int(cx - w / 2)
                y_min = int(cy - h / 2)
                x_max = int(cx + w / 2)
                y_max = int(cy + h / 2)

                # Draw bounding box
                cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)

                # Keypoints
                keypoint_entries = parts[5:]
                for i in range(len(ALL_KEYPOINTS)):
                    x = float(keypoint_entries[i * 3]) * img_width
                    y = float(keypoint_entries[i * 3 + 1]) * img_height
                    v = int(float(keypoint_entries[i * 3 + 2]))

                    kp_name = ALL_KEYPOINTS[i]
                    color = COLOR_MAP.get(kp_name, (255, 255, 255))

                    if v > 0:
                        cv2.circle(image_rgb, (int(x), int(y)), radius=6, color=color, thickness=-1)
                    else:
                        cv2.circle(image_rgb, (int(x), int(y)), radius=4, color=(120, 120, 120), thickness=-1)

                    cv2.putText(image_rgb, f"{kp_name} ({v})", (int(x + 5), int(y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Save output
        out_path = os.path.join(out_dir, f"annotated_{name}.png")
        plt.figure(figsize=(12, 6))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Annotated: {name}")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"✅ Saved: {out_path}")