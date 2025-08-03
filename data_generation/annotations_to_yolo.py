# Converts our custom per-frame keypoint JSON annotations into a COCO-style keypoint dataset.

import os
import json
from tqdm import tqdm

# === CONFIG ===
IMAGE_DIR = "/home/student/project/output/coco_data/images"
KEYPOINT_JSON_DIR = "/home/student/project/output/keypoints"
COCO_OUTPUT_PATH = "/home/student/project/output/coco_data/annotations.json"

# === TOOL CONFIGURATION ===
CATEGORY_CONFIG = {
    "needle_holder": {
        "id": 1,
        "keypoints": [
            "tip_right", "tip_left", "shaft_right", "shaft_left", "ring_right", "ring_left"
        ]
    },
    "tweezer": {
        "id": 2,
        "keypoints": [
            "tip_right", "tip_left", "stem_right", "stem_left", "base"
        ]
    }
}

# 🔒 FIXED keypoint layout for YOLOv8 Pose — must match kpt_shape: [9, 3]
ALL_KEYPOINTS = [
    "tip_right", "tip_left",
    "shaft_right", "shaft_left",
    "ring_right", "ring_left",
    "base"
]

# === Create COCO dict ===
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories
for cat_name, cat_data in CATEGORY_CONFIG.items():
    coco_output["categories"].append({
        "id": cat_data["id"],
        "name": cat_name,
        "supercategory": "tool",
        "keypoints": ALL_KEYPOINTS,
        "skeleton": []
    })

# === Process each JSON file ===
image_id = 0
annotation_id = 1

for file in tqdm(sorted(os.listdir(KEYPOINT_JSON_DIR))):
    if not file.endswith(".json"):
        continue

    name = os.path.splitext(file)[0]
    image_file = f"{name}.png"
    image_path = os.path.join(IMAGE_DIR, image_file)
    if not os.path.exists(image_path):
        continue

    # Add image entry
    coco_output["images"].append({
        "id": image_id,
        "file_name": image_file,
        "width": 960,
        "height": 544
    })

    # Parse keypoints JSON
    with open(os.path.join(KEYPOINT_JSON_DIR, file), "r") as f:
        tools = json.load(f)

    for tool in tools:
        name = tool["name"].lower()
        tool_type = "needle_holder" if name.startswith("nh") else "tweezer"
        cat_id = CATEGORY_CONFIG[tool_type]["id"]
        layout = CATEGORY_CONFIG[tool_type]["keypoints"]

        keypoints_flat = []
        visible_count = 0
        keypoint_dict = tool["keypoints"]

        for kpt in ALL_KEYPOINTS:
            if kpt in keypoint_dict:
                x, y, v = keypoint_dict[kpt]
                keypoints_flat.extend([x, y, v])
                if v > 0:
                    visible_count += 1
            else:
                keypoints_flat.extend([0, 0, 0])

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cat_id,
            "keypoints": keypoints_flat,
            "num_keypoints": visible_count,
            "bbox": [0, 0, 0, 0],  # No bbox in our case
            "iscrowd": 0,
            "area": 1
        })
        annotation_id += 1

    image_id += 1

# === Save final JSON ===
os.makedirs(os.path.dirname(COCO_OUTPUT_PATH), exist_ok=True)
with open(COCO_OUTPUT_PATH, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"✅ COCO annotation file written to: {COCO_OUTPUT_PATH}")
