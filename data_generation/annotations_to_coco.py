import os
import json
from tqdm import tqdm

# === CONFIG ===
IMAGE_DIR = "/home/student/project/output/coco_data/images"
KEYPOINT_JSON_DIR = "/home/student/project/output/keypoints"
COCO_OUTPUT_PATH = "/home/student/project/output/coco_data/annotations.json"

# === FIXED YOLOv8 keypoint layout (MUST match your YAML!)
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
    "categories": [
        {
            "id": 1,
            "name": "needle_holder",  # ✅ required for write_data_yaml()
            "supercategory": "tool",
            "keypoints": ALL_KEYPOINTS,
            "skeleton": []
        },
        {
            "id": 2,
            "name": "tweezer",  # ✅ required for write_data_yaml()
            "supercategory": "tool",
            "keypoints": ALL_KEYPOINTS,
            "skeleton": []
        }
    ]
}

# === Process each JSON file ===
image_id = 0
annotation_id = 1

for file in tqdm(sorted(os.listdir(KEYPOINT_JSON_DIR))):
    if not file.endswith(".json"):
        print(f"⚠️ Skipping non-JSON file: {file}")
        continue

    name = os.path.splitext(file)[0]
    image_file = f"{name}.png"
    image_path = os.path.join(IMAGE_DIR, image_file)
    if not os.path.exists(image_path):
        print(f"⚠️ Image file not found: {image_file}, skipping...")
        continue

    # Add image entry
    coco_output["images"].append({
        "id": image_id,
        "file_name": image_file,
        "width": 960,
        "height": 544
    })

    # Load tool annotations
    with open(os.path.join(KEYPOINT_JSON_DIR, file), "r") as f:
        tools = json.load(f)

    for tool in tools:
        keypoint_dict = tool["keypoints"]
        tool_class = 1 if tool["name"].lower().startswith("nh") else 2
        if tool_class not in [1, 2]:
            print(f"⚠️ Unknown tool class {tool_class} in {file}, skipping...")
            continue  # skip unknown

        keypoints_flat = []
        visible_count = 0

        for kpt in ALL_KEYPOINTS:
            if kpt in keypoint_dict:
                x, y, v = keypoint_dict[kpt]
            else:
                x, y, v = 0, 0, 0
            keypoints_flat.extend([x, y, v])
            if v > 0:
                visible_count += 1

        # get bbox from corresponding JSON
        bbox = tool.get("bbox", [0, 0, 0, 0])
        area = bbox[2] * bbox[3]

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": tool_class,
            "keypoints": keypoints_flat,
            "num_keypoints": visible_count,
            "bbox": bbox,
            "iscrowd": 0,
            "area": bbox[2] * bbox[3]
        })

        annotation_id += 1

    image_id += 1

# === Save final JSON ===
os.makedirs(os.path.dirname(COCO_OUTPUT_PATH), exist_ok=True)
with open(COCO_OUTPUT_PATH, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"✅ COCO annotation file written to: {COCO_OUTPUT_PATH}")
