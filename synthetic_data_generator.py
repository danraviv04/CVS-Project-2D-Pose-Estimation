import argparse
import subprocess
import os
import random
import shutil
import json
from pathlib import Path
from collections import defaultdict

def generate_data(args):
    print("\U0001F6E0️ Generating transparent tool renderings...")
    subprocess.run([
        "blenderproc", "run", "data_generation/generate_tools.py",
        "--obj_dir", "/datashare/project/surgical_tools_models",
        "--camera_params", "/datashare/project/camera.json",
        "--output_dir", args.output_dir,
        "--num_images", str(args.num_images)
    ], check=True)

    print("\U0001F5BC️ Pasting tools onto random backgrounds...")
    images_dir = os.path.join(args.output_dir, "coco_data", "images")
    subprocess.run([
        "python3", "data_generation/paste_on_random_background.py",
        "-i", images_dir,
        "-b", args.backgrounds_dir,
        "-o", os.path.join(args.output_dir, "composited")
    ], check=True)

def split_coco_annotations(coco_path, images_dir, train_ratio=0.8):
    with open(coco_path / "annotations.json") as f:
        coco = json.load(f)

    all_images = coco["images"]
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)

    subsets = {
        "train": all_images[:split_idx],
        "val": all_images[split_idx:]
    }

    for subset, images in subsets.items():
        image_ids = {img["id"] for img in images}
        annots = [ann for ann in coco["annotations"] if ann["image_id"] in image_ids]

        subset_dir = images_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            src = images_dir / img["file_name"]
            dst = subset_dir / img["file_name"]
            if src.exists():
                shutil.move(src, dst)
            img["file_name"] = f"{subset}/{img['file_name']}"

        out = {
            "images": images,
            "annotations": annots,
            "categories": coco["categories"]
        }

        with open(coco_path / f"annotations_{subset}.json", "w") as f:
            json.dump(out, f, indent=2)

    print("✅ COCO annotations split into train/val")

def write_data_yaml(
    coco_path,
    output_dir,
    expected_kpts=7,
    keypoints=None,
    add_lr_links=True
):
    """
    Create a YOLO pose data.yaml from a COCO-style annotations json.

    - Infers class names from categories, preserving category id ordering when available.
    - Writes flip_idx by pairing *_left <-> *_right automatically.
    - Builds a skeleton anchored at 'base' (index resolved from keypoints).
    - Optionally adds left-right lines ([0,1], [2,3], [4,5]) for nicer viz.

    Args:
        coco_path (Path|str): folder that contains annotations_train.json
        output_dir (Path|str): root output folder that contains yolo_data/images/{train,val}
        expected_kpts (int): expected number of keypoints
        keypoints (list[str] or None): override keypoint names/order; if None, use default layout.
        add_lr_links (bool): if True, add lines between left/right pairs in the skeleton
    """
    from pathlib import Path
    import json

    coco_path = Path(coco_path)
    output_dir = Path(output_dir)
    yolo_dir = output_dir / "yolo_data"
    images_dir = yolo_dir / "images"
    data_yaml = yolo_dir / "data.yaml"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # --- classes (preserve COCO category order; sort by id if present) ---
    ann_path = coco_path / "annotations_train.json"
    with ann_path.open("r") as f:
        sample = json.load(f)

    cats = sample.get("categories", [])
    if not cats:
        raise ValueError(f"No categories found in {ann_path}")

    if all("id" in c for c in cats):
        cats = sorted(cats, key=lambda c: c["id"])

    # names as short labels "NH"/"T" to match your setup
    classnames = []
    for c in cats:
        cname = c.get("name", "")
        short = "NH" if "needle" in cname.lower() or "nh" in cname.lower() else "T"
        classnames.append(short)

    # --- keypoints order (default to your 7) ---
    if keypoints is None:
        keypoints = [
            "tip_right",
            "tip_left",
            "shaft_right",
            "shaft_left",
            "ring_right",
            "ring_left",
            "base"
        ]

    if len(keypoints) != expected_kpts:
        raise ValueError(
            f"Expected {expected_kpts} keypoints, but got {len(keypoints)}: {keypoints}"
        )

    # --- flip_idx (auto from *_left/_right naming) ---
    name_to_idx = {k: i for i, k in enumerate(keypoints)}
    flip_idx = []
    lr_pairs = []  # will also use for optional skeleton links

    for i, k in enumerate(keypoints):
        if k.endswith("_right"):
            twin = k[:-6] + "_left"
            j = name_to_idx.get(twin, i)
            flip_idx.append(j)
            if j != i and (j, i) not in lr_pairs and (i, j) not in lr_pairs:
                lr_pairs.append((i, j))
        elif k.endswith("_left"):
            twin = k[:-5] + "_right"
            j = name_to_idx.get(twin, i)
            flip_idx.append(j)
            if j != i and (j, i) not in lr_pairs and (i, j) not in lr_pairs:
                lr_pairs.append((j, i) if j < i else (i, j))
        else:
            flip_idx.append(i)

    # --- skeleton (anchor at 'base' if present) ---
    try:
        base_idx = name_to_idx["base"]
    except KeyError:
        # fallback: center-ish anchor
        base_idx = len(keypoints) // 2

    # edges from base to all others except itself
    skeleton = [[base_idx, i] for i in range(len(keypoints)) if i != base_idx]

    # optional left-right bars for viz (only for pairs we actually found)
    if add_lr_links:
        for (a, b) in lr_pairs:
            if [a, b] not in skeleton and [b, a] not in skeleton:
                skeleton.append([a, b])

    # --- write YAML ---
    with data_yaml.open("w") as f:
        f.write(f"train: {images_dir}/train\n")
        f.write(f"val: {images_dir}/val\n\n")
        f.write(f"nc: {len(classnames)}\n")
        f.write("names:\n")
        for i, name in enumerate(classnames):
            f.write(f"  {i}: {name}\n")

        f.write(f"\nkpt_shape: [{len(keypoints)}, 3]\n")
        f.write("flip_idx: [")
        f.write(", ".join(str(i) for i in flip_idx))
        f.write("]\n")

        f.write("keypoints:\n")
        for kp in keypoints:
            f.write(f"  - {kp}\n")

        f.write("skeleton:\n")
        for a, b in skeleton:
            f.write(f"  - [{a}, {b}]\n")

    print(f"✅ Wrote data.yaml at: {data_yaml}")
    
def convert_coco_to_yolo_keypoints(json_path: Path, split_dir: Path, expected_kpts: int = 7):
    import json
    from collections import defaultdict

    with open(json_path) as f:
        data = json.load(f)

    image_info = {img['id']: img for img in data['images']}
    image_annotations = defaultdict(list)
    for ann in data['annotations']:
        image_annotations[ann['image_id']].append(ann)

    for image_id, annotations in image_annotations.items():
        img = image_info[image_id]
        img_width, img_height = img['width'], img['height']
        file_name = Path(img['file_name']).stem + ".txt"
        out_path = split_dir / file_name

        with open(out_path, 'w') as f_out:
            for ann in annotations:
                kps = ann['keypoints']
                keypoints = []

                for i in range(0, len(kps), 3):
                    x = kps[i] / img_width
                    y = kps[i + 1] / img_height
                    v = int(kps[i + 2])
                    keypoints.append((x, y, v))

                # pad or trim
                if len(keypoints) > expected_kpts:
                    keypoints = keypoints[:expected_kpts]
                elif len(keypoints) < expected_kpts:
                    keypoints += [(0.0, 0.0, 0)] * (expected_kpts - len(keypoints))

                # skip if all keypoints invisible
                visible = [kp for kp in keypoints if kp[2] > 0]
                if not visible:
                    continue

                # Step 1: compute raw keypoint bounds (even for occluded ones)
                cx, cy, w, h = ann["bbox"]

                # write YOLO format
                flat_kpts = " ".join(f"{x:.6f} {y:.6f} {v}" for (x, y, v) in keypoints)
                line = f"{ann['category_id'] - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {flat_kpts}"
                f_out.write(line + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="/home/student/project/output")
    parser.add_argument("--backgrounds_dir", type=str, default="/datashare/project/train2017")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keypoints", type=int, default=7)
    args = parser.parse_args()

    generate_data(args)

    subprocess.run([
        "python3", "data_generation/annotations_to_coco.py",
        "--output_dir", os.path.join(args.output_dir, "coco_data")
    ], check=True)

    coco_path = Path(args.output_dir) / "coco_data"
    composited_dir = Path(args.output_dir) / "composited"
    split_coco_annotations(coco_path, composited_dir)

    yolo_img_dir = Path(args.output_dir) / "yolo_data" / "images"
    yolo_lbl_dir = Path(args.output_dir) / "yolo_data" / "labels"
    shutil.copytree(composited_dir / "train", yolo_img_dir / "train", dirs_exist_ok=True)
    shutil.copytree(composited_dir / "val", yolo_img_dir / "val", dirs_exist_ok=True)
    (yolo_lbl_dir / "train").mkdir(parents=True, exist_ok=True)
    (yolo_lbl_dir / "val").mkdir(parents=True, exist_ok=True)

    convert_coco_to_yolo_keypoints(coco_path / "annotations_train.json", yolo_lbl_dir / "train", args.keypoints)
    convert_coco_to_yolo_keypoints(coco_path / "annotations_val.json", yolo_lbl_dir / "val", args.keypoints)

    write_data_yaml(coco_path, args.output_dir, args.keypoints)

    if args.debug:
        print("🔍 Debug mode enabled: generating annotated visualizations...")
        subprocess.run([
            "python3", "data_generation/place_coordinates.py",
            "--image_dir", str(yolo_img_dir / "train"),
            "--json_dir", str(coco_path / "keypoints"),
            "--annotated_dir", str(coco_path / "annotated")
        ], check=True)

    print("✅ Data generation complete!")

if __name__ == "__main__":
    main()