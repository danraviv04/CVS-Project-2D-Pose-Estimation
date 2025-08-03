# import argparse
# import subprocess
# import os
# import random
# import shutil
# import json
# from pathlib import Path

# def generate_data(args):
#     print("\U0001F6E0️ Generating transparent tool renderings...")
#     subprocess.run([
#         "blenderproc", "run", "data_generation/generate_tools.py",
#         "--obj_dir", "/datashare/project/surgical_tools_models",
#         "--camera_params", "/datashare/project/camera.json",
#         "--output_dir", args.output_dir,
#         "--num_images", str(args.num_images)
#     ], check=True)

#     print("\U0001F5BC️ Pasting tools onto random backgrounds...")
#     images_dir = os.path.join(args.output_dir, "coco_data", "images")
#     subprocess.run([
#         "python3", "data_generation/paste_on_random_background.py",
#         "-i", images_dir,
#         "-b", args.backgrounds_dir,
#         "-o", os.path.join(args.output_dir, "composited")
#     ], check=True)

# def split_coco_annotations(coco_path, images_dir, train_ratio=0.8):
#     with open(coco_path / "annotations.json") as f:
#         coco = json.load(f)

#     all_images = coco["images"]
#     random.shuffle(all_images)
#     split_idx = int(len(all_images) * train_ratio)

#     subsets = {
#         "train": all_images[:split_idx],
#         "val": all_images[split_idx:]
#     }

#     for subset, images in subsets.items():
#         image_ids = {img["id"] for img in images}
#         annots = [ann for ann in coco["annotations"] if ann["image_id"] in image_ids]

#         subset_dir = images_dir / subset
#         subset_dir.mkdir(parents=True, exist_ok=True)

#         for img in images:
#             src = images_dir / img["file_name"]
#             dst = subset_dir / img["file_name"]
#             if src.exists():
#                 shutil.move(src, dst)
#             img["file_name"] = f"{subset}/{img['file_name']}"

#         out = {
#             "images": images,
#             "annotations": annots,
#             "categories": coco["categories"]
#         }

#         with open(coco_path / f"annotations_{subset}.json", "w") as f:
#             json.dump(out, f, indent=2)

#     print("✅ COCO annotations split into train/val")

# def write_data_yaml(coco_path):
#     data_yaml = coco_path / "data.yaml"
#     with open(coco_path / "annotations_train.json") as f:
#         sample = json.load(f)
#         keypoints = sample["categories"][0]["keypoints"]
#         classnames = [cat["name"] for cat in sample["categories"]]

#     with open(data_yaml, "w") as f:
#         f.write(f"path: {str(coco_path)}\n")
#         f.write("train: ../composited/train\n")
#         f.write("val: ../composited/val\n")
#         f.write("\n")
#         f.write(f"nc: {len(classnames)}\n")
#         f.write("names:\n")
#         for i, name in enumerate(classnames):
#             short = "NH" if "needle" in name else "T"
#             f.write(f"  {i}: {short}\n")
#         f.write("\n")
#         f.write(f"kpt_shape: [{len(keypoints)}, 3]\n")
#         f.write("keypoints:\n")
#         for kp in keypoints:
#             f.write(f"  - {kp}\n")

#     print(f"✅ Wrote data.yaml at: {data_yaml}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_images", type=int, default=100)
#     parser.add_argument("--output_dir", type=str, default="/home/student/project/output")
#     parser.add_argument("--backgrounds_dir", type=str, default="/datashare/project/train2017")
#     parser.add_argument("--debug", action="store_true", help="Enable debug mode to create annotated images")
#     args = parser.parse_args()

#     generate_data(args)

#     subprocess.run([
#         "python3", "data_generation/annotations_to_coco.py",
#         "--output_dir", os.path.join(args.output_dir, "coco_data"),
#     ], check=True)

#     coco_path = Path(args.output_dir) / "coco_data"
#     images_dir = Path(args.output_dir) / "composited"
#     split_coco_annotations(coco_path, images_dir)
#     write_data_yaml(coco_path)

#     meaningless_json = "/home/student/project/output/coco_data/coco_annotations.json"

#     if os.path.exists(meaningless_json):
#         os.remove(meaningless_json)
#         print(f"✅ Removed: {meaningless_json}")
#     else:
#         print(f"⚠️ File not found: {meaningless_json}")


#     if args.debug:
#         print("Debug mode enabled, creating annotated pictures...")
#         subprocess.run([
#             "python3", "data_generation/place_coorrdinates.py",
#             "--image_dir", str(images_dir / "train"),
#             "--json_dir", str(coco_path / "keypoints"),
#             "--annotated_dir", str(coco_path / "annotated")
#         ], check=True)
#         print("✅ Data generation complete!")
#     else:
#         print("✅ Data generation complete! Run with --debug to create annotated images.")

# if __name__ == "__main__":
#     main()

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

def write_data_yaml(coco_path, output_dir, expected_kpts):
    data_yaml = Path(output_dir) / "yolo_data" / "data.yaml"
    with open(coco_path / "annotations_train.json") as f:
        sample = json.load(f)
        classnames = [cat["name"] for cat in sample["categories"]]
        keypoints = sample["categories"][0]["keypoints"]

    images_dir = Path(output_dir) / "yolo_data" / "images"
    with open(data_yaml, "w") as f:
        f.write(f"train: {images_dir}/train\n")
        f.write(f"val: {images_dir}/val\n\n")
        f.write(f"nc: {len(classnames)}\n")
        f.write("names:\n")
        for i, name in enumerate(classnames):
            short = "NH" if "needle" in name.lower() else "T"
            f.write(f"  {i}: {short}\n")
        f.write(f"\nkpt_shape: [{expected_kpts}, 3]\n")
        f.write("keypoints:\n")
        for i in range(expected_kpts):
            f.write(f"  - kp{i}\n")  # Generic names to match the fixed keypoint count

    print(f"✅ Wrote data.yaml at: {data_yaml}")

def convert_coco_to_yolo_keypoints(json_path: Path, split_dir: Path, expected_kpts: int = 10):
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
                    y = kps[i+1] / img_height
                    v = int(kps[i+2])
                    keypoints.append((x, y, v))

                # pad or trim
                if len(keypoints) > expected_kpts:
                    keypoints = keypoints[:expected_kpts]
                elif len(keypoints) < expected_kpts:
                    keypoints += [(0.0, 0.0, 0)] * (expected_kpts - len(keypoints))

                # skip if all keypoints are invisible
                visible = [kp for kp in keypoints if kp[2] > 0]
                if not visible:
                    continue

                cx = sum(kp[0] for kp in visible) / len(visible)
                cy = sum(kp[1] for kp in visible) / len(visible)

                flat_kpts = " ".join(f"{x:.6f} {y:.6f} {v}" for (x, y, v) in keypoints)
                # Add dummy width/height = 0.1
                line = f"{ann['category_id'] - 1} {cx:.6f} {cy:.6f} 0.100000 0.100000 {flat_kpts}"
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
        "python3", "data_generation/annotations_to_yolo.py",
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
            "python3", "data_generation/place_coorrdinates.py",
            "--image_dir", str(yolo_img_dir / "train"),
            "--json_dir", str(coco_path / "keypoints"),
            "--annotated_dir", str(coco_path / "annotated")
        ], check=True)

    print("✅ Data generation complete!")

if __name__ == "__main__":
    main()