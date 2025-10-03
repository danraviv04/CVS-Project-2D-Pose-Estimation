# import argparse
# import subprocess
# import os
# import random
# import shutil
# import json
# from pathlib import Path
# from collections import defaultdict

# CAT_MAP = {
#     1: {"id": 1, "name": "needle_holder", "supercategory": "surgical_tool"},
#     2: {"id": 2, "name": "tweezers",      "supercategory": "surgical_tool"},
# }

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
#     coco_file = coco_path / "coco_annotations.json"
#     with coco_file.open("r") as f:
#         coco = json.load(f)

#     all_images = coco["images"][:]
#     random.shuffle(all_images)
#     split_idx = int(len(all_images) * train_ratio)
#     subsets = {"train": all_images[:split_idx], "val": all_images[split_idx:]}

#     # index annotations by image_id once (use it!)
#     anns_by_img = defaultdict(list)
#     for ann in coco["annotations"]:
#         anns_by_img[ann["image_id"]].append(ann)

#     # build/normalize categories once
#     if coco.get("categories"):
#         cats = sorted(coco["categories"], key=lambda c: c.get("id", 0))
#     else:
#         used_ids = sorted({a.get("category_id", 0)
#                            for a in coco["annotations"] if a.get("category_id", 0) != 0})
#         cats = [CAT_MAP.get(cid, {"id": cid, "name": f"class_{cid}", "supercategory": "object"})
#                 for cid in used_ids]

#     for subset, images in subsets.items():
#         image_ids = {img["id"] for img in images}
#         # fast collect via the index
#         annots = [a for iid in image_ids for a in anns_by_img.get(iid, [])]

#         subset_dir = images_dir / subset
#         subset_dir.mkdir(parents=True, exist_ok=True)

#         # move images into subset folder and update file_name
#         for img in images:
#             src = images_dir / Path(img["file_name"]).name  # original BP puts plain filename here
#             dst = subset_dir / Path(img["file_name"]).name
#             if src.exists():
#                 shutil.move(src, dst)
#             img["file_name"] = f"{subset}/{Path(img['file_name']).name}"

#         out = {
#             "images": images,
#             "annotations": annots,
#             "categories": cats
#         }
#         with open(coco_path / f"annotations_{subset}.json", "w") as f:
#             json.dump(out, f, indent=2)

#     print("✅ COCO annotations split into train/val with categories")

# def write_data_yaml(
#     coco_path,
#     output_dir,
#     expected_kpts=7,
#     keypoints=None,
#     add_lr_links=True
# ):
#     """
#     Create a YOLO pose data.yaml from a COCO-style annotations json.

#     - Infers class names from categories, preserving category id ordering when available.
#     - Writes flip_idx by pairing *_left <-> *_right automatically.
#     - Builds a skeleton anchored at 'base' (index resolved from keypoints).
#     - Optionally adds left-right lines ([0,1], [2,3], [4,5]) for nicer viz.

#     Args:
#         coco_path (Path|str): folder that contains annotations_train.json
#         output_dir (Path|str): root output folder that contains yolo_data/images/{train,val}
#         expected_kpts (int): expected number of keypoints
#         keypoints (list[str] or None): override keypoint names/order; if None, use default layout.
#         add_lr_links (bool): if True, add lines between left/right pairs in the skeleton
#     """
#     from pathlib import Path
#     import json

#     coco_path = Path(coco_path)
#     output_dir = Path(output_dir)
#     yolo_dir = output_dir / "yolo_data"
#     images_dir = yolo_dir / "images"
#     data_yaml = yolo_dir / "data.yaml"
#     yolo_dir.mkdir(parents=True, exist_ok=True)

#     # --- classes (preserve COCO category order; sort by id if present) ---
#     ann_path = coco_path / "annotations_train.json"
#     with ann_path.open("r") as f:
#         sample = json.load(f)

#     cats = sample.get("categories", [])
#     if not cats:
#         raise ValueError(f"No categories found in {ann_path}")

#     if all("id" in c for c in cats):
#         cats = sorted(cats, key=lambda c: c["id"])

#     # names as short labels "NH"/"T" to match your setup
#     classnames = []
#     for c in cats:
#         cname = c.get("name", "")
#         short = "NH" if "needle" in cname.lower() or "nh" in cname.lower() else "T"
#         classnames.append(short)

#     # --- keypoints order (default to your 7) ---
#     if keypoints is None:
#         keypoints = [
#             "tip_right",
#             "tip_left",
#             "shaft_right",
#             "shaft_left",
#             "ring_right",
#             "ring_left",
#             "base"
#         ]

#     if len(keypoints) != expected_kpts:
#         raise ValueError(
#             f"Expected {expected_kpts} keypoints, but got {len(keypoints)}: {keypoints}"
#         )

#     # --- flip_idx (auto from *_left/_right naming) ---
#     name_to_idx = {k: i for i, k in enumerate(keypoints)}
#     flip_idx = []
#     lr_pairs = []  # will also use for optional skeleton links

#     for i, k in enumerate(keypoints):
#         if k.endswith("_right"):
#             twin = k[:-6] + "_left"
#             j = name_to_idx.get(twin, i)
#             flip_idx.append(j)
#             if j != i and (j, i) not in lr_pairs and (i, j) not in lr_pairs:
#                 lr_pairs.append((i, j))
#         elif k.endswith("_left"):
#             twin = k[:-5] + "_right"
#             j = name_to_idx.get(twin, i)
#             flip_idx.append(j)
#             if j != i and (j, i) not in lr_pairs and (i, j) not in lr_pairs:
#                 lr_pairs.append((j, i) if j < i else (i, j))
#         else:
#             flip_idx.append(i)

#     # --- skeleton (anchor at 'base' if present) ---
#     try:
#         base_idx = name_to_idx["base"]
#     except KeyError:
#         # fallback: center-ish anchor
#         base_idx = len(keypoints) // 2

#     # edges from base to all others except itself
#     skeleton = [[base_idx, i] for i in range(len(keypoints)) if i != base_idx]

#     # optional left-right bars for viz (only for pairs we actually found)
#     if add_lr_links:
#         for (a, b) in lr_pairs:
#             if [a, b] not in skeleton and [b, a] not in skeleton:
#                 skeleton.append([a, b])

#     # --- write YAML ---
#     with data_yaml.open("w") as f:
#         f.write(f"train: {images_dir}/train\n")
#         f.write(f"val: {images_dir}/val\n\n")
#         f.write(f"nc: {len(classnames)}\n")
#         f.write("names:\n")
#         for i, name in enumerate(classnames):
#             f.write(f"  {i}: {name}\n")

#         f.write(f"\nkpt_shape: [{len(keypoints)}, 3]\n")
#         f.write("flip_idx: [")
#         f.write(", ".join(str(i) for i in flip_idx))
#         f.write("]\n")

#         f.write("keypoints:\n")
#         for kp in keypoints:
#             f.write(f"  - {kp}\n")

#         f.write("skeleton:\n")
#         for a, b in skeleton:
#             f.write(f"  - [{a}, {b}]\n")

#     print(f"✅ Wrote data.yaml at: {data_yaml}")
    
# def write_seg_data_yaml(coco_path, output_dir):
#     """
#     Create a YOLO-Seg data.yaml from a COCO-style annotations json.
#     """
#     from pathlib import Path
#     import json

#     coco_path = Path(coco_path)
#     output_dir = Path(output_dir)
#     yolo_dir = output_dir / "yolo_data_seg"
#     images_dir = yolo_dir / "images"
#     data_yaml = yolo_dir / "data.yaml"
#     yolo_dir.mkdir(parents=True, exist_ok=True)

#     # --- classes (preserve COCO category order; sort by id if present) ---
#     ann_path = coco_path / "annotations_train.json"
#     with ann_path.open("r") as f:
#         sample = json.load(f)

#     cats = sample.get("categories", [])
#     if not cats:
#         raise ValueError(f"No categories found in {ann_path}")

#     if all("id" in c for c in cats):
#         cats = sorted(cats, key=lambda c: c["id"])

#     names = [c.get("name", f"class_{i}") for i, c in enumerate(cats)]

#     # --- write YAML ---
#     with data_yaml.open("w") as f:
#         f.write(f"path: {output_dir.resolve()}\n")
#         f.write(f"train: {images_dir}/train\n")
#         f.write(f"val: {images_dir}/val\n\n")
#         f.write(f"nc: {len(names)}\n")
#         f.write("names:\n")
#         for i, name in enumerate(names):
#             f.write(f"  {i}: {name}\n")

#     print(f"✅ Wrote segmentation data.yaml at: {data_yaml}")
    
# def convert_coco_to_yolo_seg(json_path: Path, split_img_dir: Path, split_lbl_dir: Path, cat_to_yolo=None):
#     """
#     Convert a *split* COCO (annotations_{train,val}.json) to YOLO-Seg txt files.
#     Expects images already in split_img_dir; only writes labels into split_lbl_dir.
#     """
#     import json
#     from collections import defaultdict

#     if cat_to_yolo is None:
#         cat_to_yolo = {1: 0, 2: 1}  # 1=needle_holder, 2=tweezers

#     with json_path.open("r") as f:
#         data = json.load(f)

#     images = {im["id"]: im for im in data["images"]}
#     anns_by_img = defaultdict(list)
#     for ann in data["annotations"]:
#         if ann.get("iscrowd", 0) == 1: 
#             continue
#         if not ann.get("segmentation"):
#             continue
#         anns_by_img[ann["image_id"]].append(ann)

#     split_lbl_dir.mkdir(parents=True, exist_ok=True)

#     def _area(pl):
#         xs, ys = pl[0::2], pl[1::2]
#         s = 0.0
#         for i in range(len(xs)):
#             j = (i + 1) % len(xs)
#             s += xs[i]*ys[j] - xs[j]*ys[i]
#         return abs(s) * 0.5

#     for img_id, im in images.items():
#         W, H = im["width"], im["height"]
#         stem = Path(im["file_name"]).stem  # file_name might include 'train/' or 'val/'
#         out_txt = split_lbl_dir / f"{stem}.txt"

#         lines = []
#         for ann in anns_by_img.get(img_id, []):
#             cat = ann["category_id"]
#             if cat not in cat_to_yolo:
#                 continue
#             seg = ann.get("segmentation", [])
#             polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
#             if not polys:
#                 continue
#             poly = max(polys, key=_area)

#             xs = [max(0.0, min(1.0, x / W)) for x in poly[0::2]]
#             ys = [max(0.0, min(1.0, y / H)) for y in poly[1::2]]
#             coords = []
#             for x, y in zip(xs, ys):
#                 coords += [f"{x:.6f}", f"{y:.6f}"]
#             lines.append(" ".join([str(cat_to_yolo[cat])] + coords))

#         out_txt.write_text("\n".join(lines), encoding="utf-8")
    
# def convert_coco_to_yolo_keypoints(json_path: Path, split_dir: Path, expected_kpts: int = 7):
#     import json
#     from collections import defaultdict

#     with open(json_path) as f:
#         data = json.load(f)

#     image_info = {img['id']: img for img in data['images']}
#     image_annotations = defaultdict(list)
#     for ann in data['annotations']:
#         image_annotations[ann['image_id']].append(ann)

#     for image_id, annotations in image_annotations.items():
#         img = image_info[image_id]
#         img_width, img_height = img['width'], img['height']
#         file_name = Path(img['file_name']).stem + ".txt"
#         out_path = split_dir / file_name

#         with open(out_path, 'w') as f_out:
#             for ann in annotations:
#                 kps = ann['keypoints']
#                 keypoints = []

#                 for i in range(0, len(kps), 3):
#                     x = kps[i] / img_width
#                     y = kps[i + 1] / img_height
#                     v = int(kps[i + 2])
#                     keypoints.append((x, y, v))

#                 # pad or trim
#                 if len(keypoints) > expected_kpts:
#                     keypoints = keypoints[:expected_kpts]
#                 elif len(keypoints) < expected_kpts:
#                     keypoints += [(0.0, 0.0, 0)] * (expected_kpts - len(keypoints))

#                 # skip if all keypoints invisible
#                 visible = [kp for kp in keypoints if kp[2] > 0]
#                 if not visible:
#                     continue

#                 # Step 1: compute raw keypoint bounds (even for occluded ones)
#                 cx, cy, w, h = ann["bbox"]

#                 # write YOLO format
#                 flat_kpts = " ".join(f"{x:.6f} {y:.6f} {v}" for (x, y, v) in keypoints)
#                 line = f"{ann['category_id'] - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {flat_kpts}"
#                 f_out.write(line + "\n")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_images", type=int, default=100)
#     parser.add_argument("--output_dir", type=str, default="/home/student/project/output")
#     parser.add_argument("--backgrounds_dir", type=str, default="/datashare/project/train2017")
#     parser.add_argument("--train_ratio", type=float, default=0.8)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()

#     random.seed(args.seed)

#     # 1) Render with BlenderProc and composite on random backgrounds
#     generate_data(args)

#     # 2) Split COCO (writes annotations_{train,val}.json) and move images into {train,val}
#     coco_path = Path(args.output_dir) / "coco_data"
#     composited_dir = Path(args.output_dir) / "composited"
#     split_coco_annotations(coco_path, composited_dir, train_ratio=args.train_ratio)

#     # 3) Build YOLO-Seg folder structure (copy images)
#     yolo_seg_dir = Path(args.output_dir) / "yolo_data_seg"
#     yolo_seg_img_dir = yolo_seg_dir / "images"
#     yolo_seg_lbl_dir = yolo_seg_dir / "labels"
#     (yolo_seg_lbl_dir / "train").mkdir(parents=True, exist_ok=True)
#     (yolo_seg_lbl_dir / "val").mkdir(parents=True, exist_ok=True)

#     # copy composited images into YOLO-Seg images/
#     shutil.copytree(composited_dir / "train", yolo_seg_img_dir / "train", dirs_exist_ok=True)
#     shutil.copytree(composited_dir / "val",   yolo_seg_img_dir / "val",   dirs_exist_ok=True)

#     # 4) COCO → YOLO-Seg labels (auto map categories by id if cat_to_yolo=None)
#     convert_coco_to_yolo_seg(
#         coco_path / "annotations_train.json",
#         yolo_seg_img_dir / "train",
#         yolo_seg_lbl_dir / "train",
#         cat_to_yolo=None
#     )
#     convert_coco_to_yolo_seg(
#         coco_path / "annotations_val.json",
#         yolo_seg_img_dir / "val",
#         yolo_seg_lbl_dir / "val",
#         cat_to_yolo=None
#     )

#     # 5) Write YOLO-Seg data.yaml that points to yolo_data_seg/images/{train,val}
#     write_seg_data_yaml(coco_path, args.output_dir)

#     # 6) Optional sanity viz (only if you added a viewer; safely no-op if missing)
#     if args.debug:
#         vis_script = Path("data_generation/visualize_coco_polygons.py")
#         if vis_script.exists():
#             print("🔍 Debug: drawing polygon overlays…")
#             subprocess.run([
#                 "python3", str(vis_script),
#                 "--json", str(coco_path / "annotations_train.json"),
#                 "--images", str(yolo_seg_img_dir / "train"),
#                 "--out", str(yolo_seg_dir / "viz_train")
#             ], check=True)
#             subprocess.run([
#                 "python3", str(vis_script),
#                 "--json", str(coco_path / "annotations_val.json"),
#                 "--images", str(yolo_seg_img_dir / "val"),
#                 "--out", str(yolo_seg_dir / "viz_val")
#             ], check=True)
#         else:
#             print("ℹ️ Skipping polygon viz (viewer script not found).")

#     print("✅ Done:")
#     print("   - COCO:", coco_path.as_posix())
#     print("   - YOLO-Seg images:", (yolo_seg_img_dir).as_posix())
#     print("   - YOLO-Seg labels:", (yolo_seg_lbl_dir).as_posix())
#     print("   - data.yaml:", (yolo_seg_dir / "data.yaml").as_posix())

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import subprocess
import random
import shutil
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from skimage import measure

# -----------------------------
# BlenderProc + Compose step
# -----------------------------
def generate_data(args):
    print("🛠️  Generating transparent tool renderings…")
    subprocess.run([
        "blenderproc", "run", "data_generation/generate_tools.py",
        "--obj_dir", "/datashare/project/surgical_tools_models",
        "--camera_params", "/datashare/project/camera.json",
        "--output_dir", args.output_dir,
        "--num_images", str(args.num_images),
    ], check=True)

    print("🖼️  Pasting tools onto random backgrounds…")
    images_dir = Path(args.output_dir) / "coco_data" / "images"
    subprocess.run([
        "python3", "data_generation/paste_on_random_background.py",
        "-i", str(images_dir),
        "-b", args.backgrounds_dir,
        "-o", str(Path(args.output_dir) / "composited")
    ], check=True)

# -----------------------------
# COCO split (train/val)
# -----------------------------
def split_coco_annotations(coco_path: Path, images_dir: Path, train_ratio=0.8):
    coco_file = coco_path / "coco_annotations.json"
    with coco_file.open("r") as f:
        coco = json.load(f)

    all_images = coco["images"][:]
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    subsets = {"train": all_images[:split_idx], "val": all_images[split_idx:]}

    # index annotations by image_id once (reused below)
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    # normalize/ensure categories
    cats = coco.get("categories") or []
    if cats:
        cats = sorted(cats, key=lambda c: c.get("id", 0))
    else:
        used_ids = sorted({a.get("category_id", 0)
                           for a in coco.get("annotations", []) if a.get("category_id", 0) != 0})
        # fallback names if BlenderProc didn't write categories
        name_map = {1: "needle_holder", 2: "tweezers"}
        cats = [{"id": cid, "name": name_map.get(cid, f"class_{cid}"), "supercategory": "object"}
                for cid in used_ids]

    for subset, images in subsets.items():
        image_ids = {img["id"] for img in images}
        annots = [a for iid in image_ids for a in anns_by_img.get(iid, [])]

        subset_dir = images_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        # move composited images → {train,val} and update file_name to "train/<file>"
        for img in images:
            src = images_dir / Path(img["file_name"]).name
            dst = subset_dir / Path(img["file_name"]).name
            if src.exists():
                shutil.move(src, dst)
            else:
                print(f"⚠️ Missing composited: {src.name} (skipping move)")
            img["file_name"] = f"{subset}/{Path(img['file_name']).name}"

        out = {"images": images, "annotations": annots, "categories": cats}
        with (coco_path / f"annotations_{subset}.json").open("w") as f:
            json.dump(out, f, indent=2)

    print("✅ COCO annotations split into train/val with categories")

# -----------------------------
# Category mapping (COCO → YOLO ids)
# -----------------------------
def cat_map_from_json(json_path: Path):
    cats = json.loads(json_path.read_text())["categories"]
    cats = sorted(cats, key=lambda c: c["id"])
    # map COCO category id → contiguous YOLO id [0..nc-1] in the same order
    return {c["id"]: i for i, c in enumerate(cats)}, [c["name"] for c in cats]

# -----------------------------
# COCO → YOLO-Seg (txt labels)
# -----------------------------

from pycocotools import mask as maskUtils
import numpy as np
from skimage import measure

def _mask_to_polys(msk):
    msk = (msk.astype(np.uint8) > 0).astype(np.uint8)
    cs = measure.find_contours(msk, 0.5)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        c = np.flip(c, axis=1)          # (row,col) -> (x,y)
        out.append(c.ravel().tolist())
    return out

def polys_from_seg(seg, H=None, W=None):
    """
    Return list of flat XY polygons for any COCO segmentation:
    - polygon lists            -> passthrough
    - single RLE dict          -> decode
    - list of RLE dicts        -> merge then decode
    Works with compressed and uncompressed RLE.
    """
    if not seg:
        return []

    # A) polygon format already
    if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
        return [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]

    # B) list of RLE dicts
    if isinstance(seg, list) and seg and isinstance(seg[0], dict):
        if H is None or W is None:
            sz = seg[0].get("size")
            if not sz or len(sz) != 2:
                raise ValueError("polys_from_seg needs H,W for list-of-RLE")
            H, W = int(sz[0]), int(sz[1])
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)       # HxW or HxWxN
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return _mask_to_polys(m)

    # C) single RLE dict
    if isinstance(seg, dict) and "counts" in seg:
        rle = seg
        if not isinstance(rle["counts"], (bytes, bytearray, str)):
            # uncompressed RLE -> need H,W (or seg['size'])
            if H is None or W is None:
                sz = rle.get("size")
                if not sz or len(sz) != 2:
                    raise ValueError("polys_from_seg needs H,W for uncompressed RLE")
                H, W = int(sz[0]), int(sz[1])
            rle = maskUtils.frPyObjects([rle], H, W)
        m = maskUtils.decode(rle)        # HxW or HxWx1
        if m.ndim == 3:
            m = m[..., 0]
        return _mask_to_polys(m)

    return []

def _area(pl):
    xs, ys = pl[0::2], pl[1::2]
    s = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        s += xs[i]*ys[j] - xs[j]*ys[i]
    return abs(s) * 0.5

def convert_coco_to_yolo_seg(json_path: Path, yolo_split_img_dir: Path, yolo_split_lbl_dir: Path, cat_to_yolo: dict):
    """
    Writes YOLO-seg .txt labels next to image split folders.
    Ensures exactly one polygon per instance (keeps the largest).
    """
    data = json.loads(json_path.read_text())
    images = {im["id"]: im for im in data["images"]}

    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        if not ann.get("segmentation"):
            continue
        anns_by_img[ann["image_id"]].append(ann)

    yolo_split_lbl_dir.mkdir(parents=True, exist_ok=True)

    def _poly_area(flat):
        xs, ys = flat[0::2], flat[1::2]
        s = 0.0
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            s += xs[i] * ys[j] - xs[j] * ys[i]
        return abs(s) * 0.5

    written, skipped = 0, 0

    for img_id, im in images.items():
        W, H = im["width"], im["height"]
        stem = Path(im["file_name"]).stem
        out_txt = yolo_split_lbl_dir / f"{stem}.txt"

        # optional: warn if image missing in split img dir
        img_path = yolo_split_img_dir / Path(im["file_name"]).name
        if not img_path.exists():
            print(f"⚠️ Label has no matching image in split dir: {img_path.name}")

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            if cid not in cat_to_yolo:
                continue
            
            # COCO polygons can be multi-polygons (list of lists). Keep the largest polygon.
            seg = ann.get("segmentation", [])
            polys = polys_from_seg(seg, H, W)
            if not polys:
                continue
            poly = max(polys, key=_area)

            # Normalize to [0,1], clamp, then format
            xs = [max(0.0, min(1.0, x / W)) for x in poly[0::2]]
            ys = [max(0.0, min(1.0, y / H)) for y in poly[1::2]]
            coords = []
            for x, y in zip(xs, ys):
                coords += [f"{x:.6f}", f"{y:.6f}"]

            lines.append(" ".join([str(cat_to_yolo[cid])] + coords))

        if lines:
            out_txt.write_text("\n".join(lines), encoding="utf-8")
            written += 1
        else:
            skipped += 1

    print(f"📝 YOLO-Seg labels: wrote {written}, skipped {skipped} (no polys) for {json_path.name}")

# -----------------------------
# data.yaml (YOLO-Seg)
# -----------------------------
def write_seg_data_yaml(train_json: Path, yolo_dir: Path):
    cat_map, names = cat_map_from_json(train_json)
    images_dir = yolo_dir / "images"
    yaml_path = yolo_dir / "data.yaml"

    with yaml_path.open("w") as f:
        f.write(f"path: {yolo_dir.resolve()}\n")
        f.write(f"train: {images_dir}/train\n")
        f.write(f"val: {images_dir}/val\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")

    print(f"✅ Wrote segmentation data.yaml at: {yaml_path}")
    return cat_map, names

# -----------------------------
# Helpers: copying / symlinking
# -----------------------------
def mirror_with_symlinks(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.is_file():
            target = (dst_dir / p.name)
            if target.exists() or target.is_symlink():
                try:
                    target.unlink()
                except Exception:
                    pass
            target.symlink_to(p.resolve())

def mirror_with_copies(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_images", type=int, default=100)
    ap.add_argument("--output_dir", type=str, default="/home/student/project/output")
    ap.add_argument("--backgrounds_dir", type=str, default="/datashare/project/train2017")
    ap.add_argument("--train_ratio", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--link_images", action="store_true",
                    help="Use symlinks instead of copying images into YOLO structure")
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) Render with BlenderProc and composite on random backgrounds
    generate_data(args)

    # 2) Split COCO (writes annotations_{train,val}.json) and move composited images into {train,val}
    coco_path = Path(args.output_dir) / "coco_data"
    composited_dir = Path(args.output_dir) / "composited"
    split_coco_annotations(coco_path, composited_dir, train_ratio=args.train_ratio)

    # 3) Build YOLO-Seg folder structure
    yolo_seg_dir = Path(args.output_dir) / "yolo_data_seg"
    yolo_seg_img_dir = yolo_seg_dir / "images"
    yolo_seg_lbl_dir = yolo_seg_dir / "labels"
    (yolo_seg_lbl_dir / "train").mkdir(parents=True, exist_ok=True)
    (yolo_seg_lbl_dir / "val").mkdir(parents=True, exist_ok=True)

    # mirror images from composited/{train,val} to yolo_data_seg/images/{train,val}
    if args.link_images:
        mirror_with_symlinks(composited_dir / "train", yolo_seg_img_dir / "train")
        mirror_with_symlinks(composited_dir / "val",   yolo_seg_img_dir / "val")
    else:
        mirror_with_copies(composited_dir / "train", yolo_seg_img_dir / "train")
        mirror_with_copies(composited_dir / "val",   yolo_seg_img_dir / "val")

    # 4) Write data.yaml (derives class names/order from training JSON)
    train_json = coco_path / "annotations_train.json"
    val_json   = coco_path / "annotations_val.json"
    cat_map, names = write_seg_data_yaml(train_json, yolo_seg_dir)

    # 5) COCO → YOLO-Seg labels (use the SAME cat_map for train + val)
    convert_coco_to_yolo_seg(train_json, yolo_seg_img_dir / "train", yolo_seg_lbl_dir / "train", cat_to_yolo=cat_map)
    convert_coco_to_yolo_seg(val_json,   yolo_seg_img_dir / "val",   yolo_seg_lbl_dir / "val",   cat_to_yolo=cat_map)

    # 6) Optional polygon overlays (requires visualize_coco_polygons.py)
    if args.debug:
        vis = Path("data_generation/visualize_coco_polygons.py")
        if vis.exists():
            print("🔍 Debug: drawing polygon overlays…")
            subprocess.run([
                "python3", "data_generation/visualize_coco_polygons.py",
                "--coco", str(train_json),
                "--images_root", str(yolo_seg_img_dir / "train"),
                "--out", str(yolo_seg_dir / "viz_train"),
                "--draw_names",
                "--label", "short",
            ], check=True)

            subprocess.run([
                "python3", "data_generation/visualize_coco_polygons.py",
                "--coco", str(val_json),
                "--images_root", str(yolo_seg_img_dir / "val"),
                "--out", str(yolo_seg_dir / "viz_val"),
                "--draw_names",
                "--label", "short",
            ], check=True)
        else:
            print("ℹ️ Skipping polygon viz (viewer script not found).")
    
    subprocess.run([
                "python3", "data_generation/make_yolo_pose_from_keypoints.py",
                "--root", "/home/student/project/output",
                "--link_images"
            ], check=True)
            
    print("✅ Done:")
    print(f"   - COCO:           {coco_path.as_posix()}")
    print(f"   - YOLO-Seg images:{yolo_seg_img_dir.as_posix()}")
    print(f"   - YOLO-Seg labels:{(yolo_seg_lbl_dir).as_posix()}")
    print(f"   - data.yaml:      {(yolo_seg_dir / 'data.yaml').as_posix()}")

if __name__ == "__main__":
    main()