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
    print("Generating transparent tool renderings…")
    subprocess.run([
        "blenderproc", "run", "data_generation/generate_tools.py",
        "--obj_dir", "/datashare/project/surgical_tools_models",
        "--camera_params", "/datashare/project/camera.json",
        "--output_dir", args.output_dir,
        "--num_images", str(args.num_images),
    ], check=True)

    print("Pasting tools onto random backgrounds…")
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
                print(f"Missing composited: {src.name} (skipping move)")
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
            print(f"Label has no matching image in split dir: {img_path.name}")

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

    print(f"YOLO-Seg labels: wrote {written}, skipped {skipped} (no polys) for {json_path.name}")

# -----------------------------
# data.yaml (YOLO-Seg)
# -----------------------------
def write_seg_data_yaml(train_json: Path, yolo_dir: Path):
    """
    Writes a data.yaml file for YOLO-Seg, deriving class names/order from the training JSON.
    Returns (cat_map, names) where cat_map is COCO category id → YOLO id and names is list of class names.
    """
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
    """
    Create symlinks for all files from src_dir to dst_dir (creates dst_dir if needed).
    """
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
    """
    Copy all files from src_dir to dst_dir (creates dst_dir if needed).
    """
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
            print("Debug: drawing polygon overlays…")
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
            print("Skipping polygon viz (viewer script not found).")
    
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