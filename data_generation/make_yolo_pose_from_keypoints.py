#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, shutil
from pathlib import Path

KP_ORDER = [
    "tip_right", "tip_left",
    "shaft_right", "shaft_left",
    "ring_right", "ring_left",
    "base",
]

COCO2YOLO = {1: 0, 2: 1}  # 1=needle_holder -> NH(0), 2=tweezers -> T(1)

def short_name(cat_or_inst: str) -> str:
    """
    Convert category or instance name to short form.
        cat_or_inst: str, category or instance name
    return: "NH" for Needle Holder, "T" for Tweezers
    """
    
    s = (cat_or_inst or "").lower()
    return "Needle Holder" if ("needle" in s or s.startswith("nh")) else "Tweezwers"

def write_yaml(yolo_root: Path):
    """
    Write a YOLO-Pose data.yaml file.
        yolo_root: Path, output directory for YOLO-Pose dataset
    """
    yaml = yolo_root / "data.yaml"
    (yolo_root / "images/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images/val").mkdir(parents=True, exist_ok=True)
    with yaml.open("w") as f:
        f.write(f"path: {yolo_root.resolve()}\n")
        f.write(f"train: {yolo_root / 'images/train'}\n")
        f.write(f"val: {yolo_root / 'images/val'}\n\n")
        f.write("nc: 2\nnames:\n  0: NH\n  1: T\n")
        f.write("\nkpt_shape: [7, 3]\n")
        f.write("flip_idx: [1, 0, 3, 2, 5, 4, 6]\n")  # swap L/R, keep base
        f.write("keypoints:\n")
        for k in KP_ORDER:
            f.write(f"  - {k}\n")
        # a simple skeleton (optional, nice for viz)
        f.write("skeleton:\n")
        for i in range(len(KP_ORDER)-1):
            f.write(f"  - [{i}, {i+1}]\n")

def make_split(split: str, root: Path, link: bool):
    """
    Create a YOLO-Pose dataset split (train or val).
        split: str, "train" or "val"
        root: Path, root directory containing coco_data/, composited/, keypoints/
        link: bool, whether to use symlinks for images instead of copies
    """
    coco = json.loads((root / "coco_data" / f"annotations_{split}.json").read_text())
    kproot = root / "keypoints"
    src_img_root = root / "composited" / split

    yolo_root = root / "yolo_data"
    img_out = yolo_root / "images" / split
    lbl_out = yolo_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = {im["id"]: im for im in coco["images"]}

    for im in images.values():
        # copy/symlink image
        src = src_img_root / Path(im["file_name"]).name
        dst = img_out / src.name
        if not src.exists():
            print(f"missing image: {src}")
            continue
        if link:
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

        # write label(s) from keypoints JSON
        stem = Path(im["file_name"]).stem
        kp_file = kproot / f"{stem}.json"
        if not kp_file.exists():
            # no keypoints for this frame – skip label
            (lbl_out / f"{stem}.txt").write_text("", encoding="utf-8")
            continue

        objs = json.loads(kp_file.read_text())
        lines = []
        W, H = im["width"], im["height"]

        for obj in objs:
            # class id → YOLO class
            cid = obj.get("class_id", 0)
            if cid not in COCO2YOLO:
                continue
            elif cid in (1, 2):
                ycls = cid - 1
            if cid in (0, 1):
                ycls = cid
            ycls = COCO2YOLO[cid]

            # bbox already normalized (cx, cy, w, h)
            cx, cy, w, h = obj.get("bbox", [0, 0, 0, 0])

            # keypoints dict → fixed order, normalize to [0,1]
            kpd = obj.get("keypoints", {})
            flat = []
            for name in KP_ORDER:
                x, y, v = kpd.get(name, [0, 0, 0])
                x = max(0.0, min(1.0, x / W)) if W else 0.0
                y = max(0.0, min(1.0, y / H)) if H else 0.0
                v = int(v) if v in (0,1,2) else (2 if (x>0 and y>0) else 0)
                flat += [f"{x:.6f}", f"{y:.6f}", str(v)]

            lines.append(" ".join([str(ycls),
                                   f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"] + flat))

        (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/student/project/output",
                    help="Your run output folder that contains coco_data/, composited/, keypoints/")
    ap.add_argument("--link_images", action="store_true", help="Use symlinks instead of copies")
    args = ap.parse_args()

    root = Path(args.root)
    write_yaml(root / "yolo_data")
    for split in ("train", "val"):
        make_split(split, root, link=args.link_images)
    print("✅ YOLO-Pose dataset at:", (root / "yolo_data").as_posix())

if __name__ == "__main__":
    main()