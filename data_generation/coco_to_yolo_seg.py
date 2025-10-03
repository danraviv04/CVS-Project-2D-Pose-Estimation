#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, random, shutil, argparse
from pathlib import Path

# COCO cat ids from your script: 1=needle holder, 2=tweezers
CAT_TO_YOLO = {1: 0, 2: 1}
NAMES = ["needle_holder", "tweezers"]

def poly_area(pl):
    x = pl[0::2]; y = pl[1::2]
    return 0.5 * abs(sum(x[i]*y[(i+1)%len(y)] - x[(i+1)%len(x)]*y[i] for i in range(len(x))))

def write_yaml(root, train, val, names):
    text = (
        f"path: {Path(root).resolve()}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"names:\n" + "\n".join([f"  {i}: {n}" for i, n in enumerate(names)]) + "\n"
    )
    Path(root, "data.yaml").write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help=".../coco_annotations.json")
    ap.add_argument("--images", required=True, help=".../images")
    ap.add_argument("--out", required=True, help="output root for YOLO-Seg")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))

    images = {im["id"]: im for im in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1: 
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    root = Path(args.out)
    for d in ["images/train","images/val","labels/train","labels/val"]:
        (root/d).mkdir(parents=True, exist_ok=True)

    ids = list(images.keys()); random.shuffle(ids)
    n_val = int(len(ids) * args.val_split)
    valset = set(ids[:n_val])

    for img_id in ids:
        im = images[img_id]
        W, H = im["width"], im["height"]
        file = Path(im["file_name"]).name
        src = Path(args.images) / file

        if img_id in valset:
            dst_img = root/"images/val"/file
            dst_lab = root/"labels/val"/(Path(file).stem + ".txt")
        else:
            dst_img = root/"images/train"/file
            dst_lab = root/"labels/train"/(Path(file).stem + ".txt")

        shutil.copy2(src, dst_img)

        lines = []
        for a in ann_by_img.get(img_id, []):
            cat = a["category_id"]
            if cat not in CAT_TO_YOLO: 
                continue
            seg = a.get("segmentation", [])
            # Choose largest polygon if multiple
            polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
            if not polys: 
                continue
            poly = max(polys, key=poly_area)

            xs = [max(0.0, min(1.0, x / W)) for x in poly[0::2]]
            ys = [max(0.0, min(1.0, y / H)) for y in poly[1::2]]
            coords = []
            for x, y in zip(xs, ys):
                coords += [f"{x:.6f}", f"{y:.6f}"]
            lines.append(" ".join([str(CAT_TO_YOLO[cat])] + coords))

        dst_lab.write_text("\n".join(lines), encoding="utf-8")

    write_yaml(root, "images/train", "images/val", NAMES)
    print("✅ YOLO-Seg dataset:", root)
    print("   data.yaml:", (root/'data.yaml').as_posix())

if __name__ == "__main__":
    main()
