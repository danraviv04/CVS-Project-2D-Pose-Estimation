#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from pycocotools import mask as maskUtils


# ---------------- utils ----------------

def _cat_palette(n: int):
    """A small stable palette; cycles if n > len(base)."""
    base = [
        (66, 135, 245),  # blue
        (245, 130, 48),  # orange
        (60, 180, 75),   # green
        (230, 25, 75),   # red
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190), # pink
        (0, 128, 128),   # teal
    ]
    return [base[i % len(base)] for i in range(n)]


def load_coco(coco_json: str):
    """Load COCO and index images/annotations."""
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        if not ann.get("segmentation"):
            continue
        anns_by_img[ann["image_id"]].append(ann)
    cats = coco.get("categories", [])
    cats_by_id = {c["id"]: c for c in cats}
    return images, anns_by_img, cats_by_id


def _mask_to_polys(msk: np.ndarray):
    """Binary mask -> list of flat [x1,y1,...] polygons."""
    msk = (msk.astype(np.uint8) > 0).astype(np.uint8)
    cs = measure.find_contours(msk, 0.5)
    polys = []
    for c in cs:
        if len(c) < 3:
            continue
        c = np.flip(c, axis=1)  # (row,col)->(x,y)
        polys.append(c.ravel().tolist())
    return polys


def seg_to_polys(seg, H: int, W: int):
    """
    Accept any COCO 'segmentation' and return a list of flat polygons.
    - polygons: list of lists [x1,y1,...]
    - single RLE dict (compressed or uncompressed)
    - list of RLE dicts (multi-part)
    """
    if not seg:
        return []

    # Polygon list case (already XY coords)
    if isinstance(seg, list):
        if seg and isinstance(seg[0], (list, tuple)):
            return [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
        if seg and isinstance(seg[0], dict):
            # list of RLE dicts -> frPyObjects -> decode -> merge
            rles = maskUtils.frPyObjects(seg, H, W)
            m = maskUtils.decode(rles)   # HxWxN or HxW
            if m.ndim == 3:
                m = np.any(m, axis=2)
            return _mask_to_polys(m)
        return []

    # Single RLE dict
    if isinstance(seg, dict) and "counts" in seg:
        rle = seg
        # If counts is a list (uncompressed), convert via frPyObjects first
        if not isinstance(rle["counts"], (bytes, bytearray, str)):
            rle = maskUtils.frPyObjects([rle], H, W)
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return _mask_to_polys(m)

    return []


def draw_polygons(pil_img: Image.Image, polygons, color, alpha=96, outline=2):
    """Draw filled + outlined polygons onto image and return composited result."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    ol = ImageDraw.Draw(overlay, "RGBA")
    line = ImageDraw.Draw(overlay)

    fill_col = (color[0], color[1], color[2], int(alpha))
    edge_col = (color[0], color[1], color[2], 255)

    for poly in polygons:
        if not isinstance(poly, (list, tuple)) or len(poly) < 6:
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        ol.polygon(pts, fill=fill_col)
        line.line(pts + [pts[0]], fill=edge_col, width=int(outline), joint="curve")
    return Image.alpha_composite(pil_img, overlay)


def short_from_category(name: str) -> str:
    n = (name or "").lower()
    return "NH" if ("needle" in n or n == "nh") else "T"


def label_for_ann(ann, cats_by_id, mode="category"):
    """
    Get a label string for an annotation.
        ann: COCO annotation dict
        cats_by_id: dict mapping category_id to category dict
        mode: "category" (full name), "short" (NH/T), or "instance" (instance name)
    returns: label string
    """
    cname = (cats_by_id.get(ann["category_id"], {}) or {}).get("name", "")
    iname = ann.get("attributes", {}).get("name", "") or ann.get("name", "")

    if mode == "short":
        def as_short(s):
            s = (s or "").lower()
            return "Needle Holder" if ("needle" in s or "nh" in s) else "Tweezers"

        s = as_short(cname)
        # if category name didn't reveal NH, try instance name
        if s == "T" and as_short(iname) == "NH":
            s = "NH"
        return s

    if mode == "instance":
        return iname or cname or str(ann["category_id"])
    return cname or str(ann["category_id"])


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True,
                    help="Path to annotations_train.json or annotations_val.json")
    ap.add_argument("--images_root", required=True,
                    help="Root where images live (e.g. output/composited/train or /val)")
    ap.add_argument("--out", required=True, help="Folder to write visualizations")
    ap.add_argument("--limit", type=int, default=0,
                    help="Visualize at most N images (0 = all)")
    ap.add_argument("--draw_names", action="store_true",
                    help="Draw a text label for each instance")
    ap.add_argument("--label", choices=["category", "short", "instance"],
                    default="category",
                    help="What to display for each label")
    ap.add_argument("--outline", type=int, default=2, help="Polygon outline width")
    ap.add_argument("--alpha", type=int, default=96, help="Polygon fill alpha (0-255)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, anns_by_img, cats_by_id = load_coco(args.coco)
    cat_ids_sorted = sorted(cats_by_id.keys())
    palette = _cat_palette(len(cat_ids_sorted))
    color_by_cat = {cid: palette[i] for i, cid in enumerate(cat_ids_sorted)}

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    img_ids = list(images.keys())
    if args.limit > 0:
        img_ids = img_ids[:args.limit]

    for img_id in img_ids:
        im_info = images[img_id]
        H = im_info.get("height")
        W = im_info.get("width")

        fn = Path(im_info["file_name"]).name  # try basename first
        src = Path(args.images_root) / fn
        if not src.exists():
            src = Path(args.images_root) / Path(im_info["file_name"])
        if not src.exists():
            print(f"missing image for {im_info['file_name']}")
            continue

        # Fallback if width/height missing in COCO
        if not H or not W:
            with Image.open(src) as _tmp:
                W, H = _tmp.size

        im = Image.open(src).convert("RGBA")
        anns = anns_by_img.get(img_id, [])

        # 1) Fill masks grouped by category (prettier color stacking)
        by_cat = defaultdict(list)
        for ann in anns:
            polys = seg_to_polys(ann.get("segmentation", []), H, W)
            if not polys:
                continue
            by_cat[ann["category_id"]].extend(polys)

        vis = im
        for cid, polys in by_cat.items():
            vis = draw_polygons(
                vis, polys, color_by_cat.get(cid, (255, 255, 255)),
                alpha=args.alpha, outline=args.outline
            )

        # 2) Labels per instance (centroid of first polygon)
        if args.draw_names:
            for ann in anns:
                polys = seg_to_polys(ann.get("segmentation", []), H, W)
                if not polys:
                    continue
                xy = np.array(polys[0], dtype=float).reshape(-1, 2)
                cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
                lab = label_for_ann(ann, cats_by_id, mode=args.label)

                ann_overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
                ann_draw = ImageDraw.Draw(ann_overlay, "RGBA")
                try:
                    bbox = ann_draw.textbbox((cx, cy), lab, font=font)
                    ann_draw.rectangle(bbox, fill=(0, 0, 0, 100))
                except Exception:
                    w, h = ann_draw.textsize(lab, font=font)
                    ann_draw.rectangle([cx, cy, cx + w, cy + h], fill=(0, 0, 0, 100))
                ann_draw.text((cx, cy), lab, fill=(255, 255, 255, 255), font=font)
                vis = Image.alpha_composite(vis, ann_overlay)

        out_path = out_dir / fn
        vis.convert("RGB").save(out_path, "PNG", optimize=True)
        print("✅", out_path)

    print(f"done. previews in: {out_dir}")


if __name__ == "__main__":
    main()
