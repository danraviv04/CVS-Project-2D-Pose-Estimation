# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Predict with a YOLO-SEG model and derive 7 keypoints per instance from masks.

# Keypoint layout (always):
#   ["tip_right","tip_left","shaft_right","shaft_left","ring_right","ring_left","base"]

# Usage:
#   python3 predict.py --weights seg_phaseB/yolo11s_seg_with_kpts/weights/best.pt \
#                      --source /path/to/image.png \
#                      --output out_vis --save_json --show
# """

# import argparse, os, json
# from pathlib import Path

# import cv2
# import numpy as np
# from ultralytics import YOLO


# # ---------------- label / kind helpers ----------------

# def short_from_category(name: str) -> str:
#     n = (name or "").lower()
#     if "needle" in n or n.startswith("nh"):
#         return "NH"
#     if "tweezer" in n or n.startswith("t"):
#         return "T"
#     if n.startswith("nh"):
#         return "NH"
#     return "T"


# # ---------------- geometry from polygons ----------------

# def _pca_axes(P: np.ndarray):
#     """Return (mean, major, minor, length, projections_along_major) for Nx2 points."""
#     mean = P.mean(0)
#     A = P - mean
#     C = np.cov(A.T)
#     evals, evecs = np.linalg.eigh(C)
#     major = evecs[:, np.argmax(evals)]
#     minor = evecs[:, np.argmin(evals)]
#     major /= (np.linalg.norm(major) + 1e-9)
#     minor /= (np.linalg.norm(minor) + 1e-9)
#     s = A @ major
#     length = float(s.max() - s.min())
#     return mean, major, minor, length, s

# def _thickness(pts: np.ndarray, axis: np.ndarray) -> float:
#     if len(pts) == 0:
#         return 0.0
#     d = (pts - pts.mean(0)) @ axis
#     return float(np.sqrt((d**2).mean() + 1e-12))

# def _farthest_from_shaft(pts: np.ndarray, base: np.ndarray, shaft: np.ndarray):
#     if len(pts) == 0:
#         return base.copy()
#     t = (pts - base) @ shaft
#     proj = base + np.outer(t, shaft)
#     d2 = ((pts - proj) ** 2).sum(1)
#     return pts[int(np.argmax(d2))].copy()

# def keypoints_from_polys(polys, kind: str, img_h: int, img_w: int):
#     """
#     polys: list of (Mi x 2) arrays in image coords (float).
#     kind: "NH" or "T"
#     returns: dict of 7 keypoints (x,y) pixels or None if cannot compute.
#     """
#     if not polys:
#         return None
#     P = np.concatenate(polys, axis=0).astype(np.float32)
#     if P.shape[0] < 3:
#         return None

#     mean, shaft, side, length, s_all = _pca_axes(P)

#     # ends of the tool along the shaft
#     smin, smax = float(s_all.min()), float(s_all.max())
#     slab = max(4.0, 0.10 * (smax - smin))
#     near_min = P[s_all <= smin + slab]
#     near_max = P[s_all >= smax - slab]

#     t_min = _thickness(near_min, side)
#     t_max = _thickness(near_max, side)

#     # base (thicker) vs tip (thinner)
#     if t_max >= t_min:
#         base = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
#         tip  = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
#         base_t = smax
#     else:
#         base = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
#         tip  = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
#         base_t = smin

#     # slab near base for ring points
#     span = (smax - smin)
#     start, end = (base_t, base_t + 0.35 * span) if base_t <= (smin + smax) / 2 else (base_t - 0.35 * span, base_t)
#     lo, hi = min(start, end), max(start, end)
#     mask = (s_all >= lo) & (s_all <= hi)
#     slab_pts = P[mask] if int(mask.sum()) >= 15 else P

#     side_sign = (slab_pts - slab_pts.mean(0)) @ side
#     left_pts  = slab_pts[side_sign < 0]
#     right_pts = slab_pts[side_sign >= 0]

#     ring_left  = _farthest_from_shaft(left_pts,  base, shaft)
#     ring_right = _farthest_from_shaft(right_pts, base, shaft)

#     # shaft midpoints around center
#     offset = 0.02 * max(img_h, img_w)
#     shaft_left  = mean - offset * side
#     shaft_right = mean + offset * side

#     # tip split slightly for visibility
#     tip_left  = tip - 0.01 * max(img_h, img_w) * side
#     tip_right = tip + 0.01 * max(img_h, img_w) * side

#     def _clamp(p):
#         return [float(np.clip(p[0], 0, img_w - 1)), float(np.clip(p[1], 0, img_h - 1))]

#     return {
#         "tip_right":   _clamp(tip_right),
#         "tip_left":    _clamp(tip_left),
#         "shaft_right": _clamp(shaft_right),
#         "shaft_left":  _clamp(shaft_left),
#         "ring_right":  _clamp(ring_right),
#         "ring_left":   _clamp(ring_left),
#         "base":        _clamp(base),
#     }


# # ---------------- drawing ----------------

# # ---------- drawing (color-coded, with skeleton + halo) ----------

# import cv2
# from typing import Dict, Tuple

# # Default colors in BGR
# DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
#     "base_rings": (0, 165, 255),  # orange
#     "shafts":     (255, 255, 0),  # cyan
#     "tips":       (0, 255, 0),    # green
#     "halo":       (255, 255, 255) # white
# }

# # Order we expect (must match your keypoint generator)
# KPT_ORDER = [
#     "tip_right", "tip_left",
#     "shaft_right", "shaft_left",
#     "ring_right", "ring_left",
#     "base",
# ]

# def _draw_text_with_bg(
#     img,
#     text: str,
#     org: Tuple[int, int],
#     scale: float = 0.5,
#     thickness: int = 1,
#     bg_color: Tuple[int, int, int] = (0, 0, 0),
#     fg_color: Tuple[int, int, int] = (255, 255, 255)
# ):
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
#     x, y = int(org[0]), int(org[1])
#     cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + 3), bg_color, -1)
#     cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
#                 scale, fg_color, thickness, cv2.LINE_AA)

# def _line_with_halo(
#     img,
#     p1: Tuple[int, int],
#     p2: Tuple[int, int],
#     color: Tuple[int, int, int],
#     thickness: int = 2,
#     halo_color: Tuple[int, int, int] = (255, 255, 255),
#     halo_gain: int = 2
# ):
#     """Draw a colored line with a thin white halo behind it to pop on busy backgrounds."""
#     cv2.line(img, p1, p2, halo_color, max(1, thickness + halo_gain), cv2.LINE_AA)
#     cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

# def _circle_with_halo(
#     img,
#     center: Tuple[int, int],
#     radius: int,
#     color: Tuple[int, int, int],
#     halo_color: Tuple[int, int, int] = (255, 255, 255)
# ):
#     """Filled dot with a 1px halo ring."""
#     cv2.circle(img, center, radius + 1, halo_color, -1, cv2.LINE_AA)
#     cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)

# def draw_keypoints_cv(
#     img_bgr,
#     kps: dict,
#     colors: Dict[str, Tuple[int, int, int]] | None = None,
#     r: int = 4,
#     thickness: int = 2,
#     draw_label: str = "",
#     draw_legend: bool = False,
#     legend_pos: Tuple[int, int] = (12, 18)
# ):
#     """
#     Color groups:
#       - base/rings: orange
#       - shafts:     cyan
#       - tips:       green

#     Skeleton connectors kept:
#       - tip_right  <-> tip_left
#       - shaft_right <-> shaft_left
#       - ring_right  <-> ring_left
#       - base -> tip_right (orientation cue)
#     """
#     if colors is None:
#         colors = DEFAULT_COLORS

#     # Safeguard: ensure required keys exist
#     for key in KPT_ORDER:
#         if key not in kps:
#             return  # silently skip if the instance is incomplete

#     # Convert once to ints
#     P = {k: (int(round(kps[k][0])), int(round(kps[k][1]))) for k in KPT_ORDER}

#     # --- skeleton lines (with halo) ---
#     _line_with_halo(img_bgr, P["tip_right"],   P["tip_left"],   colors["tips"],   thickness, colors["halo"])
#     _line_with_halo(img_bgr, P["shaft_right"], P["shaft_left"], colors["shafts"], thickness, colors["halo"])
#     _line_with_halo(img_bgr, P["ring_right"],  P["ring_left"],  colors["base_rings"], thickness, colors["halo"])
#     _line_with_halo(img_bgr, P["base"],        P["tip_right"],  colors["tips"],   thickness, colors["halo"])

#     # --- points (with halo) ---
#     for k in ("tip_right", "tip_left"):
#         _circle_with_halo(img_bgr, P[k], r, colors["tips"], colors["halo"])
#     for k in ("shaft_right", "shaft_left"):
#         _circle_with_halo(img_bgr, P[k], r, colors["shafts"], colors["halo"])
#     for k in ("ring_right", "ring_left", "base"):
#         _circle_with_halo(img_bgr, P[k], r, colors["base_rings"], colors["halo"])

#     # --- optional label (anchored at base) ---
#     if draw_label:
#         _draw_text_with_bg(img_bgr, draw_label, P["base"], scale=0.55, thickness=1)

#     # --- optional legend (small + unobtrusive) ---
#     if draw_legend:
#         lx, ly = legend_pos
#         swatch = [
#             ("tips",       colors["tips"]),
#             ("shafts",     colors["shafts"]),
#             ("base/rings", colors["base_rings"]),
#         ]
#         y = ly
#         for name, col in swatch:
#             cv2.circle(img_bgr, (lx, y - 4), 5, colors["halo"], -1, cv2.LINE_AA)
#             cv2.circle(img_bgr, (lx, y - 4), 4, col,            -1, cv2.LINE_AA)
#             _draw_text_with_bg(img_bgr, name, (lx + 14, y + 2), scale=0.5, thickness=1)
#             y += 20

# # ---------------- prediction ----------------

# def predict_one(model: YOLO, img_path: str, out_dir: Path | None, save_json: bool, conf=0.25, iou=0.5):
#     # handle names map (dict or list)
#     raw_names = model.model.names
#     names_map = {int(k): v for k, v in raw_names.items()} if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}

#     # run inference
#     res = model.predict(source=img_path, conf=conf, iou=iou, save=False, verbose=False, retina_masks=True)[0]

#     # start with YOLO's own annotated frame (boxes/masks/labels)
#     annotated = res.plot()  # BGR

#     derived = []
#     if res.masks is not None and len(res.masks) > 0:
#         H, W = res.orig_shape
#         classes = res.boxes.cls.cpu().numpy().astype(int)
#         confs   = res.boxes.conf.cpu().numpy().astype(float)

#         for i, cls_idx in enumerate(classes):
#             name = names_map.get(int(cls_idx), str(int(cls_idx)))
#             kind = short_from_category(name)

#             # collect polygon parts for this mask
#             polys_xy = []
#             parts = res.masks.xy[i]  # list of (Ni x 2)
#             if isinstance(parts, list):
#                 for arr in parts:
#                     if arr is None or len(arr) < 3:
#                         continue
#                     polys_xy.append(np.asarray(arr, dtype=np.float32))
#             else:
#                 arr = np.asarray(parts, dtype=np.float32)
#                 if len(arr) >= 3:
#                     polys_xy.append(arr)

#             kps = keypoints_from_polys(polys_xy, kind, H, W)
#             if kps is None:
#                 continue

#             draw_keypoints_cv(annotated, kps, draw_label=kind)

#             derived.append({
#                 "class_id": int(cls_idx),
#                 "class_name": name,
#                 "confidence": float(confs[i]),
#                 "kind": kind,
#                 "bbox_xyxy": res.boxes.xyxy[i].cpu().numpy().tolist(),
#                 "keypoints": {k: [float(x), float(y), 2] for k, (x, y) in kps.items()}
#             })

#     # print concise summary
#     print("🔍 Prediction summary")
#     print(f"  image: {img_path}")
#     if len(res.boxes) == 0:
#         print("  no detections")
#     else:
#         print(f"  detections: {len(res.boxes)}")
#         for d in derived:
#             print(f"   - {d['class_name']} ({d['kind']}) conf={d['confidence']:.3f} "
#                   f"xyxy={[round(v,1) for v in d['bbox_xyxy']]} "
#                   f"kps={list(d['keypoints'].keys())}")

#     # save outputs
#     if out_dir is not None:
#         out_dir.mkdir(parents=True, exist_ok=True)
#         stem = Path(img_path).name
#         vis_path = out_dir / f"annotated_{stem}"
#         cv2.imwrite(str(vis_path), annotated)
#         print(f"✅ Saved image: {vis_path}")

#         if save_json:
#             js_path = out_dir / f"{Path(stem).stem}.json"
#             js_path.write_text(json.dumps(derived, indent=2))
#             print(f"✅ Saved JSON:  {js_path}")

#     return annotated, derived


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', required=True, help="Path to seg weights (.pt)")
#     parser.add_argument('--source',  required=True, help="Path to input image")
#     parser.add_argument('--output',  default=None, help="Directory to save annotated image/JSON")
#     parser.add_argument('--show',    action='store_true', help="Display result window")
#     parser.add_argument('--save_json', action='store_true', help="Also save a JSON with derived keypoints")
#     parser.add_argument('--conf', type=float, default=0.25)
#     parser.add_argument('--iou',  type=float, default=0.5)
#     args = parser.parse_args()

#     model = YOLO(args.weights)

#     out_dir = Path(args.output) if args.output else None
#     annotated, _ = predict_one(model, args.source, out_dir, args.save_json, conf=args.conf, iou=args.iou)

#     if args.show:
#         cv2.imshow("Prediction", annotated)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict with a YOLO-SEG model and derive class-specific keypoints from masks.

Keypoint layout per class:
  T  (5):  ["tip_right","tip_left","shaft_right","shaft_left","base"]
  NH (6):  ["tip_right","tip_left","shaft_right","shaft_left","ring_right","ring_left"]

Usage:
  python3 predict.py --weights seg_phaseB/yolo11s_seg_with_kpts/weights/best.pt \
                     --source /path/to/image.png \
                     --output out_vis --save_json --show
"""

import argparse, json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def _mid(p, q):
    return (int(round((p[0]+q[0]) / 2)), int(round((p[1]+q[1]) / 2)))

# ---------------- label / kind helpers ----------------

def short_from_category(name: str) -> str:
    n = (name or "").lower()
    if "needle" in n or n.startswith("nh"):
        return "NH"
    if "tweezer" in n or n.startswith("t"):
        return "T"
    if n.startswith("nh"):
        return "NH"
    return "T"


# ---------------- per-class schemas ----------------

KPT_ORDER_BY_KIND = {
    "T":  ["tip_right","tip_left","shaft_right","shaft_left","base"],  # 5
    "NH": ["tip_right","tip_left","shaft_right","shaft_left","ring_right","ring_left"],  # 6
}

# Colors (BGR)
DEFAULT_COLORS = {
    "base_rings": (0, 165, 255),  # orange
    "shafts":     (255, 255, 0),  # cyan
    "tips":       (0, 255, 0),    # green
    "halo":       (255, 255, 255), # white
    "spine":      (128, 0, 128)    # purple
}


# ---------------- geometry from polygons ----------------

def _pca_axes(P: np.ndarray):
    """Return (mean, major, minor, length, projections_along_major) for Nx2 points."""
    mean = P.mean(0)
    A = P - mean
    C = np.cov(A.T)
    evals, evecs = np.linalg.eigh(C)
    major = evecs[:, np.argmax(evals)]
    minor = evecs[:, np.argmin(evals)]
    major /= (np.linalg.norm(major) + 1e-9)
    minor /= (np.linalg.norm(minor) + 1e-9)
    s = A @ major
    length = float(s.max() - s.min())
    return mean, major, minor, length, s

def _thickness(pts: np.ndarray, axis: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    d = (pts - pts.mean(0)) @ axis
    return float(np.sqrt((d**2).mean() + 1e-12))

def _farthest_from_shaft(pts: np.ndarray, base: np.ndarray, shaft: np.ndarray):
    if len(pts) == 0:
        return base.copy()
    t = (pts - base) @ shaft
    proj = base + np.outer(t, shaft)
    d2 = ((pts - proj) ** 2).sum(1)
    return pts[int(np.argmax(d2))].copy()

def keypoints_from_polys(polys, kind: str, img_h: int, img_w: int):
    """
    polys: list of (Mi x 2) arrays in image coords (float).
    kind: "NH" or "T"
    returns: dict of keypoints for that class (T=5, NH=6). None if cannot compute.
    """
    if not polys:
        return None
    P = np.concatenate(polys, axis=0).astype(np.float32)
    if P.shape[0] < 3:
        return None

    mean, shaft, side, length, s_all = _pca_axes(P)

    # ends of the tool along the shaft
    smin, smax = float(s_all.min()), float(s_all.max())
    slab = max(4.0, 0.10 * (smax - smin))
    near_min = P[s_all <= smin + slab]
    near_max = P[s_all >= smax - slab]

    t_min = _thickness(near_min, side)
    t_max = _thickness(near_max, side)

    # base (thicker) vs tip (thinner)
    if t_max >= t_min:
        base = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
        tip  = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
        base_t = smax
    else:
        base = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
        tip  = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
        base_t = smin

    # slab near base for ring points
    span = (smax - smin)
    start, end = (base_t, base_t + 0.35 * span) if base_t <= (smin + smax) / 2 else (base_t - 0.35 * span, base_t)
    lo, hi = min(start, end), max(start, end)
    mask = (s_all >= lo) & (s_all <= hi)
    slab_pts = P[mask] if int(mask.sum()) >= 15 else P

    side_sign = (slab_pts - slab_pts.mean(0)) @ side
    left_pts  = slab_pts[side_sign < 0]
    right_pts = slab_pts[side_sign >= 0]

    ring_left  = _farthest_from_shaft(left_pts,  base, shaft)
    ring_right = _farthest_from_shaft(right_pts, base, shaft)

    # shaft midpoints around center
    offset = 0.02 * max(img_h, img_w)
    shaft_left  = mean - offset * side
    shaft_right = mean + offset * side

    # tip split slightly for visibility
    tip_left  = tip - 0.01 * max(img_h, img_w) * side
    tip_right = tip + 0.01 * max(img_h, img_w) * side

    def _clamp(p):
        return [float(np.clip(p[0], 0, img_w - 1)), float(np.clip(p[1], 0, img_h - 1))]

    # build full set then filter to class schema
    all_kps = {
        "tip_right":   _clamp(tip_right),
        "tip_left":    _clamp(tip_left),
        "shaft_right": _clamp(shaft_right),
        "shaft_left":  _clamp(shaft_left),
        "ring_right":  _clamp(ring_right),
        "ring_left":   _clamp(ring_left),
        "base":        _clamp(base),
    }
    wanted = KPT_ORDER_BY_KIND.get(kind, KPT_ORDER_BY_KIND["T"])
    return {k: all_kps[k] for k in wanted}


# ---------------- drawing ----------------

def _draw_text_with_bg(img, text, org, scale=0.5, thickness=1,
                       bg_color=(0,0,0), fg_color=(255,255,255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + 3), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg_color, thickness, cv2.LINE_AA)

def _line_with_halo(img, p1, p2, color, thickness=2, halo_color=(255,255,255), halo_gain=2):
    cv2.line(img, p1, p2, halo_color, max(1, thickness + halo_gain), cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def _circle_with_halo(img, center, radius, color, halo_color=(255,255,255)):
    cv2.circle(img, center, radius + 1, halo_color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)

def draw_keypoints_cv(
    img_bgr, kps: dict, *, kind: str,
    colors=None, r=4, thickness=2, draw_label="", draw_legend=False, legend_pos=(12,18)
):
    if colors is None:
        colors = DEFAULT_COLORS

    # ints
    P = {k: (int(round(v[0])), int(round(v[1]))) for k, v in kps.items()}
    have = lambda *ks: all(k in P for k in ks)

    # ---------- spine ----------
    spine = []
    if "base" in P:                       # T has base, NH doesn't expose it
        spine.append(P["base"])
    if kind == "NH" and have("ring_left","ring_right"):
        spine.append(_mid(P["ring_left"], P["ring_right"]))
    if have("shaft_left","shaft_right"):
        spine.append(_mid(P["shaft_left"], P["shaft_right"]))
    if have("tip_left","tip_right"):
        spine.append(_mid(P["tip_left"], P["tip_right"]))
    elif "tip_left" in P:
        spine.append(P["tip_left"])
    elif "tip_right" in P:
        spine.append(P["tip_right"])

    for a, b in zip(spine[:-1], spine[1:]):
        _line_with_halo(img_bgr, a, b, colors["spine"], thickness, colors["halo"])

    # ---------- crossbars ----------
    if have("ring_left","ring_right"):
        _line_with_halo(img_bgr, P["ring_left"], P["ring_right"], colors["base_rings"], thickness, colors["halo"])
    if have("shaft_left","shaft_right"):
        _line_with_halo(img_bgr, P["shaft_left"], P["shaft_right"], colors["shafts"], thickness, colors["halo"])
    if have("tip_left","tip_right"):
        _line_with_halo(img_bgr, P["tip_left"],  P["tip_right"],  colors["tips"], thickness, colors["halo"])

    # ---------- points ----------
    for k in ("tip_left","tip_right"):
        if k in P: _circle_with_halo(img_bgr, P[k], r, colors["tips"], colors["halo"])
    for k in ("shaft_left","shaft_right"):
        if k in P: _circle_with_halo(img_bgr, P[k], r, colors["shafts"], colors["halo"])
    # NH has rings; T has base — draw whichever exists
    for k in ("ring_left","ring_right","base"):
        if k in P: _circle_with_halo(img_bgr, P[k], r, colors["base_rings"], colors["halo"])

    if draw_label:
        anchor = P.get("base") or spine[0]  # NH has no base → use first spine point
        _draw_text_with_bg(img_bgr, draw_label, anchor, scale=0.55, thickness=1)

    if draw_legend:
        lx, ly = legend_pos
        for name, col in [("tips", colors["tips"]), ("shafts", colors["shafts"]), ("base/rings", colors["base_rings"])]:
            cv2.circle(img_bgr, (lx, ly-4), 5, colors["halo"], -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (lx, ly-4), 4, col,            -1, cv2.LINE_AA)
            _draw_text_with_bg(img_bgr, name, (lx+14, ly+2), scale=0.5, thickness=1)
            ly += 20


# ---------------- prediction ----------------

def predict_one(model: YOLO, img_path: str, out_dir: Path | None, save_json: bool, conf=0.25, iou=0.5):
    # names map
    raw_names = model.model.names
    names_map = {int(k): v for k, v in raw_names.items()} if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}

    # inference
    res = model.predict(source=img_path, conf=conf, iou=iou, save=False, verbose=False, retina_masks=True)[0]

    annotated = res.plot()  # BGR with boxes/masks

    derived = []
    if res.masks is not None and len(res.masks) > 0:
        H, W = res.orig_shape
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs   = res.boxes.conf.cpu().numpy().astype(float)

        for i, cls_idx in enumerate(classes):
            name = names_map.get(int(cls_idx), str(int(cls_idx)))
            kind = short_from_category(name)  # "NH" or "T"

            # polygons for this mask (in original image coords)
            polys_xy = []
            parts = res.masks.xy[i]
            if isinstance(parts, list):
                for arr in parts:
                    if arr is None or len(arr) < 3: continue
                    polys_xy.append(np.asarray(arr, dtype=np.float32))
            else:
                arr = np.asarray(parts, dtype=np.float32)
                if len(arr) >= 3: polys_xy.append(arr)

            kps = keypoints_from_polys(polys_xy, kind, H, W)
            if kps is None:
                continue

            draw_keypoints_cv(annotated, kps, kind=kind, draw_label=kind)

            derived.append({
                "class_id": int(cls_idx),
                "class_name": name,
                "confidence": float(confs[i]),
                "kind": kind,
                "bbox_xyxy": res.boxes.xyxy[i].cpu().numpy().tolist(),
                "keypoints": {k: [float(x), float(y), 2] for k, (x, y) in kps.items()}
            })

    # log summary
    print("🔍 Prediction summary")
    print(f"  image: {img_path}")
    if len(res.boxes) == 0:
        print("  no detections")
    else:
        print(f"  detections: {len(res.boxes)}")
        for d in derived:
            print(f"   - {d['class_name']} ({d['kind']}) conf={d['confidence']:.3f} "
                  f"xyxy={[round(v,1) for v in d['bbox_xyxy']]} "
                  f"kps={list(d['keypoints'].keys())}")

    # outputs
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).name
        vis_path = out_dir / f"annotated_{stem}"
        cv2.imwrite(str(vis_path), annotated)
        print(f"✅ Saved image: {vis_path}")

        if save_json:
            js_path = out_dir / f"{Path(stem).stem}.json"
            js_path.write_text(json.dumps(derived, indent=2))
            print(f"✅ Saved JSON:  {js_path}")

    return annotated, derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help="Path to seg weights (.pt)")
    ap.add_argument('--source',  required=True, help="Path to input image")
    ap.add_argument('--output',  default=None, help="Directory to save annotated image/JSON")
    ap.add_argument('--show',    action='store_true', help="Display result window")
    ap.add_argument('--save_json', action='store_true', help="Also save a JSON with derived keypoints")
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou',  type=float, default=0.5)
    args = ap.parse_args()

    model = YOLO(args.weights)
    out_dir = Path(args.output) if args.output else None
    annotated, _ = predict_one(model, args.source, out_dir, args.save_json, conf=args.conf, iou=args.iou)

    if args.show:
        cv2.imshow("Prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()