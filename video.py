# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# video.py — Ultralytics-style YOLO-SEG video inference (cv2 I/O) + optional keypoint overlay

# - Matches `yolo predict` behavior: single-pass, no priors/EMA/multiscale unions.
# - Uses Ultralytics' own pre/post-processing via model.predict on each frame.
# - Writes an annotated video with masks/boxes/labels using result.plot().
# - (Optional) Saves YOLO-SEG polygon labels per frame (like `save_txt=True`).
# - (Optional) Overlays PCA-derived keypoints (NH=6, T=5) just for visualization.
# """

# from __future__ import annotations
# import argparse
# import os
# import time
# from pathlib import Path
# from typing import Dict, List, Optional

# import cv2
# import numpy as np
# from ultralytics import YOLO

# # -------------------- Keypoint utilities (optional overlay) --------------------
# LAYOUT = {
#     "NH": {"order": ["tip_right","tip_left","shaft_right","shaft_left","ring_right","ring_left"],
#            "skeleton": [(0,3),(1,2),(2,4),(3,5)]},
#     "T":  {"order": ["tip_right","tip_left","shaft_right","shaft_left","base"],
#            "skeleton": [(0,2),(1,3),(2,4),(3,4)]}
# }

# def kind_from_name(name: str) -> str:
#     n = (name or "").lower()
#     return "NH" if n.startswith("nh") else "T"

# def _pca_axes(P: np.ndarray):
#     mean = P.mean(0)
#     A = P - mean
#     C = np.cov(A.T)
#     ev, U = np.linalg.eigh(C)
#     major = U[:, int(np.argmax(ev))]; minor = U[:, int(np.argmin(ev))]
#     major /= (np.linalg.norm(major) + 1e-9); minor /= (np.linalg.norm(minor) + 1e-9)
#     s = A @ major
#     return mean, major, minor, float(s.max() - s.min()), s

# def _thickness(pts: np.ndarray, axis: np.ndarray) -> float:
#     if len(pts) == 0: return 0.0
#     d = (pts - pts.mean(0)) @ axis
#     return float(np.sqrt((d**2).mean() + 1e-12))

# def _farthest_from_shaft(pts: np.ndarray, base: np.ndarray, shaft: np.ndarray) -> np.ndarray:
#     if len(pts) == 0: return base.copy()
#     t = (pts - base) @ shaft
#     proj = base + np.outer(t, shaft)
#     d2 = ((pts - proj) ** 2).sum(1)
#     return pts[int(np.argmax(d2))].copy()

# def keypoints_from_points(P: np.ndarray, img_h: int, img_w: int, kind: str,
#                           min_pts: int = 12) -> Optional[Dict[str, List[float]]]:
#     if P is None or len(P) < min_pts:
#         return None
#     mean, shaft, side, span, s_all = _pca_axes(P)
#     smin, smax = float(s_all.min()), float(s_all.max())
#     slab_w = max(3.0, 0.12 * span)
#     near_min = P[s_all <= smin + slab_w]
#     near_max = P[s_all >= smax - slab_w]
#     t_min, t_max = _thickness(near_min, side), _thickness(near_max, side)

#     if t_max >= t_min:
#         base = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
#         tip  = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
#         base_t, tip_t = smax, smin
#     else:
#         base = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
#         tip  = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
#         base_t, tip_t = smin, smax

#     off_px = 0.02 * max(img_h, img_w)
#     shaft_left  = mean - off_px * side
#     shaft_right = mean + off_px * side
#     tip_split   = 0.01 * max(img_h, img_w)
#     tip_left  = tip - tip_split * side
#     tip_right = tip + tip_split * side

#     span_all = smax - smin
#     lo, hi = (base_t, base_t + 0.35 * span_all) if base_t <= (smin + smax)/2 else (base_t - 0.35 * span_all, base_t)
#     mask = (s_all >= min(lo,hi)) & (s_all <= max(lo,hi))
#     slab_pts = P[mask] if int(mask.sum()) >= 20 else P

#     if kind == "NH":
#         side_sign = (slab_pts - slab_pts.mean(0)) @ side
#         left_pts  = slab_pts[side_sign < 0]
#         right_pts = slab_pts[side_sign >= 0]
#         ring_left  = _farthest_from_shaft(left_pts,  base, shaft)
#         ring_right = _farthest_from_shaft(right_pts, base, shaft)
#     else:
#         shaft_dir  = shaft if base_t > tip_t else -shaft
#         anchor     = base + shaft_dir * (0.06 * span_all)
#         side_d     = 0.025 * max(img_h, img_w)
#         ring_right = anchor + side * side_d
#         ring_left  = anchor - side * side_d

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

# def draw_kpts(img: np.ndarray, kps: Dict[str, List[float]], kind: str,
#               color=(0, 220, 255), r: int = 2, t: int = 2):
#     layout = LAYOUT["NH" if kind == "NH" else "T"]
#     order = layout["order"]
#     pts = [tuple(map(int, kps[k])) for k in order]
#     for (i, j) in layout["skeleton"]:
#         cv2.line(img, pts[i], pts[j], color, t, cv2.LINE_AA)
#     for (x, y) in pts:
#         cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
#     return img

# # -------------------- TXT writer (Ultralytics-style seg labels) --------------------
# def write_seg_txt(res, names_map: Dict[int, str], lab_dir: str, stem: str, save_conf=False):
#     os.makedirs(lab_dir, exist_ok=True)
#     path = os.path.join(lab_dir, f"{stem}.txt")
#     with open(path, "w") as f:
#         if res.masks is None or len(res.masks) == 0:
#             return
#         cls = res.boxes.cls.cpu().numpy().astype(int)
#         conf = res.boxes.conf.cpu().numpy().astype(float)
#         xyn_list = res.masks.xyn
#         for i in range(len(cls)):
#             parts = xyn_list[i] if isinstance(xyn_list[i], list) else [xyn_list[i]]
#             for p in parts:
#                 if p is None or len(p) < 3:
#                     continue
#                 line = [str(int(cls[i]))]
#                 if save_conf:
#                     line.append(f"{float(conf[i]):.6f}")
#                 flat = " ".join(f"{v:.6f}" for v in p.reshape(-1))
#                 f.write(" ".join(line) + " " + flat + "\n")

# # -------------------- Helpers --------------------
# def _fmt_secs(s: float) -> str:
#     s = max(0, int(s))
#     h, r = divmod(s, 3600)
#     m, r = divmod(r, 60)
#     return f"{h:d}:{m:02d}:{r:02d}" if h else f"{m:02d}:{r:02d}"

# # -------------------- Main --------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--weights", required=True)
#     ap.add_argument("--video",   required=True)
#     ap.add_argument("--out_dir", default="out_vis_ultra_like")

#     # Ultralytics-core knobs (match CLI)
#     ap.add_argument("--imgsz", type=int, default=1280)
#     ap.add_argument("--conf",  type=float, default=0.45)
#     ap.add_argument("--iou",   type=float, default=0.55)
#     ap.add_argument("--device", default="0")
#     ap.add_argument("--max_det", type=int, default=6)
#     ap.add_argument("--retina_masks", action="store_true")
#     ap.add_argument("--half", action="store_true")
#     ap.add_argument("--agnostic_nms", action="store_true")

#     # Output/format
#     ap.add_argument("--codec", default="mp4v", help="cv2 FourCC: mp4v, avc1, XVID, MJPG, etc.")
#     ap.add_argument("--save_txt", action="store_true")
#     ap.add_argument("--save_conf", action="store_true")
#     ap.add_argument("--save_frames", action="store_true", help="also dump per-frame images alongside labels")

#     # Optional visualization of derived keypoints
#     ap.add_argument("--draw_kpts", action="store_true")

#     # Progress printing
#     ap.add_argument("--progress_every", type=int, default=50, help="print progress every N frames")

#     args = ap.parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     # Load model
#     model = YOLO(args.weights)
#     try:
#         model.to(args.device)
#         if args.half and str(args.device) != "cpu":
#             try:
#                 model.model.half()
#                 print("FP16 enabled.")
#             except Exception:
#                 pass
#     except Exception:
#         pass

#     # Names map (Ultralytics-style)
#     raw_names = getattr(model.model, "names", None)
#     names_map = {int(k): v for k, v in raw_names.items()} if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}

#     # Video IO
#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {args.video}")
#     ok, first = cap.read()
#     if not ok:
#         raise RuntimeError("Could not read first frame.")
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     H0, W0 = first.shape[:2]
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

#     base = Path(args.video).stem
#     out_path = os.path.join(
#         args.out_dir,
#         f"annotated_{base}.mp4" if args.codec.lower() == "mp4v" else f"annotated_{base}.avi"
#     )
#     fourcc = cv2.VideoWriter_fourcc(*args.codec)
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (W0, H0))
#     if not writer.isOpened():
#         raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

#     # Directories for optional per-frame saves
#     img_dir = os.path.join(args.out_dir, "images")
#     lab_dir = os.path.join(args.out_dir, "labels")
#     if args.save_frames:
#         os.makedirs(img_dir, exist_ok=True)
#     if args.save_txt:
#         os.makedirs(lab_dir, exist_ok=True)

#     print(f"▶️  Processing '{args.video}' ({W0}x{H0} @ {fps:.2f} fps, frames={total_frames or 'unknown'})")
#     print(f"    Weights: {args.weights}")
#     t0 = time.time()
#     idx = 0

#     # cumulative detection counters
#     nh_total, t_total = 0, 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Ultralytics predict on the raw frame (matches CLI pre/post)
#         res = model.predict(
#             source=frame,
#             imgsz=args.imgsz,
#             conf=args.conf,
#             iou=args.iou,
#             device=args.device,
#             max_det=args.max_det,
#             verbose=False,
#             retina_masks=args.retina_masks,
#             agnostic_nms=args.agnostic_nms,
#         )[0]

#         # Count detections by class-kind
#         nh_frame = 0; t_frame = 0
#         try:
#             if res.boxes is not None and len(res.boxes) > 0:
#                 cls_arr = res.boxes.cls.int().cpu().numpy().tolist()
#                 for c in cls_arr:
#                     kind = kind_from_name(names_map.get(int(c), str(c)))
#                     if kind == "NH":
#                         nh_frame += 1
#                     else:
#                         t_frame += 1
#         except Exception:
#             pass

#         # Ultralytics visualization
#         vis = res.plot()  # BGR numpy array with masks/boxes/labels

#         # Optional: derived keypoints overlay (only uses masks → polygons)
#         if args.draw_kpts and res.masks is not None and len(res.masks) > 0:
#             H, W = vis.shape[:2]
#             cls = res.boxes.cls.cpu().numpy().astype(int)
#             xyn_list = res.masks.xyn
#             for i in range(len(cls)):
#                 parts = xyn_list[i] if isinstance(xyn_list[i], list) else [xyn_list[i]]
#                 if not parts:
#                     continue
#                 P = []
#                 for p in parts:
#                     if p is None or len(p) < 3:
#                         continue
#                     p = np.asarray(p, dtype=np.float32)
#                     p[:, 0] *= W
#                     p[:, 1] *= H
#                     P.append(p)
#                 if not P:
#                     continue
#                 P = np.concatenate(P, axis=0)
#                 kind = kind_from_name(names_map.get(int(cls[i]), str(cls[i])))
#                 kps = keypoints_from_points(P, H, W, kind, min_pts=12)
#                 if kps is not None:
#                     draw_kpts(vis, kps, kind)

#         # Write video frame
#         writer.write(vis)

#         # Optional: save per-frame TXT (Ultralytics-style) and images
#         stem = f"{base}_{idx:06d}"
#         if args.save_txt:
#             write_seg_txt(res, names_map, lab_dir, stem, save_conf=args.save_conf)
#         if args.save_frames:
#             cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), vis)

#         # Progress print
#         if (idx % args.progress_every == 0) or (total_frames and idx + 1 == total_frames):
#             elapsed = time.time() - t0
#             fps_live = (idx + 1) / max(1e-6, elapsed)
#             if total_frames > 0:
#                 pct = 100.0 * (idx + 1) / total_frames
#                 eta = (total_frames - (idx + 1)) / max(1e-6, fps_live)
#                 print(f"[{idx+1:6d}/{total_frames}] {pct:5.1f}% | fps={fps_live:5.1f} | elapsed={_fmt_secs(elapsed)} | eta={_fmt_secs(eta)} | NH={nh_frame} T={t_frame}")
#             else:
#                 print(f"[{idx+1:6d}] fps={fps_live:5.1f} | elapsed={_fmt_secs(elapsed)} | NH={nh_frame} T={t_frame}")

#         idx += 1

#     cap.release()
#     writer.release()
#     total_elapsed = time.time() - t0
#     avg_fps = (idx / max(1e-6, total_elapsed))
#     print(f"✅ Output saved to: {out_path}")
#     if args.save_txt:
#         print(f"📝 Per-frame labels → {lab_dir}")
#     if args.save_frames:
#         print(f"🖼  Per-frame images → {img_dir}")
#     print(f"📊 Done: frames={idx} | elapsed={_fmt_secs(total_elapsed)} | avg_fps={avg_fps:.1f} | NH={nh_total} | T={t_total}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video.py — Ultralytics-style YOLO-SEG video inference (cv2 I/O) + optional keypoint overlay

- Matches `yolo predict` behavior: single-pass (no unions/EMAs).
- Uses Ultralytics' own pre/post-processing via model.predict on each frame.
- Writes an annotated video with masks/boxes/labels using result.plot().
- (Optional) Saves YOLO-SEG polygon labels per frame (like `save_txt=True`).
- (Optional) Overlays PCA-derived keypoints (NH=6, T=5) for visualization.
- (Optional) Pre-CLAHE on the Y channel to improve low-contrast OR frames.

Example:
  python video.py \
    --weights seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/weights/best.pt \
    --video /datashare/project/vids_tune/4_2_24_B_2.mp4 \
    --out_dir out_vis/ultra_like_4_2_24_B_2 \
    --imgsz 1280 --conf 0.45 --iou 0.55 --device 0 --max_det 6 \
    --retina_masks --half --save_txt --save_conf --draw_kpts --progress_every 1 \
    --pre_clahe --clahe_clip 2.0
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- Optional pre-CLAHE --------------------
def enhance_like_infer(img_bgr: np.ndarray, clip=2.0) -> np.ndarray:
    """Apply CLAHE to Y channel (YCrCb) — useful on dim/low-contrast OR footage."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(8, 8))
    y = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

# -------------------- Keypoint utilities (optional overlay) --------------------
LAYOUT = {
    "NH": {"order": ["tip_right","tip_left","shaft_right","shaft_left","ring_right","ring_left"],
           "skeleton": [(0,3),(1,2),(2,4),(3,5)]},
    "T":  {"order": ["tip_right","tip_left","shaft_right","shaft_left","base"],
           "skeleton": [(0,2),(1,3),(2,4),(3,4)]}
}

def kind_from_name(name: str) -> str:
    n = (name or "").lower()
    return "NH" if n.startswith("nh") else "T"

def _pca_axes(P: np.ndarray):
    mean = P.mean(0)
    A = P - mean
    C = np.cov(A.T)
    ev, U = np.linalg.eigh(C)
    major = U[:, int(np.argmax(ev))]; minor = U[:, int(np.argmin(ev))]
    major /= (np.linalg.norm(major) + 1e-9); minor /= (np.linalg.norm(minor) + 1e-9)
    s = A @ major
    return mean, major, minor, float(s.max() - s.min()), s

def _thickness(pts: np.ndarray, axis: np.ndarray) -> float:
    if len(pts) == 0: return 0.0
    d = (pts - pts.mean(0)) @ axis
    return float(np.sqrt((d**2).mean() + 1e-12))

def _farthest_from_shaft(pts: np.ndarray, base: np.ndarray, shaft: np.ndarray) -> np.ndarray:
    if len(pts) == 0: return base.copy()
    t = (pts - base) @ shaft
    proj = base + np.outer(t, shaft)
    d2 = ((pts - proj) ** 2).sum(1)
    return pts[int(np.argmax(d2))].copy()

def keypoints_from_points(P: np.ndarray, img_h: int, img_w: int, kind: str,
                          min_pts: int = 12) -> Optional[Dict[str, List[float]]]:
    if P is None or len(P) < min_pts:
        return None
    mean, shaft, side, span, s_all = _pca_axes(P)
    smin, smax = float(s_all.min()), float(s_all.max())
    slab_w = max(3.0, 0.12 * span)
    near_min = P[s_all <= smin + slab_w]
    near_max = P[s_all >= smax - slab_w]
    t_min, t_max = _thickness(near_min, side), _thickness(near_max, side)

    # base/tip decision
    if t_max >= t_min:
        base = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
        tip  = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
        base_t, tip_t = smax, smin
    else:
        base = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
        tip  = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
        base_t, tip_t = smin, smax

    # derived anchors
    off_px = 0.02 * max(img_h, img_w)
    shaft_left  = mean - off_px * side
    shaft_right = mean + off_px * side
    tip_split   = 0.01 * max(img_h, img_w)
    tip_left  = tip - tip_split * side
    tip_right = tip + tip_split * side

    # ring search slab near base
    span_all = smax - smin
    lo, hi = (base_t, base_t + 0.35 * span_all) if base_t <= (smin + smax)/2 else (base_t - 0.35 * span_all, base_t)
    mask = (s_all >= min(lo,hi)) & (s_all <= max(lo,hi))
    slab_pts = P[mask] if int(mask.sum()) >= 20 else P

    if kind == "NH":
        side_sign = (slab_pts - slab_pts.mean(0)) @ side
        left_pts  = slab_pts[side_sign < 0]
        right_pts = slab_pts[side_sign >= 0]
        ring_left  = _farthest_from_shaft(left_pts,  base, shaft)
        ring_right = _farthest_from_shaft(right_pts, base, shaft)
    else:
        # synth "rings" for schema stability; not drawn for T but used if needed
        shaft_dir  = shaft if base_t > tip_t else -shaft
        anchor     = base + shaft_dir * (0.06 * span_all)
        side_d     = 0.025 * max(img_h, img_w)
        ring_right = anchor + side * side_d
        ring_left  = anchor - side * side_d

    def _clamp(p):
        return [float(np.clip(p[0], 0, img_w - 1)), float(np.clip(p[1], 0, img_h - 1))]

    return {
        "tip_right":   _clamp(tip_right),
        "tip_left":    _clamp(tip_left),
        "shaft_right": _clamp(shaft_right),
        "shaft_left":  _clamp(shaft_left),
        "ring_right":  _clamp(ring_right),
        "ring_left":   _clamp(ring_left),
        "base":        _clamp(base),
    }

def draw_kpts(img: np.ndarray, kps: Dict[str, List[float]], kind: str,
              color=(0, 220, 255), r: int = 2, t: int = 2):
    layout = LAYOUT["NH" if kind == "NH" else "T"]
    order = layout["order"]
    pts = [tuple(map(int, kps[k])) for k in order]
    for (i, j) in layout["skeleton"]:
        cv2.line(img, pts[i], pts[j], color, t, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
    return img

# -------------------- TXT writer (Ultralytics-style seg labels) --------------------
def write_seg_txt(res, names_map: Dict[int, str], lab_dir: str, stem: str, save_conf=False):
    """
    Writes one .txt per frame. Each line:
      <cls> [<conf>] x1 y1 x2 y2 ... (normalized polygon coords)
    Supports multi-part polygons (writes one line per part).
    """
    os.makedirs(lab_dir, exist_ok=True)
    path = os.path.join(lab_dir, f"{stem}.txt")
    with open(path, "w") as f:
        if res.masks is None or len(res.masks) == 0:
            return
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy().astype(float)
        xyn_list = res.masks.xyn  # list of arrays per instance
        for i in range(len(cls)):
            parts = xyn_list[i] if isinstance(xyn_list[i], list) else [xyn_list[i]]
            for p in parts:
                if p is None or len(p) < 3:
                    continue
                line = [str(int(cls[i]))]
                if save_conf:
                    line.append(f"{float(conf[i]):.6f}")
                flat = " ".join(f"{v:.6f}" for v in p.reshape(-1))
                f.write(" ".join(line) + " " + flat + "\n")

# -------------------- Helpers --------------------
def _fmt_secs(s: float) -> str:
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    return f"{h:d}:{m:02d}:{r:02d}" if h else f"{m:02d}:{r:02d}"

def _choose_ext(codec: str) -> str:
    c = (codec or "").lower()
    return ".mp4" if c in ("mp4v", "avc1", "h264", "hevc") else ".avi"

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video",   required=True)
    ap.add_argument("--out_dir", default="out_vis_ultra_like")

    # Ultralytics-core knobs (match CLI)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf",  type=float, default=0.45)
    ap.add_argument("--iou",   type=float, default=0.55)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max_det", type=int, default=6)
    ap.add_argument("--retina_masks", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--agnostic_nms", action="store_true")

    # Optional pre-CLAHE
    ap.add_argument("--pre_clahe", action="store_true",
                    help="Apply CLAHE on the Y channel before inference")
    ap.add_argument("--clahe_clip", type=float, default=2.0,
                    help="CLAHE clip limit (only if --pre_clahe)")

    # Output/format
    ap.add_argument("--codec", default="mp4v", help="cv2 FourCC: mp4v, avc1, XVID, MJPG, etc.")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--save_conf", action="store_true")
    ap.add_argument("--save_frames", action="store_true", help="also dump per-frame images alongside labels")

    # Optional visualization of derived keypoints
    ap.add_argument("--draw_kpts", action="store_true")

    # Progress printing
    ap.add_argument("--progress_every", type=int, default=50, help="print progress every N frames")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = YOLO(args.weights)
    try:
        model.to(args.device)
        if args.half and str(args.device) != "cpu":
            try:
                model.model.half()
                print("FP16 enabled.")
            except Exception:
                pass
    except Exception:
        pass

    # Names map (Ultralytics-style)
    raw_names = getattr(model.model, "names", None)
    names_map = {int(k): v for k, v in raw_names.items()} if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}

    # Video IO
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    H0, W0 = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    base = Path(args.video).stem
    out_ext = _choose_ext(args.codec)
    out_path = os.path.join(args.out_dir, f"annotated_{base}{out_ext}")
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W0, H0))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    # Directories for optional per-frame saves
    img_dir = os.path.join(args.out_dir, "images")
    lab_dir = os.path.join(args.out_dir, "labels")
    if args.save_frames:
        os.makedirs(img_dir, exist_ok=True)
    if args.save_txt:
        os.makedirs(lab_dir, exist_ok=True)

    print(f"▶️  Processing '{args.video}' ({W0}x{H0} @ {fps:.2f} fps, frames={total_frames or 'unknown'})")
    print(f"    Weights: {args.weights}")
    print(f"    pre_clahe={args.pre_clahe}  clahe_clip={args.clahe_clip}  retina_masks={args.retina_masks}  half={args.half}")
    t0 = time.time()
    idx = 0
    nh_total, t_total = 0, 0  # cumulative detection counters

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional pre-CLAHE
        frame_in = enhance_like_infer(frame, args.clahe_clip) if args.pre_clahe else frame

        # Ultralytics predict on the (possibly enhanced) frame
        res = model.predict(
            source=frame_in,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
            retina_masks=args.retina_masks,
            agnostic_nms=args.agnostic_nms,
        )[0]

        # Per-frame detection counts by class-kind
        nh_frame = 0
        t_frame  = 0
        try:
            if res.boxes is not None and len(res.boxes) > 0:
                cls_arr = res.boxes.cls.int().cpu().numpy().tolist()
                for c in cls_arr:
                    kind = kind_from_name(names_map.get(int(c), str(c)))
                    if kind == "NH":
                        nh_frame += 1
                        nh_total += 1
                    else:
                        t_frame += 1
                        t_total += 1
        except Exception:
            pass

        # Visualization (on the same frame we passed to predict)
        vis = res.plot()  # BGR

        # Optional: derived keypoints overlay (uses masks → polygons)
        if args.draw_kpts and res.masks is not None and len(res.masks) > 0:
            H, W = vis.shape[:2]
            cls = res.boxes.cls.cpu().numpy().astype(int)
            xyn_list = res.masks.xyn
            for i in range(len(cls)):
                parts = xyn_list[i] if isinstance(xyn_list[i], list) else [xyn_list[i]]
                if not parts:
                    continue
                P = []
                for p in parts:
                    if p is None or len(p) < 3:
                        continue
                    p = np.asarray(p, dtype=np.float32)
                    p[:, 0] *= W
                    p[:, 1] *= H
                    P.append(p)
                if not P:
                    continue
                P = np.concatenate(P, axis=0)
                kind = kind_from_name(names_map.get(int(cls[i]), str(cls[i])))
                kps = keypoints_from_points(P, H, W, kind, min_pts=12)
                if kps is not None:
                    draw_kpts(vis, kps, kind)

        # Write video frame
        writer.write(vis)

        # Optional: save per-frame TXT and images
        stem = f"{base}_{idx:06d}"
        if args.save_txt:
            write_seg_txt(res, names_map, lab_dir, stem, save_conf=args.save_conf)
        if args.save_frames:
            cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), vis)

        # Progress print
        if (idx % args.progress_every == 0) or (total_frames and idx + 1 == total_frames):
            elapsed = time.time() - t0
            fps_live = (idx + 1) / max(1e-6, elapsed)
            if total_frames > 0:
                pct = 100.0 * (idx + 1) / total_frames
                eta = (total_frames - (idx + 1)) / max(1e-6, fps_live)
                print(f"[{idx+1:6d}/{total_frames}] {pct:5.1f}% | fps={fps_live:5.1f} | "
                      f"elapsed={_fmt_secs(elapsed)} | eta={_fmt_secs(eta)} | NH={nh_frame} T={t_frame}")
            else:
                print(f"[{idx+1:6d}] fps={fps_live:5.1f} | elapsed={_fmt_secs(elapsed)} | NH={nh_frame} T={t_frame}")

        idx += 1

    cap.release()
    writer.release()
    total_elapsed = time.time() - t0
    avg_fps = (idx / max(1e-6, total_elapsed))
    print(f"✅ Output saved to: {out_path}")
    if args.save_txt:
        print(f"📝 Per-frame labels → {lab_dir}")
    if args.save_frames:
        print(f"🖼  Per-frame images → {img_dir}")
    print(f"📊 Done: frames={idx} | elapsed={_fmt_secs(total_elapsed)} | avg_fps={avg_fps:.1f} | NH={nh_total} T={t_total}")

if __name__ == "__main__":
    main()