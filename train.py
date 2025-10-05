#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — YOLO-SEG (NH,T) with aligned preprocessing, safer OR augs,
polygon regularization, percentile-based keypoints, and union-NMS derivation.

Outputs:
- Normal YOLO-SEG training.
- After training, derives class-aware 7-kp geometry from predicted polygons.
- (Optional) writes padded 7-kpt YOLO-Pose labels (NH: base padded, T: rings padded).
"""

from __future__ import annotations
import argparse, json, os, random, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ==========================
# Repro
# ==========================
def set_seed_all(seed=0):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

# ==========================
# Names → kind
# ==========================
def kind_from_name(name: str) -> str:
    return "NH" if str(name).lower().startswith("nh") else "T"

# ==========================
# CLAHE (kept light & shared with inference)
# ==========================
def enhance_like_infer(img_bgr: np.ndarray, clip=2.0) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(8, 8))
    y = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

# ==========================
# OR augmentations (milder)
# ==========================
def _rp(p):  # unchanged
    import random
    return random.random() < p

def add_gamma(img, g_low=0.8, g_high=1.3):
    gamma = random.uniform(g_low, g_high); inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)

def add_color_cast(img, var=10):
    b, g, r = cv2.split(img.astype(np.int16))
    b = np.clip(b + random.randint(-var, var), 0, 255)
    g = np.clip(g + random.randint(-var, var), 0, 255)
    r = np.clip(r + random.randint(-var, var), 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

def add_jpeg(img, q_low=55, q_high=85):
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(q_low, q_high)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def add_motion_blur(img, max_ks=13):
    k = random.randrange(3, max_ks | 1, 2)
    kernel = np.zeros((k, k), np.float32); kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, M, (k, k)); kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)

def add_defocus(img, radius_max=4):
    k = random.randint(3, radius_max * 2 + 1) | 1
    kernel = np.zeros((k, k), np.float32)
    cv2.circle(kernel, (k // 2, k // 2), k // 2, 1, -1); kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)

def add_spotlight(img, alpha=0.25, rmin=120, rmax=380):
    """Simulate OR light hotspots / specular bloom."""
    h, w = img.shape[:2]
    cx = np.random.randint(int(0.2*w), int(0.8*w))
    cy = np.random.randint(int(0.2*h), int(0.8*h))
    r  = np.random.randint(rmin, rmax)
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), r*0.45)
    mask = np.clip(mask, 0, 1)[:, :, None]
    return cv2.addWeighted(img, 1.0, np.full_like(img, 255), alpha, 0, dst=None, mask=(mask*255).astype(np.uint8))

def add_shadow_band(img, alpha=0.35, wmin=80, wmax=260):
    """Arm/hand shadows sweeping across the field."""
    h, w = img.shape[:2]
    bw = np.random.randint(wmin, wmax)
    ang = np.random.uniform(0, 180)
    x = np.random.randint(-bw, w+bw)
    y = np.random.randint(-bw, h+bw)
    rect = ((x, y), (bw, max(h, w)*1.6), ang)
    mask = np.zeros((h, w), np.uint8)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    mask = cv2.GaussianBlur(mask, (0, 0), 21)
    dark = (img.astype(np.float32) * (1.0 - alpha*(mask[:, :, None]/255.0))).astype(np.uint8)
    return dark

def add_vignette(img, strength=0.25):
    h, w = img.shape[:2]
    xv = cv2.getGaussianKernel(w, w*0.6)
    yv = cv2.getGaussianKernel(h, h*0.6)
    mask = (yv @ xv.T)
    mask = (mask - mask.min())/(mask.max()-mask.min())
    mask = (1.0 - strength* (1.0 - mask))[:, :, None]
    return np.clip(img.astype(np.float32)*mask, 0, 255).astype(np.uint8)

def add_random_lines(img, nmin=2, nmax=7):
    """Suture/wire/cable clutter."""
    h, w = img.shape[:2]
    canvas = img.copy()
    k = np.random.randint(nmin, nmax+1)
    palette = [(40,40,40),(210,200,180),(200,80,80),(60,120,220)]  # dark, steel, blood-ish, drape seam
    for _ in range(k):
        p1 = (np.random.randint(-w//8, w+w//8), np.random.randint(0, h))
        p2 = (np.random.randint(-w//8, w+w//8), np.random.randint(0, h))
        t  = np.random.randint(1, 3)
        col= palette[np.random.randint(0, len(palette))]
        cv2.line(canvas, p1, p2, col, t, cv2.LINE_AA)
    return cv2.GaussianBlur(canvas, (3,3), 0.7)

def add_white_balance(img, ab=8):
    """Subtle WB swing toward blue/green drape or warm skin."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    a_shift = np.random.randint(-ab, ab)
    b_shift = np.random.randint(-ab, ab)
    lab[:,:,1] = np.clip(lab[:,:,1] + a_shift, 0, 255)
    lab[:,:,2] = np.clip(lab[:,:,2] + b_shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def add_sensor_noise(img, sigma=6.0):
    n = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

def or_augment_image_bgr(img_bgr, strength=1.0):
    """
    Heavier OR domain aug:
    - color/WB shifts, gamma, JPEG
    - glare hotspots, shadow bands, vignette
    - thin line clutter (sutures/cables)
    - mild motion/defocus + noise
    Probabilities scale with 'strength'.
    """
    p = lambda x: min(1.0, x * strength)

    img = img_bgr.copy()
    # base photometric
    if _rp(p(0.55)): img = add_white_balance(img, ab=10)
    if _rp(p(0.50)): img = add_gamma(img, 0.80, 1.30)
    if _rp(p(0.45)): img = add_color_cast(img, var=10)
    if _rp(p(0.35)): img = add_jpeg(img, 55, 85)

    # OR-specific scene effects
    if _rp(p(0.40)): img = add_spotlight(img, alpha=np.random.uniform(0.18, 0.38))
    if _rp(p(0.35)): img = add_shadow_band(img, alpha=np.random.uniform(0.25, 0.45))
    if _rp(p(0.30)): img = add_vignette(img, strength=np.random.uniform(0.15, 0.35))
    if _rp(p(0.35)): img = add_random_lines(img, nmin=2, nmax=8)

    # mild blur/noise (keep geometry intact)
    if _rp(p(0.18)): img = add_motion_blur(img, 11)
    if _rp(p(0.14)): img = add_defocus(img, 4)
    if _rp(p(0.30)): img = add_sensor_noise(img, sigma=np.random.uniform(3.0, 9.0))

    return img

# one safe hook with an internal guard
def make_train_callbacks(state, use_clahe=True, clahe_clip=2.0):
    """
    Create two YOLOv8 training callbacks:
    - on_train_epoch_start(tr): adjust OR aug strength over epochs
    - on_preprocess_end(tr): apply CLAHE + OR aug to batch images
    state: dict with keys:
        "epochs": int, total epochs
        "warmup_epochs": int, epochs with no OR aug
        "base_strength": float, initial OR aug strength
        "max_strength": float, final OR aug strength
        "strength": float, current OR aug strength (updated each epoch)
    use_clahe: bool, whether to apply CLAHE before OR aug
    clahe_clip: float, CLAHE clip limit
    Returns: (on_train_epoch_start, on_preprocess_end) callbacks
    """
    def on_train_epoch_start(tr):
        e = getattr(tr, "epoch", 0)
        ramp = max(1, int(0.4 * state["epochs"]))
        if e < state["warmup_epochs"]:
            frac = 0.0
        else:
            frac = min(1.0, (e - state["warmup_epochs"]) / ramp)
        state["strength"] = state["base_strength"] + (state["max_strength"] - state["base_strength"]) * frac

    def on_preprocess_end(trainer):
        # guard
        batch = getattr(trainer, "batch", None)
        if batch is None or batch.get("_or_aug_done", False):
            return
        try:
            import torch
            imgs = batch["img"]  # BCHW RGB in [0,1]
            with torch.no_grad():
                arr = (imgs.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                out = np.empty_like(arr)
                for i in range(arr.shape[0]):
                    rgb = np.transpose(arr[i], (1,2,0))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if use_clahe:
                        bgr = enhance_like_infer(bgr, clip=clahe_clip)
                    bgr = or_augment_image_bgr(bgr, state["strength"])
                    rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    out[i] = np.transpose(rgb2, (2,0,1))
                batch["img"] = torch.from_numpy(out.astype(np.float32)/255.0).to(imgs.device)
                batch["_or_aug_done"] = True
        except Exception:
            return
    return on_train_epoch_start, on_preprocess_end

# ==========================
# Geometry & polygon utils
# ==========================
def regularize_poly(pts: np.ndarray, eps_frac=0.003, max_pts=150) -> np.ndarray:
    """Douglas–Peucker + de-dup → stable PCA."""
    if pts is None or len(pts) < 3:
        return pts
    peri = cv2.arcLength(pts.reshape(-1,1,2).astype(np.float32), True)
    eps = max(0.8, eps_frac * peri)
    approx = cv2.approxPolyDP(pts.reshape(-1,1,2).astype(np.float32), eps, True).reshape(-1,2)
    if len(approx) > max_pts:
        step = int(np.ceil(len(approx)/max_pts)); approx = approx[::step]
    dedup = [approx[0]]
    for p in approx[1:]:
        if np.linalg.norm(p - dedup[-1]) >= 0.5:
            dedup.append(p)
    return np.asarray(dedup, np.float32)

def _pca_axes(P: np.ndarray):
    mean = P.mean(0); A = P - mean; C = np.cov(A.T)
    evals, evecs = np.linalg.eigh(C)
    major = evecs[:, int(np.argmax(evals))]; minor = evecs[:, int(np.argmin(evals))]
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
    d2 = ((pts - proj)**2).sum(1)
    return pts[int(np.argmax(d2))].copy()

def keypoints_from_points(P: np.ndarray, img_h: int, img_w: int, kind: str, min_pts: int = 8,
                          vis_eps: float = 0.5) -> Optional[Dict[str, List[float]]]:
    """Percentile slabs + visibility hysteresis; robust base/tip vote."""
    if P is None or len(P) < min_pts:
        return None
    P = regularize_poly(P)

    mean, shaft, side, span, s_all = _pca_axes(P)
    smin, smax = float(s_all.min()), float(s_all.max())

    # end slabs: 12th percentile of span (min 3 px)
    slab_w = max(3.0, 0.12 * span)
    near_min = P[s_all <= smin + slab_w]
    near_max = P[s_all >= smax - slab_w]

    t_min = _thickness(near_min, side); t_max = _thickness(near_max, side)

    # curvature vote (base often less curved): quick proxy via PCA residual
    def residual_curv(Q): 
        if len(Q) < 4: return 1e9
        _, u, _, _, sQ = _pca_axes(Q);  # reuse axis
        return float(np.var(((Q - Q.mean(0)) @ u) * 0.0 + ((Q - Q.mean(0)) @ side)))

    c_min, c_max = residual_curv(near_min), residual_curv(near_max)
    # votes: thickness, curvature
    vote_max = int(t_max >= t_min) + int(c_max <= c_min)
    vote_min = int(t_min >  t_max) + int(c_min <  c_max)
    base_first = vote_max >= vote_min

    if base_first:
        base = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]; tip = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]
        base_t, tip_t = smax, smin
    else:
        base = near_min.mean(0) if len(near_min) else P[np.argmin(s_all)]; tip = near_max.mean(0) if len(near_max) else P[np.argmax(s_all)]
        base_t, tip_t = smin, smax

    off_px = 0.02 * max(img_h, img_w)
    shaft_left  = mean - off_px * side
    shaft_right = mean + off_px * side
    tip_split   = 0.01 * max(img_h, img_w)
    tip_left  = tip - tip_split * side
    tip_right = tip + tip_split * side

    # ring region = points within 35% of span from base (percentile slab)
    lo, hi = (base_t, base_t + 0.35 * span) if base_t <= (smin + smax) / 2 else (base_t - 0.35 * span, base_t)
    mask = (s_all >= min(lo, hi)) & (s_all <= max(lo, hi))
    slab_pts = P[mask] if int(mask.sum()) >= 20 else P

    if kind == "NH":
        side_sign = (slab_pts - slab_pts.mean(0)) @ side
        left_pts  = slab_pts[side_sign < 0]; right_pts = slab_pts[side_sign >= 0]
        ring_left  = _farthest_from_shaft(left_pts,  base, shaft)
        ring_right = _farthest_from_shaft(right_pts, base, shaft)
    else:
        shaft_dir  = shaft if base_t > tip_t else -shaft
        anchor     = base + shaft_dir * (0.06 * span)
        side_d     = 0.025 * max(img_h, img_w)
        ring_right = anchor + side * side_d
        ring_left  = anchor - side * side_d

    def _clamp(p):
        return [float(np.clip(p[0], -vis_eps, img_w - 1 + vis_eps)),
                float(np.clip(p[1], -vis_eps, img_h - 1 + vis_eps))]

    return {
        "tip_right":   _clamp(tip_right),
        "tip_left":    _clamp(tip_left),
        "shaft_right": _clamp(shaft_right),
        "shaft_left":  _clamp(shaft_left),
        "ring_right":  _clamp(ring_right),
        "ring_left":   _clamp(ring_left),
        "base":        _clamp(base),
    }

# ==========================
# Result → polys/boxes (frame coords; letterbox-safe)
# ==========================
def poly_in_frame(res, i_det: int, W0: int, H0: int) -> Optional[np.ndarray]:
    """
    Extract polygon for detection i_det from results res, scaled to W0 x H0 frame.
    Returns Nx2 float32 array or None if no polygon found.
    """
    # prefer normalized polygons if available
    try:
        parts = res.masks.xyn[i_det]; parts = parts if isinstance(parts, list) else [parts]
        Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
        pts = [np.column_stack((p[:,0]*Ws*sx, p[:,1]*Hs*sy)).astype(np.float32) for p in parts if p is not None and len(p) >= 3]
        if pts: return regularize_poly(np.concatenate(pts, 0))
    except Exception: pass
    try:
        parts = res.masks.xy[i_det]; parts = parts if isinstance(parts, list) else [parts]
        Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
        pts = [np.column_stack((p[:,0]*sx, p[:,1]*sy)).astype(np.float32) for p in parts if p is not None and len(p) >= 3]
        if pts: return regularize_poly(np.concatenate(pts, 0))
    except Exception: pass
    try:
        m = res.masks.data[i_det].cpu().numpy().astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea).reshape(-1,2).astype(np.float32)
        Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
        cnt[:,0]*=sx; cnt[:,1]*=sy; return regularize_poly(cnt)
    except Exception:
        return None

def box_in_frame(res, i_det: int, W0: int, H0: int) -> List[float]:
    """
    Extract bounding box for detection i_det from results res, scaled to W0 x H0 frame. 
    Returns [x1,y1,x2,y2] as floats.
    """
    try:
        x1n, y1n, x2n, y2n = res.boxes.xyxyn[i_det]
        return [float(x1n*W0), float(y1n*H0), float(x2n*W0), float(y2n*H0)]
    except Exception:
        pass
    x1, y1, x2, y2 = res.boxes.xyxy[i_det].cpu().numpy().astype(float)
    Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
    return [x1*sx, y1*sy, x2*sx, y2*sy]

# ==========================
# Post-train derivation helpers
# ==========================
def iou_xyxy(a, b):
    """
    IoU of two [x1,y1,x2,y2] boxes.
    0.0 if no overlap or invalid boxes.
    """
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    iw = max(0.0, min(ax2,bx2)-max(ax1,bx1))
    ih = max(0.0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih; ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def nms_union(items, iou_thr=0.6):
    """
    Non-maximum suppression in frame space (boxes).
    items: list of dict with keys "cls", "conf", "poly", "xyxy"
    Returns: filtered list of items
    """
    keep = []
    items = sorted(items, key=lambda x: x["conf"], reverse=True)
    for it in items:
        if all(iou_xyxy(it["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(it)
    return keep

def list_images(dir_path: Path) -> List[Path]:
    """
    Recursively list all image files in dir_path.
    Returns sorted list of Paths.
    """
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in exts])

ORDER7 = ["tip_right","tip_left","shaft_right","shaft_left","ring_right","ring_left","base"]
def _norm_xywh(xyxy, W, H):
    """
    Convert [x1,y1,x2,y2] box to normalized [cx,cy,w,h].
    """
    x1,y1,x2,y2 = map(float, xyxy)
    w=max(1e-6,x2-x1); h=max(1e-6,y2-y1)
    cx=(x1+x2)*0.5; cy=(y1+y2)*0.5
    return cx/W, cy/H, w/W, h/H

def pose_line_padded(kind: str, xyxy, kps: Dict[str,List[float]], W: int, H: int) -> str:
    """
    Create a padded YOLO-Pose label line (string) for one object.
    kind: "NH" or "T"
    xyxy: [x1,y1,x2,y2] box in frame coords
    kps: dict of 7 keypoints {name:[x_pix,y_pix]}
    W, H: image width and height
    Returns: string line or raises Exception on error.
    """
    
    cls01 = 0 if kind=="NH" else 1
    cx,cy,w,h = _norm_xywh(xyxy, W, H)
    parts = [str(cls01), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for name in ORDER7:
        if (kind=="T" and name in ("ring_right","ring_left")) or (kind=="NH" and name=="base"):
            xn, yn, v = 0.0, 0.0, 0
        else:
            x_pix, y_pix = kps[name]
            v = 2 if (-0.5 <= x_pix <= W-0.5 and -0.5 <= y_pix <= H-0.5) else 1
            xn = float(np.clip(x_pix / W, 0.0, 1.0)); yn = float(np.clip(y_pix / H, 0.0, 1.0))
        parts.extend([f"{xn:.6f}", f"{yn:.6f}", str(v)])
    return " ".join(parts)

def predict_union(model: YOLO, frame_bgr: np.ndarray, scales: Tuple[float,...], confs: Tuple[float,...],
                  imgsz: int, iou: float, device: str):
    """
    Multi-scale, multi-confidence prediction.
    Returns: list of dict with keys "cls", "conf", "poly", "xyxy"
    """
    H0, W0 = frame_bgr.shape[:2]
    cands = []
    for s in scales:
        proc = frame_bgr if abs(s-1.0)<1e-6 else cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for c in confs:
            r = model.predict(source=proc, imgsz=imgsz, conf=c, iou=iou, device=device,
                              verbose=False, retina_masks=True, agnostic_nms=False)[0]
            if r.masks is None or len(r.masks)==0: 
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confb = r.boxes.conf.cpu().numpy().astype(float)
            for i in range(len(cls)):
                poly = poly_in_frame(r, i, W0, H0)
                if poly is None or len(poly) < 3:
                    continue
                xyxy = box_in_frame(r, i, W0, H0)
                cands.append({"cls": int(cls[i]), "conf": float(confb[i]), "poly": poly, "xyxy": xyxy})
    return nms_union(cands, iou_thr=0.6)

def derive_for_split(model: YOLO, split_dir: Path, names_map: Dict[int,str], out_root: Path,
                     imgsz: int, iou: float, device: str,
                     scales: Tuple[float,...], confs: Tuple[float,...],
                     write_pose: bool, pose_out: Optional[Path], pose_copy_images: bool):
    """
    Derive geometry for all images in split_dir using model.
    Saves JSON and VIS in out_root/{json,vis}.
    Optionally saves padded YOLO-Pose labels and images in pose_out.
    """

    json_dir = out_root / "json"; json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir  = out_root / "vis";  vis_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    if write_pose and pose_out is not None:
        (pose_out / "images").mkdir(parents=True, exist_ok=True)
        (pose_out / "labels").mkdir(parents=True, exist_ok=True)

    for img_path in list_images(split_dir):
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None: 
            continue
        H0, W0 = frame.shape[:2]
        dets = predict_union(model, frame, scales, confs, imgsz, iou, device)

        items = []
        for d in dets:
            raw_name = names_map.get(d["cls"], str(d["cls"]))
            kind = kind_from_name(raw_name)
            kps = keypoints_from_points(d["poly"], H0, W0, kind)
            if kps is None: 
                continue
            items.append({"kind": kind, "xyxy": d["xyxy"], "kps": kps})

        # Save JSON & VIS
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        vis_list = []
        for it in items:
            vis_list.append({"kind": it["kind"],
                             "keypoints": {k:[float(x), float(y), 2] for k,(x,y) in it["kps"].items()}})
            # simple preview
            r=4
            for k,(x,y) in it["kps"].items():
                draw.ellipse([x-r,y-r,x+r,y+r], outline=(0,170,255), width=2)
            bx, by = it["kps"]["base"]
            try:
                bb = draw.textbbox((bx, by), it["kind"], font=font)
                draw.rectangle(bb, fill=(0,0,0,128))
            except Exception: pass
            draw.text((bx, by), it["kind"], fill=(255,255,255), font=font)

        stem = img_path.stem
        (out_root/"json"/f"{stem}.json").write_text(json.dumps(vis_list, indent=2))
        pil.save(out_root/"vis"/f"{stem}.png", "PNG", optimize=True)

        # Optional pose dataset
        if write_pose and pose_out is not None:
            if "train" in str(split_dir): split_sub="train"
            elif "val" in str(split_dir): split_sub="val"
            else: split_sub="extra"
            img_dst_dir = pose_out/"images"/split_sub
            lab_dst_dir = pose_out/"labels"/split_sub
            img_dst_dir.mkdir(parents=True, exist_ok=True); lab_dst_dir.mkdir(parents=True, exist_ok=True)

            dst_img = img_dst_dir/(stem + img_path.suffix.lower())
            if pose_copy_images:
                if not dst_img.exists(): shutil.copy2(img_path, dst_img)
            else:
                try:
                    if dst_img.exists(): dst_img.unlink()
                    os.symlink(img_path, dst_img)
                except Exception:
                    if not dst_img.exists(): shutil.copy2(img_path, dst_img)

            with open(lab_dst_dir/f"{stem}.txt","w") as f:
                for it in items:
                    f.write(pose_line_padded(it["kind"], it["xyxy"], it["kps"], W0, H0) + "\n")

# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    # training
    ap.add_argument("--data",   default="/home/student/project/output/yolo_data_seg/data.yaml")
    ap.add_argument("--model",  default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--imgsz",  type=int, default=1280)
    ap.add_argument("--batch",  type=int, default=-1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="seg_phaseB")
    ap.add_argument("--name",    default="yolo11s_seg_or_aug_2super")

    # curriculum & CLAHE alignment
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--base_strength", type=float, default=0.65)
    ap.add_argument("--max_strength",  type=float, default=1.10)
    ap.add_argument("--clahe_clip", type=float, default=2.0)

    # derivation
    ap.add_argument("--derive_on", choices=["none","val","train","both"], default="none")
    ap.add_argument("--derive_imgsz", type=int, default=1536)
    ap.add_argument("--derive_iou",  type=float, default=0.55)
    ap.add_argument("--scales", type=str, default="1.25,1.0")
    ap.add_argument("--conf_ladder", type=str, default="0.36,0.32,0.28")

    # pose export
    ap.add_argument("--write_pose", action="store_true")
    ap.add_argument("--pose_out", type=str, default="/home/student/project/output/yolo_data")
    ap.add_argument("--pose_copy_images", action="store_true")

    args = ap.parse_args()
    set_seed_all(0)

    # ---- Train SEG (2 classes: NH,T) ----
    model = YOLO(args.model)
    on_epoch_start, on_preproc = make_train_callbacks(
        {"epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
         "base_strength": args.base_strength, "max_strength": args.max_strength,
         "strength": args.base_strength},
        use_clahe=True, clahe_clip=args.clahe_clip
    )
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_preprocess_end", on_preproc)  # single safe hook

    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project, name=args.name, pretrained=True, amp=True,
        optimizer="AdamW", lr0=9e-4, weight_decay=0.01, cos_lr=True,
        warmup_epochs=5, patience=12,
        mosaic=0.01, mixup=0.00, close_mosaic=max(8, args.epochs // 12),
        degrees=5.0, translate=0.03, scale=0.12, shear=0.25, perspective=0.0008,
        fliplr=0.20, flipud=0.00,
        hsv_h=0.010, hsv_s=0.26, hsv_v=0.26,
        erasing=0.14,
        workers=8, cache=True
    )
    

    # ---- Reload best & derive KP from polygons with union-NMS ----
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    seg = YOLO(str(best))
    raw_names = seg.model.names
    names_map = ({int(k): v for k, v in raw_names.items()}
                 if isinstance(raw_names, dict)
                 else {i: n for i, n in enumerate(raw_names)})
    assert len(names_map) == 2 and all(str(v).lower().startswith(("nh","t")) for v in names_map.values()), \
        f"Expected 2 classes (NH,T), got {names_map}"

    save_root = Path(model.trainer.save_dir) / "derived_kpts"
    ds_root = Path(args.data).parent
    imgs_train = ds_root / "images" / "train"
    imgs_val   = ds_root / "images" / "val"

    scales = tuple(float(s.strip()) for s in args.scales.split(",") if s.strip())
    confs  = tuple(float(s.strip()) for s in args.conf_ladder.split(",") if s.strip())

    if args.derive_on in ("train","both") and imgs_train.exists():
        derive_for_split(seg, imgs_train, names_map, save_root/"train",
            imgsz=args.derive_imgsz, iou=args.derive_iou, device=args.device,
            scales=scales, confs=confs,
            write_pose=args.write_pose, pose_out=Path(args.pose_out),
            pose_copy_images=args.pose_copy_images)

    if args.derive_on in ("val","both") and imgs_val.exists():
        derive_for_split(seg, imgs_val, names_map, save_root/"val",
            imgsz=args.derive_imgsz, iou=args.derive_iou, device=args.device,
            scales=scales, confs=confs,
            write_pose=args.write_pose, pose_out=Path(args.pose_out),
            pose_copy_images=args.pose_copy_images)

    print("\n✅ Done.")
    print(f"  Weights:     {best}")
    if (save_root/"val").exists() or (save_root/"train").exists():
        print(f"  Derived KPs: {save_root}")
    if args.write_pose:
        print(f"  Pose data →  {args.pose_out} (images/ & labels/ with padded 7-kpt)")

if __name__ == "__main__":
    main()