#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
refine.py — YOLO-SEG (NH,T) refinement + optional one-shot self-training.

- Starts from a given best.pt (pretrained=False).
- Light augs, lower LR, AdamW, cosine LR, short warmup.
- RAM-friendly: cache='disk' (or False) and fewer workers.
- Optional stage-1 backbone freeze for stability.
- Optional self-training: re-predict masks for the train split, write fresh
  Ultralytics seg .txt (normalized), then do a short top-up train.

Assumptions:
- data.yaml has exactly two classes: 0: NH, 1: T
- Your curated labels are normalized to [0,1]

Usage (refine only):
  python refine.py \
    --data /home/student/project/pseudo_ultra_ds/data.yaml \
    --base_weights /home/student/project/seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/weights/best.pt \
    --epochs 60 --imgsz 1280 --device 0 \
    --project seg_phaseB --name refined_seg

Usage (refine + one-shot self-train):
  python refine.py \
    --data /home/student/project/pseudo_ultra_ds/data.yaml \
    --base_weights /home/student/project/seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/weights/best.pt \
    --epochs 50 --topup_epochs 12 \
    --self_train --self_min_conf_nh 0.55 --self_min_conf_t 0.45 \
    --imgsz 1280 --device 0 --project seg_phaseB --name refined_seg_self
"""

from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Basics
def set_seed_all(seed=0):
    import random, numpy as _np
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(seed); _np.random.seed(seed)

def kind_from_name(name: str) -> str:
    return "NH" if str(name).lower().startswith("nh") else "T"

# ---------------- Safe polygon utils (letterbox/version tolerant)
def regularize_poly(pts: np.ndarray, eps_frac=0.003, max_pts=150) -> np.ndarray:
    if pts is None or len(pts) < 3: return pts
    peri = cv2.arcLength(pts.reshape(-1,1,2).astype(np.float32), True)
    eps  = max(0.8, eps_frac * peri)
    approx = cv2.approxPolyDP(pts.reshape(-1,1,2).astype(np.float32), eps, True).reshape(-1,2)
    if len(approx) > max_pts:
        step = int(np.ceil(len(approx)/max_pts)); approx = approx[::step]
    # de-dup
    out = [approx[0]]
    for p in approx[1:]:
        if np.linalg.norm(p - out[-1]) >= 0.5: out.append(p)
    return np.asarray(out, np.float32)

def poly_in_frame(res, i_det: int, W0: int, H0: int) -> Optional[np.ndarray]:
    try:  # normalized polys
        parts = res.masks.xyn[i_det]; parts = parts if isinstance(parts, list) else [parts]
        Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
        pts = [np.column_stack((p[:,0]*Ws*sx, p[:,1]*Hs*sy)).astype(np.float32) for p in parts if p is not None and len(p)>=3]
        if pts: return regularize_poly(np.concatenate(pts, 0))
    except Exception: pass
    try:  # absolute polys (proc space)
        parts = res.masks.xy[i_det]; parts = parts if isinstance(parts, list) else [parts]
        Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
        pts = [np.column_stack((p[:,0]*sx, p[:,1]*sy)).astype(np.float32) for p in parts if p is not None and len(p)>=3]
        if pts: return regularize_poly(np.concatenate(pts, 0))
    except Exception: pass
    try:  # bitmap fallback
        m = res.masks.data[i_det].cpu().numpy().astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea).reshape(-1,2).astype(np.float32)
        Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
        cnt[:,0]*=sx; cnt[:,1]*=sy; return regularize_poly(cnt)
    except Exception:
        return None

def box_in_frame(res, i_det: int, W0: int, H0: int) -> List[float]:
    try:
        x1n,y1n,x2n,y2n = res.boxes.xyxyn[i_det]
        return [float(x1n*W0), float(y1n*H0), float(x2n*W0), float(y2n*H0)]
    except Exception:
        pass
    x1,y1,x2,y2 = res.boxes.xyxy[i_det].cpu().numpy().astype(float)
    Hs, Ws = res.orig_shape; sx, sy = W0/float(Ws), H0/float(Hs)
    return [x1*sx, y1*sy, x2*sx, y2*sy]

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    iw=max(0.0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0.0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def nms_union(items, thr=0.6):
    keep=[]
    items=sorted(items, key=lambda x:x["conf"], reverse=True)
    for it in items:
        if all(iou_xyxy(it["xyxy"],k["xyxy"])<thr for k in keep): keep.append(it)
    return keep

def list_images(dir_path: Path) -> List[Path]:
    exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in exts])

# ---------------- Self-training: predict → write seg labels
def write_seg_label_txt(out_path: Path, cls_id: int, poly_xy: np.ndarray, W: int, H: int):
    # write in Ultralytics seg format (normalized, no confidence): "cls x1 y1 x2 y2 ..."
    p = poly_xy.copy().astype(np.float32)
    p[:,0] = np.clip(p[:,0]/max(1.0,W), 0, 1)
    p[:,1] = np.clip(p[:,1]/max(1.0,H), 0, 1)
    flat = " ".join(f"{v:.6f}" for v in p.reshape(-1))
    with open(out_path, "a") as f:
        f.write(f"{cls_id} {flat}\n")

def harvest_labels(model: YOLO, images_dir: Path, labels_dir: Path,
                   imgsz: int, iou: float,
                   scales=(1.25,1.0), confs=(0.36,0.32,0.28),
                   min_conf_nh=0.55, min_conf_t=0.45):
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw_names = model.model.names
    names_map = ({int(k):v for k,v in raw_names.items()} if isinstance(raw_names, dict)
                 else {i:n for i,n in enumerate(raw_names)})

    for img_path in list_images(images_dir):
        frame = cv2.imread(str(img_path)); 
        if frame is None: continue
        H0,W0 = frame.shape[:2]
        cands=[]
        for s in scales:
            proc = frame if abs(s-1.0)<1e-6 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for c in confs:
                r = model.predict(source=proc, imgsz=imgsz, conf=c, iou=iou,
                                  verbose=False, retina_masks=True, agnostic_nms=False)[0]
                if r.masks is None or len(r.masks)==0: continue
                cls = r.boxes.cls.cpu().numpy().astype(int)
                confb = r.boxes.conf.cpu().numpy().astype(float)
                for i in range(len(cls)):
                    poly = poly_in_frame(r, i, W0, H0)
                    if poly is None or len(poly)<3: continue
                    cands.append({"cls": int(cls[i]),
                                  "conf": float(confb[i]),
                                  "poly": poly,
                                  "xyxy": box_in_frame(r, i, W0, H0)})

        keep = nms_union(cands, thr=0.6)
        # overwrite label
        lab_path = labels_dir / (img_path.stem + ".txt")
        if lab_path.exists(): lab_path.unlink()
        for k in keep:
            nm = names_map.get(k["cls"], str(k["cls"]))
            kind = "NH" if str(nm).lower().startswith("nh") else "T"
            if kind=="NH" and k["conf"]<min_conf_nh: continue
            if kind=="T"  and k["conf"]<min_conf_t:  continue
            cls_id = 0 if kind=="NH" else 1
            write_seg_label_txt(lab_path, cls_id, k["poly"], W0, H0)

# ---------------- Main refine
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, default="/home/student/project/pseudo_ultra_ds/data.yaml")  # path to your data.yaml
    ap.add_argument("--base_weights", required=True, default="/home/student/project/seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/weights/best.pt")  # path to your previous best.pt
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="seg_phaseB")
    ap.add_argument("--name",    default="refined_seg")

    # training hygiene
    ap.add_argument("--cache", default="ram", help="disk|ram|False")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--freeze_backbone", type=int, default=0, help="e.g., 10 to freeze first 10 layers for 5 quick epochs")
    ap.add_argument("--freeze_epochs", type=int, default=5)

    # optional self-training (one-shot)
    ap.add_argument("--self_train", action="store_true")
    ap.add_argument("--topup_epochs", type=int, default=12)
    ap.add_argument("--self_imgsz", type=int, default=1536)
    ap.add_argument("--self_iou", type=float, default=0.55)
    ap.add_argument("--self_scales", type=str, default="1.25,1.0")
    ap.add_argument("--self_confs",  type=str, default="0.36,0.32,0.28")
    ap.add_argument("--self_min_conf_nh", type=float, default=0.5)
    ap.add_argument("--self_min_conf_t",  type=float, default=0.4)

    args = ap.parse_args()
    set_seed_all(0)

    # ---------------- stage A: refinement from base weights
    model = YOLO(args.base_weights)
    # light augs good for pseudo-labels; keep flips, tone down geometry
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project, name=args.name,
        pretrained=False, amp=True,
        optimizer="AdamW", lr0=args.lr0, weight_decay=args.weight_decay, cos_lr=True,
        warmup_epochs=args.warmup_epochs, patience=args.patience,
        mosaic=0.0, mixup=0.0, close_mosaic=0,
        degrees=3.0, translate=0.02, scale=0.08, shear=0.10, perspective=0.0005,
        fliplr=0.20, flipud=0.00,
        hsv_h=0.010, hsv_s=0.22, hsv_v=0.22,
        erasing=0.08,
        workers=args.workers, cache=args.cache
    )

    if args.freeze_backbone > 0 and args.freeze_epochs > 0:
        # quick stabilized warm start
        model.train(**{**train_kwargs, "epochs": args.freeze_epochs, "freeze": args.freeze_backbone, "name": args.name + "_frozen"})
        # continue unfrozen from last
        chk = Path(model.trainer.save_dir)/"weights"/"last.pt"
        model = YOLO(str(chk))

    model.train(**{**train_kwargs, "freeze": None})

    # best checkpoint after refine
    best = Path(model.trainer.save_dir)/"weights"/"best.pt"

    # ---------------- stage B: optional one-shot self-training (predict → relabel → short top-up)
    if args.self_train:
        seg = YOLO(str(best))
        ds_root = Path(args.data).parent
        imgs_train = ds_root/"images"/"train"
        labs_train = ds_root/"labels"/"train"
        # write into a clean temp labels dir to avoid mixing
        new_labels = ds_root/"labels_selftrain"/"train"
        if new_labels.exists(): shutil.rmtree(new_labels.parent)
        new_labels.mkdir(parents=True, exist_ok=True)

        scales = tuple(float(s) for s in args.self_scales.split(",") if s.strip())
        confs  = tuple(float(s) for s in args.self_confs.split(",") if s.strip())

        harvest_labels(seg, imgs_train, new_labels,
                       imgsz=args.self_imgsz, iou=args.self_iou,
                       scales=scales, confs=confs,
                       min_conf_nh=args.self_min_conf_nh, min_conf_t=args.self_min_conf_t)

        # swap label dir for train split by pointing Ultralytics to the same data.yaml
        # (Ultralytics resolves labels by relative split folder; easiest is to
        # temporarily move/backup original labels/train and replace with ours.)
        backup = ds_root/"labels_backup_train"
        if backup.exists(): shutil.rmtree(backup)
        labs_train.rename(backup)
        new_labels.rename(labs_train)

        try:
            # short top-up train from best
            model = YOLO(str(best))
            model.train(**{**train_kwargs, "epochs": args.topup_epochs, "name": args.name + "_topup"})
            best = Path(model.trainer.save_dir)/"weights"/"best.pt"
        finally:
            # restore original labels regardless
            if labs_train.exists(): shutil.rmtree(labs_train)
            backup.rename(ds_root/"labels"/"train")

    print("\n✅ Done.")
    print(f"  Final weights: {best}")

if __name__ == "__main__":
    main()