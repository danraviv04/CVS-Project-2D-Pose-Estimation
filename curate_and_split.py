# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# curate_and_split.py — curate pseudo-seg labels from video predictions and
# build a YOLO-SEG train/val dataset (images + labels + data.yaml).

# - Accepts N (video, labels_dir) pairs.
# - Parses Ultralytics-style seg .txt (optionally with confidence).
# - Per-frame filtering: min conf per class, stride, require classes.
# - Extracts frames from videos to match label stems.
# - Writes dataset:
#     out_ds/
#       images/{train,val}/<video>_<frame6>.jpg
#       labels/{train,val}/<video>_<frame6>.txt
#       data.yaml
# - Saves curation stats: stats.json and curation_log.csv

# Usage example (edit paths for your box):
#   python curate_and_split.py \
#     --pair /datashare/project/vids_tune/4_2_24_B_2.mp4::out_vis/ultra_like_4_2_24_B_2/labels \
#     --pair /datashare/project/vids_tune/20_2_24_1.mp4::out_vis/ultra_like_20_2_24_1/labels \
#     --out_ds /home/student/project/output/pseudo_ultra_ds \
#     --min_conf_nh 0.40 --min_conf_t 0.45 \
#     --stride 3 --save_frames --jpg_quality 92 \
#     --split 0.9 --split_by frame --seed 0 \
#     --clahe --clahe_clip 2.0
# """
# from __future__ import annotations
# import argparse, csv, json, os, re, random
# from collections import defaultdict, Counter
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Tuple

# import cv2
# import numpy as np

# # ------------------------ utils ------------------------
# STEM_RX = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")

# def parse_stem(path: Path) -> Tuple[str, int]:
#     m = STEM_RX.match(path.stem)
#     if not m:
#         raise ValueError(f"Label filename doesn't match '<base>_<frame>': {path.name}")
#     return m.group("base"), int(m.group("idx"))

# def enhance_like_infer(img_bgr: np.ndarray, clip=2.0) -> np.ndarray:
#     ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
#     y, cr, cb = cv2.split(ycrcb)
#     clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(8, 8))
#     y = clahe.apply(y)
#     return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

# @dataclass
# class FrameSel:
#     video: Path
#     labels_file: Path
#     frame_idx: int
#     base: str
#     nh_keep: int
#     t_keep: int

# # ------------------------ label parsing ------------------------
# def parse_seg_txt_line(line: str) -> Tuple[int, float | None, np.ndarray]:
#     """
#     return: (cls, conf or None, poly Nx2 normalized)
#     Heuristic: even number of tokens => has confidence (class + conf + 2n coords)
#                odd number => no confidence (class + 2n coords)
#     """
#     toks = line.strip().split()
#     if not toks:
#         raise ValueError("empty label line")
#     cls = int(float(toks[0]))
#     rest = list(map(float, toks[1:]))

#     has_conf = (len(toks) % 2 == 0)  # see docstring
#     if has_conf:
#         conf = rest[0]
#         coords = rest[1:]
#     else:
#         conf = None
#         coords = rest

#     if len(coords) < 6 or len(coords) % 2 != 0:
#         raise ValueError("seg coords malformed")

#     poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
#     return cls, conf, poly

# def load_and_filter_label(path: Path, min_conf_nh: float, min_conf_t: float) -> Tuple[List[str], int, int]:
#     """
#     Load a label .txt, filter lines by per-class confidence, and
#     return new lines (without confidence) + kept counts per class.
#     """
#     if not path.exists():
#         return [], 0, 0
#     kept_lines: List[str] = []
#     nh_keep = 0
#     t_keep = 0
#     with open(path, "r") as f:
#         for raw in f:
#             if not raw.strip():
#                 continue
#             try:
#                 cls, conf, poly = parse_seg_txt_line(raw)
#             except Exception:
#                 # skip malformed
#                 continue
#             # class 0 -> NH, class 1 -> T (your training)
#             kind = "NH" if cls == 0 else "T"
#             # decide pass/fail (if no conf present, keep)
#             if conf is not None:
#                 if kind == "NH" and conf < min_conf_nh:
#                     continue
#                 if kind == "T" and conf < min_conf_t:
#                     continue
#             # clamp coords to [0,1]
#             poly = np.clip(poly, 0.0, 1.0)
#             flat = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
#             kept_lines.append(f"{cls} {flat}")
#             if kind == "NH": nh_keep += 1
#             else: t_keep += 1
#     return kept_lines, nh_keep, t_keep

# # ------------------------ curation core ------------------------
# def gather_frames(
#     pairs: List[Tuple[Path, Path]],
#     stride: int,
#     min_conf_nh: float,
#     min_conf_t: float,
#     require_any: bool,
#     require_both: bool,
# ) -> Tuple[List[FrameSel], Dict[str, Counter]]:
#     """
#     Return list of selected frames + per-video counters.
#     """
#     selected: List[FrameSel] = []
#     stats: Dict[str, Counter] = defaultdict(Counter)

#     for video_path, labels_dir in pairs:
#         if not labels_dir.exists():
#             raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
#         base_to_video = video_path.stem  # e.g., 4_2_24_B_2
#         all_txt = sorted(labels_dir.glob("*.txt"))
#         for i, lab in enumerate(all_txt):
#             # stride filter
#             _, idx = parse_stem(lab)
#             if stride > 1 and (idx % stride != 0):
#                 continue

#             lines, nh_k, t_k = load_and_filter_label(lab, min_conf_nh, min_conf_t)
#             if require_both and not (nh_k > 0 and t_k > 0):
#                 continue
#             if require_any and (nh_k + t_k) == 0:
#                 continue

#             # keep this frame
#             stem_base, frame_idx = parse_stem(lab)
#             # sanity: stem_base should start with video stem
#             if not stem_base.startswith(base_to_video):
#                 # still accept; just record base from label
#                 pass

#             selected.append(FrameSel(
#                 video=video_path,
#                 labels_file=lab,
#                 frame_idx=frame_idx,
#                 base=stem_base,
#                 nh_keep=nh_k,
#                 t_keep=t_k
#             ))
#             stats[video_path.name]["frames_total"] += 1
#             stats[video_path.name]["frames_kept"] += 1
#             stats[video_path.name]["NH_kept"] += nh_k
#             stats[video_path.name]["T_kept"] += t_k

#     return selected, stats

# def extract_frames_for_video(
#     video_path: Path,
#     frame_indices: List[int],
#     out_dir: Path,
#     apply_clahe: bool,
#     clahe_clip: float,
#     jpg_quality: int,
# ) -> Dict[int, Tuple[int, int]]:
#     """
#     Extract requested frames (by exact index) into out_dir.
#     Returns dict frame_idx -> (W, H).
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     frame_indices_sorted = sorted(set(frame_indices))
#     idx_set = set(frame_indices_sorted)
#     wh_map: Dict[int, Tuple[int, int]] = {}

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

#     # walk the video once; this is usually faster than random seeks
#     wanted_ptr = 0
#     wanted = frame_indices_sorted
#     cur = -1
#     ok, frame = cap.read()
#     cur += 1
#     while ok and wanted_ptr < len(wanted):
#         tgt = wanted[wanted_ptr]
#         if cur < tgt:
#             # fast skip by seek (approx) then read forward
#             cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
#             cur = tgt
#             ok, frame = cap.read()
#             if not ok:
#                 break
#         if cur == tgt and ok:
#             if apply_clahe:
#                 frame = enhance_like_infer(frame, clip=clahe_clip)
#             H, W = frame.shape[:2]
#             wh_map[cur] = (W, H)
#             out_name = f"{video_path.stem}_{cur:06d}.jpg"
#             cv2.imwrite(str(out_dir / out_name), frame,
#                         [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
#             wanted_ptr += 1
#             # move to next
#             ok, frame = cap.read()
#             cur += 1
#         elif cur > tgt:
#             wanted_ptr += 1  # shouldn't happen, but recover
#     cap.release()
#     return wh_map

# # ------------------------ main ------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pair", action="append", required=True,
#                     help="Repeatable: '<video_path>::<labels_dir>' (labels from video.py --save_txt)")
#     ap.add_argument("--out_ds", required=True, help="Output dataset root")
#     ap.add_argument("--split", type=float, default=0.9, help="train ratio (val = 1 - split)")
#     ap.add_argument("--split_by", choices=["frame","video"], default="frame",
#                     help="frame: random per-frame split; video: split by whole videos")
#     ap.add_argument("--seed", type=int, default=0)

#     # curation
#     ap.add_argument("--min_conf_nh", type=float, default=0.40)
#     ap.add_argument("--min_conf_t",  type=float, default=0.45)
#     ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
#     ap.add_argument("--require_any", action="store_true", help="drop frames with zero kept instances")
#     ap.add_argument("--require_both", action="store_true", help="keep only frames that include NH and T")

#     # frame extraction
#     ap.add_argument("--save_frames", action="store_true", help="actually extract JPGs (required for training)")
#     ap.add_argument("--jpg_quality", type=int, default=92)
#     ap.add_argument("--clahe", action="store_true", help="apply CLAHE (same as inference)")
#     ap.add_argument("--clahe_clip", type=float, default=2.0)

#     args = ap.parse_args()
#     random.seed(args.seed)
#     np_random = np.random.default_rng(args.seed)

#     # parse pairs
#     pairs: List[Tuple[Path, Path]] = []
#     for p in args.pair:
#         try:
#             v, l = p.split("::", 1)
#         except ValueError:
#             raise SystemExit(f"--pair must be '<video>::<labels_dir>', got: {p}")
#         pairs.append((Path(v), Path(l)))

#     out_root = Path(args.out_ds)
#     (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
#     (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
#     (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
#     (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

#     # 1) scan & filter labels
#     selected, stats = gather_frames(
#         pairs=pairs,
#         stride=args.stride,
#         min_conf_nh=args.min_conf_nh,
#         min_conf_t=args.min_conf_t,
#         require_any=args.require_any,
#         require_both=args.require_both,
#     )

#     if not selected:
#         raise SystemExit("No frames selected after filtering.")

#     # 2) split (frame-level or video-level)
#     if args.split_by == "video":
#         # assign whole videos to train/val by ratio
#         vids = sorted(set(s.video for s in selected))
#         np_random.shuffle(vids)
#         n_train = max(1, int(round(args.split * len(vids))))
#         train_vids = set(vids[:n_train])
#         split_assign = ["train" if s.video in train_vids else "val" for s in selected]
#     else:
#         # frame-level random split
#         idx = list(range(len(selected)))
#         np_random.shuffle(idx)
#         n_train = max(1, int(round(args.split * len(idx))))
#         train_idx = set(idx[:n_train])
#         split_assign = ["train" if i in train_idx else "val" for i in range(len(selected))]

#     # 3) extract frames (once per video) and write labels
#     wh_cache: Dict[Tuple[str,int], Tuple[int,int]] = {}  # (video_name, frame_idx) -> (W,H)
#     extracted_by_video: Dict[str, Dict[int, Tuple[int,int]]] = {}

#     if args.save_frames:
#         for video_path, _labels_dir in pairs:
#             # collect all frame indices we need from this video
#             wanted = [s.frame_idx for s in selected if s.video == video_path]
#             if not wanted:
#                 continue
#             out_img_dir = out_root / "images" / "all_tmp"
#             wh_map = extract_frames_for_video(
#                 video_path, wanted, out_img_dir,
#                 apply_clahe=args.clahe, clahe_clip=args.clahe_clip,
#                 jpg_quality=args.jpg_quality,
#             )
#             extracted_by_video[video_path.name] = wh_map
#             # move images into train/val later

#     # 4) write labels (strip conf already done) and move images to split dirs
#     log_rows = []
#     per_split_counts = {"train": Counter(), "val": Counter()}
#     for s, split in zip(selected, split_assign):
#         # re-load and filter (to get stripped lines)
#         new_lines, nh_k, t_k = load_and_filter_label(
#             s.labels_file, args.min_conf_nh, args.min_conf_t
#         )
#         if args.require_any and (nh_k + t_k) == 0:
#             continue
#         if args.require_both and not (nh_k > 0 and t_k > 0):
#             continue

#         stem = f"{s.video.stem}_{s.frame_idx:06d}"
#         # write label
#         out_lab = out_root / "labels" / split / f"{stem}.txt"
#         out_lab.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

#         # move/copy image
#         if args.save_frames:
#             tmp_img = out_root / "images" / "all_tmp" / f"{stem}.jpg"
#             dst_img = out_root / "images" / split / f"{stem}.jpg"
#             if tmp_img.exists():
#                 tmp_img.replace(dst_img)
#             else:
#                 # fallback: attempt direct extraction if missing
#                 cap = cv2.VideoCapture(str(s.video))
#                 if cap.isOpened():
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame_idx)
#                     ok, frame = cap.read()
#                     if ok:
#                         if args.clahe:
#                             frame = enhance_like_infer(frame, clip=args.clahe_clip)
#                         cv2.imwrite(str(dst_img), frame,
#                                     [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
#                 cap.release()

#         per_split_counts[split]["frames"] += 1
#         per_split_counts[split]["NH"] += nh_k
#         per_split_counts[split]["T"] += t_k

#         log_rows.append({
#             "video": s.video.name,
#             "split": split,
#             "frame_idx": s.frame_idx,
#             "label_path": str(out_root / "labels" / split / f"{stem}.txt"),
#             "image_path": str(out_root / "images" / split / f"{stem}.jpg") if args.save_frames else "",
#             "NH": nh_k,
#             "T": t_k,
#         })

#     # cleanup tmp
#     tmp_all = out_root / "images" / "all_tmp"
#     if tmp_all.exists():
#         for p in tmp_all.glob("*.jpg"):
#             p.unlink()
#         tmp_all.rmdir()

#     # 5) write data.yaml
#     data_yaml = f"""# Autogenerated by curate_and_split.py
# path: {out_root.as_posix()}
# train: images/train
# val: images/val
# names:
#   0: NH
#   1: T
# """
#     (out_root / "data.yaml").write_text(data_yaml)

#     # 6) save stats/log
#     (out_root / "stats.json").write_text(json.dumps({
#         "per_video": {k: dict(v) for k, v in stats.items()},
#         "per_split": {k: dict(v) for k, v in per_split_counts.items()},
#         "total_selected": len(selected)
#     }, indent=2))

#     with open(out_root / "curation_log.csv", "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
#         w.writeheader()
#         w.writerows(log_rows)

#     print("✅ Dataset built at:", out_root)
#     print("   - data.yaml")
#     print("   - images/train, images/val")
#     print("   - labels/train, labels/val")
#     print("   - stats.json, curation_log.csv")
#     print("   Train next with:")
#     print(f"     python train.py --data {out_root}/data.yaml --model yolo11s-seg.pt --epochs 60 --imgsz 1280 --device 0")
#     print("   (or swap in your last best weights to fine-tune).")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate_and_split.py — curate pseudo-seg labels from video predictions and
build a YOLO-SEG train/val dataset (images + labels + data.yaml).

- Accepts N (video, labels_dir) pairs.
- Parses Ultralytics-style seg .txt (optionally with confidence).
- Per-frame filtering: min conf per class, stride, require classes.
- Extracts frames from videos to match label stems.
- Writes dataset:
    out_ds/
      images/{train,val}/<video>_<frame6>.jpg
      labels/{train,val}/<video>_<frame6>.txt
      data.yaml
- Saves curation stats: stats.json and (optional) curation_log.csv
- Progress printing for each major stage (configurable by --progress_every)
"""
from __future__ import annotations
import argparse, csv, json, os, re, random, time
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ------------------------ utils ------------------------
STEM_RX = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")

def parse_stem(path: Path) -> Tuple[str, int]:
    m = STEM_RX.match(path.stem)
    if not m:
        raise ValueError(f"Label filename doesn't match '<base>_<frame>': {path.name}")
    return m.group("base"), int(m.group("idx"))

def enhance_like_infer(img_bgr: np.ndarray, clip=2.0) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(8, 8))
    y = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

def _fmt_secs(s: float) -> str:
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    return f"{h:d}:{m:02d}:{r:02d}" if h else f"{m:02d}:{r:02d}"

@dataclass
class FrameSel:
    video: Path
    labels_file: Path
    frame_idx: int
    base: str
    nh_keep: int
    t_keep: int

# ------------------------ label parsing ------------------------
def parse_seg_txt_line(line: str) -> Tuple[int, float | None, np.ndarray]:
    """
    return: (cls, conf or None, poly Nx2 normalized)
    Heuristic: even number of tokens => has confidence (class + conf + 2n coords)
               odd number => no confidence (class + 2n coords)
    """
    toks = line.strip().split()
    if not toks:
        raise ValueError("empty label line")
    cls = int(float(toks[0]))
    rest = list(map(float, toks[1:]))

    has_conf = (len(toks) % 2 == 0)
    if has_conf:
        conf = rest[0]
        coords = rest[1:]
    else:
        conf = None
        coords = rest

    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("seg coords malformed")

    poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
    return cls, conf, poly

def needs_norm(poly: np.ndarray) -> bool:
    # if any coord > 1.001 it's almost certainly pixel-space
    return float(np.max(poly)) > 1.001

def norm_poly(poly: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
    W, H = map(float, wh)
    if W <= 1 or H <= 1:
        return poly
    poly = poly.copy()
    poly[:, 0] /= W
    poly[:, 1] /= H
    return poly

def load_and_filter_label(path: Path, min_conf_nh: float, min_conf_t: float,
                          img_wh: Tuple[int, int] | None = None) -> Tuple[List[str], int, int]:
    """
    Load a label .txt, filter lines by per-class confidence,
    optionally normalize to [0,1] using img_wh if coords look like pixels,
    and return new lines (without confidence) + kept counts per class.
    """
    if not path.exists():
        return [], 0, 0
    kept_lines: List[str] = []
    nh_keep = 0
    t_keep = 0
    with open(path, "r") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                cls, conf, poly = parse_seg_txt_line(raw)
            except Exception:
                continue

            # class 0 -> NH, class 1 -> T
            kind = "NH" if cls == 0 else "T"

            # conf filter (if present)
            if conf is not None:
                if kind == "NH" and conf < min_conf_nh:
                    continue
                if kind == "T"  and conf < min_conf_t:
                    continue

            # normalize if coords look like pixels and we know (W,H)
            if img_wh is not None and needs_norm(poly):
                poly = norm_poly(poly, img_wh)

            # finally clamp to [0,1]
            poly = np.clip(poly, 0.0, 1.0)

            flat = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
            kept_lines.append(f"{cls} {flat}")
            if kind == "NH": nh_keep += 1
            else:            t_keep += 1
    return kept_lines, nh_keep, t_keep

# ------------------------ curation core ------------------------
def gather_frames(
    pairs: List[Tuple[Path, Path]],
    stride: int,
    min_conf_nh: float,
    min_conf_t: float,
    require_any: bool,
    require_both: bool,
    progress_every: int = 200,
) -> Tuple[List[FrameSel], Dict[str, Counter]]:
    """
    Return list of selected frames + per-video counters, with progress prints.
    """
    selected: List[FrameSel] = []
    stats: Dict[str, Counter] = defaultdict(Counter)

    # count total label files for progress %
    total_txt = 0
    for _video_path, labels_dir in pairs:
        total_txt += len(list(labels_dir.glob("*.txt")))
    processed = 0
    t0 = time.time()

    print(f"🔎 Scanning labels across {len(pairs)} pair(s)...")
    for video_path, labels_dir in pairs:
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
        base_to_video = video_path.stem  # e.g., 4_2_24_B_2
        all_txt = sorted(labels_dir.glob("*.txt"))
        for lab in all_txt:
            processed += 1

            # stride filter
            _, idx = parse_stem(lab)
            if stride > 1 and (idx % stride != 0):
                if (processed % progress_every) == 0 or processed == total_txt:
                    elapsed = time.time() - t0
                    pct = 100.0 * processed / max(1, total_txt)
                    print(f"[scan {processed}/{total_txt}] {pct:5.1f}% | elapsed={_fmt_secs(elapsed)} | kept={len(selected)} frames")
                continue

            lines, nh_k, t_k = load_and_filter_label(lab, min_conf_nh, min_conf_t)
            if require_both and not (nh_k > 0 and t_k > 0):
                if (processed % progress_every) == 0 or processed == total_txt:
                    elapsed = time.time() - t0
                    pct = 100.0 * processed / max(1, total_txt)
                    print(f"[scan {processed}/{total_txt}] {pct:5.1f}% | elapsed={_fmt_secs(elapsed)} | kept={len(selected)} frames")
                continue
            if require_any and (nh_k + t_k) == 0:
                if (processed % progress_every) == 0 or processed == total_txt:
                    elapsed = time.time() - t0
                    pct = 100.0 * processed / max(1, total_txt)
                    print(f"[scan {processed}/{total_txt}] {pct:5.1f}% | elapsed={_fmt_secs(elapsed)} | kept={len(selected)} frames")
                continue

            # keep this frame
            stem_base, frame_idx = parse_stem(lab)
            selected.append(FrameSel(
                video=video_path,
                labels_file=lab,
                frame_idx=frame_idx,
                base=stem_base,
                nh_keep=nh_k,
                t_keep=t_k
            ))
            stats[video_path.name]["frames_total"] += 1
            stats[video_path.name]["frames_kept"] += 1
            stats[video_path.name]["NH_kept"] += nh_k
            stats[video_path.name]["T_kept"] += t_k

            if (processed % progress_every) == 0 or processed == total_txt:
                elapsed = time.time() - t0
                pct = 100.0 * processed / max(1, total_txt)
                print(f"[scan {processed}/{total_txt}] {pct:5.1f}% | elapsed={_fmt_secs(elapsed)} | kept={len(selected)} frames")

    print(f"✅ Scanning done: considered {processed} label files, selected {len(selected)} frame(s).")
    return selected, stats

def extract_frames_for_video(
    video_path: Path,
    frame_indices: List[int],
    out_dir: Path,
    apply_clahe: bool,
    clahe_clip: float,
    jpg_quality: int,
    progress_every: int = 200,
) -> Dict[int, Tuple[int, int]]:
    """
    Extract requested frames (by exact index) into out_dir.
    Returns dict frame_idx -> (W, H).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_indices_sorted = sorted(set(frame_indices))
    wh_map: Dict[int, Tuple[int, int]] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"🎞  Extracting {len(frame_indices_sorted)} frames from '{video_path.name}' (total frames in video={total})")
    t0 = time.time()
    saved = 0

    # walk the video by seeking to exact targets
    cur = -1
    ok, frame = cap.read()
    cur += 1
    wanted_ptr = 0
    wanted = frame_indices_sorted

    while ok and wanted_ptr < len(wanted):
        tgt = wanted[wanted_ptr]
        if cur < tgt:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
            cur = tgt
            ok, frame = cap.read()
            if not ok:
                break
        if cur == tgt and ok:
            if apply_clahe:
                frame = enhance_like_infer(frame, clip=clahe_clip)
            H, W = frame.shape[:2]
            wh_map[cur] = (W, H)
            out_name = f"{video_path.stem}_{cur:06d}.jpg"
            cv2.imwrite(str(out_dir / out_name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            saved += 1
            wanted_ptr += 1

            if (saved % progress_every) == 0 or saved == len(wanted):
                elapsed = time.time() - t0
                pct = 100.0 * saved / max(1, len(wanted))
                fps = saved / max(1e-6, elapsed)
                eta = (len(wanted) - saved) / max(1e-6, fps)
                print(f"[extract {saved}/{len(wanted)}] {pct:5.1f}% | fps={fps:5.1f} | elapsed={_fmt_secs(elapsed)} | eta={_fmt_secs(eta)}")

            ok, frame = cap.read()
            cur += 1
        elif cur > tgt:
            wanted_ptr += 1  # shouldn't happen often

    cap.release()
    print(f"✅ Extracted {saved}/{len(wanted)} frames from '{video_path.name}'.")
    return wh_map

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    help="Repeatable: '<video_path>::<labels_dir>' (labels from video.py --save_txt)")
    ap.add_argument("--out_ds", required=True, help="Output dataset root")
    ap.add_argument("--split", type=float, default=0.9, help="train ratio (val = 1 - split)")
    ap.add_argument("--split_by", choices=["frame","video"], default="frame",
                    help="frame: random per-frame split; video: split by whole videos")
    ap.add_argument("--seed", type=int, default=0)

    # curation
    ap.add_argument("--min_conf_nh", type=float, default=0.40)
    ap.add_argument("--min_conf_t",  type=float, default=0.45)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--require_any", action="store_true", help="drop frames with zero kept instances")
    ap.add_argument("--require_both", action="store_true", help="keep only frames that include NH and T")

    # frame extraction
    ap.add_argument("--save_frames", action="store_true", help="actually extract JPGs (required for training)")
    ap.add_argument("--jpg_quality", type=int, default=92)
    ap.add_argument("--clahe", action="store_true", help="apply CLAHE (same as inference)")
    ap.add_argument("--clahe_clip", type=float, default=2.0)

    # progress
    ap.add_argument("--progress_every", type=int, default=200, help="print progress every N items for each stage")

    args = ap.parse_args()
    random.seed(args.seed)
    np_random = np.random.default_rng(args.seed)

    # parse pairs
    pairs: List[Tuple[Path, Path]] = []
    for p in args.pair:
        try:
            v, l = p.split("::", 1)
        except ValueError:
            raise SystemExit(f"--pair must be '<video>::<labels_dir>', got: {p}")
        pairs.append((Path(v), Path(l)))

    out_root = Path(args.out_ds)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("⚙️  Curation config:")
    print(f"    pairs={len(pairs)} | stride={args.stride} | require_any={args.require_any} | require_both={args.require_both}")
    print(f"    min_conf: NH={args.min_conf_nh:.2f}, T={args.min_conf_t:.2f}")
    print(f"    split={args.split:.2f} by {args.split_by} | seed={args.seed}")
    if args.save_frames:
        print(f"    save_frames=True | jpg_quality={args.jpg_quality} | clahe={args.clahe} (clip={args.clahe_clip})")
    print(f"    progress_every={args.progress_every}")

    # 1) scan & filter labels
    selected, stats = gather_frames(
        pairs=pairs,
        stride=args.stride,
        min_conf_nh=args.min_conf_nh,
        min_conf_t=args.min_conf_t,
        require_any=args.require_any,
        require_both=args.require_both,
        progress_every=args.progress_every,
    )

    if not selected:
        raise SystemExit("No frames selected after filtering.")

    # 2) split (frame-level or video-level)
    print("✂️  Splitting into train/val...")
    if args.split_by == "video":
        vids = sorted(set(s.video for s in selected))
        np_random.shuffle(vids)
        n_train = max(1, int(round(args.split * len(vids))))
        train_vids = set(vids[:n_train])
        split_assign = ["train" if s.video in train_vids else "val" for s in selected]
        print(f"    split_by=video | train_videos={len(train_vids)} / total_videos={len(vids)}")
    else:
        idx = list(range(len(selected)))
        np_random.shuffle(idx)
        n_train = max(1, int(round(args.split * len(idx))))
        train_idx = set(idx[:n_train])
        split_assign = ["train" if i in train_idx else "val" for i in range(len(selected))]
        print(f"    split_by=frame | train_frames={n_train} / total_frames={len(selected)}")

    # 3) extract frames (once per video) and write labels
    extracted_by_video: Dict[str, Dict[int, Tuple[int,int]]] = {}

    if args.save_frames:
        for video_path, _labels_dir in pairs:
            wanted = [s.frame_idx for s in selected if s.video == video_path]
            if not wanted:
                continue
            out_img_dir = out_root / "images" / "all_tmp"
            wh_map = extract_frames_for_video(
                video_path, wanted, out_img_dir,
                apply_clahe=args.clahe, clahe_clip=args.clahe_clip,
                jpg_quality=args.jpg_quality,
                progress_every=args.progress_every,
            )
            extracted_by_video[video_path.name] = wh_map

    # 4) write labels (strip conf already done) and move images to split dirs
    print("📝 Writing labels and placing images into train/val...")
    t0 = time.time()
    log_rows = []
    per_split_counts = {"train": Counter(), "val": Counter()}
    for i, (s, split) in enumerate(zip(selected, split_assign), start=1):
        # re-load and filter (to get stripped lines)
        img_wh = None
        vid_name = s.video.name
        if args.save_frames and vid_name in extracted_by_video:
            wh_map = extracted_by_video[vid_name]
            if s.frame_idx in wh_map:
                img_wh = wh_map[s.frame_idx]
        if img_wh is None:
            # fallback: read props once
            cap = cv2.VideoCapture(str(s.video))
            if cap.isOpened():
                W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if W > 0 and H > 0:
                    img_wh = (W, H)
            cap.release()

        # re-load and filter WITH normalization when needed
        new_lines, nh_k, t_k = load_and_filter_label(
            s.labels_file, args.min_conf_nh, args.min_conf_t, img_wh=img_wh
        )
        if args.require_any and (nh_k + t_k) == 0:
            continue
        if args.require_both and not (nh_k > 0 and t_k > 0):
            continue

        stem = f"{s.video.stem}_{s.frame_idx:06d}"
        out_lab = out_root / "labels" / split / f"{stem}.txt"
        out_lab.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

        if args.save_frames:
            tmp_img = out_root / "images" / "all_tmp" / f"{stem}.jpg"
            dst_img = out_root / "images" / split / f"{stem}.jpg"
            if tmp_img.exists():
                tmp_img.replace(dst_img)
            else:
                # fallback: direct extraction if missing
                cap = cv2.VideoCapture(str(s.video))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame_idx)
                    ok, frame = cap.read()
                    if ok:
                        if args.clahe:
                            frame = enhance_like_infer(frame, clip=args.clahe_clip)
                        cv2.imwrite(str(dst_img), frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
                cap.release()

        per_split_counts[split]["frames"] += 1
        per_split_counts[split]["NH"] += nh_k
        per_split_counts[split]["T"] += t_k

        log_rows.append({
            "video": s.video.name,
            "split": split,
            "frame_idx": s.frame_idx,
            "label_path": str(out_root / "labels" / split / f"{stem}.txt"),
            "image_path": str(out_root / "images" / split / f"{stem}.jpg") if args.save_frames else "",
            "NH": nh_k,
            "T": t_k,
        })

        if (i % args.progress_every) == 0 or i == len(selected):
            elapsed = time.time() - t0
            pct = 100.0 * i / len(selected)
            fps = i / max(1e-6, elapsed)
            eta = (len(selected) - i) / max(1e-6, fps)
            tr = per_split_counts["train"]["frames"]
            va = per_split_counts["val"]["frames"]
            print(f"[write {i}/{len(selected)}] {pct:5.1f}% | fps={fps:5.1f} | elapsed={_fmt_secs(elapsed)} | eta={_fmt_secs(eta)} | train={tr} val={va}")

    # cleanup tmp
    tmp_all = out_root / "images" / "all_tmp"
    if tmp_all.exists():
        for p in tmp_all.glob("*.jpg"):
            try: p.unlink()
            except Exception: pass
        try: tmp_all.rmdir()
        except Exception: pass

    # 5) write data.yaml
    data_yaml = f"""# Autogenerated by curate_and_split.py
path: {out_root.as_posix()}
train: images/train
val: images/val
names:
  0: NH
  1: T
"""
    (out_root / "data.yaml").write_text(data_yaml)

    # 6) save stats/log
    (out_root / "stats.json").write_text(json.dumps({
        "per_video": {k: dict(v) for k, v in stats.items()},
        "per_split": {k: dict(v) for k, v in per_split_counts.items()},
        "total_selected": len(selected)
    }, indent=2))

    if log_rows:
        with open(out_root / "curation_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    else:
        print("⚠️  No rows to write into curation_log.csv (after final filtering).")

    print("\n✅ Dataset built at:", out_root)
    print("   - data.yaml")
    print("   - images/train, images/val")
    print("   - labels/train, labels/val")
    print("   - stats.json", "(+ curation_log.csv)" if log_rows else "")
    print("   Train next with for fine-tune, e.g.:")
    print(f"     python train.py --data {out_root}/data.yaml --model yolo11s-seg.pt --epochs 60 --imgsz 1280 --device 0")

if __name__ == "__main__":
    main()