# video.py
import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------
# Image enhancement (optional)
# --------------------------
def enhance(frame_bgr, clahe_clip=2.0, denoise=False):
    """CLAHE on Y channel (YCrCb) + optional denoise."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    y = clahe.apply(y)
    ycrcb = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    return out


# --------------------------
# Utility: count boxes per class
# --------------------------
def count_by_class(res):
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return {0: 0, 1: 0}
    cls = res.boxes.cls.int().cpu().numpy().tolist()
    return {c: cls.count(c) for c in (0, 1)}


# --------------------------
# Shape/size prior per class
# --------------------------
def filter_tooly_boxes(
    res,
    frame_shape,
    min_conf=0.22,
    relax_high_conf=0.40,
    # per-class thresholds
    nh_min_ar=1.6, nh_max_area=0.20,
    t_min_ar=2.3,  t_max_area=0.12,
    min_area=0.0004
):
    """
    Keep boxes that look like tools:
      - Aspect ratio thin/long (class-specific)
      - Area in a plausible range
      - Confidence above min_conf (unless very confident: relax_high_conf)
    """
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return res

    H, W = frame_shape[:2]
    keep = []

    xywh = res.boxes.xywh.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clses = res.boxes.cls.int().cpu().numpy()

    for i, (cx, cy, bw, bh) in enumerate(xywh):
        conf = float(confs[i])
        cls_id = int(clses[i])

        ar = float(max(bw, bh) / max(1e-6, min(bw, bh)))
        area_rel = float((bw * bh) / (W * H))

        if cls_id == 1:  # Tweezers
            min_ar = t_min_ar
            max_area = t_max_area
        else:            # Needle holder
            min_ar = nh_min_ar
            max_area = nh_max_area

        # very confident? just check area sanity
        if conf >= relax_high_conf:
            if (min_area < area_rel < max_area):
                keep.append(i)
            continue

        if (conf >= min_conf) and (ar >= min_ar) and (min_area < area_rel < max_area):
            keep.append(i)

    res.boxes = res.boxes[keep] if keep else res.boxes[:0]
    if getattr(res, "keypoints", None) is not None:
        res.keypoints = res.keypoints[keep] if keep else res.keypoints[:0]
    return res


# --------------------------
# Keep top-K per class (post filter)
# --------------------------
def keep_topk_per_class(res, k=1):
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return res
    cls = res.boxes.cls.int().cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    keep = []
    for cid in (0, 1):
        idx = np.where(cls == cid)[0].tolist()
        if idx:
            idx = sorted(idx, key=lambda i: float(conf[i]), reverse=True)[:k]
            keep += idx
    res.boxes = res.boxes[keep] if keep else res.boxes[:0]
    if getattr(res, "keypoints", None) is not None:
        res.keypoints = res.keypoints[keep] if keep else res.keypoints[:0]
    return res


# --------------------------
# Autoscale + guarded autoconf
# --------------------------
def predict_autoscale_autoconf(
    model,
    frame_bgr,
    base_conf=0.32,           # start stricter
    scales=(1.75, 1.5, 1.25, 1.0),
    max_imgsz=1536,
    iou=0.55,
    device=0,
    augment=False,
    max_det=6
):
    """
    Try a few scales and a short confidence ladder.
    We only relax confidence if no valid detections survive the shape filter
    (caller applies filters).
    """
    def _round32(x): return int(32 * np.ceil(x / 32.0))
    def pick_imgsz_for(img, max_side=1536):
        h, w = img.shape[:2]
        side = min(max_side, max(h, w))
        return _round32(side)

    # We will try the first value; lower ones only if nothing remains after filters
    conf_ladder = [base_conf, 0.30, 0.28]
    last = None
    used_s, used_c = 1.0, base_conf

    for s in scales:
        big = frame_bgr if s == 1.0 else cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        imgsz = pick_imgsz_for(big, max_side=max_imgsz)

        for c in conf_ladder:
            res = model.predict(
                source=big,
                imgsz=imgsz,
                conf=c,
                iou=iou,
                max_det=max_det,
                agnostic_nms=False,      # class-aware
                classes=[0, 1],          # NH, T
                verbose=False,
                augment=augment,
                device=device
            )[0]
            last, used_s, used_c = res, s, c
            # don't decide here; caller applies geometry filter & top-k
            return last, used_s, used_c

    return last, used_s, used_c


# --------------------------
# Main
# --------------------------
def main(
    weights,
    video,
    output_dir="vid_output",
    conf=0.32,
    imgsz=1536,
    device=0,
    clahe_clip=2.0,
    denoise=False,
    augment=False,
    debug_every=30,
    scales="1.75,1.5,1.25,1.0",
    iou=0.55,
    max_det=6,
    # shape priors (can be tuned from CLI)
    nh_min_ar=1.6, nh_max_area=0.20,
    t_min_ar=2.3,  t_max_area=0.12,
    min_area=0.0004,
    relax_high_conf=0.40
):
    # Load model
    model = YOLO(weights)
    try:
        model.to(device)
        if str(device) != "cpu":
            try:
                model.model.half()
                print("Model set to FP16 for GPU inference.")
            except Exception:
                print("FP16 switch not applied (already half or unsupported).")
    except Exception:
        pass

    print("Loaded model classes:", getattr(model, "names", None))
    print("task:", getattr(model, "task", None))
    print("ckpt kpt_shape:", getattr(getattr(model, "model", None), "kpt_shape", None))

    # Open video
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    # Probe first frame for writer sizing
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Quick sanity call
    try:
        test_res = model.predict(
            frame0, imgsz=imgsz, conf=max(0.22, conf), iou=iou,
            agnostic_nms=False, classes=[0, 1], verbose=False, device=device
        )[0]
        has_kpts = hasattr(test_res, "keypoints") and (test_res.keypoints is not None)
        print("has keypoints?:", has_kpts)
        if has_kpts and getattr(test_res.keypoints, "xy", None) is not None:
            print("kpt tensor shape:", test_res.keypoints.xy.shape)
    except Exception as e:
        print("Sanity predict failed:", e)

    # Prepare writer
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0
    if width == 0 or height == 0:
        raise RuntimeError("Video reports 0 width/height — bad file or codec?")
    if fps <= 1:
        fps = 30.0

    base = os.path.splitext(os.path.basename(video))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_{base}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    # Parse scale list
    try:
        scale_list = tuple(float(s.strip()) for s in str(scales).split(",") if s.strip())
        if not scale_list:
            scale_list = (1.75, 1.5, 1.25, 1.0)
    except Exception:
        scale_list = (1.75, 1.5, 1.25, 1.0)

    # Process frames
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        proc = enhance(frame, clahe_clip=clahe_clip, denoise=denoise)
        # one predict try (strict), caller will filter and if empty we retry at lower conf
        res, used_scale, used_conf = predict_autoscale_autoconf(
            model, proc, base_conf=conf, scales=scale_list, max_imgsz=imgsz,
            iou=iou, device=device, augment=augment, max_det=max_det
        )

        # apply geometry + area + confidence prior
        filtered = filter_tooly_boxes(
            res, proc.shape, min_conf=max(0.22, used_conf),
            relax_high_conf=relax_high_conf,
            nh_min_ar=nh_min_ar, nh_max_area=nh_max_area,
            t_min_ar=t_min_ar,  t_max_area=t_max_area,
            min_area=min_area
        )
        filtered = keep_topk_per_class(filtered, k=1)

        # If nothing left, take a softer pass (only once) at lower conf
        if filtered is None or filtered.boxes is None or len(filtered.boxes) == 0:
            for soft_c in (0.30, 0.28):
                res_soft = model.predict(
                    source=proc,
                    imgsz=imgsz,
                    conf=soft_c,
                    iou=iou,
                    max_det=max_det,
                    agnostic_nms=False,
                    classes=[0, 1],
                    verbose=False,
                    augment=augment,
                    device=device
                )[0]
                filtered = filter_tooly_boxes(
                    res_soft, proc.shape, min_conf=max(0.22, soft_c),
                    relax_high_conf=relax_high_conf,
                    nh_min_ar=nh_min_ar, nh_max_area=nh_max_area,
                    t_min_ar=t_min_ar,  t_max_area=t_max_area,
                    min_area=min_area
                )
                filtered = keep_topk_per_class(filtered, k=1)
                if filtered is not None and filtered.boxes is not None and len(filtered.boxes) > 0:
                    used_conf = soft_c
                    break

        # Debug prints
        if debug_every and idx % debug_every == 0:
            counts = count_by_class(filtered)
            print(f"[frame {idx}] boxes={sum(counts.values())} (NH={counts.get(0,0)}, T={counts.get(1,0)}), "
                  f"scale={used_scale}, conf={used_conf:.2f}")

        # Plot overlays (Ultralytics draws boxes and keypoints)
        annotated = (filtered or res).plot() if (filtered or res) is not None else proc

        # Back to original resolution
        if annotated.shape[1] != width or annotated.shape[0] != height:
            annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_AREA)

        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()
    print(f"✅ Output saved to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default="pose_phase2/yolo11s_surgical/weights/best.pt")
    p.add_argument('--video',   default="/datashare/project/vids_tune/4_2_24_B_2.mp4") #20_2_24_1
    p.add_argument('--output_dir', default="vid_output")
    p.add_argument('--conf', type=float, default=0.32)
    p.add_argument('--imgsz', type=int, default=1536)
    p.add_argument('--device', default=0)
    p.add_argument('--clahe_clip', type=float, default=2.0)
    p.add_argument('--denoise', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--debug_every', type=int, default=30)
    p.add_argument('--scales', type=str, default="1.75,1.5,1.25,1.0")
    p.add_argument('--iou', type=float, default=0.55)
    p.add_argument('--max_det', type=int, default=6)
    # class priors
    p.add_argument('--nh_min_ar', type=float, default=1.6)
    p.add_argument('--nh_max_area', type=float, default=0.20)
    p.add_argument('--t_min_ar', type=float, default=2.3)
    p.add_argument('--t_max_area', type=float, default=0.12)
    p.add_argument('--min_area', type=float, default=0.0004)
    p.add_argument('--relax_high_conf', type=float, default=0.40)
    args = p.parse_args()
    main(**vars(args))