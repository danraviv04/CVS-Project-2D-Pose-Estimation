# # video.py
# import os
# import argparse
# import cv2
# from ultralytics import YOLO

# def enhance(frame, clahe_clip=2.0, denoise=False):
#     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     y, cr, cb = cv2.split(ycrcb)
#     clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
#     y = clahe.apply(y)
#     ycrcb = cv2.merge([y, cr, cb])
#     out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
#     if denoise:
#         out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
#     return out

# def main(weights, video, output_dir=None, conf=0.20, imgsz=1280 , device=0,
#          clahe_clip=2.0, denoise=False, augment=False, debug_every=30):
#     model = YOLO(weights)
#     try:
#         model.to(device)
#     except Exception:
#         pass

#     print("Loaded model classes:", getattr(model, "names", None))
#     print("task:", getattr(model, "task", None))
#     print("ckpt kpt_shape:", getattr(getattr(model, "model", None), "kpt_shape", None))

#     cap = cv2.VideoCapture(video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {video}")

#     # read one frame for sanity, then rewind
#     ok, frame0 = cap.read()
#     if not ok:
#         raise RuntimeError("Could not read first frame")
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     # quick sanity: does this checkpoint output keypoints?
#     try:
#         test_res = model.predict(frame0, imgsz=imgsz, conf=max(0.05, conf), verbose=False, augment=False, device=device)[0]
#         has_kpts = hasattr(test_res, "keypoints") and (test_res.keypoints is not None)
#         print("has keypoints?:", has_kpts)
#         if has_kpts:
#             print("kpt tensor shape:", getattr(test_res.keypoints, "xy", None).shape)
#     except Exception as e:
#         print("Sanity predict failed:", e)

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
#     fps    = cap.get(cv2.CAP_PROP_FPS) or 0
#     if width == 0 or height == 0:
#         raise RuntimeError("Video reports 0 width/height — bad file or codec?")
#     if fps <= 1:
#         fps = 30.0

#     base = os.path.splitext(os.path.basename(video))[0]
#     output_dir = output_dir or "outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, f"annotated_{base}.mp4")

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
#     if not out.isOpened():
#         raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

#     idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         proc = enhance(frame, clahe_clip=clahe_clip, denoise=denoise)

#         res = model.predict(
#             source=proc,
#             conf=conf,
#             imgsz=imgsz,
#             verbose=False,
#             agnostic_nms=True,
#             augment=augment,   # keep False for video unless you know you need TTA
#             device=device
#         )[0]

#         if debug_every and idx % debug_every == 0:
#             n_boxes = 0 if res.boxes is None else res.boxes.shape[0]
#             kpt_info = getattr(getattr(res, "keypoints", None), "xy", None)
#             kpt_per = kpt_info.shape[1] if kpt_info is not None and kpt_info.size else 0
#             print(f"[frame {idx}] boxes={n_boxes}, kpts_per_inst={kpt_per}")

#         annotated = res.plot()  # draws boxes + kpts when present
#         out.write(annotated)
#         idx += 1

#     cap.release()
#     out.release()
#     print(f"✅ Output saved to: {out_path}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('--weights', default="pose_phase2/yolo11s_surgical/weights/best.pt")
#     p.add_argument('--video',   default="/datashare/project/vids_test/4_2_24_A_1_small.mp4")
#     p.add_argument('--output_dir', default="vid_output")
#     p.add_argument('--conf', type=float, default=0.20)
#     p.add_argument('--imgsz', type=int, default=1024)
#     p.add_argument('--device', default=0)  # 0 GPU, or 'cpu'
#     p.add_argument('--clahe_clip', type=float, default=2.0)
#     p.add_argument('--denoise', action='store_true')
#     p.add_argument('--augment', action='store_true')  # you can pass to test, but default off
#     p.add_argument('--debug_every', type=int, default=30)
#     args = p.parse_args()
#     main(**vars(args))

# video.py
import os
import argparse
import cv2
from ultralytics import YOLO

def enhance(frame, clahe_clip=2.0, denoise=False):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    y = clahe.apply(y)
    ycrcb = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    return out

def main(weights, video, output_dir=None, conf=0.18, imgsz=1280, device=0,
         clahe_clip=2.0, denoise=False, augment=False, debug_every=30, scale=1.5):
    model = YOLO(weights)
    try:
        model.to(device)
    except Exception:
        pass

    print("Loaded model classes:", getattr(model, "names", None))
    print("task:", getattr(model, "task", None))
    print("ckpt kpt_shape:", getattr(getattr(model, "model", None), "kpt_shape", None))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    # read one frame for sanity, then rewind
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # quick sanity: does this checkpoint output keypoints?
    try:
        test_res = model.predict(frame0, imgsz=imgsz, conf=max(0.05, conf),
                                 verbose=False, augment=False, device=device)[0]
        has_kpts = hasattr(test_res, "keypoints") and (test_res.keypoints is not None)
        print("has keypoints?:", has_kpts)
        if has_kpts:
            print("kpt tensor shape:", getattr(test_res.keypoints, "xy", None).shape)
    except Exception as e:
        print("Sanity predict failed:", e)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0
    if width == 0 or height == 0:
        raise RuntimeError("Video reports 0 width/height — bad file or codec?")
    if fps <= 1:
        fps = 30.0

    base = os.path.splitext(os.path.basename(video))[0]
    output_dir = output_dir or "outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_{base}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        proc = enhance(frame, clahe_clip=clahe_clip, denoise=denoise)

        # --- upscale for inference (helps tiny tools) ---
        if scale and scale != 1.0:
            big = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            big = proc

        res = model.predict(
            source=big,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            agnostic_nms=True,
            augment=augment,   # keep False for video unless you know you need TTA
            device=device
        )[0]

        if debug_every and idx % debug_every == 0:
            n_boxes = 0 if res.boxes is None else res.boxes.shape[0]
            kpt_info = getattr(getattr(res, "keypoints", None), "xy", None)
            kpt_per = kpt_info.shape[1] if kpt_info is not None and kpt_info.size else 0
            print(f"[frame {idx}] boxes={n_boxes}, kpts_per_inst={kpt_per}")

        annotated_big = res.plot()  # draws boxes + kpts when present

        # --- downscale back to original frame size for writing ---
        if annotated_big.shape[1] != width or annotated_big.shape[0] != height:
            annotated = cv2.resize(annotated_big, (width, height), interpolation=cv2.INTER_AREA)
        else:
            annotated = annotated_big

        out.write(annotated)
        idx += 1

    cap.release()
    out.release()
    print(f"✅ Output saved to: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default="pose_phase2/yolo11s_surgical/weights/best.pt")
    p.add_argument('--video',   default="/datashare/project/vids_test/4_2_24_A_1_small.mp4")
    p.add_argument('--output_dir', default="vid_output")
    p.add_argument('--conf', type=float, default=0.18)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--device', default=0)  # 0 GPU, or 'cpu'
    p.add_argument('--clahe_clip', type=float, default=2.0)
    p.add_argument('--denoise', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--debug_every', type=int, default=30)
    p.add_argument('--scale', type=float, default=1.5, help="pre-inference upscale factor (1.0=off)")
    args = p.parse_args()
    main(**vars(args))