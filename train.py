from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11s-pose.pt')
    parser.add_argument('--data', default='/home/student/project/output/yolo_data/data.yaml')
    parser.add_argument('--epochs', type=int, default=200)   # plenty of steps
    parser.add_argument('--imgsz', type=int, default=960)    # good for thin metal parts
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', default='pose_phase2')
    parser.add_argument('--name', default='yolo11s_surgical')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()

    model = YOLO(args.model)

    # Make mosaic stop ~last 10% of training (pose likes clean images late)
    close_mosaic = max(10, args.epochs // 10)

    # Pose-safe augmentation with stronger illumination robustness
    aug = dict(
        # geometry
        degrees=35,           # allow strong roll
        translate=0.10,
        scale=0.30,
        shear=3.0,
        perspective=0.0012,
        fliplr=0.50,
        flipud=0.00,

        # composition (kept modest for pose)
        mosaic=0.05,          # ↓ to avoid weird keypoint geometry
        mixup=0.05,
        close_mosaic=close_mosaic,

        # photometric (“lighting”)
        hsv_h=0.010,
        hsv_s=0.55,
        hsv_v=0.50,           # stronger brightness jitter
        erasing=0.20,         # occlusion sim but not excessive
    )

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        pretrained=True,

        # scheduler/optim
        optimizer='AdamW',
        lr0=0.002,            # if val OKS plateaus early, try 0.0015
        weight_decay=0.01,
        cos_lr=True,
        warmup_epochs=3,
        patience=60,          # early-stop on no OKS mAP improvement

        # loader/cache
        workers=8,
        cache=True,           # fastest; use 'disk' if you need exact reproducibility

        # augs
        **aug
    )

if __name__ == '__main__':
    main()