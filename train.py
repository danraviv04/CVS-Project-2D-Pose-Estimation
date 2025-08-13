from ultralytics import YOLO
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11s-pose.pt', help='Pretrained YOLO11 pose model variant')
    parser.add_argument('--data', default='/home/student/project/output/yolo_data/data.yaml', help='Path to dataset config YAML')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--project', default='pose_phase2', help='Project directory')
    parser.add_argument('--name', default='yolo11s_surgical', help='Experiment name')
    parser.add_argument('--device', default='0', help='CUDA device, e.g. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    # Load YOLO11 pose model
    model = YOLO(args.model)

    # Train model
    model.train(
    task='pose',  # ✅ This is critical!
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    project=args.project,
    name=args.name,
    device=args.device,
    pretrained=True,
    )


if __name__ == '__main__':
    main()