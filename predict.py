from ultralytics import YOLO
import cv2
import argparse
import os

def main(weights_path, image_path, output_dir=None, show=False):
    model = YOLO(weights_path)
    results = model(image_path)
    
    #print the results
    print("🔍 Prediction results:"
        f"\n{results[0].boxes.xyxy}\n"
        f"Keypoints: {results[0].keypoints}\n"
        f"Scores: {results[0].boxes.conf}\n"
        f"Labels: {results[0].boxes.cls}\n"
        f"Image size: {results[0].orig_shape}")


    annotated = results[0].plot()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(save_path, annotated)
        print(f"✅ Saved to: {save_path}")

    if show:
        cv2.imshow("Prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help="Path to model weights (.pt)")
    parser.add_argument('--source', required=True, help="Path to input image")
    parser.add_argument('--output', default=None, help="Directory to save annotated image")
    parser.add_argument('--show', action='store_true', help="Display result in window")
    args = parser.parse_args()

    main(args.weights, args.source, args.output, args.show)