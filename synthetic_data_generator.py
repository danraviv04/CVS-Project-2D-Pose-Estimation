import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100, help="Number of synthetic images to generate.")
    parser.add_argument("--output_dir", type=str, default="/home/student/project/output", help="Output directory for generated and composited images.")
    parser.add_argument("--backgrounds_dir", type=str, default="/datashare/project/train2017", help="Directory with background images.")
    args = parser.parse_args()

    images_dir = os.path.join(args.output_dir, "coco_data", "images")  # ✅ updated path

    # Step 1: Generate transparent tool renderings
    print("🛠️ Generating transparent tool renderings...")
    subprocess.run([
        "blenderproc", "run", "data_generation/generate_tools.py",
        "--obj_dir", "/datashare/project/surgical_tools_models",
        "--camera_params", "/datashare/project/camera.json",
        "--output_dir", args.output_dir,
        "--num_images", str(args.num_images)
    ], check=True)

    # Step 2: Paste those onto random backgrounds
    print("🖼️ Pasting tools onto random backgrounds...")
    subprocess.run([
        "python3", "data_generation/paste_on_random_background.py",
        "-i", images_dir,
        "-b", args.backgrounds_dir,
        "-o", os.path.join(args.output_dir, "composited")
    ], check=True)

    print("✅ Done! Composited images are ready.")

if __name__ == "__main__":
    main()