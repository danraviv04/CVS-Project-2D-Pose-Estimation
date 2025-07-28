#!/usr/bin/env python3
"""This script pastes rendered tool images with transparency onto random backgrounds."""

import os
import random
import argparse
from PIL import Image
from tqdm import tqdm

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--images", type=str, default="/home/student/project/output",
        help="Directory containing transparent object images.")
    parser.add_argument(
        "-b", "--backgrounds", type=str, default="/datashare/project/train2017",
        help="Directory containing background images.")
    parser.add_argument(
        "-t", "--types", default=('jpg', 'jpeg', 'png'), type=str, nargs='+',
        help="File types to consider. Default: jp[e]g, png.")
    parser.add_argument(
        "-w", "--overwrite", action="store_true",
        help="If set, overwrite original images. Default: False.")
    parser.add_argument(
        "-o", "--output", default="composited", type=str,
        help="Output directory for composited images.")
    args = parser.parse_args()

    # Setup output directory
    output_dir = args.images if args.overwrite else args.output
    os.makedirs(output_dir, exist_ok=True)

    # Gather image and background filenames
    input_images = [f for f in os.listdir(args.images) if f.lower().endswith(tuple(args.types))]
    backgrounds = [os.path.join(args.backgrounds, f) for f in os.listdir(args.backgrounds)
                   if f.lower().endswith(tuple(args.types))]

    if not input_images:
        raise RuntimeError(f"No valid images found in {args.images}")
    if not backgrounds:
        raise RuntimeError(f"No valid backgrounds found in {args.backgrounds}")

    # Composite loop
    for file_name in tqdm(input_images, desc="Pasting objects"):
        img_path = os.path.join(args.images, file_name)
        img = Image.open(img_path).convert("RGBA")
        img_w, img_h = img.size

        bg_path = random.choice(backgrounds)
        background = Image.open(bg_path).convert("RGB").resize((img_w, img_h))

        # Paste with transparency mask (alpha channel)
        background.paste(img, (0, 0), mask=img.split()[-1])

        # Save composited image
        save_path = os.path.join(output_dir, file_name)
        background.save(save_path)

if __name__ == "__main__":
    main()