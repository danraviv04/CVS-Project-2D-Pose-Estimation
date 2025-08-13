# #!/usr/bin/env python3
# """This script pastes rendered tool images with transparency onto random backgrounds."""

# import os
# import random
# import argparse
# from PIL import Image
# from tqdm import tqdm

# def main():
#     # Parse CLI arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-i", "--images", type=str, default="/home/student/project/output",
#         help="Directory containing transparent object images.")
#     parser.add_argument(
#         "-b", "--backgrounds", type=str, default="/datashare/project/train2017",
#         help="Directory containing background images.")
#     parser.add_argument(
#         "-t", "--types", default=('jpg', 'jpeg', 'png'), type=str, nargs='+',
#         help="File types to consider. Default: jp[e]g, png.")
#     parser.add_argument(
#         "-w", "--overwrite", action="store_true",
#         heAlp="If set, overwrite original images. Default: False.")
#     parser.add_argument(
#         "-o", "--output", default="composited", type=str,
#         help="Output directory for composited images.")
#     args = parser.parse_args()

#     # Setup output directory
#     output_dir = args.images if args.overwrite else args.output
#     os.makedirs(output_dir, exist_ok=True)

#     # Gather image and background filenames
#     input_images = [f for f in os.listdir(args.images) if f.lower().endswith(tuple(args.types))]
#     backgrounds = [os.path.join(args.backgrounds, f) for f in os.listdir(args.backgrounds)
#                    if f.lower().endswith(tuple(args.types))]

#     if not input_images:
#         raise RuntimeError(f"No valid images found in {args.images}")
#     if not backgrounds:
#         raise RuntimeError(f"No valid backgrounds found in {args.backgrounds}")

#     # Composite loop
#     for file_name in tqdm(input_images, desc="Pasting objects"):
#         img_path = os.path.join(args.images, file_name)
#         img = Image.open(img_path).convert("RGBA")
#         img_w, img_h = img.size

#         bg_path = random.choice(backgrounds)
#         background = Image.open(bg_path).convert("RGB").resize((img_w, img_h))

#         # Paste with transparency mask (alpha channel)
#         background.paste(img, (0, 0), mask=img.split()[-1])

#         # Save composited image
#         save_path = os.path.join(output_dir, file_name)
#         background.save(save_path)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""Paste rendered RGBA tool images onto random backgrounds (flat output),
centered, with simple OR-style directional lighting + soft shadow."""
import os, random, argparse, json
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import numpy as np

# ---------- helpers ----------
def tight_crop_rgba(img, pad=2):
    a = img.split()[-1]
    box = a.getbbox()
    if not box:
        return img
    x0, y0, x1, y1 = box
    return img.crop((max(0, x0 - pad), max(0, y0 - pad),
                     min(img.width, x1 + pad), min(img.height, y1 + pad)))

def make_dir_light_mask(size, direction=(0, -1), hard=1.25):
    """0..1 ramp along a direction (slightly harder falloff for OR feel)."""
    w, h = size
    v = np.array(direction, np.float32); v /= (np.linalg.norm(v) + 1e-6)
    xs = (np.arange(w, dtype=np.float32) - w/2.0)[None, :]
    ys = (np.arange(h, dtype=np.float32) - h/2.0)[:, None]
    dot = xs * v[0] + ys * v[1]
    dot = (dot - dot.min()) / (dot.max() - dot.min() + 1e-6)
    dot = np.clip(dot ** hard, 0, 1)
    return Image.fromarray((dot * 255).astype(np.uint8))

def relight_tool(tool, light_dir=(0, -1), strength=0.4, contrast_boost=1.12):
    """Directional brighten/darken + subtle cool tint in highlights."""
    tool = tool.convert("RGBA")
    rgb, a = tool.convert("RGB"), tool.split()[-1]

    ramp = make_dir_light_mask(rgb.size, light_dir).filter(
        ImageFilter.GaussianBlur(radius=max(1, min(rgb.size)//110))
    )
    arr = np.asarray(ramp, np.float32) / 255.0
    light = (arr - 0.5) * 2.0 * strength                       # [-s, s]

    rgb_arr = np.asarray(rgb, np.float32) / 255.0
    # base directional light
    lit = np.clip(rgb_arr * (1.0 + light[..., None]), 0, 1)

    # cool tint only where light > 0 (highlights)
    highlight = np.clip(light, 0, None)[..., None]
    cool = np.array([0.98, 1.00, 1.06], dtype=np.float32)      # gentle blue
    lit = lit * (1.0 - 0.25 * highlight) + (lit * cool) * (0.25 * highlight)

    rgb_lit = Image.fromarray((np.clip(lit, 0, 1) * 255).astype(np.uint8))
    rgb_lit = ImageEnhance.Contrast(rgb_lit).enhance(contrast_boost)

    return Image.merge("RGBA", (*rgb_lit.split(), a))

def add_shadow(bg_rgba, tool_rgba, pos, light_dir=(0, -1),
               opacity=0.5, blur_frac=0.018, spread=1.12):
    """More directional, slightly tighter OR-like shadow."""
    x, y = pos
    bg = bg_rgba.convert("RGBA")
    a = tool_rgba.split()[-1]

    max_dim = max(bg.width, bg.height)
    blur = max(1, int(max_dim * blur_frac))

    lx, ly = light_dir
    off_x = int(-lx * max(4, tool_rgba.width * 0.065))
    off_y = int(-ly * max(4, tool_rgba.height * 0.065))

    sh = a.resize((int(a.width * spread), int(a.height * spread)), Image.BICUBIC)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))

    shadow_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    black = Image.new("RGBA", sh.size, (0, 0, 0, int(255 * opacity)))
    sx = x + off_x - (sh.width - a.width)//2
    sy = y + off_y - (sh.height - a.height)//2
    shadow_layer.paste(black, (sx, sy), mask=sh)

    bg.alpha_composite(shadow_layer)
    return bg

def add_spot_hotspot(bg_rgb, center, radius_frac=0.18, intensity=0.08):
    """Subtle cool-ish hotspot under the tool (simulating an OR lamp spot)."""
    cx, cy = center
    W, H = bg_rgb.size
    R = int(min(W, H) * radius_frac)
    if R <= 0:
        return bg_rgb

    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = 1.0 - np.clip(mask / float(R * R), 0, 1)  # 1 in center -> 0 at edge
    spot = Image.fromarray((mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=max(2, R // 4))
    )

    bg = bg_rgb.convert("RGBA")
    tint = Image.new("RGBA", bg.size, (235, 240, 255, int(255 * intensity)))
    bg.paste(tint, (0, 0), mask=spot)  # ← fixed: use paste() with mask
    return bg.convert("RGB")

# def center_paste_with_lighting(bg_rgb, tool_rgba, min_px=180, max_px=360):
#     """Center the tool, OR-style light + shadow, return RGB bg."""
#     tool = tight_crop_rgba(tool_rgba)
#     a = tool.split()[-1]
#     if not a.getbbox():
#         return bg_rgb.convert("RGB")

#     # OR lamp: mostly overhead; allow small ±15° jitter
#     import math
#     deg = 90 + random.uniform(-15, 15)     # 90° ≈ straight down
#     ang = math.radians(deg)
#     light_dir = (np.cos(ang), -np.sin(ang))  # screen-space

#     # scale by longest side
#     # w, h = tool.size
#     # target = random.randint(min_px, max_px)
#     # s = target / float(max(w, h)) if max(w, h) > 0 else 1.0
#     # tool = tool.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BICUBIC)

#     # ensure not tiny
#     # if min(tool.width, tool.height) < 48:
#     #     s = 48.0 / float(min(tool.width, tool.height))
#     #     tool = tool.resize((int(tool.width*s), int(tool.height*s)), Image.BICUBIC)

#     # relight
#     tool = relight_tool(tool, light_dir=light_dir, strength=0.4, contrast_boost=1.12)

#     # center position
#     x = (bg_rgb.width - tool.width)//2
#     y = (bg_rgb.height - tool.height)//2

#     # subtle hotspot beneath tool
#     bg_rgb = add_spot_hotspot(bg_rgb, (x + tool.width//2, y + tool.height//2),
#                               radius_frac=0.16, intensity=0.07)

#     # shadow then paste
#     bg_rgba = add_shadow(bg_rgb, tool, (x, y), light_dir=light_dir,
#                          opacity=0.5, blur_frac=0.018, spread=1.12)
#     bg_rgba.alpha_composite(tool, (x, y))
#     return bg_rgba.convert("RGB")

def center_paste_with_lighting(bg_rgb, tool_rgba, min_px=180, max_px=360):
    # PRESERVE CANVAS: do NOT crop or resize
    tool = tool_rgba.convert("RGBA")

    # OR-style light direction (mostly overhead)
    import math, random as _r
    deg = 90 + _r.uniform(-15, 15)
    ang = math.radians(deg)
    light_dir = (np.cos(ang), -np.sin(ang))

    # relight on full canvas (alpha protects background)
    tool = relight_tool(tool, light_dir=light_dir, strength=0.4, contrast_boost=1.12)

    # paste position: top-left of the same-sized canvas → labels unchanged
    x, y = 0, 0

    # optional hotspot under existing tool center (uses the same canvas)
    cx, cy = tool.width // 2, tool.height // 2
    bg_rgb = add_spot_hotspot(bg_rgb, (cx, cy), radius_frac=0.16, intensity=0.07)

    # shadow using the tool alpha
    bg_rgba = add_shadow(bg_rgb, tool, (x, y), light_dir=light_dir,
                         opacity=0.5, blur_frac=0.018, spread=1.12)
    bg_rgba.alpha_composite(tool, (x, y))
    return bg_rgba.convert("RGB")

# ---------- main (unchanged CLI/IO) ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, default="/home/student/project/output",
                        help="Directory containing transparent object images.")
    parser.add_argument("-b", "--backgrounds", type=str, default="/datashare/project/train2017",
                        help="Directory containing background images.")
    parser.add_argument("-t", "--types", default=('jpg', 'jpeg', 'png'), type=str, nargs='+',
                        help="File types to consider. Default: jpg, jpeg, png.")
    parser.add_argument("-w", "--overwrite", action="store_true",
                        help="If set, overwrite original images. Default: False.")
    parser.add_argument("-o", "--output", default="/home/student/project/output/composited", type=str,
                        help="Output directory for composited images.")
    parser.add_argument("--camera_json", type=str, default=None,
                        help="Optional camera.json with width/height to force output size.")
    parser.add_argument("--min_px", type=int, default=180,
                        help="Min longest-side pixels for pasted tool.")
    parser.add_argument("--max_px", type=int, default=360,
                        help="Max longest-side pixels for pasted tool.")
    args = parser.parse_args()

    output_dir = args.images if args.overwrite else args.output
    os.makedirs(output_dir, exist_ok=True)

    # optional fixed output size
    target_size = None
    if args.camera_json and os.path.exists(args.camera_json):
        with open(args.camera_json, "r") as f:
            cam = json.load(f)
        target_size = (int(cam["width"]), int(cam["height"]))

    input_images = [f for f in os.listdir(args.images)
                    if f.lower().endswith(tuple(args.types))]
    backgrounds = [os.path.join(args.backgrounds, f) for f in os.listdir(args.backgrounds)
                   if f.lower().endswith(tuple(args.types))]

    if not input_images:
        raise RuntimeError(f"No valid images found in {args.images}")
    if not backgrounds:
        raise RuntimeError(f"No valid backgrounds found in {args.backgrounds}")

    for file_name in tqdm(input_images, desc="Pasting objects"):
        tool_path = os.path.join(args.images, file_name)
        tool_rgba = Image.open(tool_path).convert("RGBA")

        if target_size:
            bg = Image.open(random.choice(backgrounds)).convert("RGB").resize(target_size, Image.BICUBIC)
        else:
            bg = Image.open(random.choice(backgrounds)).convert("RGB").resize(tool_rgba.size, Image.BICUBIC)

        out = center_paste_with_lighting(bg, tool_rgba,
                                         min_px=args.min_px, max_px=args.max_px)

        out.save(os.path.join(output_dir, file_name))

if __name__ == "__main__":
    main()