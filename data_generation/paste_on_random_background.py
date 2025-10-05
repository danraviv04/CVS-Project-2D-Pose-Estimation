#!/usr/bin/env python3
"""
Compose full-frame RGBA renders onto backgrounds without cropping or re-centering.

- Keeps the tool image exactly as rendered (no tight-crop, no resize, position=(0,0)).
- Background is resized to the output frame size (camera.json W/H if given, else tool.size).
- Optional OR-style relight + soft shadow (computed from the tool alpha).
- Optional 2-D hand cutouts with alpha, placed near a keypoint if provided.
- Writes a COCO JSON of full-silhouette masks (pre-occlusion) if --coco_out is provided.
"""

import os, random, argparse, json, math, re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm
import numpy as np
import cv2
from pycocotools import mask as m
from PIL import ImageFont, ImageDraw

def draw_label_on_top(img_rgba, text, xy, fg=(255,255,255), bg=(0,0,0), pad=2):
    """Draw readable label above everything with a tiny shadow box."""
    draw = ImageDraw.Draw(img_rgba)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox(xy, text, font=font)
    # pad background rectangle
    x0, y0, x1, y1 = bbox
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    draw.rectangle([x0, y0, x1, y1], fill=bg + (180,))   # semi-transparent bg
    draw.text(xy, text, fill=fg + (255,), font=font, stroke_width=2, stroke_fill=(0,0,0,255))

# ----------------- RLE / COCO helpers -----------------

def rle_from_rgba(tool_rgba, M, out_w, out_h, thr=8):
    """
    tool_rgba: PIL.Image RGBA of the tool BEFORE occluders
        M: 2x3 affine you used to place the tool (same rotation/scale/shift)
        out_w, out_h: canvas size
        thr: alpha threshold (0-255) to consider as foreground
    returns: (rle_dict, mask_bin) where rle_dict has ascii 'counts'
    """
    # 1) alpha -> binary mask
    alpha = np.array(tool_rgba.getchannel("A"), dtype=np.uint8)
    mask = (alpha >= thr).astype(np.uint8)

    # 2) warp mask exactly like the tool image
    mask_warp = cv2.warpAffine(
        mask, M, (out_w, out_h),
        flags=cv2.INTER_NEAREST, borderValue=0
    )
    # ensure strictly 0/1 uint8
    mask_warp = (mask_warp > 0).astype(np.uint8)

    # 3) encode to COCO RLE (expects Fortran-order HxWx1)
    rle = m.encode(np.asfortranarray(mask_warp[:, :, None]))

    # pycocotools may return a list even for single mask -> take first
    if isinstance(rle, list):
        rle = rle[0]

    # 'counts' may be bytes -> decode to ascii for JSON
    if isinstance(rle.get("counts", None), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")

    return rle, mask_warp

def coco_ann_from_mask(rle, mask_bin, cat_id, image_id, ann_id):
    """
    rle: RLE dict from rle_from_rgba()
        mask_bin: binary HxW uint8 mask (0/1) corresponding to rle
        cat_id: int, COCO category id
        image_id: int, COCO image id
        ann_id: int, COCO annotation id
    returns: COCO annotation dict, or None if mask is empty
    """
    ys, xs = np.where(mask_bin > 0)
    if ys.size == 0 or xs.size == 0:
        # empty mask -> caller should handle (we'll return None)
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": cat_id,
        "iscrowd": 0,
        "segmentation": rle,
        "area": float(m.area(rle)),
        "bbox": bbox,
    }

# ----------------- lighting helpers -----------------

HOTSPOT_PROB = 0.10

def make_dir_light_mask(size, direction=(0, -1), hard=1.25):
    """
    2D directional light ramp for relighting.
        size: (w, h)
        direction: (x, y) light direction vector (should be normalized)
        hard: float, higher = sharper light transition
    returns: PIL.Image L, light ramp in [0,255]
    """
    
    w, h = size
    v = np.array(direction, np.float32); v /= (np.linalg.norm(v) + 1e-6)
    xs = (np.arange(w, dtype=np.float32) - w / 2.0)[None, :]
    ys = (np.arange(h, dtype=np.float32) - h / 2.0)[:, None]
    dot = xs * v[0] + ys * v[1]
    dot = (dot - dot.min()) / (dot.max() - dot.min() + 1e-6)
    dot = np.clip(dot ** hard, 0, 1)
    return Image.fromarray((dot * 255).astype(np.uint8))

def estimate_bg_stats(bg_rgb_img, alpha01):
    """
    Estimate background luminance and tint from visible areas.
        bg_rgb_img: PIL.Image RGB background
        alpha01: HxW float mask in [0,1] where tool is present
    returns: (L_bg, tint_bg) where L_bg is float luminance, tint_bg is [r,g,b] mean color
    """
    
    bg_np = np.asarray(bg_rgb_img, np.float32) / 255.0
    inv = (alpha01 < 0.01)
    region = bg_np[inv] if inv.any() else bg_np.reshape(-1, 3)
    L = float((0.2126*region[:,0] + 0.7152*region[:,1] + 0.0722*region[:,2]).mean())
    tint = region.mean(0)
    return L, tint

def relight_tool_fullframe(tool_rgba, light_dir=(0, -1), strength=0.35,
                           contrast_boost=1.08, bg_stats=None):
    """
    Relight the tool image using a directional light ramp and optional background stats.
        tool_rgba: PIL.Image RGBA of the tool
        light_dir: (x, y) light direction vector (should be normalized)
        strength: float, relight strength
        contrast_boost: float, contrast enhancement after relight
        bg_stats: (L_bg, tint_bg) from estimate_bg_stats(), or None to skip
    returns: PIL.Image RGBA of relit tool (same size as input)
    """
    tool_rgba = tool_rgba.convert("RGBA")
    rgb, a = tool_rgba.convert("RGB"), tool_rgba.split()[-1]

    ramp = make_dir_light_mask(rgb.size, light_dir).filter(
        ImageFilter.GaussianBlur(radius=max(1, min(rgb.size)//110))
    )
    arr = np.asarray(ramp, np.float32) / 255.0
    light = (arr - 0.5) * 2.0 * strength

    rgb_arr = np.asarray(rgb, np.float32) / 255.0
    lit = np.clip(rgb_arr * (1.0 + light[..., None]), 0, 1)

    # cool tint in highlights
    highlight = np.clip(light, 0, None)[..., None]
    cool = np.array([0.98, 1.00, 1.06], np.float32)
    lit = lit * (1.0 - 0.25 * highlight) + (lit * cool) * (0.25 * highlight)

    # match background luminance/tint
    if bg_stats is not None:
        L_bg, tint_bg = bg_stats
        L_tool = (0.2126*lit[...,0] + 0.7152*lit[...,1] + 0.0722*lit[...,2]).mean()
        gain = np.clip(L_bg / (L_tool + 1e-6), 0.87, 1.12)
        lit = np.clip(lit * gain, 0, 1)
        lit = np.clip(lit * 0.95 + tint_bg * 0.05, 0, 1)

    rgb_lit = Image.fromarray((lit * 255).astype(np.uint8))
    rgb_lit = ImageEnhance.Contrast(rgb_lit).enhance(contrast_boost)
    return Image.merge("RGBA", (*rgb_lit.split(), a))

def add_grain(rgb, amount=0.015):
    """
    Add subtle Gaussian noise to an RGB image.
        rgb: PIL.Image RGB
        amount: float, noise standard deviation
    returns: PIL.Image RGB with noise added
    """
    arr = np.asarray(rgb, np.float32) / 255.0
    noise = np.random.normal(0, amount, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

def add_shadow(bg_rgba, tool_rgba, pos=(0, 0), light_dir=(0, -1),
               opacity=0.5, blur_frac=0.018, spread=1.12):
    """Soft drop shadow using tool alpha; tinted & multiplied into BG."""
    x, y = pos
    bg = bg_rgba.convert("RGBA")
    a = tool_rgba.split()[-1]
    if not a.getbbox():
        return bg_rgba  # no alpha → skip

    max_dim = max(bg.width, bg.height)
    blur = max(1, int(max_dim * blur_frac))

    lx, ly = light_dir
    off_x = int(-lx * max(4, tool_rgba.width * 0.065))
    off_y = int(-ly * max(4, tool_rgba.height * 0.065))

    # soft mask
    sh = a.resize((int(a.width * spread), int(a.height * spread)), Image.BICUBIC)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))

    # sample local BG to tint
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(bg.width,  x + a.width); y1 = min(bg.height, y + a.height)
    patch = np.asarray(bg.crop((x0, y0, x1, y1)).convert("RGB"), np.float32) / 255.0
    c = patch.mean((0, 1)) if patch.size else np.array([0.5, 0.5, 0.5], np.float32)

    # place mask on full canvas
    shadow_mask = Image.new("L", bg.size, 0)
    sx = x + off_x - (sh.width - a.width)//2
    sy = y + off_y - (sh.height - a.height)//2
    shadow_mask.paste(sh, (sx, sy))

    # multiply darkening with slight colorization toward local BG
    k = float(opacity)
    bg_np = np.asarray(bg.convert("RGB"), np.float32) / 255.0
    msk = (np.asarray(shadow_mask, np.float32) / 255.0)[..., None]
    tint = c[None, None, :]
    darkener = 1.0 - k * msk
    out = np.clip(bg_np * (darkener * (0.85 + 0.15 * tint)), 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8), "RGB").convert("RGBA")

# ----------------- hotspot -----------------

def _as_float_mask01(mask):
    """
    Convert a PIL or numpy mask to a float32 HxW array in [0,1].
        mask: PIL.Image or np.ndarray
    returns: np.ndarray HxW float32 in [0,1]
    """
    # PIL or np -> float HxW in [0,1]
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask)
    elif isinstance(mask, np.ndarray):
        arr = mask
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    if arr.ndim == 3:
        if arr.shape[-1] == 4:      # RGBA
            arr = arr[..., 3]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:                        # RGB -> luminance
            arr = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)

def _to_pil_L(x):
    """
    Convert a PIL or numpy array to a PIL.Image L (8-bit grayscale).
        x: PIL.Image or np.ndarray
    returns: PIL.Image L
    """
    
    if isinstance(x, Image.Image):
        return x if x.mode == "L" else x.convert("L")
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="L")

def _pil_gauss(x, sigma):
    im = _to_pil_L(x)
    return im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

def add_spot_hotspot(bg_rgb_img, alpha_mask01, intensity=None,
                     minmax=(0.02, 0.08), outside_only=True, match_bg=True):
    
    """
    Add a subtle bright spot around the tool using its alpha mask.
        bg_rgb_img: PIL.Image RGB background
        alpha_mask01: HxW float mask in [0,1] where tool is present (or None to skip)
        intensity: float or None, fixed intensity for the spot (overrides minmax)
        minmax: (min, max) range of random intensity if intensity is None
        outside_only: bool, whether to restrict spot to outside tool area
        match_bg: bool, whether to adjust spot brightness based on local bg
    returns: PIL.Image RGB with spot added
    """
    
    if alpha_mask01 is None:
        return bg_rgb_img
    alpha = _as_float_mask01(alpha_mask01)
    if alpha.max() <= 0.0:
        return bg_rgb_img

    bg_np = np.asarray(bg_rgb_img, dtype=np.float32) / 255.0
    yy, xx = np.where(alpha > 0.01)
    if yy.size:
        y0, y1 = max(0, yy.min()-40), min(bg_np.shape[0], yy.max()+40)
        x0, x1 = max(0, xx.min()-40), min(bg_np.shape[1], xx.max()+40)
        local = bg_np[y0:y1, x0:x1]
        if local.std() < 0.05:
            return bg_rgb_img

    H, W = alpha.shape
    diag = float(np.hypot(H, W))
    s_outer = min(max(1.0, 0.06 * diag), 80.0)
    s_inner = min(max(1.0, 0.02 * diag), 40.0)
    outer = np.asarray(_pil_gauss(alpha, s_outer), np.float32) / 255.0
    inner = np.asarray(_pil_gauss(alpha, s_inner), np.float32) / 255.0
    ring  = np.clip(outer - inner, 0.0, 1.0)
    if outside_only:
        ring *= (1.0 - alpha)

    gain = float(np.random.uniform(*minmax)) if intensity is None else float(intensity)

    if match_bg:
        luma = 0.2126 * bg_np[...,0] + 0.7152 * bg_np[...,1] + 0.0722 * bg_np[...,2]
        near = ring > 0.01
        m = float(np.median(luma[near])) if np.any(near) else float(np.median(luma))
        scale = float(np.interp(m, [0.15, 0.80], [1.0, 0.3]))
        gain *= scale

    spot   = (ring * gain)[..., None]
    out_np = 1.0 - (1.0 - bg_np) * (1.0 - spot)  # screen
    out_np = np.clip(out_np, 0.0, 1.0)
    return Image.fromarray((out_np * 255).astype(np.uint8), mode="RGB")

# ----------------- 2D hand overlay -----------------

def load_hand_pool(hands_dir):
    """
    Load all hand cutouts from a directory.
        hands_dir: str, directory containing hand images (PNG/WebP with alpha)
    """
    if not hands_dir or not os.path.isdir(hands_dir):
        return []
    exts = (".png", ".webp")
    pool = []
    for f in os.listdir(hands_dir):
        if f.lower().endswith(exts):
            try:
                im = Image.open(os.path.join(hands_dir, f)).convert("RGBA")
                if "A" in im.getbands() and im.getchannel("A").getbbox():
                    pool.append(im)
            except Exception:
                pass
    return pool

def augment_hand(im, rot_deg_range=(-25, 25), blur_prob=0.3, brighten=(0.9, 1.1), jpeg_prob=0.3):
    """Random augmentations for hand cutouts."""
    if random.random() < 0.5:
        im = ImageOps.mirror(im)
    im = im.rotate(random.uniform(*rot_deg_range), resample=Image.BICUBIC, expand=True)
    if random.random() < blur_prob:
        im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    if random.random() < 0.8:
        im = ImageEnhance.Brightness(im).enhance(random.uniform(*brighten))
    if random.random() < jpeg_prob:
        from io import BytesIO
        rgb, a = im.convert("RGB"), im.split()[-1]
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=random.randint(75, 92))
        buf.seek(0)
        rgb = Image.open(buf).convert("RGB")
        im = Image.merge("RGBA", (*rgb.split(), a))
    return im

def place_hand(bg, hand_rgba, tool_bbox, side="auto",
               scale_range=(0.85, 1.15), overlap_frac=0.50):
    """
    Place a hand cutout near the tool bbox on the background.
        bg: PIL.Image RGBA background
        hand_rgba: PIL.Image RGBA of the hand cutout
        tool_bbox: (x0, y0, x1, y1) bbox of the tool in the bg image
        side: "left", "right", "top", "bottom", or "auto" to choose randomly
        scale_range: (min, max) range to scale hand height relative to tool height
        overlap_frac: fraction of hand width/height to overlap with tool bbox
    returns: PIL.Image RGBA with hand composited
    """
    x0, y0, x1, y1 = tool_bbox
    tw, th = x1 - x0, y1 - y0
    if tw <= 0 or th <= 0:
        return bg

    if side == "auto":
        side = random.choices(["left", "right", "top", "bottom"], weights=[2, 2, 1, 1])[0]

    target_h = th * random.uniform(*scale_range)
    s = target_h / max(1, hand_rgba.height)
    hand = hand_rgba.resize((max(1, int(hand_rgba.width * s)),
                             max(1, int(hand_rgba.height * s))), Image.BICUBIC)

    if side == "left":
        x = int(x0 - hand.width * (1 - overlap_frac))
        y = int(y0 + th * random.uniform(0.15, 0.7) - hand.height * 0.5)
    elif side == "right":
        x = int(x1 - hand.width * overlap_frac)
        y = int(y0 + th * random.uniform(0.15, 0.7) - hand.height * 0.5)
    elif side == "top":
        x = int(x0 + tw * random.uniform(0.2, 0.8) - hand.width * 0.5)
        y = int(y0 - hand.height * (1 - overlap_frac))
    else:
        x = int(x0 + tw * random.uniform(0.2, 0.8) - hand.width * 0.5)
        y = int(y1 - hand.height * overlap_frac)

    canvas = bg.copy()
    canvas.alpha_composite(hand, (x, y))
    return canvas

# ----------------- background helpers -----------------

def numeric_stem(name: str) -> str:
    """
    Extract numeric stem from filename for sorting.
        name: str, filename
    returns: numeric part of stem or full stem if no digits
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"\d+", stem)
    return m.group(0) if m else stem

def resize_bg_to(bg_rgb, target_size, mode="cover"):
    """Resize background to target frame size without touching the tool."""
    W, H = target_size
    if mode == "stretch":
        return bg_rgb.resize((W, H), Image.BICUBIC)

    w, h = bg_rgb.size
    if mode == "contain":
        scale = min(W / w, H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        scaled = bg_rgb.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(scaled, ((W - nw) // 2, (H - nh) // 2))
        return canvas

    # cover
    scale = max(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    scaled = bg_rgb.resize((nw, nh), Image.BICUBIC)
    x0 = (nw - W) // 2
    y0 = (nh - H) // 2
    return scaled.crop((x0, y0, x0 + W, y0 + H))

def feather_rgba(im, r=1.0):
    """Feather the alpha channel of an RGBA image."""
    r_, g_, b_, a_ = im.split()
    size = max(3, int(2 * max(0.1, r)) | 1)  # odd kernel ≥3
    a_ = a_.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.GaussianBlur(0.6 * r))
    return Image.merge("RGBA", (r_, g_, b_, a_))

# ----------------- composition (no crop/recenter) -----------------

def compose_fullframe(bg_rgb, tool_rgba, hand_pool=None, hand_prob=0.65, keypoint_xy=None,
                      bg_fit_mode="cover"):
    """Keep tool at (0,0); background resized to frame size."""
    frame_size = tool_rgba.size
    bg_rgb = resize_bg_to(bg_rgb, frame_size, mode=bg_fit_mode)

    # OR lamp direction (mostly overhead)
    deg = 90 + random.uniform(-15, 15)
    ang = math.radians(deg)
    light_dir = (np.cos(ang), -np.sin(ang))

    a = tool_rgba.split()[-1]
    alpha01 = np.asarray(a, np.float32) / 255.0
    bg_stats = estimate_bg_stats(bg_rgb, alpha01)
    tool_relit = relight_tool_fullframe(
        tool_rgba, light_dir=light_dir,
        strength=random.uniform(0.25, 0.40),
        contrast_boost=1.08,
        bg_stats=bg_stats
    )

    tool_relit = feather_rgba(tool_relit, r=random.uniform(0.6, 1.4))

    a = tool_relit.split()[-1]
    alpha01 = np.asarray(a, dtype=np.float32) / 255.0

    if np.random.rand() < HOTSPOT_PROB:
        bg_rgb = add_spot_hotspot(
            bg_rgb_img=bg_rgb,
            alpha_mask01=alpha01,
            minmax=(0.02, 0.06),
            outside_only=True,
            match_bg=True
        )

    # Shadow then tool (fixed at 0,0)
    bg_rgba = add_shadow(
        bg_rgb, tool_relit, pos=(0, 0), light_dir=light_dir,
        opacity=random.uniform(0.35, 0.55),
        blur_frac=random.uniform(0.014, 0.022),
        spread=random.uniform(1.08, 1.16)
    )
    bg_rgba.alpha_composite(tool_relit, (0, 0))

    # Add grain
    bg_std = np.asarray(bg_rgba.convert("RGB"), np.uint8).std()
    grain_amt = 0.006 if bg_std < 25 else 0.012
    bg_rgb = add_grain(bg_rgba.convert("RGB"), amount=grain_amt)
    bg_rgba = bg_rgb.convert("RGBA")

    # Optional hand overlay
    alpha_bbox = a.getbbox()
    if hand_pool and len(hand_pool) and random.random() < hand_prob and alpha_bbox:
        hand = augment_hand(random.choice(hand_pool))
        x0, y0, x1, y1 = alpha_bbox
        th = (y1 - y0)

        if keypoint_xy:
            # Anchor near provided keypoint in frame coords
            kx, ky = keypoint_xy
            target_h = th * random.uniform(0.8, 1.15)
            s = target_h / max(1, hand.height)
            hand = hand.resize((max(1, int(hand.width * s)),
                                max(1, int(hand.height * s))), Image.BICUBIC)
            kx += random.randint(-int(th * 0.1), int(th * 0.1))
            ky += random.randint(-int(th * 0.1), int(th * 0.1))
            hx = int(kx - hand.width * 0.45)
            hy = int(ky - hand.height * 0.55)
            bg_rgba.alpha_composite(hand, (hx, hy))
        else:
            bg_rgba = place_hand(bg_rgba, hand, alpha_bbox, side="auto",
                                 scale_range=(0.85, 1.15), overlap_frac=0.50)

    if getattr(compose_fullframe, "_label_text", None): 
        try:
            # if you pass a text into compose_fullframe, use it; otherwise infer from filename upstream
            if hasattr(compose_fullframe, "_label_text") and compose_fullframe._label_text:
                label_txt = compose_fullframe._label_text
            else:
                label_txt = "Needle Holder"  # or "Tweezers" – set per image upstream

            # put it near the tool alpha bbox center (or wherever you prefer)
            if alpha_bbox:
                x0, y0, x1, y1 = alpha_bbox
                cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
                draw_label_on_top(bg_rgba, label_txt, (cx, cy))
        except Exception:
            pass

    return bg_rgba.convert("RGB")


# ----------------- CLI -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, required=True,
                        help="Folder with full-frame transparent tool renders (PNG/JPG with alpha).")
    parser.add_argument("-b", "--backgrounds", type=str, required=True,
                        help="Folder with background photos.")
    parser.add_argument("-t", "--types", default=('jpg', 'jpeg', 'png', 'webp'), type=str, nargs='+',
                        help="Background file types.")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Output folder.")
    parser.add_argument("-w", "--overwrite", action="store_true",
                        help="Overwrite images in --images instead of writing to --output.")
    parser.add_argument("--camera_json", type=str, default=None,
                        help="If given, force output W/H (must match render size).")
    parser.add_argument("--bg_fit", type=str, default="cover", choices=["cover", "contain", "stretch"],
                        help="How to fit background to the frame size (tool stays untouched).")
    parser.add_argument("--hands", type=str, default=None,
                        help="Folder of hand PNG/WebP cutouts with alpha.")
    parser.add_argument("--hand_prob", type=float, default=0.65,
                        help="Probability to overlay a hand.")
    parser.add_argument("--keypoints_dir", type=str, default=None,
                        help="Folder with keypoint JSON per image (same numeric stem).")
    parser.add_argument("--coco_out", type=str, default=None,
                        help="If set, write a single COCO json here (full-silhouette segs).")
    parser.add_argument("--add_text_labels", action="store_true",
                    help="Overlay tool name text on composites")

    args = parser.parse_args()

    out_dir = args.images if args.overwrite else args.output
    os.makedirs(out_dir, exist_ok=True)

    # COCO buffers
    images_coco, annotations_coco = [], []
    categories_coco = [
        {"id": 1, "name": "needle_holder", "supercategory": "surgical_tool"},
        {"id": 2, "name": "tweezers",      "supercategory": "surgical_tool"},
    ]
    next_img_id = 1
    next_ann_id = 1

    def cat_id_from_filename(fname: str) -> int:
        s = os.path.basename(fname).lower()
        return 1 if s.startswith("nh") or "needle" in s else 2  # NH -> 1, T -> 2

    # Gather tool images
    tool_files = [f for f in os.listdir(args.images)
                  if f.lower().endswith((".png", ".webp", ".jpg", ".jpeg"))]
    if not tool_files:
        raise RuntimeError(f"No tool images found in {args.images}")

    # Gather backgrounds
    bg_files = [os.path.join(args.backgrounds, f) for f in os.listdir(args.backgrounds)
                if f.lower().endswith(tuple(args.types))]
    if not bg_files:
        raise RuntimeError(f"No background images found in {args.backgrounds}")

    # Optional hand pool
    hand_pool = load_hand_pool(args.hands) if args.hands else []

    # Optional fixed size from camera.json (sanity check only)
    forced_size = None
    if args.camera_json and os.path.exists(args.camera_json):
        with open(args.camera_json, "r") as f:
            cam = json.load(f)
        forced_size = (int(cam["width"]), int(cam["height"]))

    def read_ring_kp(stem):
        if not args.keypoints_dir:
            return None
        jp = os.path.join(args.keypoints_dir, f"{stem}.json")
        if not os.path.exists(jp):
            return None
        try:
            with open(jp, "r") as f:
                data = json.load(f)
            for obj in data:
                kps = obj.get("keypoints", {})
                rR = kps.get("ring_right", [0, 0, 0])
                rL = kps.get("ring_left",  [0, 0, 0])
                if rR[2] > 0 and rL[2] > 0:
                    return ((rR[0] + rL[0]) // 2, (rR[1] + rL[1]) // 2)
                if rR[2] > 0:
                    return (rR[0], rR[1])
                if rL[2] > 0:
                    return (rL[0], rL[1])
        except Exception:
            return None
        return None

    saved_count = 0
    ann_count = 0

    for name in tqdm(tool_files, desc="Compositing"):
        tool_path = os.path.join(args.images, name)
        tool_rgba = Image.open(tool_path).convert("RGBA")

        # Enforce expected frame size if supplied (we keep tool untouched regardless)
        if forced_size and tool_rgba.size != forced_size:
            # optional: warn; we still use tool size as frame
            pass

        # --- FULL-SILHOUETTE RLE (pre-occlusion) ---
        W, H = tool_rgba.size
        M_id = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # identity warp
        rle, mask_bin = rle_from_rgba(tool_rgba, M_id, W, H, thr=8)

        img_id = next_img_id
        cat_id = cat_id_from_filename(name)
        ann = coco_ann_from_mask(rle, mask_bin, cat_id=cat_id, image_id=img_id, ann_id=next_ann_id)

        # Prepare background to match full frame
        bg = Image.open(random.choice(bg_files)).convert("RGB")
        if forced_size:
            bg = resize_bg_to(bg, forced_size, mode=args.bg_fit)
        else:
            bg = resize_bg_to(bg, tool_rgba.size, mode=args.bg_fit)

        # Optional ring anchor from keypoints (frame coords)
        stem = numeric_stem(name)
        ring_xy = read_ring_kp(stem)

        # Compose (hands/shadows added AFTER silhouette capture)
        compose_fullframe._label_text = ("Needle Holder" if cat_id == 1 else "Tweezers") if args.add_text_labels else None
        
        out = compose_fullframe(
            bg_rgb=bg,
            tool_rgba=tool_rgba,
            hand_pool=hand_pool,
            hand_prob=args.hand_prob,
            keypoint_xy=ring_xy,
            bg_fit_mode=args.bg_fit
        )
        
        compose_fullframe._label_text = None  # cleanup

        # Save composited image
        out_path = os.path.join(out_dir, name)
        out.save(out_path, quality=95)
        saved_count += 1

        # Always trust the saved image size
        W, H = out.size

        # COCO image entry (always add the image)
        images_coco.append({
            "id": img_id,
            "file_name": name,  # or relpath if you prefer
            "width": W,
            "height": H
        })
        next_img_id += 1

        # COCO annotation entry (only if mask not empty)
        if ann is not None:
            # ann["image_id"] must equal img_id from above (your code sets it earlier)
            annotations_coco.append(ann)
            next_ann_id += 1
            ann_count += 1
        else:
            print(f"No foreground pixels in {name}; added image without annotation.")


    # Write COCO JSON
    if args.coco_out:
        coco = {
            "images": images_coco,
            "annotations": annotations_coco,
            "categories": categories_coco
        }
        os.makedirs(os.path.dirname(args.coco_out) or ".", exist_ok=True)
        with open(args.coco_out, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"✅ Wrote COCO: {args.coco_out} ({len(images_coco)} images, {len(annotations_coco)} annotations)")

    print(f"✅ Done. Saved {saved_count} images to: {out_dir}")

if __name__ == "__main__":
    main()