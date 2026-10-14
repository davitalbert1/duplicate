import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def extract_film_from_slide(img: Image.Image) -> Image.Image:
    """
    Detects the film aperture window inside a photographed slide frame
    and extracts only the actual photo content, discarding the plastic/cardboard
    slide mount, backlighting background, and digitizer borders.
    
    If the image is already cropped (e.g. standard flatbed scan or document),
    returns the original image.
    """
    orig_w, orig_h = img.size
    
    # Slides photographed with DSLR/camera usually have high resolution and horizontal aspect ratio
    if orig_w > 1800 and orig_h > 1200 and (orig_w / orig_h) > 1.2:
        # Downscale for fast, robust contour analysis
        tw = 400
        scale = tw / orig_w
        th = int(orig_h * scale)
        small = img.resize((tw, th), Image.Resampling.BILINEAR)
        gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
        
        # In typical slide setups (like in E:\backup\digitalização), the slide frame is in the left 75%
        roi_w = int(tw * 0.72)
        roi_gray = gray[:, :roi_w]

        # Filter and edge detection
        edges = cv2.Canny(roi_gray, 35, 110)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            asp = cw / float(ch)
            area = cw * ch
            # Check standard 35mm film aperture in this rig:
            # Horizontal aperture: asp in [1.1, 1.75], cw in [70, 240], ch in [50, 180]
            # Vertical aperture: asp in [0.55, 0.90], cw in [50, 180], ch in [70, 240]
            is_h = (1.10 <= asp <= 1.75 and 70 <= cw <= 240 and 50 <= ch <= 180)
            is_v = (0.55 <= asp <= 0.90 and 50 <= cw <= 180 and 70 <= ch <= 240)
            if (is_h or is_v) and (10 <= x <= 250 and 20 <= y <= 240):
                candidates.append((area, x, y, cw, ch, is_h))

        if candidates:
            # Pick largest candidate contour matching aperture proportions
            candidates.sort(key=lambda c: c[0], reverse=True)
            _, x, y, cw, ch, is_h = candidates[0]

            # Map back to original full-resolution coordinates
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + cw) / scale)
            y2 = int((y + ch) / scale)

            # Shave 6% inner margin to guarantee no plastic mount / bezel borders enter the visual features
            pad_x = int((x2 - x1) * 0.06)
            pad_y = int((y2 - y1) * 0.06)

            crop_box = (
                max(0, x1 + pad_x),
                max(0, y1 + pad_y),
                min(orig_w, x2 - pad_x),
                min(orig_h, y2 - pad_y)
            )
            return img.crop(crop_box)
        else:
            # Fallback based on digitizer rig geometry
            fallback_box = (
                int(0.12 * orig_w),
                int(0.24 * orig_h),
                int(0.55 * orig_w),
                int(0.76 * orig_h)
            )
            return img.crop(fallback_box)
            
    # For scanned documents or non-slide formats
    return img


def load_film_crop_fast(img_path: str, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Fast loader using JPEG draft subsampling to load high-resolution camera slides
    in under ~40ms instead of 300ms, while preserving crop fidelity.
    """
    with Image.open(img_path) as raw:
        w, h = raw.size
        if w > 1800 and h > 1200 and (w / h) > 1.2:
            # Subsample directly in JPEG decoder
            raw.draft('RGB', (800, int(800 * (h / w))))
            rgb = raw.convert("RGB")
            crop = extract_film_from_slide(rgb)
            return crop.resize(target_size, Image.Resampling.BILINEAR)
        else:
            raw.draft('RGB', (target_size[0] * 2, target_size[1] * 2))
            rgb = raw.convert("RGB")
            return rgb.resize(target_size, Image.Resampling.BILINEAR)
