from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageStat


def _bucket(value: float, cuts: List[float], labels: List[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1] if labels else "mid"


def _analyze_image_impl(path: str) -> Tuple[float, ...]:
    p = Path(path)
    if not p.exists():
        return (0.0,) * 12

    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        stat = ImageStat.Stat(im)
        mean_r, mean_g, mean_b = [float(x) / 255.0 for x in stat.mean[:3]]
        std_r, std_g, std_b = [float(x) / 255.0 for x in stat.stddev[:3]]
        brightness = 0.2126 * mean_r + 0.7152 * mean_g + 0.0722 * mean_b
        contrast = (std_r + std_g + std_b) / 3.0

        ratio = float(w) / float(max(1, h))
        log_aspect = 0.0
        try:
            import math

            log_aspect = max(-2.0, min(2.0, math.log(max(1e-6, ratio))))
        except Exception:
            pass

        gray = im.convert("L").resize((min(64, max(1, w)), min(64, max(1, h))))
        px = list(gray.getdata())
        gw, gh = gray.size
        diff_sum = 0.0
        diff_count = 0
        for y in range(gh):
            row = y * gw
            for x in range(gw - 1):
                diff_sum += abs(float(px[row + x]) - float(px[row + x + 1]))
                diff_count += 1
        for y in range(gh - 1):
            row = y * gw
            next_row = (y + 1) * gw
            for x in range(gw):
                diff_sum += abs(float(px[row + x]) - float(px[next_row + x]))
                diff_count += 1
        edge_density = (diff_sum / max(1, diff_count)) / 255.0

        area_proxy = min(1.0, (float(w) * float(h)) / float(1024 * 1024))
        max_c = max(mean_r, mean_g, mean_b)
        min_c = min(mean_r, mean_g, mean_b)
        saturation_proxy = 0.0 if max_c < 1e-6 else (max_c - min_c) / max_c
        red_dom = max(0.0, mean_r - max(mean_g, mean_b))
        green_dom = max(0.0, mean_g - max(mean_r, mean_b))
        blue_dom = max(0.0, mean_b - max(mean_r, mean_g))

        return (
            float(w) / 1024.0,
            float(h) / 1024.0,
            log_aspect / 2.0,
            area_proxy,
            brightness,
            contrast,
            edge_density,
            saturation_proxy,
            mean_r,
            mean_g,
            mean_b,
            max(red_dom, green_dom, blue_dom),
        )


@lru_cache(maxsize=512)
def extract_image_numeric_features(path: str) -> Tuple[float, ...]:
    """
    Compact numeric image features for multimodal fusion in the text model.
    """
    try:
        return _analyze_image_impl(path)
    except Exception:
        return (0.0,) * 12


@lru_cache(maxsize=512)
def describe_image_for_text(path: str) -> str:
    """
    Lightweight image descriptor for text-only training:
    converts a local image into a compact textual hint (size/color/brightness/edges).
    """
    p = Path(path)
    if not p.exists():
        return f"missing_image:{path}"

    nums = extract_image_numeric_features(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        stat = ImageStat.Stat(im)
        mean_r, mean_g, mean_b = [float(x) for x in stat.mean[:3]]
        # Contrast proxy from per-channel stddev.
        std_r, std_g, std_b = [float(x) for x in stat.stddev[:3]]
        brightness = float(nums[4])
        contrast = float(nums[5]) * 2.0

        # Dominant color heuristic.
        if max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b) < 12:
            color_family = "gray"
        else:
            if mean_r >= mean_g and mean_r >= mean_b:
                color_family = "red"
            elif mean_g >= mean_r and mean_g >= mean_b:
                color_family = "green"
            else:
                color_family = "blue"

        aspect = "square"
        ratio = float(w) / float(max(1, h))
        if ratio > 1.15:
            aspect = "landscape"
        elif ratio < 0.87:
            aspect = "portrait"

        # Edge density proxy: grayscale adjacent differences.
        edge_density = float(nums[6])

        bright_label = _bucket(brightness, [0.25, 0.45, 0.70], ["dark", "dim", "mid", "bright"])
        contrast_label = _bucket(contrast, [0.08, 0.16, 0.28], ["flat", "soft", "medium", "high"])
        edge_label = _bucket(edge_density, [0.05, 0.10, 0.18], ["smooth", "low_edges", "medium_edges", "high_edges"])

        return (
            f"{aspect} {w}x{h}; color={color_family}; brightness={bright_label}; "
            f"contrast={contrast_label}; texture={edge_label}"
        )
