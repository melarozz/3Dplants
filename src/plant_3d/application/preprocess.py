from __future__ import annotations

import numpy as np


def center_rgba_canvas(rgba: np.ndarray, *, target_size: int | None = None) -> np.ndarray:
    """Crop to non-transparent alpha bbox and center on a square RGBA canvas."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA array, got shape {rgba.shape}")
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return rgba.copy()
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgba[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = int(max(ch, cw) if target_size is None else target_size)
    canvas = np.zeros((side, side, 4), dtype=rgba.dtype)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    canvas[y_off : y_off + ch, x_off : x_off + cw] = crop
    return canvas
