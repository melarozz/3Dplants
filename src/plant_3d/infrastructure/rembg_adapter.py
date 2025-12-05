"""E0 image preprocessing via rembg (background removal + centering)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from plant_3d.application.generation import PreprocessResult
from plant_3d.application.preprocess import center_rgba_canvas
from plant_3d.application.ports import PreprocessPort


def _remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
    except ImportError as exc:
        raise ImportError(
            "rembg is required for prep-images. Install with: pip install -e '.[generation]'",
        ) from exc
    return remove(image)


class RembgAdapter(PreprocessPort):
    def prep_images(
        self,
        images: list[str | Path],
        output_dir: str | Path,
        *,
        target_size: int | None = None,
    ) -> PreprocessResult:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        prepared: list[Path] = []
        for idx, raw in enumerate(images):
            src = Path(raw).expanduser().resolve()
            with Image.open(src) as im:
                rgba = _remove_background(im.convert("RGBA"))
            arr = np.asarray(rgba, dtype=np.uint8)
            centered = center_rgba_canvas(arr, target_size=target_size)
            dest = out / f"{src.stem}_prepared.png"
            Image.fromarray(centered, mode="RGBA").save(dest)
            prepared.append(dest)
        return PreprocessResult(output_dir=out, prepared_images=prepared)
