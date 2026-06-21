from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    octree_resolution: int = 256
    mc_level: float = 0.0
    target_image_size: int | None = None


@dataclass(frozen=True)
class PreprocessResult:
    output_dir: Path
    prepared_images: list[Path]


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    stl_path: Path
    glb_path: Path | None
    config: GenerationConfig
