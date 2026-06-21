from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from plant_3d.application.generation import GenerationConfig, GenerationResult, PreprocessResult
from plant_3d.application.models import (
    MorphResult,
    PostSegResult,
    PrepSegResult,
    SegmentationSessionResult,
    TrackResult,
)
from plant_3d.application.types import LabeledPointCloud


@dataclass(frozen=True)
class LightSimulationResult:
    output_glb: Path
    output_hdf5: Path
    mean_ppfd: float
    dli: float


class PreprocessPort(Protocol):
    def prep_images(
        self,
        images: list[str | Path],
        output_dir: str | Path,
        *,
        target_size: int | None = None,
    ) -> PreprocessResult: ...


class GenerationPort(Protocol):
    def generate_3d(
        self,
        images: list[str | Path],
        output_dir: str | Path,
        config: GenerationConfig,
    ) -> GenerationResult: ...


class SegmentationPort(Protocol):
    def run_prep(self, input_stl: str | Path, output_dir: str | Path, voxel_size: float) -> PrepSegResult: ...

    def run_post(
        self,
        points_source: str | Path,
        labels_txt: str | Path,
        output_dir: str | Path,
        class_map: dict[int, tuple[int, int, int]],
    ) -> PostSegResult: ...

    def run_session(
        self,
        input_stl: str | Path,
        output_dir: str | Path,
        voxel_size: float,
        class_map: dict[int, tuple[int, int, int]],
        labels_txt: str | Path | None,
        editor_dir: str | Path | None,
        auto_wait_seconds: int,
        run_editor: bool,
        run_in_background: bool,
    ) -> SegmentationSessionResult: ...


class RegistrationPort(Protocol):
    def register_any(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_dir: str | Path,
        config: object | None = None,
    ) -> object: ...


class LightingPort(Protocol):
    def run_light(
        self,
        input_glb: str | Path,
        output_dir: str | Path,
        light_type: str,
        light_pos: tuple[float, float, float],
        light_dir: tuple[float, float, float],
        power: float,
        scale_to_m: float,
        photoperiod: float,
        add_lamp: bool,
        tone_map_mode: str,
    ) -> LightSimulationResult: ...


class CloudReaderPort(Protocol):
    def load_labeled(self, path: str | Path, labels_txt: str | Path | None = None) -> LabeledPointCloud: ...


class TrackingPort(Protocol):
    def track(
        self,
        source: LabeledPointCloud,
        target: LabeledPointCloud,
        output_dir: str | Path,
        mode: str,
    ) -> TrackResult: ...


class MorphometricsPort(Protocol):
    def compute(
        self,
        cloud: LabeledPointCloud,
        output_dir: str | Path,
        stem_label: int,
        *,
        displacement_npz: str | Path | None = None,
        delta_time_hours: float | None = None,
        skeleton_bins: int = 30,
    ) -> MorphResult: ...
