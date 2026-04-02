from __future__ import annotations

from pathlib import Path

from plant_3d.application.models import (
    MorphResult,
    PostSegResult,
    PrepSegResult,
    SegmentationSessionResult,
    TrackResult,
)
from plant_3d.application.ports import (
    CloudReaderPort,
    LightSimulationResult,
    LightingPort,
    MorphometricsPort,
    RegistrationPort,
    SegmentationPort,
    TrackingPort,
)
from plant_3d.application.types import LabeledPointCloud
from plant_3d.infrastructure.cpd_bridge import register_any_format
from plant_3d.infrastructure.growth_tracking.tracking import track_displacement
from plant_3d.infrastructure.io.formats import load_labeled_ply
from plant_3d.infrastructure.light_bridge import run_light_simulation
from plant_3d.infrastructure.morphometrics.metrics import compute_morphometrics
from plant_3d.infrastructure.seg_prep.pipeline import run_post_seg, run_prep_seg
from plant_3d.infrastructure.seg_prep.session import run_segmentation_session


class SegmentationAdapter(SegmentationPort):
    def run_prep(self, input_stl: str | Path, output_dir: str | Path, voxel_size: float) -> PrepSegResult:
        return run_prep_seg(input_stl=input_stl, output_dir=output_dir, voxel_size=voxel_size)

    def run_post(
        self,
        points_source: str | Path,
        labels_txt: str | Path,
        output_dir: str | Path,
        class_map: dict[int, tuple[int, int, int]],
    ) -> PostSegResult:
        return run_post_seg(
            points_source=points_source,
            labels_txt=labels_txt,
            output_dir=output_dir,
            class_map=class_map,
        )

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
    ) -> SegmentationSessionResult:
        return run_segmentation_session(
            input_stl=input_stl,
            output_dir=output_dir,
            voxel_size=voxel_size,
            class_map=class_map,
            labels_txt=labels_txt,
            editor_dir=editor_dir,
            auto_wait_seconds=auto_wait_seconds,
            run_editor=run_editor,
            run_in_background=run_in_background,
        )


class RegistrationAdapter(RegistrationPort):
    def register_any(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_dir: str | Path,
        config: object | None = None,
    ) -> object:
        return register_any_format(source_path, target_path, output_dir=output_dir, config=config)


class LightingAdapter(LightingPort):
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
    ) -> LightSimulationResult:
        result = run_light_simulation(
            input_glb=input_glb,
            output_dir=output_dir,
            light_type=light_type,
            light_pos=light_pos,
            light_dir=light_dir,
            power=power,
            scale_to_m=scale_to_m,
            photoperiod=photoperiod,
            add_lamp=add_lamp,
            tone_map_mode=tone_map_mode,
        )
        return LightSimulationResult(
            output_glb=result.output_glb,
            output_hdf5=result.output_hdf5,
            mean_ppfd=result.mean_ppfd,
            dli=result.dli,
        )


class CloudReaderAdapter(CloudReaderPort):
    def load_labeled(self, path: str | Path, labels_txt: str | Path | None = None) -> LabeledPointCloud:
        return load_labeled_ply(path, labels_txt=labels_txt)


class TrackingAdapter(TrackingPort):
    def track(
        self,
        source: LabeledPointCloud,
        target: LabeledPointCloud,
        output_dir: str | Path,
        mode: str,
    ) -> TrackResult:
        return track_displacement(source=source, target=target, output_dir=output_dir, mode=mode)


class MorphometricsAdapter(MorphometricsPort):
    def compute(
        self,
        cloud: LabeledPointCloud,
        output_dir: str | Path,
        stem_label: int,
        *,
        displacement_npz: str | Path | None = None,
        delta_time_hours: float | None = None,
        skeleton_bins: int = 30,
    ) -> MorphResult:
        return compute_morphometrics(
            cloud=cloud,
            output_dir=output_dir,
            stem_label=stem_label,
            displacement_npz=displacement_npz,
            delta_time_hours=delta_time_hours,
            skeleton_bins=skeleton_bins,
        )
