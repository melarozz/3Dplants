from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plant_3d.application.generation import GenerationConfig
from plant_3d.application.models import (
    MorphResult,
    PostSegResult,
    PrepSegResult,
    SegmentationSessionResult,
    TrackResult,
)
from plant_3d.application.ports import (
    CloudReaderPort,
    GenerationPort,
    LightingPort,
    LightSimulationResult,
    MorphometricsPort,
    PreprocessPort,
    RegistrationPort,
    SegmentationPort,
    TrackingPort,
)
from plant_3d.infrastructure.metric_calibration import calibrate_files


@dataclass
class Plant3DService:
    preprocess: PreprocessPort
    generation: GenerationPort
    segmentation: SegmentationPort
    registration: RegistrationPort
    lighting: LightingPort
    clouds: CloudReaderPort
    tracking: TrackingPort
    morph: MorphometricsPort

    def prep_images(
        self,
        images: list[str | Path],
        output_dir: str | Path,
        *,
        target_size: int | None = None,
    ):
        return self.preprocess.prep_images(images, output_dir, target_size=target_size)

    def generate_3d(
        self,
        images: list[str | Path],
        output_dir: str | Path,
        config: GenerationConfig,
    ):
        return self.generation.generate_3d(images, output_dir, config)

    def prep_seg(self, input_stl: str | Path, output_dir: str | Path, voxel_size: float) -> PrepSegResult:
        return self.segmentation.run_prep(input_stl, output_dir, voxel_size)

    def post_seg(
        self,
        points_source: str | Path,
        labels_txt: str | Path,
        output_dir: str | Path,
        class_map: dict[int, tuple[int, int, int]],
    ) -> PostSegResult:
        return self.segmentation.run_post(points_source, labels_txt, output_dir, class_map)

    def segment_session(
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
        return self.segmentation.run_session(
            input_stl,
            output_dir,
            voxel_size,
            class_map,
            labels_txt,
            editor_dir,
            auto_wait_seconds,
            run_editor,
            run_in_background,
        )

    def light(
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
        return self.lighting.run_light(
            input_glb,
            output_dir,
            light_type,
            light_pos,
            light_dir,
            power,
            scale_to_m,
            photoperiod,
            add_lamp,
            tone_map_mode,
        )

    def track_growth(
        self,
        source_ply: str | Path,
        target_ply: str | Path,
        source_labels: str | Path | None,
        target_labels: str | Path | None,
        output_dir: str | Path,
    ) -> TrackResult:
        source = self.clouds.load_labeled(source_ply, source_labels)
        target = self.clouds.load_labeled(target_ply, target_labels)
        return self.tracking.track(source, target, output_dir, mode="track-growth")

    def track_original(
        self,
        source: str | Path,
        reference: str | Path,
        output_dir: str | Path,
        registration_config: object | None,
        source_ply: str | Path | None,
        reference_ply: str | Path | None,
        source_labels: str | Path | None,
        reference_labels: str | Path | None,
    ) -> tuple[object, TrackResult | None]:
        cpd_result = self.registration.register_any(
            source,
            reference,
            output_dir,
            config=registration_config,
        )
        track_result = None
        if source_ply and reference_ply:
            source_cloud = self.clouds.load_labeled(source_ply, source_labels)
            target_cloud = self.clouds.load_labeled(reference_ply, reference_labels)
            track_result = self.tracking.track(
                source_cloud,
                target_cloud,
                output_dir,
                mode="track-original",
            )
        return cpd_result, track_result

    def calibrate_metric(
        self,
        inputs: list[str | Path],
        output_dir: str | Path,
        reference_mesh: str | Path | None,
        physical_extent: float | None,
        unit: str,
        axis: str,
        uniform_scale: float | None,
        shift_ground: bool,
        suffix: str,
    ):
        paths = [Path(p).expanduser().resolve() for p in inputs]
        ref = Path(reference_mesh).expanduser().resolve() if reference_mesh else None
        return calibrate_files(
            paths,
            Path(output_dir),
            reference_path=ref,
            physical_extent=physical_extent,
            unit=unit,
            axis=axis,
            uniform_scale=uniform_scale,
            shift_min_to_zero=shift_ground,
            name_suffix=suffix,
        )

    def morphometrics(
        self,
        input_ply: str | Path,
        labels: str | Path | None,
        stem_label: int,
        output_dir: str | Path,
        displacement_npz: str | Path | None = None,
        delta_time_hours: float | None = None,
        skeleton_bins: int = 30,
    ) -> MorphResult:
        cloud = self.clouds.load_labeled(input_ply, labels)
        return self.morph.compute(
            cloud,
            output_dir,
            stem_label,
            displacement_npz=displacement_npz,
            delta_time_hours=delta_time_hours,
            skeleton_bins=skeleton_bins,
        )
