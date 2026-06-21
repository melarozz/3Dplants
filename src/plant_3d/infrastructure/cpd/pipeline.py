"""Main registration pipeline orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from plant_3d.infrastructure.cpd.config import PipelineConfig
from plant_3d.infrastructure.cpd.cpd_nonrigid import NonrigidResult, nonrigid_cpd
from plant_3d.infrastructure.cpd.cpd_rigid import RigidResult, rigid_cpd
from plant_3d.infrastructure.cpd.hdf5_store import HDF5Store
from plant_3d.infrastructure.cpd.io_glb import (
    export_glb,
    export_merged_glb,
    load_glb,
)
from plant_3d.infrastructure.cpd.metrics import compute_all_metrics
from plant_3d.infrastructure.cpd.preprocessing import preprocess

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Aggregated result of the full registration pipeline.
    """

    rigid_result: RigidResult
    nonrigid_result: NonrigidResult | None
    aligned_source: np.ndarray
    target: np.ndarray
    displacement_field: np.ndarray | None
    metrics: dict[str, float] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("."))

    def show(self) -> None:
        """Open interactive PyVista visualisation."""
        from plant_3d.infrastructure.cpd.visualization import show_registration
        show_registration(
            source=self.aligned_source,
            target=self.target,
            displacement_field=self.displacement_field,
        )


def register(
    source_path: str | PathLike,
    target_path: str | PathLike,
    config: PipelineConfig | None = None,
    output_dir: str | PathLike = "./results",
) -> RegistrationResult:
    """Run the full registration pipeline.
    """
    if config is None:
        config = PipelineConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Loading source: %s", source_path)
    source_raw = load_glb(source_path)

    logger.info("Loading target: %s", target_path)
    target_raw = load_glb(target_path)

    logger.info("Pre-processing source…")
    source_pp, source_meta = preprocess(
        source_raw, config.preprocess,
    )

    logger.info("Pre-processing target…")
    target_pp, target_meta = preprocess(
        target_raw, config.preprocess,
    )

    logger.info("Running rigid CPD…")
    r_result = rigid_cpd(source_pp, target_pp, config.rigid)
    current_aligned = r_result.aligned

    nr_result: NonrigidResult | None = None
    displacement: np.ndarray | None = None

    if not config.rigid_only:
        logger.info("Running non-rigid CPD…")
        nr_result = nonrigid_cpd(
            current_aligned, target_pp, config.nonrigid,
        )
        current_aligned = nr_result.deformed
        displacement = nr_result.displacement_field

    metrics = compute_all_metrics(current_aligned, target_pp)
    logger.info("Metrics: %s", metrics)

    if config.save_hdf5:
        h5_path = output / config.hdf5_filename
        logger.info("Saving HDF5: %s", h5_path)
        with HDF5Store(h5_path) as store:
            # Raw & preprocessed clouds
            store.save_pointcloud("source/raw", source_raw)
            store.save_pointcloud("source/preprocessed", source_pp)
            store.save_pointcloud("target/raw", target_raw)
            store.save_pointcloud("target/preprocessed", target_pp)

            # Rigid results
            store.save_dict("rigid", {
                "rotation": r_result.rotation,
                "translation": r_result.translation,
                "scale": r_result.scale,
                "aligned": r_result.aligned,
            })

            # Non-rigid results
            if nr_result is not None:
                store.save_dict("nonrigid", {
                    "W": nr_result.W,
                    "G": nr_result.G,
                    "deformed": nr_result.deformed,
                    "displacement_field": displacement,
                })

            # Metrics
            store.save_dict("metrics", metrics)

    # Export GLB files
    export_glb(
        current_aligned,
        output / "aligned_source.glb",
        color=(80, 200, 120, 255),
    )
    export_glb(
        target_pp,
        output / "target.glb",
        color=(100, 149, 237, 255),
    )
    export_merged_glb(
        current_aligned,
        target_pp,
        output / "merged.glb",
    )

    result = RegistrationResult(
        rigid_result=r_result,
        nonrigid_result=nr_result,
        aligned_source=current_aligned,
        target=target_pp,
        displacement_field=displacement,
        metrics=metrics,
        output_dir=output,
    )

    if config.visualize:
        result.show()

    logger.info("Pipeline complete. Outputs in '%s'", output)
    return result