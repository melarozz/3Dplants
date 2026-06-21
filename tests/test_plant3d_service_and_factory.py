from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from plant_3d.application.factory import build_default_service
from plant_3d.application.service import Plant3DService
from plant_3d.application.types import LabeledPointCloud


@dataclass
class _PreprocessStub:
    called: dict
    def prep_images(self, images, output_dir, *, target_size=None):
        self.called["prep_images"] = True
        return type("R", (), {"prepared_images": []})()


@dataclass
class _GenerationStub:
    called: dict
    def generate_3d(self, images, output_dir, config):
        self.called["generate_3d"] = True
        return type("R", (), {"stl_path": Path("x"), "glb_path": None})()


@dataclass
class _SegStub:
    called: dict
    def run_prep(self, *a, **k):
        self.called["prep"] = True
        return type("R", (), {"pcd_path": Path("p"), "points_count": 1})()
    def run_post(self, *a, **k):
        return type("R", (), {"ply_path": Path("p"), "labels_json": Path("l"), "class_map_json": Path("c")})()
    def run_session(self, *a, **k):
        return type("R", (), {"prep": type("P", (), {"pcd_path": Path("p")})(), "post": None, "editor_started": False, "labels_used": None})()


@dataclass
class _RegStub:
    called: dict
    def register_any(self, source_path, target_path, output_dir, config=None):
        self.called["register"] = (source_path, target_path)
        return type("R", (), {"output_dir": Path(output_dir)})()


@dataclass
class _LightStub:
    called: dict
    def run_light(self, *a, **k):
        self.called["light"] = True
        return type("R", (), {"output_glb": Path("a"), "output_hdf5": Path("b"), "mean_ppfd": 1.0, "dli": 1.0})()


@dataclass
class _CloudsStub:
    cloud: LabeledPointCloud
    def load_labeled(self, path, labels_txt=None):
        return self.cloud


@dataclass
class _TrackStub:
    called: dict
    def track(self, source, target, output_dir, mode):
        self.called["track"] = mode
        return type("R", (), {"vectors_path": Path("v"), "heatmap_path": Path("h"), "report_path": Path("r")})()


@dataclass
class _MorphStub:
    called: dict
    def compute(self, cloud, output_dir, stem_label, *, displacement_npz=None, delta_time_hours=None, skeleton_bins=30):
        self.called["morph"] = (stem_label, str(displacement_npz) if displacement_npz else None, delta_time_hours, skeleton_bins)
        return type("R", (), {"csv_path": Path("c"), "json_path": Path("j")})()


def test_factory_build_default_service_smoke() -> None:
    svc = build_default_service()
    assert isinstance(svc, Plant3DService)
    assert svc.preprocess is not None
    assert svc.generation is not None


def test_service_track_original_calls_registration(tmp_path: Path) -> None:
    cloud = LabeledPointCloud(points=np.zeros((3, 3)), labels=np.ones((3,), dtype=np.int32))
    called: dict = {}
    svc = Plant3DService(
        preprocess=_PreprocessStub(called), generation=_GenerationStub(called), segmentation=_SegStub(called),
        registration=_RegStub(called), lighting=_LightStub(called), clouds=_CloudsStub(cloud),
        tracking=_TrackStub(called), morph=_MorphStub(called),
    )
    cpd_result, track_result = svc.track_original(
        source="s.stl", reference="r.stl", output_dir=str(tmp_path),
        registration_config={"x": 1}, source_ply="s.ply", reference_ply="r.ply",
        source_labels=None, reference_labels=None,
    )
    assert hasattr(cpd_result, "output_dir")
    assert track_result is not None
    assert called["register"][0] == "s.stl"


def test_service_calibrate_metric_smoke(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 1))
    p = tmp_path / "a.stl"
    mesh.export(str(p))
    called: dict = {}
    svc = Plant3DService(
        preprocess=_PreprocessStub(called), generation=_GenerationStub(called), segmentation=_SegStub(called),
        registration=_RegStub(called), lighting=_LightStub(called),
        clouds=_CloudsStub(LabeledPointCloud(points=np.zeros((3, 3)), labels=np.ones(3, dtype=np.int32))),
        tracking=_TrackStub(called), morph=_MorphStub(called),
    )
    written, summary = svc.calibrate_metric(
        inputs=[str(p)], output_dir=str(tmp_path / "out"), reference_mesh=str(p),
        physical_extent=10, unit="cm", axis="z", uniform_scale=None, shift_ground=False, suffix="metric",
    )
    assert written and written[0].exists()
    assert summary.mode == "reference_extent"
