from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepSegResult:
    pcd_path: Path
    points_count: int


@dataclass
class PostSegResult:
    ply_path: Path
    labels_json: Path
    class_map_json: Path


@dataclass
class SegmentationSessionResult:
    prep: PrepSegResult
    post: PostSegResult | None
    editor_started: bool
    labels_used: Path | None


@dataclass
class TrackResult:
    vectors_path: Path
    heatmap_path: Path
    report_path: Path


@dataclass
class MorphResult:
    csv_path: Path
    json_path: Path
