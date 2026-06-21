from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from plant_3d.application.models import SegmentationSessionResult
from plant_3d.infrastructure.seg_prep.pipeline import run_post_seg, run_prep_seg


def _start_editor(
    editor_dir: Path,
    project_data_dir: Path,
    run_in_background: bool,
) -> subprocess.Popen[bytes] | None:
    cmd = "npm start"
    env = None
    if (editor_dir / "settings.json").exists():
        env = dict(os.environ)
        env["SSE_IMAGES"] = str(project_data_dir)
    if run_in_background:
        return subprocess.Popen(
            cmd,
            cwd=str(editor_dir),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    subprocess.run(cmd, cwd=str(editor_dir), shell=True, check=False, env=env)
    return None


def run_segmentation_session(
    input_stl: str | Path,
    output_dir: str | Path,
    voxel_size: float,
    class_map: dict[int, tuple[int, int, int]],
    labels_txt: str | Path | None = None,
    editor_dir: str | Path | None = None,
    auto_wait_seconds: int = 0,
    run_editor: bool = False,
    run_in_background: bool = True,
) -> SegmentationSessionResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    prep = run_prep_seg(input_stl=input_stl, output_dir=output, voxel_size=voxel_size)
    editor_started = False
    labels_path: Path | None = Path(labels_txt).expanduser().resolve() if labels_txt else None
    pcd_path = output / "prepared_cloud.pcd"

    if run_editor and editor_dir is not None:
        _start_editor(
            editor_dir=Path(editor_dir).expanduser().resolve(),
            project_data_dir=output,
            run_in_background=run_in_background,
        )
        editor_started = True

    if labels_path is None and auto_wait_seconds > 0:
        candidates = [
            output / "labels.txt",
            pcd_path.with_suffix(".pcd.labels"),
        ]
        if editor_dir is not None:
            internal_dir = Path(editor_dir).expanduser().resolve() / "pc_folder"
            candidates.append(internal_dir / f"{pcd_path.name}.labels")
            candidates.append(internal_dir / f"{pcd_path.stem}.pcd.labels")
        start = time.time()
        while time.time() - start <= auto_wait_seconds:
            for candidate in candidates:
                if candidate.exists():
                    labels_path = candidate
                    break
            if labels_path is None:
                dynamic_matches: list[Path] = []
                dynamic_matches.extend(output.glob(f"{pcd_path.stem}*.pcd.labels"))
                if editor_dir is not None:
                    internal_dir = Path(editor_dir).expanduser().resolve() / "pc_folder"
                    if internal_dir.exists():
                        dynamic_matches.extend(internal_dir.glob(f"{pcd_path.stem}*.pcd.labels"))
                if dynamic_matches:
                    labels_path = max(dynamic_matches, key=lambda p: p.stat().st_mtime)
            if labels_path is not None:
                break
            time.sleep(2)

    post = None
    if labels_path is not None and labels_path.exists():
        post = run_post_seg(
            points_source=pcd_path,
            labels_txt=labels_path,
            output_dir=output,
            class_map=class_map,
        )
    return SegmentationSessionResult(
        prep=prep,
        post=post,
        editor_started=editor_started,
        labels_used=labels_path,
    )
