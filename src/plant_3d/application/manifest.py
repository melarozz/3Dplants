from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from plant_3d import __version__
from plant_3d.application.profiles import PipelineProfile


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    return value


def _safe_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_checksums(paths: list[str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.exists() and p.is_file():
            rows.append({"path": str(p), "sha256": sha256_file(p)})
    return rows


def build_manifest(
    *,
    command: str,
    profile_name: str | None,
    profile: PipelineProfile | None,
    profile_config: str | Path | None,
    resolved_params: dict[str, Any],
    artifacts: dict[str, Any],
    input_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plant3d_version": __version__,
        "command": command,
        "profile": profile_name,
        "profile_config": str(profile_config) if profile_config else None,
        "profile_snapshot": _to_jsonable(profile) if profile else None,
        "resolved_params": _to_jsonable(resolved_params),
        "artifacts": _to_jsonable(artifacts),
        "inputs": input_checksums(input_paths or []),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _safe_version("numpy"),
            "scipy": _safe_version("scipy"),
            "trimesh": _safe_version("trimesh"),
            "pycpd": _safe_version("pycpd"),
        },
    }


def write_manifest(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
