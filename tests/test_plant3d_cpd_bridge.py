from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from plant_3d.infrastructure.cpd_bridge import register_any_format


def test_cpd_bridge_converts_stl(tmp_path: Path) -> None:
    stl = tmp_path / "a.stl"
    trimesh.creation.box().export(str(stl))
    with patch("plant_3d.infrastructure.cpd_bridge.register") as reg:
        reg.return_value = type("R", (), {"output_dir": tmp_path / "out"})()
        register_any_format(stl, stl, output_dir=tmp_path / "out")
        assert reg.called
