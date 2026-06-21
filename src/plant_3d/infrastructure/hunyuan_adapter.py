"""E1 3D reconstruction via Hunyuan3D."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import trimesh

from plant_3d.application.generation import GenerationConfig, GenerationResult
from plant_3d.application.ports import GenerationPort


class HunyuanAdapter(GenerationPort):
    """
    Wraps Hunyuan3D-2mv inference.
   """

    def generate_3d(
        self,
        images: list[str | Path],
        output_dir: str | Path,
        config: GenerationConfig,
    ) -> GenerationResult:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        image_paths = [str(Path(p).expanduser().resolve()) for p in images]
        if len(image_paths) < 1:
            raise ValueError("At least one prepared image is required for generate-3d.")

        cmd_template = os.environ.get("PLANT3D_HUNYUAN_CMD")
        if cmd_template:
            return self._run_external(cmd_template, out, image_paths, config)

        if os.environ.get("PLANT3D_HUNYUAN_STUB") == "1":
            return self._write_stub_mesh(out, config)

        raise RuntimeError(
            "Hunyuan3D inference is not configured. Set PLANT3D_HUNYUAN_CMD to your "
            "inference script, or PLANT3D_HUNYUAN_STUB=1 for development/testing.",
        )

    def _run_external(
        self,
        cmd_template: str,
        out: Path,
        image_paths: list[str],
        config: GenerationConfig,
    ) -> GenerationResult:
        cfg_path = out / "generation_config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "seed": config.seed,
                    "octree_resolution": config.octree_resolution,
                    "mc_level": config.mc_level,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd = cmd_template.format(output_dir=out, config_json=cfg_path)
        argv = shlex.split(cmd) + image_paths
        subprocess.run(argv, check=True, cwd=str(out))
        stl_path = out / "model.stl"
        glb_path = out / "model.glb"
        if not stl_path.exists():
            candidates = list(out.glob("*.stl"))
            if not candidates:
                raise FileNotFoundError(f"No STL produced in {out}")
            stl_path = candidates[0]
        glb_resolved = glb_path if glb_path.exists() else None
        return GenerationResult(
            output_dir=out,
            stl_path=stl_path,
            glb_path=glb_resolved,
            config=config,
        )

    def _write_stub_mesh(self, out: Path, config: GenerationConfig) -> GenerationResult:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        stl_path = out / "model.stl"
        glb_path = out / "model.glb"
        mesh.export(str(stl_path))
        mesh.export(str(glb_path))
        meta = out / "generation_config.json"
        meta.write_text(json.dumps({"seed": config.seed, "stub": True}, indent=2), encoding="utf-8")
        return GenerationResult(
            output_dir=out,
            stl_path=stl_path,
            glb_path=glb_path,
            config=config,
        )
