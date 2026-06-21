from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

from plant_3d import __version__
from plant_3d.application.factory import build_default_service
from plant_3d.application.generation import GenerationConfig
from plant_3d.application.manifest import build_manifest, write_manifest
from plant_3d.application.profiles import (
    load_profile,
    profile_names,
    registration_config_from_profile,
)
from plant_3d.infrastructure.metric_calibration import (
    calibrate_files,
    calibrate_from_photo_measurements,
)


def _parse_class_map(raw: str | None) -> dict[int, tuple[int, int, int]]:
    if raw is None:
        return {0: (128, 128, 128), 1: (40, 180, 99), 2: (52, 152, 219), 3: (231, 76, 60)}
    data = json.loads(Path(raw).read_text(encoding="utf-8"))
    return {int(k): tuple(v["rgb"]) for k, v in data["classes"].items()}


def _resolve_value(ctx: click.Context, param_name: str, cli_value: Any, profile_value: Any) -> Any:
    source = ctx.get_parameter_source(param_name)
    if source is None or source == ParameterSource.DEFAULT:
        return profile_value
    return cli_value


def _collect_input_paths(resolved_params: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("input", "source", "reference", "input_glb", "input_ply", "source_ply", "target_ply"):
        val = resolved_params.get(key)
        if isinstance(val, str) and val:
            paths.append(val)
    for key in ("inputs", "images"):
        val = resolved_params.get(key)
        if isinstance(val, list):
            paths.extend(str(p) for p in val if p)
    return paths


def _save_run_manifest(
    *,
    output_dir: str | Path,
    command: str,
    profile_name: str | None,
    profile_config: str | None,
    profile_obj: Any,
    resolved_params: dict[str, Any],
    artifacts: dict[str, Any],
    input_paths: list[str | Path] | None = None,
) -> Path:
    payload = build_manifest(
        command=command,
        profile_name=profile_name,
        profile=profile_obj,
        profile_config=profile_config,
        resolved_params=resolved_params,
        artifacts=artifacts,
        input_paths=input_paths or _collect_input_paths(resolved_params),
    )
    return write_manifest(output_dir, payload)


@click.group()
@click.version_option(__version__)
def main() -> None:
    """plant3d unified interface."""


@main.command("prep-images")
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/prep", show_default=True)
@click.option("--target-size", type=int, default=None)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def prep_images_cmd(
    ctx: click.Context,
    images: tuple[str, ...],
    output: str,
    target_size: int | None,
    profile: str,
    profile_config: str | None,
) -> None:
    selected = load_profile(profile, profile_config)
    target_size = _resolve_value(ctx, "target_size", target_size, selected.generation.target_image_size)
    service = build_default_service()
    result = service.prep_images(list(images), output, target_size=target_size)
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="prep-images",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"images": list(images), "output_dir": output, "target_size": target_size},
        artifacts={"prepared_images": [str(p) for p in result.prepared_images]},
        input_paths=list(images),
    )
    for p in result.prepared_images:
        click.echo(f"Prepared: {p}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("generate-3d")
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/generate", show_default=True)
@click.option("--num-inference-steps", default=50, show_default=True)
@click.option("--guidance-scale", default=7.5, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--octree-resolution", default=256, show_default=True)
@click.option("--mc-level", default=0.0, show_default=True)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def generate_3d_cmd(ctx: click.Context, images: tuple[str, ...], output: str, **kwargs: Any) -> None:
    profile = kwargs.pop("profile")
    profile_config = kwargs.pop("profile_config")
    selected = load_profile(profile, profile_config)
    config = GenerationConfig(
        num_inference_steps=int(_resolve_value(ctx, "num_inference_steps", kwargs["num_inference_steps"], selected.generation.num_inference_steps)),
        guidance_scale=float(_resolve_value(ctx, "guidance_scale", kwargs["guidance_scale"], selected.generation.guidance_scale)),
        seed=int(_resolve_value(ctx, "seed", kwargs["seed"], selected.generation.seed)),
        octree_resolution=int(_resolve_value(ctx, "octree_resolution", kwargs["octree_resolution"], selected.generation.octree_resolution)),
        mc_level=float(_resolve_value(ctx, "mc_level", kwargs["mc_level"], selected.generation.mc_level)),
    )
    service = build_default_service()
    result = service.generate_3d(list(images), output, config)
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="generate-3d",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"images": list(images), "output_dir": output, **kwargs},
        artifacts={"stl_path": str(result.stl_path), "glb_path": str(result.glb_path) if result.glb_path else None},
        input_paths=list(images),
    )
    click.echo(f"STL: {result.stl_path}")
    if result.glb_path:
        click.echo(f"GLB: {result.glb_path}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("prep-seg")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/prep", show_default=True)
@click.option("--voxel-size", default=0.001, show_default=True)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def prep_seg_cmd(ctx: click.Context, input: str, output: str, voxel_size: float, profile: str, profile_config: str | None) -> None:
    selected = load_profile(profile, profile_config)
    voxel_size = float(_resolve_value(ctx, "voxel_size", voxel_size, selected.segmentation.voxel_size))
    service = build_default_service()
    result = service.prep_seg(input_stl=input, output_dir=output, voxel_size=voxel_size)
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="prep-seg",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"input": input, "output_dir": output, "voxel_size": voxel_size},
        artifacts={"pcd_path": str(result.pcd_path), "points_count": result.points_count},
    )
    click.echo(f"PCD saved: {result.pcd_path}")
    click.echo(f"Points: {result.points_count}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("post-seg")
@click.argument("input", type=click.Path(exists=True))
@click.argument("labels_txt", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/post", show_default=True)
@click.option("--class-map", default=None, help="JSON class map file")
def post_seg_cmd(input: str, labels_txt: str, output: str, class_map: str | None) -> None:
    cmap = _parse_class_map(class_map)
    service = build_default_service()
    result = service.post_seg(points_source=input, labels_txt=labels_txt, output_dir=output, class_map=cmap)
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="post-seg",
        profile_name=None,
        profile_config=None,
        profile_obj=None,
        resolved_params={"input": input, "labels_txt": labels_txt, "output_dir": output, "class_map": cmap},
        artifacts={"labeled_ply": str(result.ply_path), "labels_json": str(result.labels_json)},
    )
    click.echo(f"Labeled PLY: {result.ply_path}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("segment-session")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/seg", show_default=True)
@click.option("--voxel-size", default=0.001, show_default=True)
@click.option("--class-map", default=None)
@click.option("--labels-txt", type=click.Path(exists=True), default=None)
@click.option("--editor-dir", default=None)
@click.option("--run-editor/--no-run-editor", default=False, show_default=True)
@click.option("--auto-wait-seconds", default=0, show_default=True)
@click.option("--foreground-editor/--background-editor", default=False, show_default=True)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def segment_session_cmd(ctx: click.Context, input: str, **opts: Any) -> None:
    profile = opts.pop("profile")
    profile_config = opts.pop("profile_config")
    selected = load_profile(profile, profile_config)
    voxel_size = float(_resolve_value(ctx, "voxel_size", opts["voxel_size"], selected.segmentation.voxel_size))
    cmap = _parse_class_map(opts.pop("class_map"))
    service = build_default_service()
    result = service.segment_session(
        input_stl=input,
        output_dir=opts["output"],
        voxel_size=voxel_size,
        class_map=cmap,
        labels_txt=opts["labels_txt"],
        editor_dir=opts["editor_dir"] or _default_editor_dir(),
        auto_wait_seconds=opts["auto_wait_seconds"],
        run_editor=opts["run_editor"],
        run_in_background=not opts["foreground_editor"],
    )
    artifacts: dict[str, Any] = {
        "prepared_pcd": str(result.prep.pcd_path),
        "editor_started": result.editor_started,
        "labels_used": str(result.labels_used) if result.labels_used else None,
    }
    if result.post:
        artifacts["labeled_ply"] = str(result.post.ply_path)
    manifest_path = _save_run_manifest(
        output_dir=opts["output"],
        command="segment-session",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"input": input, **opts, "voxel_size": voxel_size},
        artifacts=artifacts,
    )
    click.echo(f"Prepared PCD: {result.prep.pcd_path}")
    click.echo(f"Manifest: {manifest_path}")


def _default_editor_dir() -> str | None:
    return os.environ.get("PLANT3D_SSE_DIR")


@main.command("light")
@click.argument("input_glb", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/light", show_default=True)
@click.option("--light-type", type=click.Choice(["point", "directional"]), default="point")
@click.option("--light-pos", nargs=3, type=float, default=(0.0, 0.0, 0.6))
@click.option("--light-dir", nargs=3, type=float, default=(0.0, 0.0, -1.0))
@click.option("--power", default=20.0, show_default=True)
@click.option("--scale-to-m", default=0.1, show_default=True)
@click.option("--photoperiod", default=12.0, show_default=True)
@click.option("--add-lamp/--no-lamp", default=False, show_default=True)
@click.option("--tone-map-mode", default="relative", show_default=True)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def light_cmd(ctx: click.Context, input_glb: str, output: str, **opts: Any) -> None:
    profile = opts.pop("profile")
    profile_config = opts.pop("profile_config")
    selected = load_profile(profile, profile_config)
    power = float(_resolve_value(ctx, "power", opts["power"], selected.lighting.power))
    scale_to_m = float(_resolve_value(ctx, "scale_to_m", opts["scale_to_m"], selected.lighting.scale_to_m))
    photoperiod = float(_resolve_value(ctx, "photoperiod", opts["photoperiod"], selected.lighting.photoperiod))
    tone_map_mode = str(_resolve_value(ctx, "tone_map_mode", opts["tone_map_mode"], selected.lighting.tone_map_mode))
    service = build_default_service()
    result = service.light(
        input_glb=input_glb,
        output_dir=output,
        light_type=opts["light_type"],
        light_pos=tuple(opts["light_pos"]),
        light_dir=tuple(opts["light_dir"]),
        power=power,
        scale_to_m=scale_to_m,
        photoperiod=photoperiod,
        add_lamp=opts["add_lamp"],
        tone_map_mode=tone_map_mode,
    )
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="light",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"input_glb": input_glb, "output_dir": output, **opts},
        artifacts={
            "output_glb": str(result.output_glb),
            "output_hdf5": str(result.output_hdf5),
            "mean_ppfd": result.mean_ppfd,
            "dli": result.dli,
        },
    )
    click.echo(f"Light GLB: {result.output_glb}")
    click.echo(f"Mean PPFD: {result.mean_ppfd:.4f}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("track-growth")
@click.argument("source_ply", type=click.Path(exists=True))
@click.argument("target_ply", type=click.Path(exists=True))
@click.option("--source-labels", type=click.Path(exists=True), default=None)
@click.option("--target-labels", type=click.Path(exists=True), default=None)
@click.option("--output", "-o", default="./outputs/track", show_default=True)
def track_growth_cmd(source_ply: str, target_ply: str, source_labels: str | None, target_labels: str | None, output: str) -> None:
    service = build_default_service()
    result = service.track_growth(
        source_ply=source_ply,
        target_ply=target_ply,
        source_labels=source_labels,
        target_labels=target_labels,
        output_dir=output,
    )
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="track-growth",
        profile_name=None,
        profile_config=None,
        profile_obj=None,
        resolved_params={
            "source_ply": source_ply,
            "target_ply": target_ply,
            "output_dir": output,
        },
        artifacts={
            "vectors_path": str(result.vectors_path),
            "heatmap_path": str(result.heatmap_path),
        },
    )
    click.echo(f"Vectors: {result.vectors_path}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("track-original")
@click.argument("source", type=click.Path(exists=True))
@click.argument("reference", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/cpd", show_default=True)
@click.option("--source-ply", type=click.Path(exists=True), default=None)
@click.option("--reference-ply", type=click.Path(exists=True), default=None)
@click.option("--source-labels", type=click.Path(exists=True), default=None)
@click.option("--reference-labels", type=click.Path(exists=True), default=None)
@click.option("--downsample-target", default=5000, show_default=True)
@click.option("--rigid-only/--full", default=False, show_default=True)
@click.option("--rigid-w", default=0.1, show_default=True)
@click.option("--nonrigid-beta", default=3.0, show_default=True)
@click.option("--nonrigid-lambda", default=2.0, show_default=True)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def track_original_cmd(ctx: click.Context, source: str, reference: str, output: str, **opts: Any) -> None:
    profile = opts.pop("profile")
    profile_config = opts.pop("profile_config")
    selected = load_profile(profile, profile_config)
    selected = replace(
        selected,
        registration=replace(
            selected.registration,
            downsample_target=int(_resolve_value(ctx, "downsample_target", opts["downsample_target"], selected.registration.downsample_target)),
            rigid_only=bool(_resolve_value(ctx, "rigid_only", opts["rigid_only"], selected.registration.rigid_only)),
            rigid_w=float(_resolve_value(ctx, "rigid_w", opts["rigid_w"], selected.registration.rigid_w)),
            nonrigid_beta=float(_resolve_value(ctx, "nonrigid_beta", opts["nonrigid_beta"], selected.registration.nonrigid_beta)),
            nonrigid_lambda=float(_resolve_value(ctx, "nonrigid_lambda", opts["nonrigid_lambda"], selected.registration.nonrigid_lambda)),
        ),
    )
    registration_config = registration_config_from_profile(selected)
    service = build_default_service()
    cpd_result, track_result = service.track_original(
        source=source,
        reference=reference,
        output_dir=output,
        registration_config=registration_config,
        source_ply=opts["source_ply"],
        reference_ply=opts["reference_ply"],
        source_labels=opts["source_labels"],
        reference_labels=opts["reference_labels"],
    )
    artifacts: dict[str, Any] = {"cpd_output_dir": str(cpd_result.output_dir)}
    if track_result:
        artifacts["heatmap_path"] = str(track_result.heatmap_path)
    manifest_path = _save_run_manifest(
        output_dir=output,
        command="track-original",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"source": source, "reference": reference, "output_dir": output, **opts},
        artifacts=artifacts,
    )
    click.echo(f"CPD aligned output: {cpd_result.output_dir}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("calibrate-metric")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/metric", show_default=True)
@click.option("--reference-mesh", type=click.Path(exists=True), default=None)
@click.option("--physical-extent", type=float, default=None)
@click.option("--unit", default="cm", show_default=True)
@click.option("--axis", default="z", show_default=True)
@click.option("--uniform-scale", type=float, default=None)
@click.option("--shift-ground/--no-shift-ground", default=False, show_default=True)
@click.option("--suffix", default="metric", show_default=True)
@click.option("--pot-height-px", type=float, default=None)
@click.option("--plant-height-px", type=float, default=None)
@click.option("--pot-height-phys", default=12.5, show_default=True)
@click.option("--pixel-sigma", default=2.0, show_default=True)
def calibrate_metric_cmd(inputs: tuple[str, ...], output: str, **opts: Any) -> None:
    if bool(opts["pot_height_px"]) ^ bool(opts["plant_height_px"]):
        raise click.ClickException("Provide both --pot-height-px and --plant-height-px for photo calibration.")
    photo_estimate = None
    if opts["pot_height_px"] and opts["plant_height_px"]:
        written, summary, photo_estimate = calibrate_from_photo_measurements(
            list(inputs),
            output,
            pot_height_px=opts["pot_height_px"],
            plant_height_px=opts["plant_height_px"],
            pot_height_phys=opts["pot_height_phys"],
            pixel_sigma=opts["pixel_sigma"],
            axis=opts["axis"],
            shift_min_to_zero=opts["shift_ground"],
            name_suffix=opts["suffix"],
        )
    else:
        written, summary = calibrate_files(
            [Path(p) for p in inputs],
            Path(output),
            reference_path=Path(opts["reference_mesh"]) if opts["reference_mesh"] else None,
            physical_extent=opts["physical_extent"],
            unit=opts["unit"],
            axis=opts["axis"],
            uniform_scale=opts["uniform_scale"],
            shift_min_to_zero=opts["shift_ground"],
            name_suffix=opts["suffix"],
        )
    report: dict[str, Any] = {
        "mode": summary.mode,
        "scale_factor": summary.scale_factor,
        "axis": summary.axis,
        "outputs": [str(p) for p in written],
    }
    if photo_estimate:
        report["photo_calibration"] = {
            "plant_height_phys": photo_estimate.plant_height_phys,
            "unit": photo_estimate.unit,
        }
    out_dir = Path(output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "metric_calibration.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path = _save_run_manifest(
        output_dir=str(out_dir),
        command="calibrate-metric",
        profile_name=None,
        profile_config=None,
        profile_obj=None,
        resolved_params={"inputs": list(inputs), **opts},
        artifacts={"calibration_report": str(report_path), "outputs": report["outputs"]},
    )
    click.echo(f"Report: {report_path}")
    click.echo(f"Manifest: {manifest_path}")


@main.command("morph")
@click.argument("input_ply", type=click.Path(exists=True))
@click.option("--labels", type=click.Path(exists=True), default=None)
@click.option("--stem-label", default=1, show_default=True)
@click.option("--vectors-npz", type=click.Path(exists=True), default=None)
@click.option("--delta-hours", type=float, default=None)
@click.option("--skeleton-bins", default=30, show_default=True)
@click.option("--output", "-o", default="./outputs/morph", show_default=True)
@click.option("--profile", default="research", show_default=True, type=click.Choice(profile_names()))
@click.option("--profile-config", type=click.Path(exists=True), default=None)
@click.pass_context
def morph_cmd(ctx: click.Context, input_ply: str, **opts: Any) -> None:
    profile = opts.pop("profile")
    profile_config = opts.pop("profile_config")
    selected = load_profile(profile, profile_config)
    stem_label = int(_resolve_value(ctx, "stem_label", opts["stem_label"], selected.morph.stem_label))
    service = build_default_service()
    result = service.morphometrics(
        input_ply=input_ply,
        labels=opts["labels"],
        stem_label=stem_label,
        output_dir=opts["output"],
        displacement_npz=opts["vectors_npz"],
        delta_time_hours=opts["delta_hours"],
        skeleton_bins=opts["skeleton_bins"],
    )
    manifest_path = _save_run_manifest(
        output_dir=opts["output"],
        command="morph",
        profile_name=profile,
        profile_config=profile_config,
        profile_obj=selected,
        resolved_params={"input_ply": input_ply, **opts, "stem_label": stem_label},
        artifacts={"csv_path": str(result.csv_path), "json_path": str(result.json_path)},
    )
    click.echo(f"CSV: {result.csv_path}")
    click.echo(f"Manifest: {manifest_path}")
