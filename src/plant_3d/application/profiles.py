from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plant_3d.application.generation import GenerationConfig
from plant_3d.infrastructure.cpd.config import (
    NonrigidCPDConfig,
    PipelineConfig,
    PreprocessConfig,
    RigidCPDConfig,
)


@dataclass
class SegmentationProfile:
    voxel_size: float = 0.001


@dataclass
class RegistrationProfile:
    downsample_target: int = 5000
    rigid_only: bool = False
    rigid_w: float = 0.1
    nonrigid_beta: float = 3.0
    nonrigid_lambda: float = 2.0


@dataclass
class LightingProfile:
    power: float = 20.0
    scale_to_m: float = 0.1
    photoperiod: float = 12.0
    tone_map_mode: str = "relative"


@dataclass
class MorphProfile:
    stem_label: int = 1


@dataclass
class GenerationProfile:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    octree_resolution: int = 256
    mc_level: float = 0.0
    target_image_size: int | None = None


@dataclass
class PipelineProfile:
    segmentation: SegmentationProfile = field(default_factory=SegmentationProfile)
    registration: RegistrationProfile = field(default_factory=RegistrationProfile)
    lighting: LightingProfile = field(default_factory=LightingProfile)
    morph: MorphProfile = field(default_factory=MorphProfile)
    generation: GenerationProfile = field(default_factory=GenerationProfile)


_BUILTIN_PROFILES: dict[str, PipelineProfile] = {
    "research": PipelineProfile(),
    "fast": PipelineProfile(
        segmentation=SegmentationProfile(voxel_size=0.0025),
        registration=RegistrationProfile(
            downsample_target=2000,
            rigid_only=True,
            rigid_w=0.15,
            nonrigid_beta=3.0,
            nonrigid_lambda=2.0,
        ),
        lighting=LightingProfile(power=15.0, tone_map_mode="relative"),
    ),
    "high-accuracy": PipelineProfile(
        segmentation=SegmentationProfile(voxel_size=0.0007),
        registration=RegistrationProfile(
            downsample_target=12000,
            rigid_only=False,
            rigid_w=0.05,
            nonrigid_beta=2.5,
            nonrigid_lambda=3.0,
        ),
        lighting=LightingProfile(power=20.0, tone_map_mode="absolute"),
    ),
}


def profile_names() -> list[str]:
    return list(_BUILTIN_PROFILES.keys())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_dict(profile: PipelineProfile) -> dict[str, Any]:
    return {
        "segmentation": {"voxel_size": profile.segmentation.voxel_size},
        "registration": {
            "downsample_target": profile.registration.downsample_target,
            "rigid_only": profile.registration.rigid_only,
            "rigid_w": profile.registration.rigid_w,
            "nonrigid_beta": profile.registration.nonrigid_beta,
            "nonrigid_lambda": profile.registration.nonrigid_lambda,
        },
        "lighting": {
            "power": profile.lighting.power,
            "scale_to_m": profile.lighting.scale_to_m,
            "photoperiod": profile.lighting.photoperiod,
            "tone_map_mode": profile.lighting.tone_map_mode,
        },
        "morph": {"stem_label": profile.morph.stem_label},
        "generation": {
            "num_inference_steps": profile.generation.num_inference_steps,
            "guidance_scale": profile.generation.guidance_scale,
            "seed": profile.generation.seed,
            "octree_resolution": profile.generation.octree_resolution,
            "mc_level": profile.generation.mc_level,
            "target_image_size": profile.generation.target_image_size,
        },
    }


def _from_dict(raw: dict[str, Any]) -> PipelineProfile:
    seg = raw.get("segmentation", {})
    reg = raw.get("registration", {})
    light = raw.get("lighting", {})
    morph = raw.get("morph", {})
    gen = raw.get("generation", {})
    target_size = gen.get("target_image_size")
    return PipelineProfile(
        segmentation=SegmentationProfile(voxel_size=float(seg.get("voxel_size", 0.001))),
        registration=RegistrationProfile(
            downsample_target=int(reg.get("downsample_target", 5000)),
            rigid_only=bool(reg.get("rigid_only", False)),
            rigid_w=float(reg.get("rigid_w", 0.1)),
            nonrigid_beta=float(reg.get("nonrigid_beta", 3.0)),
            nonrigid_lambda=float(reg.get("nonrigid_lambda", 2.0)),
        ),
        lighting=LightingProfile(
            power=float(light.get("power", 20.0)),
            scale_to_m=float(light.get("scale_to_m", 0.1)),
            photoperiod=float(light.get("photoperiod", 12.0)),
            tone_map_mode=str(light.get("tone_map_mode", "relative")),
        ),
        morph=MorphProfile(stem_label=int(morph.get("stem_label", 1))),
        generation=GenerationProfile(
            num_inference_steps=int(gen.get("num_inference_steps", 50)),
            guidance_scale=float(gen.get("guidance_scale", 7.5)),
            seed=int(gen.get("seed", 42)),
            octree_resolution=int(gen.get("octree_resolution", 256)),
            mc_level=float(gen.get("mc_level", 0.0)),
            target_image_size=int(target_size) if target_size is not None else None,
        ),
    )


def load_profile(name: str, profile_config: str | Path | None = None) -> PipelineProfile:
    if name not in _BUILTIN_PROFILES:
        available = ", ".join(profile_names())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    base = _as_dict(_BUILTIN_PROFILES[name])
    if not profile_config:
        return _from_dict(base)
    payload = json.loads(Path(profile_config).expanduser().resolve().read_text(encoding="utf-8"))
    merged = _deep_merge(base, payload)
    return _from_dict(merged)


def generation_config_from_profile(profile: PipelineProfile) -> GenerationConfig:
    g = profile.generation
    return GenerationConfig(
        num_inference_steps=g.num_inference_steps,
        guidance_scale=g.guidance_scale,
        seed=g.seed,
        octree_resolution=g.octree_resolution,
        mc_level=g.mc_level,
        target_image_size=g.target_image_size,
    )


def registration_config_from_profile(profile: PipelineProfile) -> PipelineConfig:
    return PipelineConfig(
        preprocess=PreprocessConfig(downsample_target=profile.registration.downsample_target),
        rigid=RigidCPDConfig(w=profile.registration.rigid_w),
        nonrigid=NonrigidCPDConfig(
            beta=profile.registration.nonrigid_beta,
            lmbda=profile.registration.nonrigid_lambda,
        ),
        rigid_only=profile.registration.rigid_only,
    )
