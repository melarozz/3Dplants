"""Command-line interface for plant-cpd."""

from __future__ import annotations

import logging
import sys

import click

from plant_cpd import __version__


def _setup_logging(verbose: bool) -> None:
    """Configure root logger for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """plant-cpd: CPD-based registration of 3D plant models."""


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    default="./results",
    help="Output directory.",
    show_default=True,
)
@click.option(
    "--downsample-target", "-d",
    default=5000,
    type=int,
    help="Target point count after downsampling.",
    show_default=True,
)
@click.option(
    "--rigid-only",
    is_flag=True,
    default=False,
    help="Skip non-rigid registration.",
)
@click.option(
    "--rigid-w",
    default=0.1,
    type=float,
    help="Rigid CPD outlier weight.",
    show_default=True,
)
@click.option(
    "--nonrigid-beta",
    default=3.0,
    type=float,
    help="Non-rigid Gaussian kernel width.",
    show_default=True,
)
@click.option(
    "--nonrigid-lambda", "nonrigid_lmbda",
    default=2.0,
    type=float,
    help="Non-rigid regularisation weight.",
    show_default=True,
)
@click.option(
    "--visualize", "-V",
    is_flag=True,
    default=False,
    help="Open interactive PyVista window.",
)
@click.option(
    "--no-hdf5",
    is_flag=True,
    default=False,
    help="Disable HDF5 output.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def register_cmd(
    source: str,
    target: str,
    output_dir: str,
    downsample_target: int,
    rigid_only: bool,
    rigid_w: float,
    nonrigid_beta: float,
    nonrigid_lmbda: float,
    visualize: bool,
    no_hdf5: bool,
    verbose: bool,
) -> None:
    """Register SOURCE.glb onto TARGET.glb using CPD.

    Runs rigid alignment followed by non-rigid deformation
    (unless --rigid-only).  Outputs aligned GLB files, merged
    scene, HDF5 with all artefacts, and quality metrics.
    """
    _setup_logging(verbose)

    # Late imports so --help stays fast
    from plant_cpd.config import (
        NonrigidCPDConfig,
        PipelineConfig,
        PreprocessConfig,
        RigidCPDConfig,
    )
    from plant_cpd.pipeline import register

    cfg = PipelineConfig(
        preprocess=PreprocessConfig(
            downsample_target=downsample_target,
        ),
        rigid=RigidCPDConfig(w=rigid_w),
        nonrigid=NonrigidCPDConfig(
            beta=nonrigid_beta,
            lmbda=nonrigid_lmbda,
        ),
        rigid_only=rigid_only,
        save_hdf5=not no_hdf5,
        visualize=visualize,
    )

    result = register(
        source_path=source,
        target_path=target,
        config=cfg,
        output_dir=output_dir,
    )

    # Print metrics summary
    click.echo("\n" + "═" * 50)
    click.echo("  Registration complete")
    click.echo("═" * 50)
    for name, value in result.metrics.items():
        click.echo(f"  {name:<20s} {value:.6f}")
    click.echo("═" * 50)
    click.echo(f"  Output dir: {result.output_dir}")
    click.echo()