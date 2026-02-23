#!/usr/bin/env python
"""
Command-line helper to visualize a spin texture and test symmetries.

Usage
-----
python scripts/visualize_texture.py path/to/texture.csv \
    --beta 1.5708 \
    --detect-rotation \
    --detect-mirror
"""

from __future__ import annotations
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from spintext import SpinTexture, TextureSymmetry
from spintext.plotting import (plot_texture,plot_symmetry_axes,plot_rotation_operation,plot_mirror_operation,
                               plot_transformed_section)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize spin textures and test symmetries."
    )
    parser.add_argument(
        "texture",
        type=Path,
        help="Path to CSV file with columns x,y,z,sx,sy,sz.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.57079632679,  # ~ pi/2
        help="DMI angle beta (rad). Default: pi/2 (Néel).",
    )
    parser.add_argument(
        "--detect-rotation",
        action="store_true",
        help="Attempt to detect rotation symmetries.",
    )
    parser.add_argument(
        "--detect-mirror",
        action="store_true",
        help="Attempt to detect mirror symmetries (requires rotation detection first).",
    )
    parser.add_argument(
        "--geodesic-crit",
        type=float,
        default=1.0e-3,
        help="Geodesic distance threshold for symmetry detection.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--show-rotation",
        type=int,
        default=None,
        help="Visualize rotation by 2pi/n (give n).",
    )
    parser.add_argument(
        "--show-transformedsector",
        type=int,
        default=None,
        help="Visualize the rotational domain stretched around 0",
    )
    parser.add_argument(
        "--show-mirror",
        action="store_true",
        help="Visualize mirror operation based on mz2 fixpoint.",
    )
    parser.add_argument(
        "--mirror-mode",
        type=str,
        default="dmi",
        choices=["dmi", "spinflip"],
        help="Which mirror operation to visualize: 'dmi' or 'spinflip'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))

    # Silence matplotlib font debug spam
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
    # Load texture
    tex = SpinTexture(path_texture=args.texture)

    # Create symmetry analyzer
    tsym = TextureSymmetry(texture=tex, beta=args.beta, verbose=getattr(logging, args.loglevel))

    # Detect symmetries if requested
    if args.detect_rotation:
        tsym.detect_rotation_symmetries(geodesic_crit=args.geodesic_crit)
    if args.detect_mirror:
        tsym.detect_mirror_symmetries(geodesic_crit=args.geodesic_crit)

    # Visualize
    ax = plot_texture(tex, show=False, title=str(args.texture.name))

    # Basic texture plot
    ax = plot_texture(tex, show=False, title=str(args.texture.name))

    # Overlay symmetry axes if requested
    if args.detect_rotation or args.detect_mirror:
        plot_symmetry_axes(tex, tsym, ax=ax, show=False)

    # Explicit rotation visualization
    if args.show_rotation is not None:
        n = args.show_rotation
        angle = 2 * np.pi / n
        plot_rotation_operation(tex, angle=angle, show=False)

    # Visulation of extruded section.
    if args.show_transformedsector is not None:
        section_index = args.show_transformedsector
        tsym.detect_rotation_symmetries(geodesic_crit=args.geodesic_crit)
        section = tsym.sections[section_index]
        l_ext_pts, l_ext_spins = tsym.extrude_section(section=section.points,section_spins=section.spins,
                                                      section_angle=2 * np.pi / tsym.rot_syms[0])
        transformed_section = SpinTexture(points=l_ext_pts, spins=l_ext_spins)
        plot_transformed_section(section=section,transformed_section=transformed_section,show=False)


    # Explicit mirror visualization
    if args.show_mirror:
        # Use same recipe as detect_mirror_symmetries for no-rotation case:
        center = tex.fixpoint_mz
        fixpoint = tex.second_fixpoints["mz2"]
        mirror_dir = fixpoint - center
        mirror_dir /= np.linalg.norm(mirror_dir)
        normal = np.cross(np.array([0.0, 0.0, 1.0]), mirror_dir)

        plot_mirror_operation(texture=tex,center=center,normal=normal,beta=args.beta,mode=args.mirror_mode,show=False)

    plt.show()
    """
    # Overlay symmetry info
    if args.detect_rotation or args.detect_mirror:
        plot_symmetry_axes(tex, tsym, ax=ax, show=True)
    else:
        plt.show()"""


if __name__ == "__main__":
    main()
