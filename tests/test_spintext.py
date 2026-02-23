import json
from pathlib import Path
import numpy as np
import pytest
from spintext import SpinTexture, TextureSymmetry

def _cross_texture():
    """
    Helper: create a simple 2-fold symmetric texture in the xy-plane.

    Points: four sites on the axes at unit distance:
        (±1, 0, 0), (0, ±1, 0)

    Spins: radial (pointing away from origin), which makes the texture
    invariant under a 180° rotation around the z-axis and symmetric under
    mirror w.r.t. xz-plane (normal along y).
    """
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    # radial spins: same as points (already unit vectors here)
    spins = points.copy()
    return points, spins


# --------------------------------------------------------------------------- #
# SpinTexture tests
# --------------------------------------------------------------------------- #

def test_spintexture_self_distance_zero():
    """Geodesic distance of a texture to itself should be ~0."""
    rng = np.random.default_rng(123)
    points = rng.normal(size=(20, 3))
    points[:, 2] = 0.0  # planar texture

    # random normalized spins
    spins = rng.normal(size=(20, 3))
    spins /= np.linalg.norm(spins, axis=1)[:, None]

    tex = SpinTexture(points=points, spins=spins)
    dist = tex.geodesic_distance(tex)

    assert dist == pytest.approx(0.0, abs=1e-10)


def test_spintexture_midpoint_and_npoints():
    """Basic properties: midpoint and number of points."""
    points, spins = _cross_texture()
    tex = SpinTexture(points=points, spins=spins)

    assert tex.n_points == 4
    # midpoint of symmetric cross is exactly at origin
    assert tex.midpoint == pytest.approx(np.array([0.0, 0.0, 0.0]), abs=1e-12)


# --------------------------------------------------------------------------- #
# TextureSymmetry: rotation symmetry
# --------------------------------------------------------------------------- #

def test_texture_symmetry_detect_rotation_order_2():
    """
    The cross texture is invariant under a 180° rotation around z,
    so we should detect a 2-fold rotation symmetry.
    """
    points, spins = _cross_texture()
    tex = SpinTexture(points=points, spins=spins)

    tsym = TextureSymmetry(texture=tex, beta=np.pi / 2, verbose=40)  # quiet logger

    rot_syms = tsym.detect_rotation_symmetries(
        prime_rotations=(2,),  # only test n=2 to keep things simple
        geodesic_crit=1e-6,
    )

    assert rot_syms == [2]
    assert tsym.rot_syms == [2]
    # Should also have defined sections
    #assert tsym.sections is not None
    #assert len(tsym.sections) == 1  # with n=2, we keep only the largest symmetry


# --------------------------------------------------------------------------- #
# TextureSymmetry: direct mirror test
# --------------------------------------------------------------------------- #

def test_try_mirror_symmetry_radial_cross():
    """
    For the radial cross texture, mirror w.r.t. the xz-plane (normal along y)
    should be a symmetry if we choose beta = pi/2 (gamma = 0), i.e. pure mirror.
    """
    points, spins = _cross_texture()
    tex = SpinTexture(points=points, spins=spins)

    tsym = TextureSymmetry(texture=tex, beta=np.pi / 2, verbose=40)

    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])  # mirror plane: xz-plane

    found, dist = tsym.try_mirror_symmetry(
        points=points,
        spins=spins,
        center=center,
        normal=normal,
        crit=1e-6,
    )
    assert dist == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Saving symmetries
# --------------------------------------------------------------------------- #

def test_save_symmetries(tmp_path: Path):
    """
    Save symmetries to JSON and check basic structure.

    We don't depend on detect_mirror_symmetries() here to keep the test simple:
    we set a trivial configuration with no rotation symmetries and one mirror
    axis by hand.
    """
    points, spins = _cross_texture()
    tex = SpinTexture(points=points, spins=spins)
    tsym = TextureSymmetry(texture=tex, beta=np.pi / 2, verbose=40)

    # Mimic a "no rotation symmetries" situation
    tsym._rot_syms = []              # no rotations
    tsym._sections = [tex]           # single section = full texture
    tsym._mirrors = [np.array([1.0, 0.0, 0.0])]  # example mirror axis

    outpath = tmp_path / "symmetries.json"
    tsym.save_symmetries(outpath)

    assert outpath.is_file()

    with outpath.open("r") as f:
        data = json.load(f)

    # Basic keys should exist
    assert "rad_embedding_circle" in data
    assert "rotations" in data
    assert "mirrors" in data

    # There should be at least one section entry
    section_keys = [k for k in data.keys() if k.startswith("section_")]
    assert len(section_keys) >= 1

    # Mirrors stored as list-of-lists
    assert isinstance(data["mirrors"], list)
    assert all(isinstance(m, list) and len(m) == 3 for m in data["mirrors"])
