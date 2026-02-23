"""
spintext.plotting

Simple visualizations for SpinTexture and TextureSymmetry.
"""
from __future__ import annotations
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import colorsys

from .texture import SpinTexture
from .symmetry import TextureSymmetry


def get_color_hsl(vec: np.ndarray, max_length: float = 1.0) -> np.ndarray:
    r"""
    Maps a vector to the HSL-colorspace and returns the subsequent transformation into the rgb space. The HSL colorspace
    can be see as a cylinder. The inplane (vx,vy) angle codes the phi angle of the cylinder. The lightness is coded by
    the out of plane component and the length of the vector codes the saturation. In case a normalized vector is given
    we are always on the surface of the cylinder. In case the vector has a certain length we want to move closer and
    closer to the gray point in the middle of the cylinder as the length vanishes.
    :param vec: vector (not necessarily normalized)
    :param max_length: max length of the vectors (normalization of the colormaps)
    :return: the rgb color
    """
    vec_norm = np.linalg.norm(vec)
    vec = vec/vec_norm
    #if vec_norm > max_length:
    #    raise ValueError("Length of vector above colormap normalization")
    # The saturation will be set to the length of the vector, map intervall [0,max_length] to [0,1]
    s = 1.0#vec_norm# / max_length
    # The light is controlled by the v_z component. As the cylinder is mapped to the interval [0,1] we have to transform
    # the z component from [-max_length,max_length].

    l = (vec[2] + vec_norm) / (2*vec_norm)
    l = (l-0.5)+0.5
    # The colorcircle adressed by the in-plane angle
    phi = np.arctan2(vec[1], vec[0])
    # map to interval [0,1]
    h = phi / (2.0 * np.pi)
    #print(f"h={h}, l={l}, s={s}")
    rgb = colorsys.hls_to_rgb(h,l,s)
    gray_scale = vec_norm/max_length * 255
    #max_rgb = np.max(rgb)
    return np.array([abs(rgb[0]),abs(rgb[1]),abs(rgb[2])])#/max_rgb

def plot_texture(texture: SpinTexture,ax: Optional[plt.Axes] = None,scale_spins: float = 0.6,
                 title: Optional[str] = None,show_embedding_circle: bool = True,cmap: str = "coolwarm",
                 show: bool = True) -> plt.Axes:
    """
    Plot a spin texture in the xy-plane.

    - Points are shown in (x, y)
    - In-plane spin components as arrows (quiver)
    - Color encodes m_z (out-of-plane)

    Parameters
    ----------
    texture:
        SpinTexture instance.

    ax:
        Existing matplotlib Axes. If None, a new figure + axes is created.

    scale_spins:
        Multiplicative factor for arrow length (relative to spacing).

    title:
        Optional plot title.

    show_embedding_circle:
        If True, draw a circle with radius `texture.rad_embedding_circle`.

    cmap:
        Colormap name for m_z values.

    show:
        If True, call `plt.show()` at the end.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    pts = texture.points
    spins = texture.spins

    x = pts[:, 0]
    y = pts[:, 1]
    mz = spins[:, 2]
    mx = spins[:, 0]
    my = spins[:, 1]

    color_m = []
    for idx, m in enumerate(spins):
        color_m.append(get_color_hsl(vec=m, max_length=1.00))
    color_m = np.asarray(color_m)

    # scatter colored by m_z
    sc = ax.scatter(x, y, c=color_m, s=30, alpha=0.9, edgecolor="none")

    # quiver for in-plane components
    ax.quiver(x,y,mx,my,angles="xy",scale_units="xy",scale=1.0 / scale_spins,alpha=0.8)

    # embedding circle (using centered points around fixpoint_mz)
    if show_embedding_circle:
        center = texture.fixpoint_mz
        R = texture.rad_embedding_circle
        circle = Circle((center[0], center[1]), R, fill=False, linestyle="--", linewidth=1.0)
        ax.add_patch(circle)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is not None:
        ax.set_title(title)

    # add colorbar for m_z
    #cbar = fig.colorbar(sc, ax=ax)
    #cbar.set_label("$m_z$")

    if show:
        plt.show()

    return ax


def plot_symmetry_axes(texture: SpinTexture,tsym: TextureSymmetry,ax: Optional[plt.Axes] = None,
                       show: bool = True) -> plt.Axes:
    """
    Overlay detected rotation and mirror information onto the texture plot.

    - For rotation symmetry: draw lines along the mirror axes and/or indicate
      the smallest angular sector.
    - For mirror symmetry: draw the mirror axes as lines through the fixpoint.

    Parameters
    ----------
    texture:
        SpinTexture instance.

    tsym:
        TextureSymmetry instance on that texture (should have run
        detect_rotation_symmetries() and optionally detect_mirror_symmetries()).

    ax:
        Existing matplotlib Axes. If None, a new plot_texture() is created first.

    show:
        If True, call `plt.show()` at the end.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot and symmetry overlays.
    """
    if ax is None:
        ax = plot_texture(texture, show=False)

    center = texture.fixpoint_mz
    R = texture.rad_embedding_circle * 1.1  # slightly larger than texture extent

    # --- plot mirror axes (if any) ---
    if tsym.mirrors:
        for mirror in tsym.mirrors:
            v = mirror / np.linalg.norm(mirror)
            # only take xy components
            dx, dy = v[0], v[1]
            xs = [center[0] - R * dx, center[0] + R * dx]
            ys = [center[1] - R * dy, center[1] + R * dy]
            ax.plot(xs, ys, linestyle="-", linewidth=1.5,color="tab:cyan")

    # --- mark rotation sectors (if any finite rotation symmetry) ---
    if tsym.rot_syms and not tsym.rot_sym_inf:
        n = tsym.rot_syms[0]
        dphi = 2 * np.pi / n
        # draw boundaries of the "fundamental" sector around phi=0
        for k in range(n):
            phi = k * dphi
            # this phi uses the same convention as texture.phi: atan2(x, y)
            dx = np.sin(phi)
            dy = np.cos(phi)
            xs = [center[0], center[0] + R * dx]
            ys = [center[1], center[1] + R * dy]
            ax.plot(xs, ys, linestyle="--", linewidth=0.8)

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    return ax


def plot_texture_comparison(tex_original: SpinTexture,tex_transformed: SpinTexture,title: str = "",
                            show: bool = True) -> plt.Axes:
    """
    Plot original and transformed textures on the same axes.

    Original: faint
    Transformed: stronger, with different alpha/size, so you can see the change.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    pts0, s0 = tex_original.points_embedding_circle, tex_original.spins_embedding_circle
    pts1, s1 = tex_transformed.points_embedding_circle, tex_transformed.spins_embedding_circle
    # Original
    ax.quiver(pts0[:, 0],pts0[:, 1],s0[:, 0],s0[:, 1],angles="xy",scale_units="xy",scale=1.0,alpha=0.3,
              label="original",color="r")

    # Transformed
    ax.quiver(pts1[:, 0],pts1[:, 1],s1[:, 0],s1[:, 1],angles="xy",scale_units="xy",scale=1.0,alpha=0.9,
              label="transformed",color="tab:gray")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend()

    if show:
        plt.show()

    return ax

def plot_transformed_section(section: SpinTexture,transformed_section: SpinTexture, title: str = "",
                             show: bool = True) -> plt.Axes:
    """
    Plot original and transformed section on the same axes.

    Original: faint
    Transformed: stronger, with different alpha/size, so you can see the change.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    pts0, s0 = section.points, section.spins
    pts1, s1 = transformed_section.points, transformed_section.spins

    fixpoint = transformed_section.second_fixpoints[transformed_section.scnd_fixpoint_selection]
    mirror_dir = fixpoint - transformed_section.fixpoint_mz
    mirror_dir /= np.linalg.norm(mirror_dir)
    plot_endpoint1 = fixpoint-transformed_section.rad_embedding_circle*mirror_dir
    plot_endpoint2 = fixpoint+transformed_section.rad_embedding_circle*mirror_dir
    # Plot mirror axes in transformed sector
    ax.plot([plot_endpoint1[0],plot_endpoint2[0]],[plot_endpoint1[1],plot_endpoint2[1]],
            linestyle="-",color="tab:cyan")
    # Original sector
    ax.quiver(pts0[:, 0],pts0[:, 1],s0[:, 0],s0[:, 1],angles="xy",scale_units="xy",scale=1.0,alpha=0.3,
              label="original",color="r")

    # Transformed sector
    ax.quiver(pts1[:, 0],pts1[:, 1],s1[:, 0],s1[:, 1],angles="xy",scale_units="xy",scale=1.0,alpha=0.9,
              label="transformed",color="tab:gray")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend()

    if show:
        plt.show()

    return ax

def plot_rotation_operation(texture: SpinTexture,angle: float,show: bool = True) -> plt.Axes:
    """
    Show how the texture looks after a rotation by `angle` around the z-axis.

    Uses the same rotation as TextureSymmetry uses for testing.
    """
    rot_points, rot_spins = texture.apply_texture_rotation(alpha=angle)
    tex_rot = SpinTexture(points=rot_points, spins=rot_spins)

    geodist_per_spin = texture.geodesic_distance(other_texture=tex_rot)

    ax = plot_texture_comparison(tex_original=texture,tex_transformed=tex_rot,
                                 title=f"Rotation by {angle:.3f} rad, $D/N$: {geodist_per_spin:.5f}",
                                 show=show)
    return ax

def mirror_transform(points: np.ndarray,spins: np.ndarray,center: np.ndarray,normal: np.ndarray,
                     beta: float,mode: str = "dmi") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same mirror operations as used in TextureSymmetry:

    - mode="dmi": mirror + helicity rotation (gamma = pi/2 - beta)
    - mode="spinflip": mirror + in-plane spin flip (used in _try_mirror_spinflip_symmetry)
    """
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(center, dtype=float)

    points_centered = points - center

    # Mirror positions
    p_mirrored = points_centered - 2 * np.dot(points_centered, normal)[:, None] * normal

    if mode == "dmi":
        gamma = np.pi / 2 - beta
        rotmat = np.array(
            [
                [np.cos(2 * gamma), -np.sin(2 * gamma), 0.0],
                [np.sin(2 * gamma), np.cos(2 * gamma), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        s_mirrored = []
        for s in spins:
            s_mirrored.append(rotmat.T @ (s - 2 * np.dot(s, normal) * normal))
        s_mirrored = np.asarray(s_mirrored)
    elif mode == "spinflip":
        s_mirrored = []
        for s in spins:
            s_mirror = s - 2 * np.dot(s, normal) * normal
            s_flipped = s_mirror.copy()
            s_flipped[:2] *= -1  # flip in-plane components
            s_mirrored.append(s_flipped)
        s_mirrored = np.asarray(s_mirrored)
    else:
        raise ValueError("mode must be 'dmi' or 'spinflip'.")

    # Shift mirrored positions back by center
    p_mirrored += center
    return p_mirrored, s_mirrored


def plot_mirror_operation(texture: SpinTexture,center: np.ndarray,normal: np.ndarray,beta: float,mode: str = "dmi",
                          show: bool = True) -> plt.Axes:
    """
    Show how the texture looks after a mirror operation:

    Parameters
    ----------
    texture:
        Original SpinTexture.

    center:
        Fixpoint used in the mirror operation.

    normal:
        Normal vector of the mirror plane.

    beta:
        DMI angle beta (same meaning as in TextureSymmetry).

    mode:
        "dmi"      -> mirror + helicity rotation (same as try_mirror_symmetry)
        "spinflip" -> mirror + in-plane spin flip (same as _try_mirror_spinflip_symmetry).
    """
    pts, spins = texture.points, texture.spins
    p_m, s_m = mirror_transform(points=pts,spins=spins,center=center,normal=normal,beta=beta,mode=mode)
    tex_mirrored = SpinTexture(points=p_m, spins=s_m)

    title = f"Mirror ({mode}), normal={normal/np.linalg.norm(normal)}"
    ax = plot_texture_comparison(tex_original=texture,tex_transformed=tex_mirrored,title=title,show=show)
    return ax

