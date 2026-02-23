"""
spin_texture.py

Standalone utilities for describing localized magnetic spin textures.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import json
from scipy.interpolate import griddata


class SpinTexture:
    r"""
    Class describing a single localized (typically topological) spin texture.

    A texture consists of:
      - positions ``points`` with shape (N, 3)
      - spins     ``spins``  with shape (N, 3), assumed to be normalized.

    The class can be initialized either from a CSV file or directly from arrays.
    It computes several "fixpoints" derived from the magnetization profile and
    provides a method to estimate the geodesic distance between two textures.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(
            self,
            path_texture: Optional[Union[Path, str]] = None,
            points: Optional[np.ndarray] = None,
            spins: Optional[np.ndarray] = None,
            scnd_fixpoint_selection: str = "mz2",
            lattice: Optional[Path] = Path.cwd() / "lattice.json"
    ) -> None:
        r"""
        Initialize a spin texture.

        Parameters
        ----------
        path_texture:
            Path to a CSV file containing the texture. If given, the file must
            contain the columns: ``x, y, z, sx, sy, sz``.
            If ``None``, you must provide `points` and `spins` directly.

        points:
            Array of shape (N, 3) with the positions
            ``[[p1x, p1y, p1z], [p2x, p2y, p2z], ...]``.
            Used if ``path_texture`` is not given.

        spins:
            Array of shape (N, 3) with the corresponding spins
            ``[[s1x, s1y, s1z], [s2x, s2y, s2z], ...]``.
            Used if ``path_texture`` is not given.

        scnd_fixpoint_selection:
            Key specifying which second fixpoint to use.
            Must be one of: ``"mz2"``, ``"mrmz"``, ``"mr"``, ``"mr/r"`` or ``"mr_abs"``.
            (Default: ``"mz2"``)

        Raises
        ------
        ValueError
            If neither a valid file path nor both `points` and `spins`
            are provided, or if `scnd_fixpoint_selection` is invalid.
        """
        # --- input validation / loading ----------------------------------- #
        self._lattice = lattice
        if points is None and spins is None and path_texture is None:
            raise ValueError(
                "Provide either a path to a texture file or `points` and `spins` arrays."
            )

        if path_texture is not None:
            # Expect CSV with columns x,y,z,sx,sy,sz
            df = pd.read_csv(path_texture)
            try:
                self._points = np.column_stack(
                    (df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy())
                )
                self._spins = np.column_stack(
                    (df["sx"].to_numpy(), df["sy"].to_numpy(), df["sz"].to_numpy())
                )
            except KeyError as exc:
                raise ValueError(
                    "Texture file must contain columns: 'x', 'y', 'z', 'sx', 'sy', 'sz'."
                ) from exc
        else:
            if points is None:
                raise ValueError("Array `points` is required when `path_texture` is None.")
            if spins is None:
                raise ValueError("Array `spins` is required when `path_texture` is None.")
            if points.shape != spins.shape:
                raise ValueError(
                    f"`points` and `spins` must have the same shape; "
                    f"got {points.shape} and {spins.shape}."
                )
            if points.shape[1] != 3:
                raise ValueError("`points` must be of shape (N, 3).")
            if spins.shape[1] != 3:
                raise ValueError("`spins` must be of shape (N, 3).")

            self._points = np.asarray(points, dtype=float)
            self._spins = np.asarray(spins, dtype=float)

        # --- derived quantities ------------------------------------------- #

        # Shift the mz-magnetization to avoid dividing by zero in later steps
        self._mz_magnetization = self._spins[:, 2] - 1.0

        # First fixpoint: mz-weighted center of mass
        self._fixpoint_mz = np.average(self._points, weights=self._mz_magnetization, axis=0)

        # Centered points (w.r.t. first fixpoint)
        self._centered_points = self._points - self._fixpoint_mz

        # "Embedding circle" — minimal circle that contains all centered points.
        # In the original code this was implemented using an external lattice
        # object. Here we simply keep the centered texture as-is, and define
        # the radius as the maximal distance from the first fixpoint.
        self._rad_embedding_circle, self._points_embedding_circle, self._spins_embedding_circle = self._padding()


        # Polar coordinates of centered points
        self._phi = np.arctan2(self._centered_points[:, 1], self._centered_points[:, 0])

        self._r = np.linalg.norm(self._centered_points, axis=1)

        # Second fixpoint selection
        self._scnd_fixpoint_selection = scnd_fixpoint_selection
        if scnd_fixpoint_selection not in ["mz2", "mrmz", "mr", "mr/r", "mr_abs"]:
            raise ValueError("Not a valid second fixpoint selector.")

        # Calculate all fixpoints
        self._define_fps()

        # Possible rotational symmetries (user-defined)
        self._rot_syms: Optional[List[int]] = None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _padding(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
    Reconstructs the full lattice grid from the provided lattice.json,
    places the texture spins on the corresponding lattice sites,
    and fills the remaining sites inside the embedding circle with mz=1.

    Returns
    -------
    R : float
        Embedding radius (max radial distance from fixpoint)
    pts_out : np.ndarray
        Padded lattice points inside embedding circle
    spins_out : np.ndarray
        Corresponding spins (texture spins or background spins)
    """
        # --- 1. Load lattice metadata ---
        if isinstance(self._lattice, (str, Path)):
            with open(self._lattice, "r") as f:
                latt = json.load(f)
        else:
            raise ValueError("`lattice` must be a path to a lattice.json file.")

        Nx, Ny, Nz = latt["latt_dim"]
        a1 = np.array(latt["latt_vecs"]["a1"], float)
        a2 = np.array(latt["latt_vecs"]["a2"], float)
        a3 = np.array(latt["latt_vecs"]["a3"], float)

        # (We assume Nz = 1 for planar textures)
        if Nz != 1:
            raise ValueError("This padding implementation currently supports Nz=1 only.")

        # To make sure that padding always works: consider a lattice 4 times as large and center it at the midpoint
        Nx = Nx * 2
        Ny = Ny * 2
        # --- 2. Build full lattice grid ---
        xs = np.arange(-Nx/2,Nx/2)
        ys = np.arange(-Ny/2,Nx/2)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")  # (Nx, Ny)

        # Convert integer lattice coordinates (i,j) to actual coordinates
        pts_grid = (
            XX[..., None] * a1[None,None,:]
            + YY[..., None] * a2[None,None,:]
        ).reshape(-1, 3)  # (Nx*Ny, 3)

        # --- 3. Compute embedding radius / fixpoint ---
        R = np.max(np.linalg.norm(self._centered_points, axis=1))
        # Center lattice grid around fixpoint_mz
        pts_grid_centered = pts_grid #- self._fixpoint_mz

        # Mask: keep only points within embedding circle
        inside = np.linalg.norm(pts_grid_centered[:, :2], axis=1) <= R

        pts_inside = pts_grid_centered[inside]      # (M, 3)

        # --- 4. Create spin array with all background mz=1 ---
        spins_inside = np.zeros_like(pts_inside)
        spins_inside[:, 2] = 1.0   # uniform up background

        # --- 5. Overwrite grid spins with texture spins exactly at texture points ---

        # Texture points (already centered)
        tex_pts = self._centered_points
        tex_spins = self._spins

        # We need to match integer lattice sites:
        # Convert lattice points and texture points to integer indices
        # using inverse lattice vectors.
        # For orthogonal grids like in your JSON, this is simple:
        invA = np.linalg.inv(np.stack([a1, a2, a3], axis=1))  # columns = vectors

        # Compute fractional coordinates (i,j,k)
        tex_ijk = (invA @ tex_pts.T).T   # shape (Ntex, 3)
        pts_ijk = (invA @ pts_grid_centered.T).T  # full grid but centered

        # Round to nearest integer lattice site
        tex_idx = np.round(tex_ijk[:, :2]).astype(int)   # discard z
        pts_idx = np.round(pts_ijk[:, :2]).astype(int)

        # Hash lattice indices to dictionary for fast lookup:
        # key = (i,j), value = position in pts_inside
        idx_map = {}
        for pos, (ix, iy) in enumerate(pts_idx[inside]):
            idx_map[(ix, iy)] = pos

        # Replace background spins by actual texture spins
        for p, s in zip(tex_idx, tex_spins):
            key = (p[0], p[1])
            if key in idx_map:
                spins_inside[idx_map[key]] = s

        # Done
        return R, pts_inside, spins_inside

    def _define_fps(self) -> None:
        r"""
        Compute and cache the various fixpoints used in the analysis.

        This recomputes:
          - centered points and unit vectors
          - radial magnetization components
          - different fixpoints based on combinations of radial and z components
        """
        # Make sure centered points are up-to-date
        self._centered_points = self._points - self._fixpoint_mz

        norms = np.linalg.norm(self._centered_points, axis=1)
        # Avoid division by zero for points exactly at the fixpoint:
        norms_safe = np.where(norms == 0.0, 1.0, norms)
        self._centered_points_unitvecs = self._centered_points / norms_safe[:, np.newaxis]

        # Radial magnetization component: m · r_hat
        self._rad_magnetization = np.sum(
            self._spins * self._centered_points_unitvecs, axis=1
        )

        # Various second fixpoints
        self._fixpoint_radmz = np.sum(
            self._centered_points
            * self._rad_magnetization[:, None]
            * self._mz_magnetization[:, None],
            axis=0,
        )
        self._fixpoint_rad = np.sum(
            self._centered_points * self._rad_magnetization[:, None], axis=0
        )
        self._fixpoint_rad_r = np.sum(
            self._centered_points_unitvecs * self._rad_magnetization[:, None], axis=0
        )
        self._fixpoint_abs_rad = np.sum(
            self._centered_points_unitvecs * np.abs(self._rad_magnetization[:, None]),
            axis=0,
        )
        self._fixpoint_mz2 = np.average(
            self._centered_points, weights=self._mz_magnetization ** 2, axis=0
        )

        self._second_fixpoints: Dict[str, np.ndarray] = {
            "mrmz": self._fixpoint_radmz,
            "mr": self._fixpoint_rad,
            "mr/r": self._fixpoint_rad_r,
            "mr_abs": self._fixpoint_abs_rad,
            "mz2": self._fixpoint_mz2,
        }

        # Connection vector between first fixpoint and the "rad" fixpoint
        rad_norm = np.linalg.norm(self._fixpoint_rad)
        if rad_norm == 0.0:
            self._fixpoint_connection = np.zeros(3)
        else:
            self._fixpoint_connection = self._fixpoint_rad / rad_norm

    # --------------------------------------------------------------------- #
    # Public methods
    # --------------------------------------------------------------------- #

    def geodesic_distance(self, other_texture: "SpinTexture") -> float:
        r"""
        Calculate the geodesic distance between this texture and another one.

        Procedure:
          1. Interpolate the magnetization of *this* texture onto the positions
             of ``other_texture``, using the x–y coordinates of the embedded
             points.
          2. Compute the geodesic (angular) distance between corresponding
             spins.
          3. Return the root-mean-square of these angles.

        Parameters
        ----------
        other_texture:
            Another :class:`SpinTexture` instance.

        Returns
        -------
        float
            Geodesic distance divided by the number of points in this texture.
        """
        this_texture_spins_ipol = griddata(
            self.points_embedding_circle[:, :2],
            self.spins_embedding_circle,
            other_texture.points_embedding_circle[:, :2],
            method="cubic",
            fill_value=0.0,
        )

        dotprod = np.sum(this_texture_spins_ipol * other_texture.spins_embedding_circle, axis=1)
        cross = np.cross(this_texture_spins_ipol, other_texture.spins_embedding_circle)
        norm_cp = np.linalg.norm(cross, axis=1)

        # arctan2(norm(m1 × m2), m1 · m2) = angle between m1 and m2
        angles = np.arctan2(norm_cp, dotprod)
        return float(np.sqrt(np.sum(angles ** 2))) / self.n_points

    def apply_texture_rotation(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Rotate the embedded positions and spins around the z-axis by an angle.

        Parameters
        ----------
        alpha:
            Rotation angle in radians. Positive angles correspond to
            counter-clockwise rotation in the x–y plane.

        Returns
        -------
        rot_points : ndarray, shape (N, 3)
            Rotated embedded positions.

        rot_spins : ndarray, shape (N, 3)
            Rotated embedded spins.
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        rotmat = np.array(
            [
                [ca, -sa, 0.0],
                [sa, ca, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot_points = self.points_embedding_circle @ rotmat.T
        rot_spins = self.spins_embedding_circle @ rotmat.T
        return rot_points, rot_spins

    def save_texture(self, outpath: Union[Path, str] = Path.cwd() / "texture.dat") -> None:
        r"""
        Save points and spins of this texture to a CSV file.

        The file will contain the columns: ``x, y, z, sx, sy, sz``.

        Parameters
        ----------
        outpath:
            Output path for the CSV file.
        """
        outpath = Path(outpath)
        df = pd.DataFrame(
            data={
                "x": self.points[:, 0],
                "y": self.points[:, 1],
                "z": self.points[:, 2],
                "sx": self.spins[:, 0],
                "sy": self.spins[:, 1],
                "sz": self.spins[:, 2],
            }
        )
        df.to_csv(outpath, index=False)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def second_fixpoints(self) -> Dict[str, np.ndarray]:
        r"""Dictionary with all computed second fixpoints."""
        return self._second_fixpoints

    @property
    def scnd_fixpoint_selection(self) -> str:
        r"""Return the current selector key for the second fixpoint."""
        return self._scnd_fixpoint_selection

    @scnd_fixpoint_selection.setter
    def scnd_fixpoint_selection(self, technique: str) -> None:
        r"""
        Set which second fixpoint to use.

        Parameters
        ----------
        technique:
            One of ``"mrmz"``, ``"mr"``, ``"mr/r"``, ``"mr_abs"``, ``"mz2"``.
        """
        if technique not in ["mrmz", "mr", "mr/r", "mr_abs", "mz2"]:
            raise ValueError("Not a valid second fixpoint selector.")
        self._scnd_fixpoint_selection = technique

    @property
    def scnd_fixpoint(self) -> np.ndarray:
        r"""Return the active second fixpoint according to the current selection."""
        return self._second_fixpoints[self._scnd_fixpoint_selection]

    @property
    def fixpoint_connection(self) -> np.ndarray:
        r"""Unit vector describing the connection based on the radial fixpoint."""
        return self._fixpoint_connection

    @property
    def phi(self) -> np.ndarray:
        r"""Polar angle (phi) of the centered points."""
        return self._phi

    @property
    def r(self) -> np.ndarray:
        r"""Radial distance of the centered points."""
        return self._r

    @property
    def rot_syms(self) -> Optional[List[int]]:
        r"""List of integer rotation symmetries for the structure (user-defined)."""
        return self._rot_syms

    @rot_syms.setter
    def rot_syms(self, rots: List[int]) -> None:
        r"""Set the rotation symmetries for this texture."""
        self._rot_syms = rots

    @property
    def centered_points(self) -> np.ndarray:
        r"""Coordinates relative to the mz-fixpoint."""
        return self._centered_points

    @property
    def centered_points_unitvecs(self) -> np.ndarray:
        r"""Normalized centered coordinates (direction vectors)."""
        return self._centered_points_unitvecs

    @property
    def rad_embedding_circle(self) -> float:
        r"""Radius of the embedding circle (max distance to the mz fixpoint)."""
        return float(self._rad_embedding_circle)

    @property
    def points_embedding_circle(self) -> np.ndarray:
        r"""Embedded points (here identical to the centered points)."""
        return self._points_embedding_circle

    @property
    def spins_embedding_circle(self) -> np.ndarray:
        r"""Spins at the embedded points."""
        return self._spins_embedding_circle

    @property
    def fixpoint_mz(self) -> np.ndarray:
        r"""First fixpoint based on mz magnetization."""
        return self._fixpoint_mz

    @property
    def fixpoint_mz2(self) -> np.ndarray:
        r"""Fixpoint based on mz² magnetization."""
        return self._fixpoint_mz2

    @property
    def fixpoint_rad(self) -> np.ndarray:
        r"""Fixpoint based on radial magnetization."""
        return self._fixpoint_rad

    @property
    def mz_magnetization(self) -> np.ndarray:
        r"""mz component for all spins (shifted by -1)."""
        return self._mz_magnetization

    @property
    def rad_magnetization(self) -> np.ndarray:
        r"""Radial component of the magnetization for all spins."""
        return self._rad_magnetization

    @property
    def n_points(self) -> int:
        r"""Number of magnetic moments in the spin texture."""
        return int(len(self._spins))

    @property
    def points(self) -> np.ndarray:
        r"""Coordinates of the magnetic moments of the spin texture."""
        return self._points

    @points.setter
    def points(self, new_points: np.ndarray) -> None:
        r"""
        Overwrite point coordinates and recompute fixpoints.

        Parameters
        ----------
        new_points:
            Array of shape (N, 3) with new positions.
        """
        new_points = np.asarray(new_points, dtype=float)
        if new_points.shape != self._spins.shape:
            raise ValueError(
                f"New `points` must have shape {self._spins.shape}, got {new_points.shape}."
            )
        self._points = new_points
        self._define_fps()

    @property
    def midpoint(self) -> np.ndarray:
        r"""Mean position of the original (not centered) points."""
        return np.round(np.mean(self.points, axis=0), 12)

    @property
    def spins(self) -> np.ndarray:
        r"""Spins of the spin texture."""
        return self._spins

