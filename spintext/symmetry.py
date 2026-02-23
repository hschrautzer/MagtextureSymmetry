"""
Module for analyzing the rotation and mirror symmetries of a planar spin texture.
"""

from __future__ import annotations
import json
import logging as lg
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from .texture import SpinTexture


class TextureSymmetry:
    r"""
    Analyze rotation and mirror symmetries of a (planar) spin texture.

    Parameters
    ----------
    texture:
        Spin texture to be analyzed.

    beta:
        Angle of the DMI vector. This sets the helicity
        :math:`\gamma = \pi/2 - \beta` and affects the mirror operation.
        ``beta = 0``  → Bloch DMI,  
        ``beta = pi/2`` → Néel DMI.

    verbose:
        Logging level (10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR). Only used
        when no external logger is provided.

    logger:
        Optional external :class:`logging.Logger`. If ``None``, a local logger is created.

    logfilepath:
        Optional path to a log file. Used only if ``logger`` is ``None``.
    """

    def __init__(
        self,
        texture: SpinTexture,
        beta: float = np.pi / 2,
        verbose: int = 10,
        logger: Union[None, lg.Logger] = None,
        logfilepath: Union[Path, None] = None,
    ) -> None:
        self._texture = texture
        self._beta = beta
        self._verbose = verbose
        self._logfilepath = logfilepath

        if logger is None:
            self._logger = self._add_logger()
        else:
            self._logger = logger

        self._rot_syms: Union[None, List[int]] = None
        self._rot_sym_inf: bool = False
        self._i_extrusion: bool = False
        self._sections: Union[None, List[SpinTexture]] = None
        self._extruded_sections: Union[None, List[SpinTexture]] = None
        self._extruded_mirror_ax: Union[None, List[np.ndarray]] = None
        self._mirrors: Union[None, List[np.ndarray]] = None

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #

    def _add_logger(self) -> lg.Logger:
        r"""
        Create and configure a logger, using the class' ``verbose`` and
        optional ``logfilepath`` attributes.

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        l_logger = lg.getLogger(__name__)
        if self._verbose in [10, 20, 30, 40]:
            formatter = lg.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
            )
            console_handler = lg.StreamHandler()
            console_handler.setLevel(self._verbose)
            console_handler.setFormatter(formatter)
            l_logger.addHandler(console_handler)

            if self._logfilepath is not None:
                file_handler = lg.FileHandler(self._logfilepath)
                file_handler.setLevel(self._verbose)
                file_handler.setFormatter(formatter)
                l_logger.addHandler(file_handler)

            l_logger.setLevel(self._verbose)
        else:
            raise ValueError("Not a valid verbose-level")
        return l_logger

    # ------------------------------------------------------------------ #
    # Rotation symmetries
    # ------------------------------------------------------------------ #

    def detect_rotation_symmetries(
        self,
        prime_rotations: Union[int, Tuple[int, ...]] = (2, 3, 5, 7, 11, 13),
        geodesic_crit: float = 1.0e-3,
    ) -> List[int]:
        r"""
        Try to detect discrete rotation symmetries of the texture.

        Parameters
        ----------
        prime_rotations:
            Candidate rotation numbers :math:`n` for which a symmetry
            under :math:`2\pi / n` is tested.

        geodesic_crit:
            Threshold for the geodesic distance below which the rotated
            texture is considered equivalent to the original.

        Returns
        -------
        list[int]
            List of candidate rotation numbers for which a symmetry was found.
        """
        self._logger.info("Detect rotation symmetries...")
        rot_syms: List[int] = []

        if isinstance(prime_rotations, int):
            prime_rotations = (prime_rotations,)

        for prime_rotation in prime_rotations:
            self._logger.info(f"Test rotation number: {prime_rotation}")
            rot_angle = 2 * np.pi / prime_rotation
            i_rot_ax, dist = self._try_rotation_symmetry(angle=rot_angle, crit=geodesic_crit)
            self._logger.info(f"Considered symmetry: {i_rot_ax}")
            if i_rot_ax:
                rot_syms.append(prime_rotation)

        self._rot_syms = rot_syms
        self._texture.rot_syms = rot_syms

        if not self._rot_syms:
            self._logger.info("No rotation symmetry detected.")
        else:
            self._logger.info(
                f"Detected rotation symmetry with rotation number(s): {self._rot_syms}"
            )

        if len(self._rot_syms) == len(prime_rotations):
            self._rot_sym_inf = True
            self._logger.info(
                f"C_inf or D_inf rotation (rotation number > {max(prime_rotations)})"
            )
        elif len(self._rot_syms) <= 1:
            self._logger.info("Define non-rotationally-symmetric reference sections...")
            self._sections = self.define_prime_angle_sections()
            self._logger.info("... done.")
        else:
            self._logger.info(
                "Multiple symmetries; choose the one with the largest rotation number "
                "(smallest distinct region)."
            )
            self._rot_syms = [max(self._rot_syms)]
            self._sections = self.define_prime_angle_sections()

        self._logger.info("... done.")
        return rot_syms

    def define_prime_angle_sections(self) -> List[SpinTexture]:
        r"""
        Define angular sections according to the detected rotation symmetries.

        Returns
        -------
        list[SpinTexture]
            List of textures representing sections in angle.
            If no rotation symmetries are found, the full texture is returned as a single section.
        """
        l_sections: List[SpinTexture] = []
        current_start_angle = 0.0

        if self._rot_syms is not None:
            for rot_sym in self._rot_syms:
                current_end_angle = current_start_angle + 2 * np.pi / rot_sym
                l_cond = (self._texture.phi >= current_start_angle) & (self._texture.phi <= current_end_angle)
                l_sections.append(SpinTexture(points=self._texture.centered_points[l_cond],
                                              spins=self._texture.spins[l_cond]))
                current_start_angle = 0.0
        else:
            l_sections.append(SpinTexture(points=self._texture.centered_points, spins=self._texture.spins))
        return l_sections

    def _try_rotation_symmetry(self, angle: float, crit: float = 1.0e-3) -> Tuple[bool, float]:
        r"""
        Test a given rotation angle for rotational symmetry.

        Parameters
        ----------
        angle:
            Rotation angle in radians.

        crit:
            Threshold for the geodesic distance below which the rotated texture
            is considered equivalent.

        Returns
        -------
        (bool, float)
            Tuple of (symmetry_found, geodesic_distance).
        """
        rot_points, rot_spins = self._texture.apply_texture_rotation(alpha=angle)
        rotated_texture = SpinTexture(points=rot_points, spins=rot_spins)
        dist = self._texture.geodesic_distance(other_texture=rotated_texture)
        self._logger.info(f"Geodesic distance per spin between rotated and not rotated: {dist}")
        return dist <= crit, dist

    @property
    def rot_syms(self) -> Union[None, List[int]]:
        r"""List of detected rotation symmetry orders (if any)."""
        return self._rot_syms

    @property
    def sections(self) -> Union[None, List[SpinTexture]]:
        r"""Sections of the texture corresponding to the detected rotation symmetries."""
        return self._sections

    @property
    def rot_sym_inf(self) -> bool:
        r"""Whether the structure has effectively infinite rotation symmetry."""
        return self._rot_sym_inf

    # ------------------------------------------------------------------ #
    # Mirror symmetries
    # ------------------------------------------------------------------ #

    def detect_mirror_symmetries(self, geodesic_crit: float = 1.0) -> None:
        r"""
        Detect mirror symmetries of the texture (possibly in reduced sections).

        Parameters
        ----------
        geodesic_crit:
            Maximum allowed geodesic distance between mirrored and original texture
            to be considered symmetric.
        """
        if self._rot_syms is None:
            self._logger.error("Please determine rotation symmetries first.")
            raise ValueError("Please determine rotation symmetries first.")

        # Infinite rotational symmetry: either D_inf or C_inf group
        if self.rot_sym_inf:
            self._logger.info("Infinite rotation symmetries: group is either D_inf or C_inf.")
            self._logger.info("D_inf (e.g. Bloch/Neel skyrmion): infinite mirror axes.")
            self._logger.info("C_inf (something else): no mirror axes.")

            # Test a default mirror axis along x (normal in y direction)
            default_axis = np.array([1.0, 0.0, 0.0])
            normal = np.array([0.0, 1.0, 0.0])

            i_found, _ = self.try_mirror_symmetry(
                points=self._texture.centered_points,
                spins=self._texture.spins,
                center=np.array([0.0, 0.0, 0.0]),
                normal=normal,
                crit=geodesic_crit,
            )
            if i_found:
                self._logger.info("Result: structure belongs to D_inf.")
                self._mirrors = [default_axis]
            else:
                self._mirrors = None
                i_found, _ = self._try_mirror_spinflip_symmetry(
                    points=self._texture.centered_points,
                    spins=self._texture.spins,
                    center=np.array([0.0, 0.0, 0.0]),
                    normal=normal,
                    crit=geodesic_crit,
                )
                if i_found:
                    self._mirrors = [default_axis]
                    self._logger.info("Invariant under mirror + spin flip.")
                else:
                    self._logger.info("Not invariant under mirror + spin flip.")
                    self._logger.info("Result: structure belongs to C_inf.")
            return

        self._extruded_sections = []

        # No rotation symmetry: test mirror directly on original texture
        if not self._rot_syms:
            self._logger.info("Search for mirror symmetry in original texture.")
            self._extruded_sections = self._sections

            label = "mz2"
            fixpoint = self._texture.second_fixpoints[label]
            mirror_dir = fixpoint - self._texture.fixpoint_mz
            mirror_dir /= np.linalg.norm(mirror_dir)

            self._logger.debug(
                f"Test {label} fixpoint (Dir: {mirror_dir}) for mirror axis."
            )
            normal = np.cross(np.array([0.0, 0.0, 1.0]), mirror_dir)

            i_found, _ = self.try_mirror_symmetry(
                points=self._texture.centered_points,
                spins=self._texture.spins,
                center=np.array([0.0, 0.0, 0.0]),
                normal=normal,
                crit=geodesic_crit,
            )
            if i_found:
                self._mirrors = [mirror_dir]
                self._logger.info(f"Found mirror direction {self._mirrors[0]}")
        else:
            # Finite rotation symmetry: extrude smallest domain and look for mirror there
            self._logger.info("Search for mirror symmetry in rotational domain.")
            self._i_extrusion = True
            for idx, section in enumerate(self._sections):
                self._logger.info(
                    f"Extrude section with rotation number: {self._rot_syms[idx]}"
                )
                l_ext_pts, l_ext_spins = self.extrude_section(
                    section=section.points,
                    section_spins=section.spins,
                    section_angle=2 * np.pi / self._rot_syms[idx],
                )
                self._extruded_sections.append(
                    SpinTexture(points=l_ext_pts, spins=l_ext_spins)
                )

            self._extruded_mirror_ax = []
            for idx, ext_sect in enumerate(self._extruded_sections):
                label = "mz2"
                fixpoint = ext_sect.second_fixpoints[label]
                mirror_dir = fixpoint - ext_sect.fixpoint_mz
                mirror_dir /= np.linalg.norm(mirror_dir)

                self._logger.debug(
                    f"Test {label} fixpoint (Dir: {mirror_dir}) for mirror axis in extruded section."
                )
                normal = np.cross(np.array([0.0, 0.0, 1.0]), mirror_dir)

                i_found, _ = self.try_mirror_symmetry(
                    points=ext_sect.centered_points,
                    spins=ext_sect.spins,
                    center=ext_sect.fixpoint_mz,
                    normal=normal,
                    crit=geodesic_crit,
                )
                if i_found:
                    self._extruded_mirror_ax.append(mirror_dir)
                    self._logger.info(
                        f"Found mirror direction in extruded section: {mirror_dir}"
                    )

            self._mirrors = self._rotate_mirrorax_fromextruded(
                extruded_axes=self._extruded_mirror_ax, rotation_numbers=self._rot_syms
            )

    @staticmethod
    def extrude_section(section: np.ndarray, section_spins: np.ndarray,
                        section_angle: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Extrude a section of the texture over the full :math:`[0, 2\pi]` range.

        Parameters
        ----------
        section:
            Positions in the section.

        section_spins:
            Spins in the section.

        section_angle:
            Angular size of the section.

        Returns
        -------
        (points, spins):
            Arrays with points/spins mapped to the full angular domain.
        """
        l_scale = 2 * np.pi / section_angle # corresponds to n
        l_phi = np.arctan2(section[:, 1], section[:, 0])
        rot_angles = (l_scale-1) * l_phi # we have to map the point of the section with the biggest phi angle back
        # to (1,0). Thus point corresponds to phi_max = 2pi/n and thus we have to map it to 2pi which corresponds to
        # rotating (incrementing the angle) (by n-1)*2pi/n. Now replace for every smaller phi: 2pi/n by phi and you get
        # the above formula
        #rot_angles = l_phi - l_scale * l_phi
        rot_points, rot_spins = [], []

        for idx, point in enumerate(section):
            rotmat = np.array(
                [
                    [np.cos(rot_angles[idx]), -np.sin(rot_angles[idx]), 0.0],
                    [np.sin(rot_angles[idx]), np.cos(rot_angles[idx]), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            rot_points.append(rotmat @ point)
            rot_spins.append(rotmat @ section_spins[idx])

        return np.asarray(rot_points), np.asarray(rot_spins)

    @property
    def extruded_sections(self) -> Union[None, List[SpinTexture]]:
        r"""Extruded sections corresponding to detected rotation symmetries."""
        return self._extruded_sections

    @property
    def extruded_mirror_ax(self) -> Union[None, List[np.ndarray]]:
        r"""
        Mirror axis directions found in the extruded section(s), as unit vectors
        in the extruded frame.
        """
        return self._extruded_mirror_ax

    # ------------------------------------------------------------------ #
    # Mirror tests
    # ------------------------------------------------------------------ #

    def try_mirror_symmetry(
        self,
        points: np.ndarray,
        spins: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        crit: float = 5.0,
    ) -> Tuple[bool, float]:
        r"""
        Test a given mirror symmetry.

        Parameters
        ----------
        points:
            Points of the structure to investigate.

        spins:
            Spins at the given points.

        center:
            Fixpoint of the mirror operation.

        normal:
            Normal vector of the mirror plane.

        crit:
            Threshold for the geodesic distance below which the mirror is
            considered a symmetry.

        Returns
        -------
        (bool, float)
            Tuple of (symmetry_found, geodesic_distance).
        """
        normal = normal / np.linalg.norm(normal)
        self._logger.debug(f"Try mirror with normal: {normal}")

        points_centered = points - center

        # mirror positions
        p_mirrored = []
        for p in points_centered:
            p_mirrored.append(p - 2 * np.dot(p, normal) * normal)
        p_mirrored = np.asarray(p_mirrored)

        spins_interpolated = griddata(
            points_centered[:, :2],
            spins,
            p_mirrored[:, :2],
            method="cubic",
            fill_value=0.0,
        )

        s_mirrored = []
        gamma = np.pi / 2 - self._beta
        rotmat = np.array(
            [
                [np.cos(2 * gamma), -np.sin(2 * gamma), 0.0],
                [np.sin(2 * gamma), np.cos(2 * gamma), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        for s in spins:
            s_mirrored.append(rotmat.T @ (s - 2 * np.dot(s, normal) * normal))
        s_mirrored = np.asarray(s_mirrored)

        dotprod = np.sum(spins_interpolated * s_mirrored, axis=1)
        norm_cp = np.linalg.norm(np.cross(spins_interpolated, s_mirrored), axis=1)
        dist = np.sqrt(np.sum(np.arctan2(norm_cp, dotprod) ** 2)) / len(points_centered)

        self._logger.debug(
            f"Geodesic distance between mirrored and not mirrored: {dist}"
        )
        print(dist), print(crit)
        return dist <= crit, dist

    def _try_mirror_spinflip_symmetry(
        self,
        points: np.ndarray,
        spins: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        crit: float = 5.0,
    ) -> Tuple[bool, float]:
        r"""
        Test a mirror symmetry combined with an in-plane spin flip.

        Parameters
        ----------
        points:
            Points of the structure to investigate.

        spins:
            Spins at the given points.

        center:
            Fixpoint of the mirror operation.

        normal:
            Normal of the mirror plane.

        crit:
            Threshold for the geodesic distance below which the mirror+flip
            is considered a symmetry.

        Returns
        -------
        (bool, float)
            Tuple of (symmetry_found, geodesic_distance).
        """
        normal = normal / np.linalg.norm(normal)
        self._logger.debug(f"Try mirror+spinflip with normal: {normal}")

        points_centered = points - center

        # mirror positions
        p_mirrored = []
        for p in points_centered:
            p_mirrored.append(p - 2 * np.dot(p, normal) * normal)
        p_mirrored = np.asarray(p_mirrored)

        spins_interpolated = griddata(
            points_centered[:, :2],
            spins,
            p_mirrored[:, :2],
            method="cubic",
            fill_value=0.0,
        )

        s_mirrored = []
        for s in spins:
            s_mirror = s - 2 * np.dot(s, normal) * normal
            s_flipped = s_mirror.copy()
            s_flipped[:2] *= -1
            s_mirrored.append(s_flipped)
        s_mirrored = np.asarray(s_mirrored)

        dotprod = np.sum(spins_interpolated * s_mirrored, axis=1)
        norm_cp = np.linalg.norm(np.cross(spins_interpolated, s_mirrored), axis=1)
        dist = np.sqrt(np.sum(np.arctan2(norm_cp, dotprod) ** 2))

        self._logger.debug(
            f"Geodesic distance between mirrored+spinflip and not mirrored: {dist}"
        )
        return dist <= crit, dist

    @property
    def i_extrusion(self) -> bool:
        r"""Whether extrusion was necessary to detect mirror symmetries."""
        return self._i_extrusion

    @property
    def mirrors(self) -> Union[None, List[np.ndarray]]:
        r"""List of mirror axis directions (unit vectors) in the original frame."""
        return self._mirrors

    # ------------------------------------------------------------------ #
    # Back-transform mirror axes from extruded frame
    # ------------------------------------------------------------------ #

    def _rotate_mirrorax_fromextruded(self, extruded_axes: List[np.ndarray],
                                      rotation_numbers: List[int]) -> List[np.ndarray]:
        r"""
        Rotate mirror axes extracted in the extruded section back to the
        original texture frame.

        Parameters
        ----------
        extruded_axes:
            Mirror axes in the extruded section (one per section).

        rotation_numbers:
            Rotation number for each section.

        Returns
        -------
        list[np.ndarray]
            All mirror axes in the coordinate frame of the original texture,
            including degenerate mirrors for all rotational domains.
        """
        l_mirror_ax: List[np.ndarray] = []
        for idx, ext_ax in enumerate(extruded_axes):
            # orientation mirror axes in extruded coordinate frame
            l_phi = np.arctan2(ext_ax[1], ext_ax[0])
            angle = -(rotation_numbers[idx]-1)/(rotation_numbers[idx])*l_phi
            rotmat = np.array([
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ])
            mirror_orig = rotmat@ext_ax
            # orientation mirror axes in original coordinate system
            l_phi_orig = l_phi/rotation_numbers[idx]
            mirror_orig = np.array([np.cos(l_phi_orig),np.sin(l_phi_orig),0])
            for j in range(rotation_numbers[idx]):
                # j=0 , ... j=n-1
                current_mirror_angle = l_phi_orig + j*np.pi/rotation_numbers[idx]
                current_mirror = np.array([np.cos(current_mirror_angle),np.sin(current_mirror_angle),0])
                l_mirror_ax.append(current_mirror)
        return l_mirror_ax

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save_symmetries(self, outpath: Union[Path, str] = Path.cwd() / "symmetries.json") -> None:
        r"""
        Save detected symmetries to a JSON file.

        The JSON contains:
          - ``rad_embedding_circle``: radius of the embedding circle
          - ``rotations``: list of rotation symmetry orders
          - ``section_i``: convex hull points of each section in 2D (x, y)
          - ``mirrors``: list of mirror axis directions

        Parameters
        ----------
        outpath:
            Output file path.
        """
        out_data = {
            "rad_embedding_circle": self._texture.rad_embedding_circle,
            "rotations": self._rot_syms,
        }

        if self._sections is not None:
            for idx, section in enumerate(self._sections):
                hull_section = ConvexHull(section.points[:, :2])
                hull_points = section.points[hull_section.vertices, :]
                out_data[f"section_{idx}"] = [list(pt) for pt in hull_points]

        if self._mirrors is not None:
            out_data["mirrors"] = [list(mirror) for mirror in self._mirrors]
        else:
            out_data["mirrors"] = None

        outpath = Path(outpath)
        with outpath.open("w") as ff:
            json.dump(out_data, ff, indent=4)
