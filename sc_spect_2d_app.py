#!/usr/bin/env python3
"""
SC-SPECT 2D Interactive Explorer  (Trame / PyVista)
===================================================

Browser-streamed viewer for the **2D** multi-pinhole SC-SPECT scanner and its
ray-traced system response matrix (PPDFs). It is the 2D analogue of the 3D
``sc_spect_app.py`` and keeps the same Trame/PyVista pattern so it runs headless
on CCR compute nodes (server-side off-screen rendering streamed to Firefox on
``localhost``) with none of the on-screen GLX/llvmpipe problems.

Pipeline it visualises
----------------------
1. ``generate_mph_scanner_circularfov.py`` writes a ``.tensor`` file
   (a ``torch.save`` of a plain dict) describing the geometry per motion pose:

     scanner MD5        : geometry hash (matches the HDF5 ``layouts_md5`` attr)
     paper_config       : detector ring diameters, detectors_per_ring, aperture
                          ring, aperture count/diameter, fov_diameter_mm ...
     motion_parameters  : n rotational poses (collimator rotates, detectors fixed)
     layouts["position NNN"]:
         "detector units" : (Ncry, 4, 2)  crystal rectangles  (FIXED per pose)
         "plate circles"  : {"centers": (Nap, 2), "radius_mm": r}   (ROTATED)
         "plate segments" : (Nseg, 4, 2)  tungsten wedge walls       (ROTATED)

2. ``arg_ppdf_t8.py`` writes, per (layout_idx, pose_idx), an HDF5 file
   ``position_{layout:03d}_ppdfs_t8_{pose:02d}.hdf5`` with one dataset

         "ppdfs" : (Ncry, Npix)  float32     Npix = nx*ny  (e.g. 200*200=40000)

   plus attrs layout_idx, layouts_md5, pose_idx, dx_mm, dy_mm, pose_tag.
   Row ``i`` of "ppdfs" is crystal ``i``'s PPDF over the FOV grid, and crystal
   ``i`` is exactly ``detector units[i]`` — same order, same count. That 1:1 link
   is what makes click-to-pick work.

The scale problem & the fix
---------------------------
The interesting signal lives inside a ~5 mm-radius FOV while the detector rings
sit at ~330 mm (a ~66x span). Drawn in one frame the FOV collapses to a dot, so
the viewer uses **two linked panels**:
  * LEFT  — full-scanner overview: all crystals coloured by ring, tungsten walls,
            apertures, FOV marker, and the selected crystal's beam (line of
            response). Click any crystal here to select it.
  * RIGHT — FOV zoom: the selected crystal's PPDF as a heat-map over the 10 mm
            box (log or linear), or the summed-sensitivity map across all crystals.

Because everything is planar we render flat quads at z=0 and a flat ``ImageData``
heat-map — no volume rendering, no 3D camera juggling.

Usage
-----
  # serve interactively (open Firefox at http://localhost:8080 inside the desktop)
  python sc_spect_2d_app.py --data-dir ./data --serve --port 8080

  # headless proof screenshot (no browser needed)
  python sc_spect_2d_app.py --data-dir ./data --layout 0 --pose 0 \
                            --crystal 1700 --screenshot proof.png

  # static single-file HTML export of the current scene
  python sc_spect_2d_app.py --data-dir ./data --export-html scene.html

  # self-test with synthetic geometry + HDF5 (no real files, no torch needed)
  python sc_spect_2d_app.py --self-test --screenshot selftest.png

Interaction (serve mode)
------------------------
  * click a crystal (left panel)      -> select it, show its PPDF + beam
  * Crystal / Layout / Pose controls  -> step precisely; pose reloads the HDF5
  * Log / Linear switch               -> PPDF transfer function
  * Threshold slider                  -> raise/lower the PPDF display floor
  * Ring / Apertures / Tungsten / FOV / Beam switches -> toggle overview layers
  * Summed sensitivity switch         -> right panel shows sum over all crystals
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

# Ring colours (up to 8 rings; we have 4).
RING_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
               "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]
TUNGSTEN_COLOR = "#9ca3af"
APERTURE_COLOR = "#111827"
FOV_COLOR = "#ef4444"
BEAM_COLOR = "#fde047"


# ---------------------------------------------------------------------------
# Data model  (pure numpy; torch is confined to load_tensor below)
# ---------------------------------------------------------------------------

@dataclass
class Layout2D:
    """Geometry for a single motion pose."""
    position: np.ndarray            # [angle_rad, x_mm, y_mm]
    detector_units: np.ndarray      # (Ncry, 4, 2)  FIXED across poses
    aperture_centers: np.ndarray    # (Nap, 2)      ROTATED
    aperture_radius_mm: float
    plate_segments: np.ndarray      # (Nseg, 4, 2)  ROTATED (tungsten walls)


@dataclass
class Geometry2D:
    """Full decoded scanner: config + per-pose layouts + derived helpers."""
    scanner_md5: str
    paper_config: Dict
    motion_parameters: Dict
    layouts: List[Layout2D]
    detectors_per_ring: List[int]
    fov_diameter_mm: float

    # derived
    ring_of_crystal: np.ndarray = field(default=None)   # (Ncry,) int ring id
    crystal_centroids: np.ndarray = field(default=None)  # (Ncry, 2) mm

    def __post_init__(self):
        u = self.layouts[0].detector_units            # (Ncry,4,2)
        self.crystal_centroids = u.mean(axis=1)       # (Ncry,2)
        # ring membership from cumulative detectors_per_ring
        bounds = np.cumsum(self.detectors_per_ring)
        self.ring_of_crystal = np.searchsorted(bounds, np.arange(u.shape[0]),
                                               side="right").astype(int)

    @property
    def n_crystals(self) -> int:
        return self.layouts[0].detector_units.shape[0]

    @property
    def n_poses(self) -> int:
        return len(self.layouts)


def _to_numpy(x):
    """Convert a torch tensor (or anything array-like) to a float numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_tensor(path: str) -> Geometry2D:
    """
    Load a ``scanner_layouts_*.tensor`` (a torch.save'd dict) into Geometry2D.

    torch is imported lazily and only here, so the rest of the module (and the
    self-test) works in environments without torch. In the user's environment
    torch is present, so this path is the real one.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - only in torch-less envs
        raise RuntimeError(
            "Reading a .tensor requires PyTorch (torch.load). Install torch, or "
            "use --self-test to exercise the viewer without a .tensor."
        ) from e

    blob = torch.load(path, map_location="cpu", weights_only=False)

    paper = dict(blob.get("paper_config", {}))
    motion = dict(blob.get("motion_parameters", {}))
    dets_per_ring = [int(v) for v in paper.get("detectors_per_ring", [])]
    fov_d = float(paper.get("fov_diameter_mm", 10.0))

    layouts_dict = blob["layouts"]
    layouts: List[Layout2D] = []
    for key in sorted(layouts_dict.keys()):        # "position 000", "position 001", ...
        L = layouts_dict[key]
        circles = L["plate circles"]
        layouts.append(Layout2D(
            position=_to_numpy(L["position"]).reshape(-1).astype(float),
            detector_units=_to_numpy(L["detector units"]).astype(float),
            aperture_centers=_to_numpy(circles["centers"]).astype(float),
            aperture_radius_mm=float(circles["radius_mm"]),
            plate_segments=_to_numpy(L["plate segments"]).astype(float),
        ))

    if not dets_per_ring:
        # Fallback: treat all crystals as one ring if config lacks the field.
        dets_per_ring = [layouts[0].detector_units.shape[0]]

    return Geometry2D(
        scanner_md5=str(blob.get("scanner MD5", "")),
        paper_config=paper,
        motion_parameters=motion,
        layouts=layouts,
        detectors_per_ring=dets_per_ring,
        fov_diameter_mm=fov_d,
    )


# ---------------------------------------------------------------------------
# System matrix (HDF5) access
# ---------------------------------------------------------------------------

class SysMat2D:
    """
    Lazy accessor for the per-(layout,pose) HDF5 PPDF files produced by
    arg_ppdf_t8.py. Each file is ~0.5 GB, so we open the one we need and slice
    single crystal rows on demand (or sum all rows for the sensitivity map).
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._h5 = None
        self._path: Optional[Path] = None
        self.layout_idx: Optional[int] = None
        self.pose_idx: Optional[int] = None
        self.n_crystals: Optional[int] = None
        self.n_pixels: Optional[int] = None
        self.dx_mm: float = 0.0
        self.dy_mm: float = 0.0
        self.layouts_md5: str = ""

    @staticmethod
    def filename(layout_idx: int, pose_idx: int) -> str:
        return f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"

    def available(self) -> List[Tuple[int, int]]:
        """List (layout_idx, pose_idx) pairs present on disk."""
        pairs = []
        for p in sorted(self.data_dir.glob("position_*_ppdfs_t8_*.hdf5")):
            try:
                stem = p.stem  # position_000_ppdfs_t8_00
                parts = stem.split("_")
                layout_idx = int(parts[1])
                pose_idx = int(parts[-1])
                pairs.append((layout_idx, pose_idx))
            except Exception:
                continue
        return pairs

    def open(self, layout_idx: int, pose_idx: int):
        """Open (and cache) the HDF5 for a given layout/pose."""
        import h5py
        if (self._h5 is not None and self.layout_idx == layout_idx
                and self.pose_idx == pose_idx):
            return
        self.close()
        path = self.data_dir / self.filename(layout_idx, pose_idx)
        if not path.exists():
            raise FileNotFoundError(f"PPDF file not found: {path}")
        self._h5 = h5py.File(path, "r")
        self._path = path
        self.layout_idx = layout_idx
        self.pose_idx = pose_idx
        dset = self._h5["ppdfs"]
        self.n_crystals, self.n_pixels = int(dset.shape[0]), int(dset.shape[1])
        a = self._h5.attrs
        self.dx_mm = float(a.get("dx_mm", 0.0))
        self.dy_mm = float(a.get("dy_mm", 0.0))
        self.layouts_md5 = str(a.get("layouts_md5", ""))

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def ppdf(self, crystal_idx: int) -> np.ndarray:
        """Return crystal_idx's PPDF as a flat (n_pixels,) float32 array."""
        if self._h5 is None:
            raise RuntimeError("SysMat2D.open() must be called first")
        return np.asarray(self._h5["ppdfs"][crystal_idx], dtype=np.float32)

    def sensitivity(self) -> np.ndarray:
        """Sum of all crystal PPDFs -> flat (n_pixels,) sensitivity map."""
        if self._h5 is None:
            raise RuntimeError("SysMat2D.open() must be called first")
        dset = self._h5["ppdfs"]
        acc = np.zeros(dset.shape[1], dtype=np.float64)
        chunk = 512
        for start in range(0, dset.shape[0], chunk):
            acc += np.asarray(dset[start:start + chunk], dtype=np.float64).sum(axis=0)
        return acc.astype(np.float32)


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------

def _quads_to_polydata(quads: np.ndarray) -> pv.PolyData:
    """(N,4,2) quads -> flat PolyData at z=0 with one quad cell each."""
    n = quads.shape[0]
    pts = np.zeros((n * 4, 3), dtype=float)
    pts[:, :2] = quads.reshape(-1, 2)
    faces = np.hstack([[4, 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3] for k in range(n)]) \
        if n else np.empty(0, dtype=int)
    return pv.PolyData(pts, faces)


def build_detector_mesh(geom: Geometry2D, layout_idx: int) -> pv.PolyData:
    """All crystals as one PolyData; cell_data 'ring' and 'crystal'."""
    units = geom.layouts[layout_idx].detector_units       # (Ncry,4,2)
    poly = _quads_to_polydata(units)
    poly.cell_data["ring"] = geom.ring_of_crystal.astype(float)
    poly.cell_data["crystal"] = np.arange(units.shape[0], dtype=float)
    return poly


def build_plate_mesh(geom: Geometry2D, layout_idx: int) -> pv.PolyData:
    return _quads_to_polydata(geom.layouts[layout_idx].plate_segments)


def build_aperture_mesh(geom: Geometry2D, layout_idx: int, n_sides: int = 20) -> pv.PolyData:
    """Aperture circles as small filled discs merged into one PolyData."""
    L = geom.layouts[layout_idx]
    centers = L.aperture_centers
    r = L.aperture_radius_mm
    ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * r      # (n_sides,2)
    blocks = []
    for cx, cy in centers:
        blocks.append(_disc_polydata(cx, cy, ring))
    if not blocks:
        return pv.PolyData()
    merged = blocks[0]
    for b in blocks[1:]:
        merged = merged.merge(b)
    return merged


def _disc_polydata(cx: float, cy: float, ring_xy: np.ndarray) -> pv.PolyData:
    n = ring_xy.shape[0]
    pts = np.zeros((n, 3), float)
    pts[:, 0] = ring_xy[:, 0] + cx
    pts[:, 1] = ring_xy[:, 1] + cy
    face = np.hstack([[n], np.arange(n)])
    return pv.PolyData(pts, face)


def circle_polyline(cx: float, cy: float, r: float, n: int = 128) -> pv.PolyData:
    """A closed circle as a line loop (for FOV outline)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=True)
    pts = np.zeros((n, 3), float)
    pts[:, 0] = cx + r * np.cos(ang)
    pts[:, 1] = cy + r * np.sin(ang)
    lines = np.hstack([[n], np.arange(n)])
    pd = pv.PolyData(pts)
    pd.lines = lines
    return pd


def box_outline(cx: float, cy: float, w: float, h: float) -> pv.PolyData:
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[cx - hw, cy - hh, 0], [cx + hw, cy - hh, 0],
                    [cx + hw, cy + hh, 0], [cx - hw, cy + hh, 0],
                    [cx - hw, cy - hh, 0]], float)
    pd = pv.PolyData(pts)
    pd.lines = np.hstack([[5], np.arange(5)])
    return pd


# ---------------------------------------------------------------------------
# PPDF -> flat image grid
# ---------------------------------------------------------------------------

def ppdf_to_grid(flat: np.ndarray, npix: Tuple[int, int], size_mm: Tuple[float, float],
                 center_mm: Tuple[float, float], transpose: bool = False,
                 flip_x: bool = False, flip_y: bool = False) -> pv.ImageData:
    """
    Flat PPDF (nx*ny,) -> pv.ImageData heat-map over the FOV box.

    Memory order from arg_ppdf_t8.py is index = a*nx + b (b fastest), which maps
    directly onto VTK ImageData cell ordering (i fastest) with i<->b, j<->a. The
    transpose/flip toggles exist because the a<->x vs a<->y convention lives in
    scanner_modeling.geometry_2d (not in the files we have), so the viewer lets
    you fix the orientation visually if needed.
    """
    nx, ny = npix
    img = flat.reshape(ny, nx)              # [a, b] with b fastest
    if transpose:
        img = img.T
        nx, ny = ny, nx
    if flip_x:
        img = img[:, ::-1]
    if flip_y:
        img = img[::-1, :]

    cx, cy = center_mm
    sx, sy = size_mm
    spacing = (sx / nx, sy / ny, 1.0)
    origin = (cx - sx / 2.0, cy - sy / 2.0, 0.0)
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, 1), spacing=spacing, origin=origin)
    grid.cell_data["ppdf"] = img.ravel(order="C")
    return grid


def log_scale(flat: np.ndarray, floor_frac: float = 1e-4) -> np.ndarray:
    """log10 transfer function (matches the 3D tool's professor-requested log view)."""
    m = float(flat.max())
    if m <= 0:
        return flat.copy()
    floor = m * floor_frac
    return np.log10(np.clip(flat, floor, None))


def intensity_centroid(flat: np.ndarray, npix: Tuple[int, int],
                       size_mm: Tuple[float, float], center_mm: Tuple[float, float]
                       ) -> Optional[np.ndarray]:
    """Intensity-weighted centroid of a PPDF, in world mm (for the beam line)."""
    nx, ny = npix
    img = flat.reshape(ny, nx)
    w = img.sum()
    if w <= 0:
        return None
    cx, cy = center_mm
    sx, sy = size_mm
    xs = (np.arange(nx) + 0.5) * (sx / nx) + (cx - sx / 2.0)
    ys = (np.arange(ny) + 0.5) * (sy / ny) + (cy - sy / 2.0)
    gx = (img.sum(axis=0) * xs).sum() / w
    gy = (img.sum(axis=1) * ys).sum() / w
    return np.array([gx, gy], float)


# ---------------------------------------------------------------------------
# Explorer: owns the two-panel Plotter and all scene state
# ---------------------------------------------------------------------------

class Explorer2D:
    def __init__(self, geom: Geometry2D, sysmat: Optional[SysMat2D],
                 layout_idx: int = 0, pose_idx: int = 0, crystal: int = 0,
                 fov_npix: Tuple[int, int] = (200, 200),
                 fov_size_mm: Tuple[float, float] = (10.0, 10.0),
                 off_screen: bool = True, window_size=(1280, 640)):
        self.geom = geom
        self.sysmat = sysmat
        self.layout_idx = layout_idx
        self.pose_idx = pose_idx
        self.crystal = int(crystal)
        self.fov_npix = fov_npix
        self.fov_size_mm = fov_size_mm

        self.show_apertures = True
        self.show_tungsten = True
        self.show_fov = True
        self.show_beam = True
        self.use_log = True
        self.threshold = 0.0
        self.show_sensitivity = False
        self.transpose = False
        self.flip_x = False
        self.flip_y = False

        from scipy.spatial import cKDTree
        self._kdtree = cKDTree(geom.crystal_centroids)

        self.p = pv.Plotter(off_screen=off_screen, shape=(1, 2),
                            window_size=list(window_size))
        self.p.set_background("white")
        self._actors: Dict[str, object] = {}

        if self.sysmat is not None:
            self._ensure_open()
        self._build_overview()
        self._build_fov_panel()

    # ---- data helpers ----
    def _ensure_open(self):
        if self.sysmat is not None:
            self.sysmat.open(self.layout_idx, self.pose_idx)

    def _fov_center(self) -> Tuple[float, float]:
        if self.sysmat is not None and self.sysmat.n_pixels is not None:
            return (self.sysmat.dx_mm, self.sysmat.dy_mm)
        return (0.0, 0.0)

    def _current_ppdf(self) -> Optional[np.ndarray]:
        if self.sysmat is None:
            return None
        self._ensure_open()
        if self.show_sensitivity:
            return self.sysmat.sensitivity()
        return self.sysmat.ppdf(self.crystal)

    # ---- overview (left panel) ----
    def _build_overview(self):
        self.p.subplot(0, 0)
        g = self.geom
        det = build_detector_mesh(g, self.layout_idx)
        self._actors["detectors"] = self.p.add_mesh(
            det, scalars="ring", cmap=RING_COLORS[:max(g.detectors_per_ring.__len__(), 1)],
            show_scalar_bar=False, show_edges=False, name="detectors")

        plate = build_plate_mesh(g, self.layout_idx)
        self._actors["tungsten"] = self.p.add_mesh(
            plate, color=TUNGSTEN_COLOR, show_edges=False, name="tungsten")

        aps = build_aperture_mesh(g, self.layout_idx)
        self._actors["apertures"] = self.p.add_mesh(
            aps, color=APERTURE_COLOR, name="apertures")

        cx, cy = self._fov_center()
        fov = circle_polyline(cx, cy, g.fov_diameter_mm / 2.0)
        self._actors["fov_overview"] = self.p.add_mesh(
            fov, color=FOV_COLOR, line_width=2, name="fov_overview")

        self._update_beam()
        self.p.view_xy()
        self.p.camera.zoom(1.1)

    def _update_beam(self):
        for key in ("beam", "sel_crystal"):
            if key in self._actors:
                self.p.remove_actor(self._actors.pop(key), render=False)
        self.p.subplot(0, 0)

        # highlight selected crystal
        units = self.geom.layouts[self.layout_idx].detector_units[self.crystal]
        sel = _quads_to_polydata(units[None, ...])
        self._actors["sel_crystal"] = self.p.add_mesh(
            sel, color="black", line_width=3, style="wireframe", name="sel_crystal")

        if not self.show_beam:
            return
        flat = self._current_ppdf()
        if flat is None or self.show_sensitivity:
            return
        tgt = intensity_centroid(flat, self.fov_npix, self.fov_size_mm, self._fov_center())
        if tgt is None:
            return
        src = self.geom.crystal_centroids[self.crystal]
        # extend the line a little past the FOV for readability
        d = tgt - src
        p1 = src
        p2 = tgt + 0.15 * d
        line = pv.Line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
        self._actors["beam"] = self.p.add_mesh(
            line, color=BEAM_COLOR, line_width=3, name="beam")

    # ---- FOV zoom (right panel) ----
    def _build_fov_panel(self):
        self.p.subplot(0, 1)
        self._render_fov()

    def _render_fov(self):
        self.p.subplot(0, 1)
        for key in ("ppdf", "fov_box", "fov_circle"):
            if key in self._actors:
                self.p.remove_actor(self._actors.pop(key), render=False)

        cx, cy = self._fov_center()
        box = box_outline(cx, cy, *self.fov_size_mm)
        self._actors["fov_box"] = self.p.add_mesh(box, color="#666666", line_width=1,
                                                   name="fov_box")
        circ = circle_polyline(cx, cy, self.geom.fov_diameter_mm / 2.0)
        self._actors["fov_circle"] = self.p.add_mesh(circ, color=FOV_COLOR,
                                                      line_width=2, name="fov_circle")

        flat = self._current_ppdf()
        if flat is not None:
            disp = log_scale(flat) if self.use_log else flat.astype(float)
            if self.threshold > 0:
                lo, hi = float(disp.min()), float(disp.max())
                floor = lo + self.threshold * (hi - lo)
                disp = np.where(disp >= floor, disp, np.nan)
            grid = ppdf_to_grid(disp, self.fov_npix, self.fov_size_mm, (cx, cy),
                                transpose=self.transpose, flip_x=self.flip_x,
                                flip_y=self.flip_y)
            label = ("log10 sensitivity" if self.show_sensitivity and self.use_log else
                     "sensitivity" if self.show_sensitivity else
                     "log10 PPDF" if self.use_log else "PPDF")
            self._actors["ppdf"] = self.p.add_mesh(
                grid, scalars="ppdf", cmap="inferno", nan_opacity=0.0,
                scalar_bar_args={"title": label}, name="ppdf")

        self.p.subplot(0, 1)
        self.p.view_xy()
        pad = 0.5
        self.p.reset_camera()

    # ---- public state setters (used by Trame callbacks) ----
    def set_crystal(self, idx: int):
        self.crystal = int(np.clip(idx, 0, self.geom.n_crystals - 1))
        self._update_beam()
        self._render_fov()

    def pick_world(self, xy: Tuple[float, float]):
        """Map a clicked world point to the nearest crystal and select it."""
        _, idx = self._kdtree.query([xy[0], xy[1]])
        self.set_crystal(int(idx))
        return int(idx)

    def set_layout_pose(self, layout_idx: int, pose_idx: int):
        self.layout_idx = int(layout_idx)
        self.pose_idx = int(pose_idx)
        if self.sysmat is not None:
            self._ensure_open()
        # rebuild overview meshes that depend on pose (plate/apertures rotate)
        for key in ("detectors", "tungsten", "apertures", "fov_overview"):
            if key in self._actors:
                self.p.remove_actor(self._actors.pop(key), render=False)
        self._build_overview()
        self._render_fov()

    def set_visibility(self, **kw):
        for k, v in kw.items():
            setattr(self, k, bool(v))
        for key, flag in (("apertures", self.show_apertures),
                          ("tungsten", self.show_tungsten),
                          ("fov_overview", self.show_fov)):
            if key in self._actors:
                self._actors[key].SetVisibility(flag)
        self._update_beam()

    def set_log(self, use_log: bool):
        self.use_log = bool(use_log)
        self._render_fov()

    def set_threshold(self, t: float):
        self.threshold = float(np.clip(t, 0.0, 0.99))
        self._render_fov()

    def set_sensitivity(self, on: bool):
        self.show_sensitivity = bool(on)
        self._update_beam()
        self._render_fov()

    def set_orientation(self, transpose=None, flip_x=None, flip_y=None):
        if transpose is not None:
            self.transpose = bool(transpose)
        if flip_x is not None:
            self.flip_x = bool(flip_x)
        if flip_y is not None:
            self.flip_y = bool(flip_y)
        self._render_fov()

    # ---- outputs ----
    def screenshot(self, path: str):
        self.p.screenshot(path)
        return path

    def export_html(self, path: str):
        self.p.export_html(path)
        return path


# ---------------------------------------------------------------------------
# Trame server (mirrors the 3D sc_spect_app.py pattern)
# ---------------------------------------------------------------------------

def build_server(explorer: Explorer2D):
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import vuetify3 as v3
    from pyvista.trame.ui import plotter_ui

    server = get_server()
    server.client_type = "vue3"
    state, ctrl = server.state, server.controller
    geom = explorer.geom

    pose_pairs = explorer.sysmat.available() if explorer.sysmat is not None else []
    layout_opts = sorted({li for li, _ in pose_pairs}) or list(range(geom.n_poses))
    pose_opts = sorted({pi for _, pi in pose_pairs}) or [0]

    state.update(dict(
        crystal=explorer.crystal, layout_idx=explorer.layout_idx,
        pose_idx=explorer.pose_idx, use_log=explorer.use_log,
        threshold=int(explorer.threshold * 100), show_sensitivity=False,
        show_apertures=True, show_tungsten=True, show_fov=True, show_beam=True,
        transpose=False, flip_x=False, flip_y=False,
        n_crystals=geom.n_crystals, layout_opts=layout_opts, pose_opts=pose_opts,
    ))

    # picking: nearest crystal to the clicked point
    def on_pick(point, *args, **kwargs):
        if point is not None:
            idx = explorer.pick_world((point[0], point[1]))
            state.crystal = idx
            ctrl.view_update()
    explorer.p.subplot(0, 0)
    try:
        explorer.p.enable_point_picking(callback=on_pick, show_message=False,
                                        use_picker="hardware", left_clicking=True)
    except Exception:
        pass

    @state.change("crystal")
    def _c(crystal, **kw):
        explorer.set_crystal(int(crystal)); ctrl.view_update()

    @state.change("layout_idx", "pose_idx")
    def _lp(layout_idx, pose_idx, **kw):
        explorer.set_layout_pose(int(layout_idx), int(pose_idx)); ctrl.view_update()

    @state.change("use_log")
    def _log(use_log, **kw):
        explorer.set_log(use_log); ctrl.view_update()

    @state.change("threshold")
    def _t(threshold, **kw):
        explorer.set_threshold(float(threshold) / 100.0); ctrl.view_update()

    @state.change("show_sensitivity")
    def _s(show_sensitivity, **kw):
        explorer.set_sensitivity(show_sensitivity); ctrl.view_update()

    @state.change("show_apertures", "show_tungsten", "show_fov", "show_beam")
    def _vis(show_apertures, show_tungsten, show_fov, show_beam, **kw):
        explorer.set_visibility(show_apertures=show_apertures, show_tungsten=show_tungsten,
                                show_fov=show_fov, show_beam=show_beam)
        ctrl.view_update()

    @state.change("transpose", "flip_x", "flip_y")
    def _orient(transpose, flip_x, flip_y, **kw):
        explorer.set_orientation(transpose=transpose, flip_x=flip_x, flip_y=flip_y)
        ctrl.view_update()

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("SC-SPECT 2D Explorer")
        with layout.drawer:
            v3.VSlider(v_model=("crystal", explorer.crystal), min=0,
                       max=geom.n_crystals - 1, step=1, label="Crystal",
                       thumb_label="always", density="compact")
            with v3.VRow(classes="pa-2"):
                v3.VSelect(v_model=("layout_idx",), items=("layout_opts",),
                           label="Layout (rotation)", density="compact")
                v3.VSelect(v_model=("pose_idx",), items=("pose_opts",),
                           label="Pose (bed)", density="compact")
            v3.VSwitch(v_model=("use_log",), label="Log scale", density="compact")
            v3.VSlider(v_model=("threshold",), min=0, max=99, step=1,
                       label="Threshold %", density="compact")
            v3.VSwitch(v_model=("show_sensitivity",), label="Summed sensitivity",
                       density="compact")
            v3.VDivider()
            v3.VSwitch(v_model=("show_apertures",), label="Apertures", density="compact")
            v3.VSwitch(v_model=("show_tungsten",), label="Tungsten walls", density="compact")
            v3.VSwitch(v_model=("show_fov",), label="FOV", density="compact")
            v3.VSwitch(v_model=("show_beam",), label="Beam / LOR", density="compact")
            v3.VDivider()
            v3.VSwitch(v_model=("transpose",), label="Transpose PPDF", density="compact")
            v3.VSwitch(v_model=("flip_x",), label="Flip X", density="compact")
            v3.VSwitch(v_model=("flip_y",), label="Flip Y", density="compact")
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = plotter_ui(explorer.p)
                ctrl.view_update = view.update

    return server


# ---------------------------------------------------------------------------
# Self-test data (synthetic geometry + HDF5; no torch, no real files)
# ---------------------------------------------------------------------------

def _synthetic_geometry(dets_per_ring=(48, 72, 96, 120)) -> Geometry2D:
    """Small stand-in scanner with the real structure (4 rings, aperture ring)."""
    ring_inner = [260.0, 390.0, 520.0, 650.0]
    W, H = 0.84, 6.0
    units = []
    ring_ids = []
    for ri, (inner_d, n) in enumerate(zip(ring_inner, dets_per_ring)):
        r_c = inner_d / 2.0 + H / 2.0
        for i in range(n):
            th = 2 * math.pi * i / n
            t = np.array([-math.sin(th), math.cos(th)])
            r = np.array([math.cos(th), math.sin(th)])
            c = r * r_c
            v1 = c + (W / 2) * t + (H / 2) * r
            v2 = c - (W / 2) * t + (H / 2) * r
            v3 = c - (W / 2) * t - (H / 2) * r
            v4 = c + (W / 2) * t - (H / 2) * r
            units.append(np.stack([v1, v2, v3, v4]))
            ring_ids.append(ri)
    units = np.stack(units)                               # (Ncry,4,2)

    n_ap = 60
    r_in = 67.5 / 2.0
    r_ap = 0.2
    r_center = r_in + 2.5 / 2.0
    ang = np.linspace(0, 2 * math.pi, n_ap, endpoint=False)
    centers = np.stack([r_center * np.cos(ang), r_center * np.sin(ang)], axis=1)

    r_out = r_in + 2.5
    segs = []
    for a in ang:
        da = (2 * math.pi / n_ap) * 0.3
        for s in (-1, 1):
            a1 = a + s * da
            a2 = a + s * da * 2
            q = np.array([[r_in * math.cos(a1), r_in * math.sin(a1)],
                          [r_out * math.cos(a1), r_out * math.sin(a1)],
                          [r_out * math.cos(a2), r_out * math.sin(a2)],
                          [r_in * math.cos(a2), r_in * math.sin(a2)]])
            segs.append(q)
    segs = np.stack(segs)

    layout = Layout2D(position=np.array([0.0, 0.0, 0.0]), detector_units=units,
                      aperture_centers=centers, aperture_radius_mm=r_ap,
                      plate_segments=segs)
    return Geometry2D(scanner_md5="SELFTEST", paper_config={"detectors_per_ring": list(dets_per_ring)},
                      motion_parameters={}, layouts=[layout, layout],
                      detectors_per_ring=list(dets_per_ring), fov_diameter_mm=10.0)


def _write_synthetic_hdf5(path: str, geom: Geometry2D, npix=(200, 200),
                          size_mm=(10.0, 10.0), dx=0.0, dy=0.0):
    """Write a plausible HDF5: each crystal projects a small blob toward center."""
    import h5py
    nx, ny = npix
    ncry = geom.n_crystals
    xs = (np.arange(nx) + 0.5) * (size_mm[0] / nx) - size_mm[0] / 2 + dx
    ys = (np.arange(ny) + 0.5) * (size_mm[1] / ny) - size_mm[1] / 2 + dy
    X, Y = np.meshgrid(xs, ys)                            # (ny,nx) [a=y, b=x]
    cen = geom.crystal_centroids
    with h5py.File(path, "w") as f:
        f.attrs["layout_idx"] = 0
        f.attrs["pose_idx"] = 0
        f.attrs["dx_mm"] = float(dx)
        f.attrs["dy_mm"] = float(dy)
        f.attrs["layouts_md5"] = geom.scanner_md5
        f.attrs["pose_tag"] = "t8"
        dset = f.create_dataset("ppdfs", (ncry, nx * ny), dtype="f4")
        for i in range(ncry):
            direction = -cen[i] / (np.linalg.norm(cen[i]) + 1e-9)
            bx, by = 2.5 * direction[0], 2.5 * direction[1]
            sig = 1.2
            blob = np.exp(-(((X - bx) ** 2 + (Y - by) ** 2) / (2 * sig ** 2)))
            blob *= (0.4 + 0.6 * ((i % 7) / 6.0))
            dset[i] = blob.astype(np.float32).ravel(order="C")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_tensor(data_dir: str, tensor: Optional[str]) -> Optional[str]:
    if tensor:
        return tensor
    hits = sorted(glob.glob(os.path.join(data_dir, "scanner_layouts_*.tensor")))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="SC-SPECT 2D Trame/PyVista explorer")
    ap.add_argument("--data-dir", default="./data",
                    help="dir with scanner_layouts_*.tensor and position_*_ppdfs_t8_*.hdf5")
    ap.add_argument("--tensor", default=None, help="explicit path to .tensor (else auto-find)")
    ap.add_argument("--layout", type=int, default=0, help="layout_idx (collimator rotation)")
    ap.add_argument("--pose", type=int, default=0, help="pose_idx (bed position 0-7)")
    ap.add_argument("--crystal", type=int, default=0, help="initial crystal index")
    ap.add_argument("--serve", action="store_true", help="start Trame server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--screenshot", default=None, help="write an offscreen PNG and exit")
    ap.add_argument("--export-html", default=None, help="write a static HTML scene and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="use synthetic geometry + HDF5 (no .tensor, no torch)")
    args = ap.parse_args()

    off_screen = not args.serve

    if args.self_test:
        geom = _synthetic_geometry()
        tmp = os.path.join(os.getcwd(), "_selftest_data")
        os.makedirs(tmp, exist_ok=True)
        h5 = os.path.join(tmp, SysMat2D.filename(0, 0))
        if not os.path.exists(h5):
            _write_synthetic_hdf5(h5, geom)
        sysmat = SysMat2D(tmp)
    else:
        tpath = _resolve_tensor(args.data_dir, args.tensor)
        if tpath is None:
            raise SystemExit(f"No .tensor found in {args.data_dir} (use --tensor or --self-test)")
        print(f"Loading geometry: {tpath}")
        geom = load_tensor(tpath)
        sysmat = SysMat2D(args.data_dir)
        if geom.scanner_md5 and sysmat.available():
            li, pi = args.layout, args.pose
            try:
                sysmat.open(li, pi)
                if sysmat.layouts_md5 and geom.scanner_md5 and \
                        sysmat.layouts_md5 != geom.scanner_md5:
                    print(f"[warn] md5 mismatch: tensor={geom.scanner_md5} "
                          f"hdf5={sysmat.layouts_md5}")
                if sysmat.n_crystals != geom.n_crystals:
                    print(f"[warn] crystal count mismatch: tensor={geom.n_crystals} "
                          f"hdf5={sysmat.n_crystals}")
            except FileNotFoundError as e:
                print(f"[warn] {e}")

    print(f"Geometry: {geom.n_crystals} crystals, {len(geom.detectors_per_ring)} rings, "
          f"{geom.n_poses} poses, FOV Ø {geom.fov_diameter_mm} mm")

    explorer = Explorer2D(geom, sysmat, layout_idx=args.layout, pose_idx=args.pose,
                          crystal=args.crystal, off_screen=off_screen)

    if args.screenshot:
        explorer.screenshot(args.screenshot)
        print(f"wrote {args.screenshot}")
        return
    if args.export_html:
        explorer.export_html(args.export_html)
        print(f"wrote {args.export_html}")
        return
    if args.serve:
        server = build_server(explorer)
        server.start(port=args.port)
        return

    print("Nothing to do. Pass --serve, --screenshot, or --export-html.")


if __name__ == "__main__":
    main()