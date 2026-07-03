#!/usr/bin/env python3
"""
SC-SPECT Interactive 3D Explorer
================================

Interactive desktop viewer (PyVista / VTK) for a self-collimating SPECT
geometry and its ray-traced system response matrix (.sysmat).

Decoded geometry layout (from Params_*.dat):

  Params_Image.dat      [nx,ny,nz, dx,dy,dz, n_rot, rot_step_rad,
                         shift_x,shift_y,shift_z, fov_to_collimator]
  Params_Detector.dat   [n_det, then n_det * 12 floats:
                         cx, cy(layer, FOV-relative), cz,
                         size_x, size_y, size_z, p0..p3, flag]
  Params_Collimator.dat [n_col_layers, ... , @10: n_holes, plate_w, plate_t,
                         plate_h, offset_y, ... , @100: n_holes * 9 floats:
                         x, y1, y2, z, radius, ...]

World frame: FOV is centred at the origin. Detector cy and the collimator
are shifted by +fov_to_collimator so the stack reads
FOV  ->  collimator plate  ->  4 detector layers  along +Y (depth).

The .sysmat is memory-mapped, shape (n_rot, n_det, nx*ny*nz), float32, with
voxel memory order Z,Y,X (X fastest). Only one detector's slice
(nx*ny*nz floats ~ 8 MB at 128^3) is read at a time, so a 47 GB matrix is
fine.

Usage
-----
  python sc_spect_explorer.py --data-dir PEGen_RayTracing_CircularHole
  python sc_spect_explorer.py --data-dir DIR --sysmat FILE.sysmat --detector 1400
  python sc_spect_explorer.py --data-dir DIR --screenshot out.png   # headless proof

Interaction
-----------
  * Left-click any detector  -> select it, show its PPDF beam + lines of response
  * "Detector" slider        -> step through detector index precisely
  * "Threshold" slider       -> raise/lower the PPDF rendering floor
  * "Rotation" slider        -> change rotation index (if n_rot > 1)
  * Layer / collimator / FOV check-buttons -> toggle visibility
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

import pyvista as pv
pv.OFF_SCREEN = True
pv.start_xvfb()

# ----------------------------------------------------------------------------
# Geometry parsing
# ----------------------------------------------------------------------------

LAYER_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
                "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]


@dataclass
class Image:
    nx: int; ny: int; nz: int
    dx: float; dy: float; dz: float
    n_rot: int
    rot_step: float
    shift_x: float; shift_y: float; shift_z: float
    fov2col: float

    @property
    def nvox(self) -> int:
        return self.nx * self.ny * self.nz


@dataclass
class Geometry:
    image: Image
    det_xyz: np.ndarray      # (n_det, 3) world coords
    det_size: np.ndarray     # (n_det, 3)
    det_layer: np.ndarray    # (n_det,) integer layer index 0..L-1
    layer_depths: np.ndarray # (L,) sorted world Y of each layer
    plate_center: np.ndarray # (3,)
    plate_size: np.ndarray   # (3,)
    hole_xyz: np.ndarray     # (n_holes, 3) world coords (channel centre)
    hole_r: float
    hole_h: float

    @property
    def n_det(self) -> int:
        return self.det_xyz.shape[0]


def parse_geometry(data_dir: Path) -> Geometry:
    img_raw = np.fromfile(data_dir / "Params_Image.dat", dtype=np.float32)
    det_raw = np.fromfile(data_dir / "Params_Detector.dat", dtype=np.float32)
    col_raw = np.fromfile(data_dir / "Params_Collimator.dat", dtype=np.float32)

    image = Image(
        nx=int(img_raw[0]), ny=int(img_raw[1]), nz=int(img_raw[2]),
        dx=float(img_raw[3]), dy=float(img_raw[4]), dz=float(img_raw[5]),
        n_rot=int(img_raw[6]), rot_step=float(img_raw[7]),
        shift_x=float(img_raw[8]), shift_y=float(img_raw[9]), shift_z=float(img_raw[10]),
        fov2col=float(img_raw[11]),
    )

    n_det = int(det_raw[0])
    D = det_raw[1:1 + n_det * 12].reshape(-1, 12).astype(np.float64)
    det_xyz = D[:, 0:3].copy()
    det_xyz[:, 1] += image.fov2col              # FOV-relative depth -> world
    det_size = D[:, 3:6].copy()

    layer_depths = np.unique(det_xyz[:, 1])
    depth_to_idx = {d: i for i, d in enumerate(layer_depths)}
    det_layer = np.array([depth_to_idx[d] for d in det_xyz[:, 1]], dtype=int)

    # Collimator plate
    n_holes = int(col_raw[10])
    plate_w, plate_t, plate_h, offset_y = (float(col_raw[11]), float(col_raw[12]),
                                           float(col_raw[13]), float(col_raw[14]))
    plate_center = np.array([0.0, image.fov2col + offset_y, 0.0])
    plate_size = np.array([plate_w, plate_t, plate_h])

    if n_holes > 0:
        H = col_raw[100:100 + n_holes * 9].reshape(-1, 9).astype(np.float64)
        hx = H[:, 0]
        hy = (H[:, 1] + H[:, 2]) / 2.0 + image.fov2col
        hz = H[:, 3]
        hole_xyz = np.column_stack([hx, hy, hz])
        hole_r = float(np.median(H[:, 4]))
        hole_h = float(np.median(H[:, 2] - H[:, 1]))
    else:
        hole_xyz = np.empty((0, 3)); hole_r = 0.5; hole_h = plate_t

    return Geometry(image, det_xyz, det_size, det_layer, layer_depths,
                    plate_center, plate_size, hole_xyz, hole_r, hole_h)


# ----------------------------------------------------------------------------
# System matrix (.sysmat) access
# ----------------------------------------------------------------------------

class SysMat:
    """Lazy, memory-mapped access to one detector's PPDF at a time."""

    def __init__(self, path: Path, image: Image, n_det: int):
        self.image = image
        self.path = Path(path)
        nbytes = self.path.stat().st_size
        per_det = image.nvox * 4
        expected = image.n_rot * n_det * per_det
        self.n_det = n_det
        if nbytes != expected:
            inferred = nbytes // (image.n_rot * per_det)
            print(f"[SysMat] size {nbytes} != expected {expected}; "
                  f"inferring n_det={inferred} from file size.")
            self.n_det = int(inferred)
        self.mm = np.memmap(self.path, dtype=np.float32, mode="r",
                            shape=(image.n_rot, self.n_det, image.nvox))

    def ppdf(self, rotation: int, detector: int) -> np.ndarray:
        return np.asarray(self.mm[rotation, detector], dtype=np.float32)


def ppdf_to_grid(ppdf_1d: np.ndarray, image: Image) -> pv.ImageData:
    """Build a VTK uniform grid (point scalars) from a flat PPDF in Z,Y,X order."""
    p3d = ppdf_1d.reshape(image.nz, image.ny, image.nx).transpose(2, 1, 0)  # -> [ix,iy,iz]
    ox = (-(image.nx - 1) / 2.0) * image.dx + image.shift_x
    oy = (-(image.ny - 1) / 2.0) * image.dy + image.shift_y
    oz = (-(image.nz - 1) / 2.0) * image.dz + image.shift_z
    grid = pv.ImageData(dimensions=(image.nx, image.ny, image.nz),
                        spacing=(image.dx, image.dy, image.dz),
                        origin=(ox, oy, oz))
    grid.point_data["ppdf"] = p3d.flatten(order="F")  # VTK wants X fastest
    return grid


def beam_centroids(ppdf_1d: np.ndarray, image: Image, thr: float) -> np.ndarray:
    """Intensity-weighted centroids of each connected PPDF blob (for LOR rays)."""
    p3d = ppdf_1d.reshape(image.nz, image.ny, image.nx).transpose(2, 1, 0)
    mx = float(p3d.max())
    if mx <= 0:
        return np.empty((0, 3))
    mask = p3d >= thr * mx
    if not mask.any():
        return np.empty((0, 3))
    ix = (np.arange(image.nx) - (image.nx - 1) / 2.0) * image.dx + image.shift_x
    iy = (np.arange(image.ny) - (image.ny - 1) / 2.0) * image.dy + image.shift_y
    iz = (np.arange(image.nz) - (image.nz - 1) / 2.0) * image.dz + image.shift_z
    X, Y, Z = np.meshgrid(ix, iy, iz, indexing="ij")
    lab, n = label(mask)
    cents = []
    for b in range(1, n + 1):
        m = lab == b
        if m.sum() < 2:
            continue
        w = p3d[m]
        cents.append([np.average(X[m], weights=w),
                      np.average(Y[m], weights=w),
                      np.average(Z[m], weights=w)])
    return np.array(cents) if cents else np.empty((0, 3))


# ----------------------------------------------------------------------------
# Mesh builders
# ----------------------------------------------------------------------------

# unit-cube corners and quad faces (centre at origin, edge length 1)
_CORNERS = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                     [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
_QUADS = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                   [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]])


def boxes_polydata(centers: np.ndarray, sizes: np.ndarray) -> pv.PolyData:
    """One merged PolyData containing an axis-aligned box per (center, size)."""
    n = len(centers)
    if n == 0:
        return pv.PolyData()
    pts = (_CORNERS[None] * sizes[:, None, :] + centers[:, None, :]).reshape(-1, 3)
    base = (np.arange(n) * 8)[:, None, None]
    faces_idx = _QUADS[None] + base                      # (n, 6, 4)
    faces = np.empty((n * 6, 5), dtype=np.int64)
    faces[:, 0] = 4
    faces[:, 1:] = faces_idx.reshape(-1, 4)
    return pv.PolyData(pts, faces.ravel())


# ----------------------------------------------------------------------------
# The interactive viewer
# ----------------------------------------------------------------------------

class Explorer:
    def __init__(self, geom: Geometry, sysmat: Optional[SysMat],
                 detector: int = 0, rotation: int = 0, threshold: float = 0.05,
                 off_screen: bool = False):
        self.g = geom
        self.sm = sysmat
        self.det = int(detector)
        self.rot = int(rotation)
        self.thr = float(threshold)
        self.kdtree = cKDTree(geom.det_xyz)
        self.layer_visible = [True] * len(geom.layer_depths)

        self.p = pv.Plotter(off_screen=off_screen, window_size=(1500, 950),
                            title="SC-SPECT Explorer")
        self.p.set_background("#0e1117")
        self._actors = {}          # dynamic actors we replace on update
        self._layer_actors = {}

        self._build_static_scene()
        self._update_ppdf()
        self._add_widgets()
        self._add_hud()

    # ---- static geometry -------------------------------------------------
    def _build_static_scene(self):
        g = self.g
        # detectors, one actor per layer (toggleable, coloured)
        for li, depth in enumerate(g.layer_depths):
            m = g.det_layer == li
            mesh = boxes_polydata(g.det_xyz[m], g.det_size[m])
            a = self.p.add_mesh(mesh, color=LAYER_COLORS[li % len(LAYER_COLORS)],
                                opacity=0.55, show_edges=False,
                                name=f"layer{li}",
                                label=f"Layer {li}  (Y={depth:.0f} mm, n={m.sum()})")
            self._layer_actors[li] = a

        # collimator plate (translucent) + hole channels
        plate = pv.Cube(center=g.plate_center,
                        x_length=g.plate_size[0], y_length=g.plate_size[1],
                        z_length=g.plate_size[2])
        self.p.add_mesh(plate, color="#9aa0a6", opacity=0.18, name="plate",
                        label="Tungsten plate")
        if len(g.hole_xyz):
            cyl = pv.Cylinder(direction=(0, 1, 0), radius=g.hole_r,
                              height=g.hole_h, resolution=12)
            holes = pv.PolyData(g.hole_xyz).glyph(geom=cyl, scale=False, orient=False)
            self.p.add_mesh(holes, color="#11151c", name="holes",
                            label=f"Collimator holes (n={len(g.hole_xyz)})")

        # FOV wireframe
        im = g.image
        fov = pv.Box(bounds=(im.shift_x - im.nx * im.dx / 2, im.shift_x + im.nx * im.dx / 2,
                             im.shift_y - im.ny * im.dy / 2, im.shift_y + im.ny * im.dy / 2,
                             im.shift_z - im.nz * im.dz / 2, im.shift_z + im.nz * im.dz / 2))
        self.p.add_mesh(fov, style="wireframe", color="#22c55e", line_width=2,
                        name="fov", label="FOV")

        try:
            self.p.add_axes(color="white", viewport=(0.82, 0.0, 1.0, 0.18))
        except TypeError:
            self.p.add_axes(color="white")
        self.p.add_legend(bcolor="#0e1117", border=True, size=(0.20, 0.24),
                          loc="upper right", face=None)
        self.p.camera_position = "yz"
        self.p.camera.azimuth = 35
        self.p.camera.elevation = 18

    # ---- dynamic PPDF beam ----------------------------------------------
    def _update_ppdf(self):
        for nm in ("vol", "lor", "sel"):
            if nm in self._actors:
                self.p.remove_actor(self._actors.pop(nm))

        g = self.g
        # highlight selected detector
        sel = boxes_polydata(g.det_xyz[self.det:self.det + 1],
                             g.det_size[self.det:self.det + 1] * 1.15)
        self._actors["sel"] = self.p.add_mesh(sel, color="#ff2d55", name="sel",
                                              show_edges=True, edge_color="white",
                                              line_width=2)
        if self.sm is None:
            self._refresh_hud(maxv=None)
            return

        ppdf = self.sm.ppdf(self.rot, self.det)
        maxv = float(ppdf.max())
        if maxv > 0:
            grid = ppdf_to_grid(ppdf, g.image)
            opacity = [0.0, 0.015, 0.06, 0.16, 0.4, 0.85]
            self._actors["vol"] = self.p.add_volume(
                grid, scalars="ppdf", cmap="hot", opacity=opacity,
                clim=[self.thr * maxv, maxv], name="vol",
                scalar_bar_args=dict(title="PPDF", color="white"))
            cents = beam_centroids(ppdf, g.image, self.thr)
            if len(cents):
                d = g.det_xyz[self.det]
                segpts = np.empty((len(cents) * 2, 3))
                segpts[0::2] = d
                segpts[1::2] = cents
                self._actors["lor"] = self.p.add_lines(
                    segpts, color="#ffb000", width=3, name="lor")
        self._refresh_hud(maxv=maxv)

    # ---- HUD / text ------------------------------------------------------
    def _add_hud(self):
        self._refresh_hud(None)

    def _refresh_hud(self, maxv):
        g = self.g
        d = g.det_xyz[self.det]
        li = g.det_layer[self.det]
        txt = (f"Detector {self.det} / {g.n_det - 1}   layer {li} "
               f"(Y={g.layer_depths[li]:.0f} mm)\n"
               f"pos = ({d[0]:.0f}, {d[1]:.0f}, {d[2]:.0f}) mm   "
               f"rotation {self.rot}/{g.image.n_rot - 1}")
        if maxv is not None:
            txt += f"\nPPDF max = {maxv:.3e}   threshold = {self.thr:.0%}"
        elif self.sm is None:
            txt += "\n(no .sysmat loaded - geometry only)"
        self.p.add_text(txt, position="upper_left", font_size=11,
                        color="white", name="hud")

    # ---- widgets ---------------------------------------------------------
    def _add_widgets(self):
        g = self.g
        self.p.enable_point_picking(callback=self._on_pick, show_message=False,
                                    use_picker="point", left_clicking=True,
                                    show_point=False)

        self.p.add_slider_widget(
            self._on_det_slider, [0, g.n_det - 1], value=self.det,
            title="Detector index", fmt="%.0f",
            pointa=(0.03, 0.86), pointb=(0.30, 0.86), style="modern")
        if self.sm is not None:
            self.p.add_slider_widget(
                self._on_thr_slider, [0.005, 0.5], value=self.thr,
                title="PPDF threshold", fmt="%.3f",
                pointa=(0.03, 0.74), pointb=(0.30, 0.74), style="modern")
        if g.image.n_rot > 1:
            self.p.add_slider_widget(
                self._on_rot_slider, [0, g.image.n_rot - 1], value=self.rot,
                title="Rotation", fmt="%.0f",
                pointa=(0.03, 0.62), pointb=(0.30, 0.62), style="modern")

        # layer visibility toggles (bottom-left row, with a caption)
        self.p.add_text("Toggle layers:", position=(20, 58), font_size=9,
                        color="white", name="lblabel")
        for li in range(len(g.layer_depths)):
            self.p.add_checkbox_button_widget(
                lambda flag, i=li: self._toggle_layer(i, flag), value=True,
                position=(20 + li * 50, 12), size=38,
                color_on=LAYER_COLORS[li % len(LAYER_COLORS)], color_off="grey")

    # ---- callbacks -------------------------------------------------------
    def _on_pick(self, point, *args):
        if point is None:
            return
        pt = np.asarray(point).reshape(-1)[:3]
        _, idx = self.kdtree.query(pt)
        self.det = int(idx)
        self._update_ppdf()

    def _on_det_slider(self, value):
        self.det = int(round(value))
        self._update_ppdf()

    def _on_thr_slider(self, value):
        self.thr = float(value)
        self._update_ppdf()

    def _on_rot_slider(self, value):
        self.rot = int(round(value))
        self._update_ppdf()

    def _toggle_layer(self, li, flag):
        self._layer_actors[li].SetVisibility(bool(flag))
        self.p.render()

    # ---- run -------------------------------------------------------------
    def show(self):
        self.p.show()

    def screenshot(self, path: str):
        self.p.show(auto_close=False)
        self.p.screenshot(path)
        self.p.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SC-SPECT interactive 3D explorer.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--sysmat", type=Path, default=None,
                    help="Path to .sysmat; if omitted, auto-globbed from data-dir.")
    ap.add_argument("--detector", type=int, default=1400)
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="Render off-screen to this PNG and exit (no window).")
    ap.add_argument("--export-html", type=Path, default=None,
                    help="Export an interactive 3D scene to this HTML file and exit.")
    args = ap.parse_args()

    geom = parse_geometry(args.data_dir)
    print(f"Loaded geometry: {geom.n_det} detectors, {len(geom.layer_depths)} layers "
          f"at Y={geom.layer_depths}, {len(geom.hole_xyz)} holes.")

    sm = None
    sysmat_path = args.sysmat
    if sysmat_path is None:
        hits = list(args.data_dir.glob("*.sysmat"))
        sysmat_path = hits[0] if hits else None
    if sysmat_path and Path(sysmat_path).exists():
        sm = SysMat(sysmat_path, geom.image, geom.n_det)
        print(f"Loaded system matrix: {sysmat_path} (n_det={sm.n_det})")
    else:
        print("No .sysmat found - running in geometry-only mode.")

    off = args.screenshot is not None or args.export_html is not None
    exp = Explorer(geom, sm, detector=args.detector, rotation=args.rotation,
                   threshold=args.threshold, off_screen=off)
    if args.export_html is not None:
        exp.p.show(auto_close=False)        # render pass to populate the scene
        exp.p.export_html(str(args.export_html))
        exp.p.close()
        print(f"Saved interactive HTML -> {args.export_html}")
    elif args.screenshot is not None:
        exp.screenshot(str(args.screenshot))
        print(f"Saved screenshot -> {args.screenshot}")
    else:
        exp.show()


if __name__ == "__main__":
    main()
