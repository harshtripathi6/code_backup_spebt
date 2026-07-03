#!/usr/bin/env python3
"""
SC-SPECT Interactive 3D Explorer — Trame web app
================================================

Browser-based, fully interactive companion to sc_spect_explorer.py.

Unlike the static HTML export (--export-html), this keeps the controls LIVE:
  * Detector slider          -> re-renders the PPDF beam + LORs for that detector
  * Threshold slider         -> raises/lowers the PPDF floor
  * Rotation slider          -> steps rotation index (if n_rot > 1)
  * Layer switches           -> toggle each detector layer on/off
  * Left-click a detector    -> selects it (syncs the slider too)

The heavy VTK rendering happens SERVER-SIDE (off-screen, software GL via
llvmpipe), and frames are streamed to the browser. So it works on a headless /
software-only node where an on-screen window can't open.

Run (on the SAME node as your OnDemand desktop):
    python sc_spect_app.py --data-dir DIR --detector 1400

Then open the printed URL  ->  http://localhost:8080  in a browser running
INSIDE that desktop session (e.g. the desktop's Firefox).

Requires: trame, trame-vtk, trame-vuetify
    pip install "trame>=3.6.5" "trame-vtk>=2.8.10" trame-vuetify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Reuse the existing module (sc_spect_explorer.py must sit in the same folder).
# Importing it also sets pv.OFF_SCREEN = True and starts Xvfb, which is exactly
# what we want for server-side rendering.
from sc_spect_explorer import (
    parse_geometry, SysMat, ppdf_to_grid, beam_centroids,
    boxes_polydata, LAYER_COLORS,
)

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3, html
from pyvista.trame.ui import plotter_ui

pv.OFF_SCREEN = True


# ----------------------------------------------------------------------------
# Scene: builds the plotter and exposes update hooks (reuses module functions)
# ----------------------------------------------------------------------------

class Scene:
    def __init__(self, geom, sysmat, detector, rotation, threshold):
        self.g = geom
        self.sm = sysmat
        self.det = int(detector)
        self.rot = int(rotation)
        self.thr = float(threshold)
        self.kdtree = cKDTree(geom.det_xyz)
        self._dyn = {}
        self._layer_actors = {}

        self.pl = pv.Plotter(off_screen=True)
        self.pl.set_background("#0e1117")
        self._build_static()
        self.update_ppdf()
        self.pl.camera_position = "yz"
        self.pl.camera.azimuth = 35
        self.pl.camera.elevation = 18
        self.pl.reset_camera()

    def _build_static(self):
        g = self.g
        for li, depth in enumerate(g.layer_depths):
            m = g.det_layer == li
            mesh = boxes_polydata(g.det_xyz[m], g.det_size[m])
            self._layer_actors[li] = self.pl.add_mesh(
                mesh, color=LAYER_COLORS[li % len(LAYER_COLORS)],
                opacity=0.55, name=f"layer{li}", reset_camera=False)

        plate = pv.Cube(center=g.plate_center, x_length=g.plate_size[0],
                        y_length=g.plate_size[1], z_length=g.plate_size[2])
        self.pl.add_mesh(plate, color="#9aa0a6", opacity=0.18,
                         name="plate", reset_camera=False)

        if len(g.hole_xyz):
            cyl = pv.Cylinder(direction=(0, 1, 0), radius=g.hole_r,
                              height=g.hole_h, resolution=12)
            holes = pv.PolyData(g.hole_xyz).glyph(geom=cyl, scale=False, orient=False)
            self.pl.add_mesh(holes, color="#11151c", name="holes", reset_camera=False)

        im = g.image
        fov = pv.Box(bounds=(im.shift_x - im.nx * im.dx / 2, im.shift_x + im.nx * im.dx / 2,
                             im.shift_y - im.ny * im.dy / 2, im.shift_y + im.ny * im.dy / 2,
                             im.shift_z - im.nz * im.dz / 2, im.shift_z + im.nz * im.dz / 2))
        self.pl.add_mesh(fov, style="wireframe", color="#22c55e",
                         line_width=2, name="fov", reset_camera=False)

    def update_ppdf(self):
        """Rebuild the selected-detector highlight, PPDF volume, and LORs."""
        for nm in ("vol", "lor", "sel"):
            if nm in self._dyn:
                self.pl.remove_actor(self._dyn.pop(nm))

        g = self.g
        sel = boxes_polydata(g.det_xyz[self.det:self.det + 1],
                             g.det_size[self.det:self.det + 1] * 1.15)
        self._dyn["sel"] = self.pl.add_mesh(
            sel, color="#ff2d55", show_edges=True, edge_color="white",
            line_width=2, name="sel", reset_camera=False)

        if self.sm is None:
            return None

        ppdf = self.sm.ppdf(self.rot, self.det)
        maxv = float(ppdf.max())
        if maxv > 0:
            grid = ppdf_to_grid(ppdf, g.image)
            opacity = [0.0, 0.015, 0.06, 0.16, 0.4, 0.85]
            self._dyn["vol"] = self.pl.add_volume(
                grid, scalars="ppdf", cmap="hot", opacity=opacity,
                clim=[self.thr * maxv, maxv], name="vol", reset_camera=False)
            cents = beam_centroids(ppdf, g.image, self.thr)
            if len(cents):
                d = g.det_xyz[self.det]
                seg = np.empty((len(cents) * 2, 3))
                seg[0::2] = d
                seg[1::2] = cents
                self._dyn["lor"] = self.pl.add_lines(
                    seg, color="#ffb000", width=3, name="lor")
        return maxv

    def set_layer(self, li, flag):
        self._layer_actors[li].SetVisibility(bool(flag))


# ----------------------------------------------------------------------------
# Trame wiring: state, callbacks, and UI
# ----------------------------------------------------------------------------

def build_ui(scene: Scene, server):
    state, ctrl = server.state, server.controller
    g = scene.g
    n_layers = len(g.layer_depths)

    state.detector = scene.det
    state.threshold = scene.thr
    state.rotation = scene.rot
    for li in range(n_layers):
        setattr(state, f"layer{li}", True)
    state.info = ""

    def refresh_info(maxv=None):
        d = g.det_xyz[scene.det]
        li = g.det_layer[scene.det]
        s = (f"Detector {scene.det} / {g.n_det - 1}  ·  layer {li} "
             f"(Y={g.layer_depths[li]:.0f} mm)  ·  "
             f"pos ({d[0]:.0f}, {d[1]:.0f}, {d[2]:.0f}) mm")
        if maxv:
            s += f"  ·  PPDF max {maxv:.2e}"
        state.info = s

    @state.change("detector")
    def _on_det(detector, **kw):
        scene.det = int(detector)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    @state.change("threshold")
    def _on_thr(threshold, **kw):
        scene.thr = float(threshold)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    @state.change("rotation")
    def _on_rot(rotation, **kw):
        scene.rot = int(rotation)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    for li in range(n_layers):
        @state.change(f"layer{li}")
        def _on_layer(li=li, **kw):
            scene.set_layer(li, kw[f"layer{li}"])
            ctrl.view_update()

    # Left-click picking -> set the detector (which re-fires _on_det).
    def on_pick(point, *args):
        if point is None:
            return
        pt = np.asarray(point).reshape(-1)[:3]
        _, idx = scene.kdtree.query(pt)
        state.detector = int(idx)
        state.flush()
    scene.pl.enable_point_picking(callback=on_pick, use_picker="point",
                                  left_clicking=True, show_message=False,
                                  show_point=False)

    refresh_info(scene.update_ppdf())

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("SC-SPECT Explorer")

        with layout.drawer:
            with vuetify3.VContainer(fluid=True):
                html.Div("{{ info }}", classes="text-caption mb-4 mt-2")

                vuetify3.VSlider(
                    v_model=("detector",), min=0, max=g.n_det - 1, step=1,
                    label="Detector", thumb_label="always", classes="mt-8")

                if scene.sm is not None:
                    vuetify3.VSlider(
                        v_model=("threshold",), min=0.005, max=0.5, step=0.005,
                        label="Threshold", thumb_label="always", classes="mt-6")

                if g.image.n_rot > 1:
                    vuetify3.VSlider(
                        v_model=("rotation",), min=0, max=g.image.n_rot - 1, step=1,
                        label="Rotation", thumb_label="always", classes="mt-6")

                html.Div("Layers", classes="text-subtitle-2 mt-6 mb-1")
                for li in range(n_layers):
                    vuetify3.VSwitch(
                        v_model=(f"layer{li}",),
                        label=f"Layer {li}  (Y={g.layer_depths[li]:.0f} mm)",
                        color=LAYER_COLORS[li % len(LAYER_COLORS)],
                        hide_details=True, density="compact")

        with layout.content:
            with vuetify3.VContainer(fluid=True, classes="fill-height pa-0"):
                # mode="server": always render server-side so the PPDF *volume*
                # renders correctly (client-side vtk.js volume rendering is flaky).
                view = plotter_ui(scene.pl, mode="server")
                ctrl.view_update = view.update


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SC-SPECT interactive Trame web app.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--sysmat", type=Path, default=None,
                    help="Path to .sysmat; if omitted, auto-globbed from data-dir.")
    ap.add_argument("--detector", type=int, default=1400)
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    geom = parse_geometry(args.data_dir)
    print(f"Loaded geometry: {geom.n_det} detectors, {len(geom.layer_depths)} layers "
          f"at Y={geom.layer_depths}, {len(geom.hole_xyz)} holes.")

    sysmat_path = args.sysmat
    if sysmat_path is None:
        hits = list(args.data_dir.glob("*.sysmat"))
        sysmat_path = hits[0] if hits else None
    sm = None
    if sysmat_path and Path(sysmat_path).exists():
        sm = SysMat(sysmat_path, geom.image, geom.n_det)
        print(f"Loaded system matrix: {sysmat_path} (n_det={sm.n_det})")
    else:
        print("No .sysmat found - running in geometry-only mode.")

    scene = Scene(geom, sm, args.detector, args.rotation, args.threshold)

    server = get_server()
    server.client_type = "vue3"
    build_ui(scene, server)

    print("\n" + "=" * 60)
    print(f"  Open this in the desktop's browser:  http://localhost:{args.port}")
    print("=" * 60 + "\n")
    server.start(port=args.port, open_browser=False)


if __name__ == "__main__":
    main()