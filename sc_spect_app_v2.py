#!/usr/bin/env python3
"""
SC-SPECT Interactive 3D Explorer — Trame web app  (v2)
======================================================

Browser-based interactive companion to sc_spect_explorer.py, with live
controls and a depth-aware PPDF rendering.

What changed vs v1 (addressing professor feedback):
  * PPDF transfer function is now LOG-SCALED (toggle). On a linear scale the
    faint deep-FOV voxels were clipped to transparent, so the beam looked
    truncated, slim, and single-colored. Log scaling reveals the full depth,
    the true penumbra width, and an intensity color gradient.
  * A scalar bar shows the intensity scale.
  * Toggle switches added for: collimator plate, collimator holes, FOV box,
    LOR lines, and the PPDF volume itself (in addition to the per-layer
    switches that were already there).

Controls (left drawer):
  Detector / Threshold / Rotation sliders, Log-scale switch,
  per-layer switches, and visibility switches for plate / holes / FOV /
  LORs / volume. Left-click a detector in the scene to select it.

Rendering is server-side (off-screen, software GL via llvmpipe), streamed to
the browser — works on a headless / software-only node.

Run (on the SAME node as your OnDemand desktop):
    python sc_spect_app.py --data-dir DIR --detector 1400
Then open  http://localhost:8080  in a browser INSIDE that desktop session.

Requires: trame, trame-vtk, trame-vuetify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Reuse the existing module (must sit in the same folder). Importing it also
# sets pv.OFF_SCREEN = True and starts Xvfb, which is what we want here.
from sc_spect_explorer import (
    parse_geometry, SysMat, ppdf_to_grid, beam_centroids,
    boxes_polydata, LAYER_COLORS,
)

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3, html
from pyvista.trame.ui import plotter_ui

pv.OFF_SCREEN = True

SB_TITLE = "PPDF"
# Gentle opacity ramp; works well once the scalars are log-compressed.
OPACITY_RAMP = [0.0, 0.04, 0.10, 0.22, 0.45, 0.85]


class Scene:
    def __init__(self, geom, sysmat, detector, rotation, threshold,
                 log_scale=True, cmap="hot"):
        self.g = geom
        self.sm = sysmat
        self.det = int(detector)
        self.rot = int(rotation)
        self.thr = float(threshold)
        self.log_scale = bool(log_scale)
        self.cmap = cmap
        self.kdtree = cKDTree(geom.det_xyz)

        self._static = {}        # plate / holes / fov actors
        self._layer_actors = {}  # per-layer detector actors
        self._dyn = {}           # sel / vol / lor actors (rebuilt on change)
        self._sb_title = None

        # which things are visible (driven by the UI switches)
        self.show = dict(vol=True, lor=True, sel=True,
                         plate=True, holes=True, fov=True)

        self.pl = pv.Plotter(off_screen=True)
        self.pl.set_background("#0e1117")
        self._build_static()
        self.update_ppdf()
        self.pl.camera_position = "yz"
        self.pl.camera.azimuth = 35
        self.pl.camera.elevation = 18
        self.pl.reset_camera()

    # ---- static geometry -------------------------------------------------
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
        self._static["plate"] = self.pl.add_mesh(
            plate, color="#9aa0a6", opacity=0.18, name="plate", reset_camera=False)

        if len(g.hole_xyz):
            cyl = pv.Cylinder(direction=(0, 1, 0), radius=g.hole_r,
                              height=g.hole_h, resolution=12)
            holes = pv.PolyData(g.hole_xyz).glyph(geom=cyl, scale=False, orient=False)
            self._static["holes"] = self.pl.add_mesh(
                holes, color="#11151c", name="holes", reset_camera=False)

        im = g.image
        fov = pv.Box(bounds=(im.shift_x - im.nx * im.dx / 2, im.shift_x + im.nx * im.dx / 2,
                             im.shift_y - im.ny * im.dy / 2, im.shift_y + im.ny * im.dy / 2,
                             im.shift_z - im.nz * im.dz / 2, im.shift_z + im.nz * im.dz / 2))
        self._static["fov"] = self.pl.add_mesh(
            fov, style="wireframe", color="#22c55e", line_width=2,
            name="fov", reset_camera=False)

    # ---- dynamic PPDF beam ----------------------------------------------
    def update_ppdf(self):
        # clear previous dynamic actors + scalar bar
        for nm in ("vol", "lor", "sel"):
            if nm in self._dyn:
                self.pl.remove_actor(self._dyn.pop(nm))
        if self._sb_title is not None:
            try:
                self.pl.remove_scalar_bar(self._sb_title)
            except Exception:
                pass
            self._sb_title = None

        g = self.g

        # selected-detector highlight
        sel = boxes_polydata(g.det_xyz[self.det:self.det + 1],
                             g.det_size[self.det:self.det + 1] * 1.15)
        a = self.pl.add_mesh(sel, color="#ff2d55", show_edges=True,
                             edge_color="white", line_width=2, name="sel",
                             reset_camera=False)
        a.SetVisibility(self.show["sel"])
        self._dyn["sel"] = a

        if self.sm is None:
            return None

        ppdf = self.sm.ppdf(self.rot, self.det)
        maxv = float(ppdf.max())
        if maxv <= 0:
            return maxv

        floor = self.thr * maxv
        clipped = np.clip(ppdf, floor, maxv)         # elementwise; order preserved
        if self.log_scale:
            scal = np.log10(clipped)
            clim = [float(np.log10(floor)), float(np.log10(maxv))]
        else:
            scal = clipped
            clim = [floor, maxv]

        # ppdf_to_grid only reshapes/transposes/flattens, so applying a
        # monotonic elementwise transform to the 1-D array first is safe.
        grid = ppdf_to_grid(scal, g.image)

        if self.show["vol"]:
            bar_title = ("log10 " + SB_TITLE) if self.log_scale else SB_TITLE
            self._dyn["vol"] = self.pl.add_volume(
                grid, scalars="ppdf", cmap=self.cmap, opacity=OPACITY_RAMP,
                clim=clim, name="vol", reset_camera=False,
                scalar_bar_args=dict(title=bar_title, color="white"))
            self._sb_title = bar_title

        if self.show["lor"]:
            cents = beam_centroids(ppdf, g.image, self.thr)
            if len(cents):
                d = g.det_xyz[self.det]
                seg = np.empty((len(cents) * 2, 3))
                seg[0::2] = d
                seg[1::2] = cents
                self._dyn["lor"] = self.pl.add_lines(
                    seg, color="#ffb000", width=3, name="lor")
        return maxv

    # ---- visibility toggles ---------------------------------------------
    def set_layer(self, li, flag):
        self._layer_actors[li].SetVisibility(bool(flag))

    def set_show(self, key, flag):
        self.show[key] = bool(flag)
        act = self._dyn.get(key) or self._static.get(key)
        if act is not None:
            act.SetVisibility(bool(flag))

    def set_log(self, flag):
        self.log_scale = bool(flag)


def build_ui(scene: Scene, server):
    state, ctrl = server.state, server.controller
    g = scene.g
    n_layers = len(g.layer_depths)

    state.detector = scene.det
    state.threshold = scene.thr
    state.rotation = scene.rot
    state.logscale = scene.log_scale
    for li in range(n_layers):
        setattr(state, f"layer{li}", True)
    for key in ("vol", "lor", "plate", "holes", "fov"):
        setattr(state, f"show_{key}", True)
    state.info = ""

    def refresh_info(maxv=None):
        d = g.det_xyz[scene.det]
        li = g.det_layer[scene.det]
        s = (f"Detector {scene.det} / {g.n_det - 1}  -  layer {li} "
             f"(Y={g.layer_depths[li]:.0f} mm)  -  "
             f"pos ({d[0]:.0f}, {d[1]:.0f}, {d[2]:.0f}) mm")
        if maxv:
            s += f"  -  PPDF max {maxv:.2e}"
        state.info = s

    @state.change("detector")
    def _d(detector, **kw):
        scene.det = int(detector)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    @state.change("threshold")
    def _t(threshold, **kw):
        scene.thr = float(threshold)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    @state.change("rotation")
    def _r(rotation, **kw):
        scene.rot = int(rotation)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    @state.change("logscale")
    def _l(logscale, **kw):
        scene.set_log(logscale)
        refresh_info(scene.update_ppdf())
        ctrl.view_update()

    for li in range(n_layers):
        @state.change(f"layer{li}")
        def _ly(li=li, **kw):
            scene.set_layer(li, kw[f"layer{li}"])
            ctrl.view_update()

    for key in ("vol", "lor", "plate", "holes", "fov"):
        @state.change(f"show_{key}")
        def _sh(key=key, **kw):
            scene.set_show(key, kw[f"show_{key}"])
            if key in ("vol", "lor"):
                scene.update_ppdf()
            ctrl.view_update()

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
                html.Div("{{ info }}", classes="text-caption mb-3 mt-2")

                vuetify3.VSlider(
                    v_model=("detector",), min=0, max=g.n_det - 1, step=1,
                    label="Detector", thumb_label="always", classes="mt-8")

                if scene.sm is not None:
                    vuetify3.VSlider(
                        v_model=("threshold",), min=0.001, max=0.5, step=0.001,
                        label="Threshold (x peak)", thumb_label="always",
                        classes="mt-6")
                    vuetify3.VSwitch(
                        v_model=("logscale",),
                        label="Log intensity scale", color="#f59e0b",
                        hide_details=True, density="compact", classes="mt-2")

                if g.image.n_rot > 1:
                    vuetify3.VSlider(
                        v_model=("rotation",), min=0, max=g.image.n_rot - 1, step=1,
                        label="Rotation", thumb_label="always", classes="mt-6")

                html.Div("Detector layers", classes="text-subtitle-2 mt-6 mb-1")
                for li in range(n_layers):
                    vuetify3.VSwitch(
                        v_model=(f"layer{li}",),
                        label=f"Layer {li}  (Y={g.layer_depths[li]:.0f} mm)",
                        color=LAYER_COLORS[li % len(LAYER_COLORS)],
                        hide_details=True, density="compact")

                html.Div("Scene elements", classes="text-subtitle-2 mt-5 mb-1")
                if scene.sm is not None:
                    vuetify3.VSwitch(v_model=("show_vol",), label="PPDF volume",
                                     color="#ef4444", hide_details=True, density="compact")
                    vuetify3.VSwitch(v_model=("show_lor",), label="LOR lines",
                                     color="#ffb000", hide_details=True, density="compact")
                vuetify3.VSwitch(v_model=("show_plate",), label="Collimator plate",
                                 color="#9aa0a6", hide_details=True, density="compact")
                vuetify3.VSwitch(v_model=("show_holes",), label="Collimator holes",
                                 color="#6b7280", hide_details=True, density="compact")
                vuetify3.VSwitch(v_model=("show_fov",), label="FOV box",
                                 color="#22c55e", hide_details=True, density="compact")

        with layout.content:
            with vuetify3.VContainer(fluid=True, classes="fill-height pa-0"):
                view = plotter_ui(scene.pl, mode="server")
                ctrl.view_update = view.update


def main():
    ap = argparse.ArgumentParser(description="SC-SPECT interactive Trame web app.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--sysmat", type=Path, default=None)
    ap.add_argument("--detector", type=int, default=1400)
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--linear", action="store_true",
                    help="Start with a linear (not log) intensity scale.")
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

    scene = Scene(geom, sm, args.detector, args.rotation, args.threshold,
                  log_scale=not args.linear)

    server = get_server()
    server.client_type = "vue3"
    build_ui(scene, server)

    print("\n" + "=" * 60)
    print(f"  Open in the desktop's browser:  http://localhost:{args.port}")
    print("=" * 60 + "\n")
    server.start(port=args.port, open_browser=False)


if __name__ == "__main__":
    main()