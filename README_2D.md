# SC-SPECT 2D Explorer

Browser-streamed viewer (Trame / PyVista) for the **2D** multi-pinhole SC-SPECT
scanner and its ray-traced system response matrix (PPDFs). It is the 2D analogue
of the 3D `sc_spect_app.py` and deliberately keeps the same Trame/PyVista
pattern, so it runs headless on CCR compute nodes (server-side off-screen
rendering streamed to Firefox on `localhost`) — but because the scene is planar
it avoids the on-screen GLX / llvmpipe problems entirely.

## What it reads

- **`scanner_layouts_*.tensor`** — the geometry, written by
  `generate_mph_scanner_circularfov.py`. It's a `torch.save` of a plain dict, so
  the loader just calls `torch.load(...)` and converts tensors to numpy. Per
  motion pose (`"position NNN"`) it holds the fixed `detector units` (Ncry,4,2),
  the rotated `plate circles` (aperture centers + radius), and the rotated
  `plate segments` (Nseg,4,2) tungsten walls, plus the `paper_config` scalars
  (`detectors_per_ring`, `fov_diameter_mm`, …) and the `scanner MD5`.
- **`position_{layout:03d}_ppdfs_t8_{pose:02d}.hdf5`** — the PPDFs, written by
  `arg_ppdf_t8.py`. One dataset `"ppdfs"` of shape `(Ncry, nx*ny)` float32 (e.g.
  `(3360, 40000)` for the 200×200 FOV), plus attrs `layout_idx`, `pose_idx`,
  `dx_mm`, `dy_mm`, `layouts_md5`. **Row `i` is crystal `i`'s PPDF**, in the same
  order as `detector units` — that 1:1 link drives click-to-pick.

The full response set is indexed by two motion axes: `layout_idx` (collimator
rotation) × `pose_idx` (8-point elliptical bed shift), i.e. one HDF5 per pair.
Each file is ~0.5 GB, so the tool opens the one it needs and slices single rows
on demand (no whole-matrix load).

## The two-panel layout (why)

The signal lives inside a ~5 mm-radius FOV while the detector rings sit at
~330 mm — a ~66× span. In one frame the FOV collapses to a dot, so:

- **Left — overview:** all crystals coloured by ring, tungsten walls, apertures,
  FOV marker, and the selected crystal's beam (line of response). **Click any
  crystal here** to select it.
- **Right — FOV zoom:** the selected crystal's PPDF as a heat-map over the 10 mm
  box (log or linear), or the summed-sensitivity map across all crystals.

## Install

```bash
pip install pyvista "trame>=3.6.5" "trame-vtk>=2.8.10" trame-vuetify h5py scipy numpy torch
```

`torch` is only needed to read the `.tensor`. On a headless node install `xvfb`;
the app already renders server-side off-screen.

## Run

```bash
# interactive server — open Firefox INSIDE the CCR desktop at http://localhost:8080
python sc_spect_2d_app.py --data-dir ./data --serve --port 8080

# headless proof PNG (no browser)
python sc_spect_2d_app.py --data-dir ./data --layout 0 --pose 0 --crystal 1700 \
                          --screenshot proof.png

# static single-file HTML of the current scene
python sc_spect_2d_app.py --data-dir ./data --export-html scene.html

# synthetic self-test (no .tensor, no torch needed) — sanity-checks the pipeline
python sc_spect_2d_app.py --self-test --screenshot selftest.png
```

`--data-dir` should contain both the `.tensor` and the `position_*_ppdfs_t8_*.hdf5`
files. The `.tensor` is auto-discovered (or pass `--tensor`). On load the tool
warns if the HDF5 `layouts_md5` or crystal count disagrees with the `.tensor`.

## Controls (serve mode)

- **Crystal** slider — precise selection (click on the left panel also works).
- **Layout / Pose** selects — switch rotation and bed pose; the pose reloads the
  matching HDF5 and shifts the FOV center by its `dx_mm`/`dy_mm`.
- **Log scale** switch and **Threshold %** slider — the PPDF transfer function
  (log is the default, matching the 3D tool's professor-requested view).
- **Summed sensitivity** switch — right panel shows the sum over all crystals.
- **Apertures / Tungsten / FOV / Beam** switches — toggle overview layers.
- **Transpose / Flip X / Flip Y** — fix PPDF orientation if needed (see below).

## One orientation caveat

`arg_ppdf_t8.py` flattens with `index = a*nx + b` (b fastest), which the viewer
maps straight onto the VTK image grid. Whether the first pixel axis `a` is world
X or world Y is decided inside `scanner_modeling/geometry_2d.py` (`fov_tensor_dict`
/ `sfov_properties`), which isn't in the files provided. The default assumes the
natural raveled order; if a real PPDF looks mirrored or rotated versus the beam
direction, toggle **Transpose / Flip X / Flip Y** — or send that module and I'll
pin the default exactly and drop the toggles.

## Notes

- Beam / LOR is the intensity-weighted centroid of the PPDF, drawn from the
  crystal to that point — same approach as the 3D explorer's beam centroid.
- `--self-test` builds a small 4-ring synthetic scanner and a plausible HDF5 so
  the whole render path (geometry, ring colouring, HDF5 slicing, heat-map, beam,
  picking) is exercised without any real data. `selftest_proof.png` was produced
  this way.
