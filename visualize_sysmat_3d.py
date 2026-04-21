#!/usr/bin/env python3
"""
Publication-Quality Interactive 3D viewer for GPUPTS SPECT/PET geometry.

Upgrades:
1. Detectors are rendered as solid 3D blocks (meshes) instead of points.
2. PPDF is rendered as a continuous 3D Volume/Iso-surface instead of a point cloud.
3. Automatically computes beam centroids and draws explicit Lines of Response (rays).
4. Collimator is rendered as a semi-transparent solid plate with distinct holes.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import label


@dataclass
class ImageParams:
    nx: int; ny: int; nz: int
    dx: float; dy: float; dz: float
    n_rot: int
    shift_x: float; shift_y: float; shift_z: float
    fov_to_collimator: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D geometry + volumetric beam visualizer.")
    parser.add_argument("--data-dir", type=Path, default=Path("PEGen_RayTracing_CircularHole"))
    parser.add_argument("--sysmat", type=Path, default=None)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--detector", type=int, default=1400)
    parser.add_argument("--beam-threshold", type=float, default=0.05, help="Relative threshold for PPDF volume rendering.")
    parser.add_argument("--detector-step", type=int, default=16, help="Sub-sampling step for background detectors.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Open interactive window automatically.")
    return parser.parse_args()


def read_params(data_dir: Path):
    col = np.fromfile(data_dir / "Params_Collimator.dat", dtype=np.float32)
    det = np.fromfile(data_dir / "Params_Detector.dat", dtype=np.float32)
    img = np.fromfile(data_dir / "Params_Image.dat", dtype=np.float32)
    return col, det, img


def parse_image_params(img_raw: np.ndarray) -> ImageParams:
    return ImageParams(
        nx=int(img_raw[0]), ny=int(img_raw[1]), nz=int(img_raw[2]),
        dx=float(img_raw[3]), dy=float(img_raw[4]), dz=float(img_raw[5]),
        n_rot=int(img_raw[6]),
        shift_x=float(img_raw[8]), shift_y=float(img_raw[9]), shift_z=float(img_raw[10]),
        fov_to_collimator=float(img_raw[11]),
    )


def generate_boxes_mesh(centers_and_sizes: np.ndarray, color: str, opacity: float, name: str) -> go.Mesh3d:
    """Generates a single Plotly Mesh3d containing multiple 3D boxes for extreme performance."""
    if len(centers_and_sizes) == 0:
        return None
    
    verts = []
    i_faces, j_faces, k_faces = [], [],[]
    v_idx = 0
    
    for row in centers_and_sizes:
        cx, cy, cz = row[0:3]
        dx, dy, dz = row[3:6]
        
        # 8 corners of the box
        x0, x1 = cx - dx/2, cx + dx/2
        y0, y1 = cy - dy/2, cy + dy/2
        z0, z1 = cz - dz/2, cz + dz/2
        
        box_verts =[
            [x0, y0, z0], [x1, y0, z0],[x1, y1, z0], [x0, y1, z0],[x0, y0, z1], [x1, y0, z1], [x1, y1, z1],[x0, y1, z1]
        ]
        verts.extend(box_verts)
        
        # 12 triangles for the 6 faces
        faces = [
            [0,1,2],[0,2,3], # Bottom
            [4,5,6], [4,6,7], # Top
            [0,1,5], [0,5,4], # Front
            [2,3,7], [2,7,6], # Back
            [1,2,6],[1,6,5], # Right
            [0,3,7], [0,7,4]  # Left
        ]
        for f in faces:
            i_faces.append(f[0] + v_idx)
            j_faces.append(f[1] + v_idx)
            k_faces.append(f[2] + v_idx)
        v_idx += 8
        
    verts = np.array(verts)
    return go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=i_faces, j=j_faces, k=k_faces,
        color=color, opacity=opacity, name=name,
        flatshading=True, showlegend=True
    )


def extract_volumetric_beam(ppdf_1d: np.ndarray, img: ImageParams, threshold: float):
    """Extracts a volumetric sub-grid of the PPDF for continuous 3D rendering."""
    # GPUPTS usually maps memory Z, Y, X. We reshape and transpose to X, Y, Z.
    ppdf_3d = ppdf_1d.reshape((img.nz, img.ny, img.nx)).transpose(2, 1, 0)
    max_val = float(np.max(ppdf_3d))
    
    if max_val <= 0: return None, None, None, None, None, []

    # Get Voxel Center Coordinates
    cx = (np.arange(img.nx) - (img.nx - 1) / 2.0) * img.dx + img.shift_x
    #cy = (np.arange(img.ny) - (img.ny - 1) / 2.0) * img.dy - img.fov_to_collimator + img.shift_y
    cy = (np.arange(img.ny) - (img.ny - 1) / 2.0) * img.dy + img.shift_y
    cz = (np.arange(img.nz) - (img.nz - 1) / 2.0) * img.dz + img.shift_z
    X_grid, Y_grid, Z_grid = np.meshgrid(cx, cy, cz, indexing="ij")

    mask = ppdf_3d >= (threshold * max_val)
    if not np.any(mask): return None, None, None, None, None,[]

    # Extract bounding box to prevent Plotly from freezing on 0.0 values
    ix, iy, iz = np.where(mask)
    padding = 2
    min_x, max_x = max(0, ix.min() - padding), min(img.nx, ix.max() + padding + 1)
    min_y, max_y = max(0, iy.min() - padding), min(img.ny, iy.max() + padding + 1)
    min_z, max_z = max(0, iz.min() - padding), min(img.nz, iz.max() + padding + 1)

    sub_ppdf = ppdf_3d[min_x:max_x, min_y:max_y, min_z:max_z]
    sub_X = X_grid[min_x:max_x, min_y:max_y, min_z:max_z]
    sub_Y = Y_grid[min_x:max_x, min_y:max_y, min_z:max_z]
    sub_Z = Z_grid[min_x:max_x, min_y:max_y, min_z:max_z]

    # Find individual beam centroids for Ray Lines
    labeled_blobs, num_beams = label(mask)
    centroids =[]
    for b_id in range(1, num_beams + 1):
        b_mask = (labeled_blobs == b_id)
        if np.sum(b_mask) > 1:
            centroids.append([
                np.average(X_grid[b_mask], weights=ppdf_3d[b_mask]),
                np.average(Y_grid[b_mask], weights=ppdf_3d[b_mask]),
                np.average(Z_grid[b_mask], weights=ppdf_3d[b_mask])
            ])

    return sub_X.flatten(), sub_Y.flatten(), sub_Z.flatten(), sub_ppdf.flatten(), max_val, centroids

def add_wireframe_box(fig, cx, cy, cz, sx, sy, sz, color, name):
    x1, x2 = cx - sx/2, cx + sx/2
    y1, y2 = cy - sy/2, cy + sy/2
    z1, z2 = cz - sz/2, cz + sz/2
    edges = [
        [x1,x2,x2,x1,x1], [y1,y1,y1,y1,y1], [z1,z1,z2,z2,z1],  # bottom
        [x1,x2,x2,x1,x1], [y2,y2,y2,y2,y2], [z1,z1,z2,z2,z1],  # top
    ]
    for ex, ey, ez in zip(*[iter(edges)]*3):
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=color, width=3), showlegend=False))
    # vertical edges
    for x, z in [(x1,z1),(x2,z1),(x2,z2),(x1,z2)]:
        fig.add_trace(go.Scatter3d(x=[x,x], y=[y1,y2], z=[z,z], mode="lines", line=dict(color=color, width=3), showlegend=False, name=name))

def build_figure(data_dir, args) -> go.Figure:
    col_raw, det_raw, img_raw = read_params(data_dir)
    img = parse_image_params(img_raw)
    
    n_det = int(det_raw[0])
    #det_data = det_raw[1: 1 + n_det*12].reshape(-1, 12)
    det_data = det_raw[1: 1 + n_det*12].reshape(-1, 12).copy()
    det_data[:, 1] += img.fov_to_collimator

    # Locate SysMat
    sysmat_path = list(data_dir.glob("*.sysmat"))[0] if args.sysmat is None else args.sysmat
    sysmat = np.memmap(sysmat_path, dtype=np.float32, mode="r", shape=(img.n_rot, n_det, img.nx * img.ny * img.nz))
    ppdf_1d = np.array(sysmat[args.rotation, args.detector, :])
    print(f"Detector {args.detector}: max={np.max(ppdf_1d):.6e}, min={np.min(ppdf_1d):.6e}, nonzero={np.count_nonzero(ppdf_1d)}")

    fig = go.Figure()

    # 1. FOV Wireframe
    fov_sx, fov_sy, fov_sz = img.nx * img.dx, img.ny * img.dy, img.nz * img.dz
    fov_cy = img.shift_y  # FOV center Y (no fov_dist subtraction)
    add_wireframe_box(fig, img.shift_x, fov_cy, img.shift_z, fov_sx, fov_sy, fov_sz, "#2ca02c", "FOV Boundary")

    # 2. Collimator Plate & Holes
    n_layers = int(col_raw[0])
    base = 10
    n_holes, cw, ct, ch, offset_y = col_raw[base:base+5]
    
    # fig.add_trace(generate_boxes_mesh(
    #     np.array([[0.0, offset_y + ct/2.0, 0.0, cw, ct, ch]]), 
    #     color="gray", opacity=0.15, name="Tungsten Plate"
    # ))
    fig.add_trace(generate_boxes_mesh(
    np.array([[0.0, img.fov_to_collimator + offset_y, 0.0, cw, ct, ch]]),
    color="gray", opacity=0.15, name="Tungsten Plate"
    ))

    if n_holes > 0:
        holes = col_raw[100: 100 + int(n_holes)*9].reshape(-1, 9)
        holes_y = (holes[:, 1] + holes[:, 2]) / 2.0 + img.fov_to_collimator
        fig.add_trace(go.Scatter3d(
            x=holes[:, 0], y=holes_y, z=holes[:, 3], # y2 center (end of hole)
            mode="markers", marker=dict(size=3, color="black", opacity=0.6),
            name=f"Collimator Holes ({int(n_holes)})"
        ))

    # 3. Detectors (Solid Mosaic)
    step = max(1, args.detector_step)
    bg_dets =[d for i, d in enumerate(det_data) if i % step == 0 and i != args.detector]
    fig.add_trace(generate_boxes_mesh(np.array(bg_dets), color="#1f77b4", opacity=0.3, name="Background Detectors (Sampled)"))
    
    sel = det_data[args.detector]
    fig.add_trace(generate_boxes_mesh(np.array([sel]), color="red", opacity=1.0, name=f"Target Detector [{args.detector}]"))

    # 4. Volumetric Beam (PPDF)
    vol_x, vol_y, vol_z, vol_val, max_val, centroids = extract_volumetric_beam(ppdf_1d, img, args.beam_threshold)
    
    if vol_x is not None:
        fig.add_trace(go.Volume(
            x=vol_x, y=vol_y, z=vol_z, value=vol_val,
            isomin=args.beam_threshold * max_val, isomax=max_val,
            opacity=0.3, surface_count=10, colorscale="Hot",
            name="PPDF Iso-Surfaces", showlegend=True,
            colorbar=dict(title="Probability", len=0.5, x=0.9)
        ))

        # 5. Explicit Lines of Response (Detector -> FOV Centroids)
        lx, ly, lz = [],[], []
        for cx, cy, cz in centroids:
            lx.extend([sel[0], cx, None])
            ly.extend([sel[1], cy, None])
            lz.extend([sel[2], cz, None])
            
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line=dict(color="orange", width=4, dash='dash'),
            name="Lines of Response (Rays)"
        ))

    # Aesthetic Polish
    fig.update_layout(
        title=dict(text=f"<b>SC-SPECT 3D Geometry</b><br>Rotation {args.rotation} | Target Detector: {args.detector}", font=dict(size=18)),
        scene=dict(
            xaxis_title="X (Transverse) [mm]",
            yaxis_title="Y (Depth) [mm]",
            zaxis_title="Z (Axial)[mm]",
            aspectmode="data",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0)) # Professional viewing angle
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir.resolve()
    fig = build_figure(data_dir, args)
    
    out_path = args.output if args.output else data_dir / f"viewer_rot{args.rotation:03d}_det{args.detector:04d}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Generated high-quality 3D report: {out_path}")
    
    if args.show: fig.show()


#!/usr/bin/env python3
"""
Interactive 3D viewer for GPUPTS SPECT/PET geometry + detector-wise beams.

Reads:
- Params_Collimator.dat
- Params_Detector.dat
- Params_Image.dat
- Params_Physics.dat
- *.sysmat

Outputs:
- Interactive HTML (Plotly) showing geometry and selected detector response.
"""
'''
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


@dataclass
class ImageParams:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    n_rot: int
    shift_x: float
    shift_y: float
    shift_z: float
    fov_to_collimator: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D geometry + detector-beam visualizer for .sysmat outputs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("PEGen_RayTracing_CircularHole"),
        help="Directory containing Params_*.dat and .sysmat files.",
    )
    parser.add_argument(
        "--sysmat",
        type=Path,
        default=None,
        help="Optional explicit .sysmat path. If omitted, first *.sysmat in --data-dir is used.",
    )
    parser.add_argument("--rotation", type=int, default=0, help="Rotation index.")
    parser.add_argument("--detector", type=int, default=0, help="Detector index.")
    parser.add_argument(
        "--beam-threshold",
        type=float,
        default=0.05,
        help="Relative threshold in [0,1] versus max detector response.",
    )
    parser.add_argument(
        "--max-beam-points",
        type=int,
        default=20000,
        help="Maximum number of beam voxels to render.",
    )
    parser.add_argument(
        "--detector-step",
        type=int,
        default=16,
        help="Sub-sampling step for plotting non-selected detector centers.",
    )
    parser.add_argument(
        "--hole-step",
        type=int,
        default=2,
        help="Sub-sampling step for plotting collimator hole centers.",
    )
    parser.add_argument(
        "--reshape-order",
        choices=["xyz", "zyx"],
        default="xyz",
        help="How to reshape detector PPDF from .sysmat.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <data-dir>/viewer_rotXXX_detXXXX.html",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive window in addition to writing HTML.",
    )
    return parser.parse_args()


def read_params(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col = np.fromfile(data_dir / "Params_Collimator.dat", dtype=np.float32)
    det = np.fromfile(data_dir / "Params_Detector.dat", dtype=np.float32)
    img = np.fromfile(data_dir / "Params_Image.dat", dtype=np.float32)
    phy = np.fromfile(data_dir / "Params_Physics.dat", dtype=np.float32)
    return col, det, img, phy


def parse_image_params(img_raw: np.ndarray) -> ImageParams:
    return ImageParams(
        nx=int(img_raw[0]),
        ny=int(img_raw[1]),
        nz=int(img_raw[2]),
        dx=float(img_raw[3]),
        dy=float(img_raw[4]),
        dz=float(img_raw[5]),
        n_rot=int(img_raw[6]),
        shift_x=float(img_raw[8]),
        shift_y=float(img_raw[9]),
        shift_z=float(img_raw[10]),
        fov_to_collimator=float(img_raw[11]),
    )


def parse_detector_data(det_raw: np.ndarray) -> np.ndarray:
    n_det = int(det_raw[0])
    return det_raw[1 : 1 + n_det * 12].reshape(-1, 12)


def parse_collimator_data(col_raw: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], np.ndarray]:
    n_layers = int(col_raw[0])
    layers: List[Tuple[float, float, float, float]] = []
    total_holes = 0

    for i in range(n_layers):
        base = (i + 1) * 10
        n_holes = int(col_raw[base + 0])
        width_x = float(col_raw[base + 1])
        thickness_y = float(col_raw[base + 2])
        height_z = float(col_raw[base + 3])
        offset_y = float(col_raw[base + 4])
        total_holes += max(n_holes, 0)
        layers.append((width_x, thickness_y, height_z, offset_y))

    if total_holes <= 0:
        return layers, np.zeros((0, 9), dtype=np.float32)

    hole_flat = col_raw[100 : 100 + total_holes * 9]
    hole_data = hole_flat.reshape(-1, 9)
    return layers, hole_data


def resolve_sysmat_path(data_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    matches = sorted(glob.glob(str(data_dir / "*.sysmat")))
    if not matches:
        raise FileNotFoundError(f"No .sysmat file found in {data_dir}")
    return Path(matches[0])


def voxel_centers(img: ImageParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (np.arange(img.nx) - (img.nx - 1) / 2.0) * img.dx + img.shift_x
    y = (np.arange(img.ny) - (img.ny - 1) / 2.0) * img.dy - img.fov_to_collimator + img.shift_y
    z = (np.arange(img.nz) - (img.nz - 1) / 2.0) * img.dz + img.shift_z
    return x, y, z


def add_box_edges(
    fig: go.Figure,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    color: str,
    name: str,
    width: int = 2,
    showlegend: bool = False,
) -> None:
    x1, x2 = cx - sx / 2.0, cx + sx / 2.0
    y1, y2 = cy - sy / 2.0, cy + sy / 2.0
    z1, z2 = cz - sz / 2.0, cz + sz / 2.0

    corners = np.array(
        [
            [x1, y1, z1],
            [x2, y1, z1],
            [x2, y2, z1],
            [x1, y2, z1],
            [x1, y1, z2],
            [x2, y1, z2],
            [x2, y2, z2],
            [x1, y2, z2],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for i0, i1 in edges:
        xs.extend([corners[i0, 0], corners[i1, 0], None])
        ys.extend([corners[i0, 1], corners[i1, 1], None])
        zs.extend([corners[i0, 2], corners[i1, 2], None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
            showlegend=showlegend,
            hoverinfo="skip",
        )
    )


def build_figure(
    detector_data: np.ndarray,
    col_layers: List[Tuple[float, float, float, float]],
    hole_data: np.ndarray,
    img: ImageParams,
    detector_idx: int,
    beam_points_xyz: np.ndarray,
    beam_values: np.ndarray,
    rotation_idx: int,
    detector_step: int,
    hole_step: int,
) -> go.Figure:
    fig = go.Figure()

    fov_sx = img.nx * img.dx
    fov_sy = img.ny * img.dy
    fov_sz = img.nz * img.dz
    add_box_edges(
        fig,
        cx=img.shift_x,
        cy=-img.fov_to_collimator + img.shift_y,
        cz=img.shift_z,
        sx=fov_sx,
        sy=fov_sy,
        sz=fov_sz,
        color="#3ba272",
        name="FOV",
        width=3,
        showlegend=True,
    )

    for layer_id, (sx, sy, sz, offset_y) in enumerate(col_layers):
        add_box_edges(
            fig,
            cx=0.0,
            cy=offset_y + sy / 2.0,
            cz=0.0,
            sx=sx,
            sy=sy,
            sz=sz,
            color="#7f7f7f",
            name=f"Collimator L{layer_id}",
            width=2,
            showlegend=(layer_id == 0),
        )

    if hole_data.size > 0:
        sampled_holes = hole_data[:: max(1, hole_step)]
        fig.add_trace(
            go.Scatter3d(
                x=sampled_holes[:, 0],
                y=(sampled_holes[:, 1] + sampled_holes[:, 2]) / 2.0,
                z=sampled_holes[:, 3],
                mode="markers",
                marker=dict(size=2, color="#333333", opacity=0.5),
                name="Hole centers",
                showlegend=True,
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            )
        )

    step = max(1, detector_step)
    others = np.arange(detector_data.shape[0]) % step == 0
    others[detector_idx] = False
    fig.add_trace(
        go.Scatter3d(
            x=detector_data[others, 0],
            y=detector_data[others, 1],
            z=detector_data[others, 2],
            mode="markers",
            marker=dict(size=2, color="#4f81bd", opacity=0.35),
            name="Detector centers (sampled)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    sel = detector_data[detector_idx]
    fig.add_trace(
        go.Scatter3d(
            x=[sel[0]],
            y=[sel[1]],
            z=[sel[2]],
            mode="markers",
            marker=dict(size=8, color="#d62728"),
            name=f"Selected detector {detector_idx}",
            showlegend=True,
            hovertemplate=(
                f"Detector {detector_idx}<br>x={sel[0]:.2f}<br>y={sel[1]:.2f}<br>z={sel[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )
    add_box_edges(
        fig,
        cx=float(sel[0]),
        cy=float(sel[1]),
        cz=float(sel[2]),
        sx=float(sel[3]),
        sy=float(sel[4]),
        sz=float(sel[5]),
        color="#d62728",
        name="Selected detector volume",
        width=4,
        showlegend=False,
    )

    if beam_points_xyz.size > 0:
        norm = beam_values / np.max(beam_values)
        fig.add_trace(
            go.Scatter3d(
                x=beam_points_xyz[:, 0],
                y=beam_points_xyz[:, 1],
                z=beam_points_xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=norm,
                    colorscale="Turbo",
                    opacity=0.75,
                    colorbar=dict(title="Norm. PPDF"),
                ),
                name="Beam voxels",
                showlegend=True,
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>norm=%{marker.color:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        centroid = np.average(beam_points_xyz, axis=0, weights=beam_values)
        fig.add_trace(
            go.Scatter3d(
                x=[sel[0], centroid[0]],
                y=[sel[1], centroid[1]],
                z=[sel[2], centroid[2]],
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=5),
                marker=dict(size=5, color="#ff7f0e"),
                name="Detector to beam centroid",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"3D System View | Rotation {rotation_idx} | Detector {detector_idx}",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm, depth)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            bgcolor="white",
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def extract_beam_points(
    ppdf_1d: np.ndarray,
    img: ImageParams,
    threshold: float,
    max_points: int,
    reshape_order: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if ppdf_1d.size != img.nx * img.ny * img.nz:
        raise ValueError(
            "Detector vector length does not match Params_Image.dat dimensions."
        )

    if reshape_order == "xyz":
        ppdf_3d = ppdf_1d.reshape((img.nx, img.ny, img.nz))
    else:
        ppdf_3d = ppdf_1d.reshape((img.nz, img.ny, img.nx)).transpose(2, 1, 0)

    max_val = float(np.max(ppdf_3d))
    if max_val <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    x, y, z = voxel_centers(img)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")

    mask = ppdf_3d >= (threshold * max_val)
    values = ppdf_3d[mask]
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    coords = np.column_stack([gx[mask], gy[mask], gz[mask]])

    if values.size > max_points:
        keep_idx = np.argpartition(values, -max_points)[-max_points:]
        values = values[keep_idx]
        coords = coords[keep_idx]

    return coords.astype(np.float32), values.astype(np.float32)


def main() -> None:
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    col_raw, det_raw, img_raw, _phy_raw = read_params(data_dir)
    img = parse_image_params(img_raw)
    detector_data = parse_detector_data(det_raw)
    col_layers, hole_data = parse_collimator_data(col_raw)

    if not (0 <= args.rotation < img.n_rot):
        raise ValueError(f"--rotation must be in [0, {img.n_rot - 1}]")
    if not (0 <= args.detector < detector_data.shape[0]):
        raise ValueError(f"--detector must be in [0, {detector_data.shape[0] - 1}]")
    if not (0.0 < args.beam_threshold <= 1.0):
        raise ValueError("--beam-threshold must be in (0, 1]")

    sysmat_path = resolve_sysmat_path(data_dir, args.sysmat.resolve() if args.sysmat else None)
    n_voxels = img.nx * img.ny * img.nz
    sysmat = np.memmap(
        sysmat_path,
        dtype=np.float32,
        mode="r",
        shape=(img.n_rot, detector_data.shape[0], n_voxels),
    )
    ppdf_1d = np.array(sysmat[args.rotation, args.detector, :], dtype=np.float32)

    beam_points_xyz, beam_values = extract_beam_points(
        ppdf_1d=ppdf_1d,
        img=img,
        threshold=args.beam_threshold,
        max_points=args.max_beam_points,
        reshape_order=args.reshape_order,
    )

    fig = build_figure(
        detector_data=detector_data,
        col_layers=col_layers,
        hole_data=hole_data,
        img=img,
        detector_idx=args.detector,
        beam_points_xyz=beam_points_xyz,
        beam_values=beam_values,
        rotation_idx=args.rotation,
        detector_step=args.detector_step,
        hole_step=args.hole_step,
    )

    output_path = args.output
    if output_path is None:
        output_path = data_dir / f"viewer_rot{args.rotation:03d}_det{args.detector:04d}.html"
    output_path = output_path.resolve()
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    print(f"Data directory: {data_dir}")
    print(f"System matrix: {sysmat_path.resolve()}")
    print(f"Rendered beam points: {beam_points_xyz.shape[0]}")
    print(f"Output HTML: {output_path}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    args = parse_args()
    main()
'''

