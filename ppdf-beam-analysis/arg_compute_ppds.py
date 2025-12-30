#!/usr/bin/env python3
import os, sys, h5py, torch
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

# --- your modules ----------------------------------------------------
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict, pixels_coordinates, pixels_to_detector_unit_rads
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from ppdf_io import load_ppdfs_data_from_hdf5
from beam_property_extract import (
    sample_ppdf_on_arc_2d_local,
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,                 # tangential FWHM (perpendicular to axis)
    beam_samples_on_points_batch,   # reuse for radial sampling
    full_width_half_maximum_1d_batch,
)
# ---------------------------------------------------------------------

# ------------------------- helpers -----------------------------------
@torch.no_grad()
def _beam_axis_sampling_line_batch(
    detector_center: torch.Tensor,
    beam_centers: torch.Tensor,
    n_samples: int = 4096,
    length_mm: float = 64.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampling line *along* the beam axis (radial)."""
    n_beams = beam_centers.shape[0]
    axis = torch.atan2(beam_centers[:, 1] - detector_center[1],
                       beam_centers[:, 0] - detector_center[0])
    kx, ky = torch.cos(axis), torch.sin(axis)
    dist = torch.linspace(-length_mm/2, length_mm/2, n_samples, dtype=torch.float32, device=beam_centers.device)
    pts  = torch.stack((kx, ky), dim=1).unsqueeze(1).expand(-1, n_samples, -1) \
           * dist.view(1, n_samples, 1) + beam_centers.unsqueeze(1).expand(-1, n_samples, -1)
    return pts, dist

@torch.no_grad()
def _get_beam_width_radial(
    beams_centers: torch.Tensor,
    detector_center: torch.Tensor,
    beams_masks: torch.Tensor,
    ppdf_2d: torch.Tensor,
    fov_dict: dict,
    line_n_samples: int = 4096,
):
    """Radial FWHM by sampling *along* the axis."""
    pts_batch, dist = _beam_axis_sampling_line_batch(
        detector_center, beams_centers, n_samples=line_n_samples, length_mm=64.0
    )
    n_beams = beams_centers.shape[0]
    beams_data_2d_batch = (
        ppdf_2d.view(-1).unsqueeze(0).expand(n_beams, -1).clone().masked_fill_(~beams_masks, 0)
    ).view(n_beams, int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1]))
    sampled = beam_samples_on_points_batch(beams_data_2d_batch, pts_batch, fov_dict)
    fwhm, x_bounds = full_width_half_maximum_1d_batch(
        dist.view(1, -1).expand(n_beams, -1), sampled
    )
    return fwhm, x_bounds, sampled, dist
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_ppds_for_layout(
    layout_idx: int,
    scanner_layouts_dir: str,
    scanner_layouts_filename: str,
    ppdfs_dir: str,
    out_dir: str,
    threshold_relative: float = 0.01,
    line_n_samples: int = 4096,
):
    os.makedirs(out_dir, exist_ok=True)

    # --- load meta and data ------------------------------------------
    layouts_data, layouts_uid = load_scanner_layouts(scanner_layouts_dir, scanner_layouts_filename)
    fov = fov_tensor_dict(n_pixels=(280, 280), mm_per_pixel=(0.25, 0.25), center_coordinates=(0.0, 0.0))
    _, det_units = load_scanner_layout_geometries(layout_idx, layouts_data)

    # match your Block-2 naming convention
    ppdf_h5 = f"scanner_layouts_{layouts_uid}_layout_{layout_idx:03d}_3x3_subvoxels.hdf5"
    ppdfs = load_ppdfs_data_from_hdf5(ppdfs_dir, ppdf_h5, fov)  # (n_crystals, npx, npy)
    n_det = int(det_units.shape[0])

    det_centers = det_units.mean(dim=1)
    fov_corners = (torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float32) * fov["size in mm"] * 0.5)
    hull_points_batch = torch.cat(
        (fov_corners.unsqueeze(0).expand(n_det, -1, -1), det_centers.unsqueeze(1)),
        dim=1
    )
    hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

    # outputs
    PPDS = torch.zeros(int(fov["n pixels"][0]), int(fov["n pixels"][1]), dtype=torch.float32)
    SENS = torch.zeros_like(PPDS)
    V_i  = torch.zeros(n_det, dtype=torch.float32)

    for i in range(n_det):
        ppdf2d = ppdfs[i].view(int(fov["n pixels"][0]), int(fov["n pixels"][1]))
        center = det_centers[i]
        hull   = convex_hull_2d(hull_points_batch[i])

        # arc sampling -> beam boundaries
        sampled, rads, _ = sample_ppdf_on_arc_2d_local(ppdf2d, center, hull, fov)
        # IMPORTANT: your dependency has only (arc, rads, threshold)
        bounds = beams_boundaries_radians(sampled, rads, threshold=threshold_relative)

        # conventional sensitivity always sums PPDFs
        SENS += ppdf2d

        if bounds.numel() == 0:
            continue

        # masks & centers
        fov_xy   = pixels_coordinates(fov)
        fov_rads = pixels_to_detector_unit_rads(fov_xy, center)
        masks    = get_beams_masks(fov_rads, bounds)           # (n_beams, n_pixels)
        centers  = get_beams_weighted_center(masks, fov_xy, ppdf2d)

        # tangential width (perpendicular)
        fwhm_tan, _, _, _ = get_beam_width(
            centers, center, masks, ppdf2d, fov, line_n_samples=line_n_samples
        )

        # radial width (along axis)
        fwhm_rad, _, _, _ = _get_beam_width_radial(
            centers, center, masks, ppdf2d, fov, line_n_samples=line_n_samples
        )

        eps = 1e-6
        fwhm_tan = torch.nan_to_num(fwhm_tan, nan=0.0).clamp_min(0.0)
        fwhm_rad = torch.nan_to_num(fwhm_rad, nan=0.0).clamp_min(0.0)

        Vi_b = fwhm_tan * fwhm_rad
        Vi   = float(torch.sum(Vi_b).item())
        if Vi <= eps:
            continue

        V_i[i] = Vi
        PPDS += ppdf2d / Vi

    # ---- save --------------------------------------------------------
    out_h5 = os.path.join(out_dir, f"ppds_layout_{layout_idx:03d}.hdf5")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("PPDS", data=PPDS.numpy())
        f.create_dataset("SENS", data=SENS.numpy())
        f.create_dataset("V_i",  data=V_i.numpy())
        f["PPDS"].attrs["definition"] = "sum_i PPDF_i / V_i, with V_i=sum_b FWHM_tan(i,b)*FWHM_rad(i,b)"
        f["SENS"].attrs["definition"] = "conventional sensitivity: sum_i PPDF_i"
        f.attrs["layout_idx"] = layout_idx
        f.attrs["layouts_uid"] = layouts_uid
        f.attrs["threshold_relative"] = threshold_relative
        f.attrs["line_n_samples"] = line_n_samples

    # ---- histogram of V_i -------------------------------------------
    vals = V_i[V_i > 0]
    if vals.numel() > 0:
        fig, ax = plt.subplots(figsize=(7,5), layout="constrained")
        ax.hist(vals.cpu().numpy(), bins=50)
        ax.set_xlabel(r"$V_i$  (mm$^2$)  =  $\sum_b$ FWHM$_{i,b}^{\rm tan}$ × FWHM$_{i,b}^{\rm rad}$")
        ax.set_ylabel("Count")
        ax.set_title(f"Layout {layout_idx:03d}: histogram of $V_i$ (n={vals.numel()})")
        plt.savefig(os.path.join(out_dir, f"V_i_hist_layout_{layout_idx:03d}.png"), dpi=200)
        plt.close(fig)

    print(f"[OK] Wrote: {out_h5}")
    if vals.numel() > 0:
        print(f"[OK] Saved V_i histogram with {vals.numel()} nonzero entries.")

# -------------------- NEW: average across layouts --------------------
@torch.no_grad()
def compute_ppds_average(
    layout_indices,
    ppds_dir: str,
    out_dir: str,
):
    """
    Build an average PPDS (and SENS) map across the given layouts.
    Expects per-layout files named 'ppds_layout_XXX.hdf5' already present in ppds_dir.

    Saves:
      out_dir/ppds_layout_AVG.hdf5 with datasets:
        - 'PPDS'  (float32): arithmetic mean over layouts of per-layout PPDS maps
        - 'SENS'  (float32): arithmetic mean over layouts of per-layout SENS maps
      and attributes: 'n_layouts', 'layout_indices', 'layouts_uid'
    """
    assert len(layout_indices) > 0, "No layout indices provided for averaging."

    sum_PPDS = None
    sum_SENS = None
    layouts_uid = None
    shape = None

    found = []
    for li in layout_indices:
        h5_path = os.path.join(ppds_dir, f"ppds_layout_{li:03d}.hdf5")
        if not os.path.exists(h5_path):
            print(f"[WARN] Missing {h5_path}; skipping.")
            continue

        with h5py.File(h5_path, "r") as f:
            PPDS_l = torch.from_numpy(f["PPDS"][...]).to(torch.float32)
            SENS_l = torch.from_numpy(f["SENS"][...]).to(torch.float32)
            uid    = f.attrs.get("layouts_uid", None)

            if shape is None:
                shape = PPDS_l.shape
                sum_PPDS = torch.zeros_like(PPDS_l, dtype=torch.float32)
                sum_SENS = torch.zeros_like(SENS_l, dtype=torch.float32)
                layouts_uid = uid
            else:
                if PPDS_l.shape != shape:
                    raise ValueError(f"Shape mismatch for layout {li}: {PPDS_l.shape} vs {shape}")

            sum_PPDS += PPDS_l
            sum_SENS += SENS_l
            found.append(li)

    n = len(found)
    if n == 0:
        raise FileNotFoundError("No per-layout PPDS files found; nothing to average.")

    mean_PPDS = sum_PPDS / float(n)
    mean_SENS = sum_SENS / float(n)

    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, "ppds_layout_AVG.hdf5")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("PPDS", data=mean_PPDS.cpu().numpy())
        f.create_dataset("SENS", data=mean_SENS.cpu().numpy())
        f.attrs["n_layouts"] = n
        f.attrs["layout_indices"] = found
        if layouts_uid is not None:
            f.attrs["layouts_uid"] = layouts_uid
        f["PPDS"].attrs["definition"] = "Arithmetic mean over layouts of per-layout PPDS maps"
        f["SENS"].attrs["definition"] = "Arithmetic mean over layouts of per-layout sensitivity (sum of PPDFs)"

    print(f"[OK] Wrote average PPDS: {out_h5} (n={n}, layouts={found})")
# ---------------------------------------------------------------------



if __name__ == "__main__":
    # Two modes:
    #   1) Per-layout: python arg_compute_ppds.py <layout_idx>
    #   2) Average:    python arg_compute_ppds.py avg <N>
    #                  python arg_compute_ppds.py avg <start> <end>   (inclusive)

    SCANNER_LAYOUTS_DIR  = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon"
    SCANNER_LAYOUTS_FILE = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/3mm_config/scanner_layouts_636bed318120192d3b39fb33a7655ea8_rot20_trans1x1_step0x0.tensor"
    PPDFS_DIR            = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/ppdfs_20rots_3mm_aperture"
    OUT_DIR              = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/outputs"

    if len(sys.argv) >= 2 and sys.argv[1].lower() in ("avg", "average", "mean"):
        # Average mode
        if len(sys.argv) == 3:
            # avg N  -> layouts 0..N-1
            try:
                N = int(sys.argv[2])
            except ValueError:
                print("Error: N must be an integer.")
                sys.exit(1)
            indices = list(range(N))
        elif len(sys.argv) == 4:
            # avg start end  (inclusive)
            try:
                start = int(sys.argv[2]); end = int(sys.argv[3])
            except ValueError:
                print("Error: start/end must be integers.")
                sys.exit(1)
            if end < start:
                print("Error: end must be >= start.")
                sys.exit(1)
            indices = list(range(start, end + 1))
        else:
            print("Usage:\n  python arg_compute_ppds.py avg <N>\n  python arg_compute_ppds.py avg <start> <end>")
            sys.exit(1)

        # Build average from already-computed per-layout files in OUT_DIR
        compute_ppds_average(indices, OUT_DIR, OUT_DIR)
        sys.exit(0)

    # Per-layout mode
    if len(sys.argv) != 2:
        print("Usage: python arg_compute_ppds.py <layout_idx>\n   or : python arg_compute_ppds.py avg <N|start end>")
        sys.exit(1)
    try:
        layout_idx = int(sys.argv[1])
    except ValueError:
        print("Error: <layout_idx> must be an integer.")
        sys.exit(1)

    compute_ppds_for_layout(
        layout_idx,
        SCANNER_LAYOUTS_DIR,
        SCANNER_LAYOUTS_FILE,
        PPDFS_DIR,
        OUT_DIR,
        threshold_relative=0.01,
        line_n_samples=4096,
    )