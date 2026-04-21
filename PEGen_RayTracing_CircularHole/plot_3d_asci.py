import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -------------------------------------------------------------------
METRICS_DIR = "metrics"
PLOT_DIR    = "plots"
DEG_RES     = 1.0
N_BINS      = int(180 / DEG_RES) * int(360 / DEG_RES)
# -------------------------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_3d_asci():
    # 1. Load System Dimensions
    img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
    nx, ny, nz = int(img_raw[0]), int(img_raw[1]), int(img_raw[2])
    dx, dy, dz = img_raw[3], img_raw[4], img_raw[5]
    num_rot = int(img_raw[6])
    n_voxels = nx * ny * nz

    fov_x, fov_y, fov_z = nx * dx, ny * dy, nz * dz

    # 2. Sum Histograms across all rotations
    print(f"Aggregating ASCI histograms over {num_rot} rotations...")
    asci_hist_total = np.zeros((n_voxels, N_BINS), dtype=np.uint32)

    for rot_idx in range(num_rot):
        filepath = os.path.join(METRICS_DIR, f"asci_histogram_rot_{rot_idx:03d}.hdf5")
        if os.path.exists(filepath):
            with h5py.File(filepath, "r") as f:
                asci_hist_total += f["asci_histogram"][:]
        else:
            print(f"Warning: Missing histogram for rotation {rot_idx}")

    # 3. Calculate ASCI Metric (Fraction of sphere covered)
    # Count how many angular bins have at least 1 ray hitting that voxel
    asci_1d = np.count_nonzero(asci_hist_total, axis=1) / N_BINS
    asci_3d = asci_1d.reshape((nz, ny, nx)) # Z, Y, X order

    print(f"ASCI Stats - Max: {asci_3d.max():.2%}, Min: {asci_3d.min():.2%}, Mean: {asci_3d.mean():.2%}")

    # 4. Plot 3 Orthogonal Slices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # Slice indices (Middle of the FOV)
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

    extent_xy =[-fov_x/2, fov_x/2, -fov_y/2, fov_y/2]
    extent_xz =[-fov_x/2, fov_x/2, -fov_z/2, fov_z/2]
    extent_yz =[-fov_y/2, fov_y/2, -fov_z/2, fov_z/2]

    # Panel 1: Transverse (X-Y plane, slicing through Z)
    im1 = axes[0].imshow(asci_3d[mid_z, :, :].T, origin="lower", extent=extent_xy, cmap="viridis", vmin=0, vmax=asci_3d.max())
    axes[0].set_title(f"Transverse Slice (Z = {mid_z})")
    axes[0].set_xlabel("X (mm)"); axes[0].set_ylabel("Y (mm)")

    # Panel 2: Coronal (X-Z plane, slicing through Y)
    im2 = axes[1].imshow(asci_3d[:, mid_y, :].T, origin="lower", extent=extent_xz, cmap="viridis", vmin=0, vmax=asci_3d.max())
    axes[1].set_title(f"Coronal Slice (Y = {mid_y})")
    axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Z (mm)")

    # Panel 3: Sagittal (Y-Z plane, slicing through X)
    im3 = axes[2].imshow(asci_3d[:, :, mid_x].T, origin="lower", extent=extent_yz, cmap="viridis", vmin=0, vmax=asci_3d.max())
    axes[2].set_title(f"Sagittal Slice (X = {mid_x})")
    axes[2].set_xlabel("Y (mm)"); axes[2].set_ylabel("Z (mm)")

    # Shared Colorbar
    cbar = fig.colorbar(im3, ax=axes, label="ASCI (Angular Completeness)")
    cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
    cbar.update_ticks()

    fig.suptitle(f"3D Angular Sampling Completeness Index (ASCI Map)\nMax: {asci_3d.max():.2%}", fontsize=16)

    out_file = os.path.join(PLOT_DIR, "asci_3d_orthogonal_map.png")
    fig.savefig(out_file, dpi=300)
    print(f"Success! Plot saved to {out_file}")
    plt.close(fig)

if __name__ == "__main__":
    plot_3d_asci()