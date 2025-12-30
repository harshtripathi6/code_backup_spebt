import argparse
import os
import glob
import h5py
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_asci_for_run(run_id, work_dir, out_dir=None):
    if out_dir is None:
        out_dir = work_dir

    print(f"--- Generating ASCI Plot for Run ID: {run_id} ---")

    # --- Configuration ---
    # Based on your previous scripts: 280 pixels * 0.25mm = 70mm FOV
    N_PIXELS = 280
    N_BINS = 360
    FOV_SIDE_MM = 70.0 

    # --- 1. Find all histogram files for this Run ID ---
    # Matches: asci_histogram_d1p0_disp0p0_layout_000.hdf5, ..._layout_001.hdf5, etc.
    search_pattern = os.path.join(work_dir, f"asci_histogram_{run_id}_layout_*.hdf5")
    hist_files = sorted(glob.glob(search_pattern))

    if not hist_files:
        print(f"Error: No ASCI histogram files found for {run_id} in {work_dir}")
        return

    print(f"Found {len(hist_files)} histogram files to aggregate.")

    # --- 2. Aggregate Histograms ---
    # Initialize accumulator
    total_asci_hist = torch.zeros((N_PIXELS * N_PIXELS, N_BINS), dtype=torch.int32)

    for fpath in hist_files:
        try:
            with h5py.File(fpath, "r") as f:
                # Load and add to total
                data = torch.from_numpy(f["asci_histogram"][...])
                total_asci_hist += data
        except Exception as e:
            print(f"Warning: Could not read {os.path.basename(fpath)}: {e}")

    # --- 3. Compute ASCI Map ---
    # Count non-zero bins for every pixel
    # Result is (N_pixels,) tensor with values 0 to N_BINS
    filled_bins_count = torch.count_nonzero(total_asci_hist, dim=1)
    
    # Normalize to 0.0 - 1.0 (Percentage of angles covered)
    asci_map = filled_bins_count.float() / N_BINS

    # Reshape for plotting: (280, 280)
    # Note: .T (transpose) is often needed depending on how row/col were saved vs plotted
    asci_map_2d = asci_map.view(N_PIXELS, N_PIXELS).T

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")

    # Extent = [left, right, bottom, top]
    extent = (-FOV_SIDE_MM/2, FOV_SIDE_MM/2, -FOV_SIDE_MM/2, FOV_SIDE_MM/2)

    im = ax.imshow(
        asci_map_2d.numpy(),
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0, 
        max=1.0  # Fix scale 0-100% for consistency across runs
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label="ASCI (Angular Sampling Completeness)")
    cbar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    cbar.update_ticks()

    # Labels
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    # Extract params from run_id for title if possible (e.g., d1p0_disp10p0)
    title_str = f"ASCI Map: {run_id}\n(Aggregated {len(hist_files)} layouts)"
    
    # Stats for title
    max_val = asci_map.max().item()
    mean_val = asci_map.mean().item()
    ax.set_title(f"{title_str}\nMax Coverage: {max_val:.1%}, Mean: {mean_val:.1%}")

    # Optional: Draw circular FOV boundary for reference
    circle = plt.Circle((0, 0), FOV_SIDE_MM/2, color='white', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)

    # --- 5. Save ---
    out_filename = f"asci_map_plot_{run_id}.png"
    out_path = os.path.join(out_dir, out_filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Unique Run ID (e.g. d3p0_disp10p0)")
    parser.add_argument("--work_dir", type=str, required=True, help="Directory containing asci_histogram files")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional output directory for images")
    
    args = parser.parse_args()

    plot_asci_for_run(args.run_id, args.work_dir, args.out_dir)
