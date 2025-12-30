#!/usr/bin/env python3
"""
check_cumulative_beam_widths.py
--------------------------------
Reads beam properties from ALL rotation files in a directory and generates a single,
cumulative histogram of beam FWHM values.

* **Figure 1:** Cumulative histogram of all valid beam FWHM values (mm) across
    all configurations, with mean/median markers.
* Prints the total number of beams found and the count of those within the
    2–5 mm “good-width” window.

Run e.g.
```bash
python check_cumulative_beam_widths.py \
       --props-dir ../../../data/scspect_4rings_biconicalmph_01pos_24pinholes_T4_protocol/outputs/ \
       --out       ../../../data/scspect_4rings_biconicalmph_01pos_24pinholes_T4_protocol/plots
```
"""
import argparse
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------------ #

def read_fwhm_column(h5_path: str) -> np.ndarray:
    """
    Return the 'FWHM (mm)' column from a beam_properties HDF5 file.
    
    Args:
        h5_path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: An array containing the FWHM values.
    """
    with h5py.File(h5_path, "r") as f:
        # Handle case where header might be bytes and needs decoding
        header_raw = f["beam_properties"].attrs["Header"]
        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in header_raw]
        
        if "FWHM (mm)" not in header:
            raise RuntimeError(f"Column 'FWHM (mm)' not found in {h5_path}")
            
        col_index = header.index("FWHM (mm)")
        # Use numpy directly which is sufficient for this task
        fwhm_data = f["beam_properties"][:, col_index]
        
    return fwhm_data

# ------------------------------------------------------------------ #

def main(args):
    """
    Main function to find files, aggregate data, and generate the plot.
    """
    # -- Find all property files in the input directory -------------
    search_pattern = os.path.join(args.props_dir, "beams_properties_*.hdf5")
    property_files = sorted(glob.glob(search_pattern))

    if not property_files:
        print(f"Error: No 'beams_properties_*.hdf5' files found in '{args.props_dir}'")
        return

    print(f"Found {len(property_files)} beam property files to process.")

    # -- Read FWHM from all files and aggregate --------------------
    all_fwhm_values = []
    total_beams_read = 0

    for file_path in property_files:
        print(f"  - Reading properties from: {os.path.basename(file_path)}")
        try:
            fwhm_raw = read_fwhm_column(file_path)
            total_beams_read += len(fwhm_raw)
            
            # IMPORTANT: Filter out NaN values from FWHM for this file.
            # This handles cases where FWHM calculation may have failed.
            valid_fwhm_mask = ~np.isnan(fwhm_raw)
            valid_fwhm = fwhm_raw[valid_fwhm_mask]
            
            all_fwhm_values.extend(valid_fwhm)
        except Exception as e:
            print(f"    Could not process file {os.path.basename(file_path)}: {e}")

    # Consolidate into a single NumPy array for efficient analysis
    cumulative_fwhm = np.array(all_fwhm_values)
    
    print("-" * 50)
    print(f"Total beams read across all files: {total_beams_read}")
    print(f"Found {len(cumulative_fwhm)} total beams with a valid FWHM value.")

    # -- Analyze cumulative data ------------------------------------
    if len(cumulative_fwhm) == 0:
        print("No valid FWHM values found across all files. Cannot generate plot.")
        return
        
    good_width_mask = (cumulative_fwhm >= 2.0) & (cumulative_fwhm <= 5.0)
    n_good_beams = np.sum(good_width_mask)
    print(f"Total beams in the 2–5 mm 'good-width' window: {n_good_beams}")

    # -------------- Figure 1 – Cumulative FWHM histogram -----------
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    
    # Plot the main histogram
    ax.hist(cumulative_fwhm, bins=50, color="#4c72b0", alpha=0.9, label=f'{len(cumulative_fwhm)} total beams')
    
    # Calculate and plot mean/median lines for reference
    mean_v = cumulative_fwhm.mean()
    med_v = np.median(cumulative_fwhm)
    max_v = np.max(cumulative_fwhm)
    print(f"Max FWHM - {max_v}")
    
    ax.axvline(mean_v, color="#d62728", ls="--", lw=2, label=f"Mean: {mean_v:.2f} mm")
    ax.axvline(med_v, color="#2ca02c", ls=":", lw=2, label=f"Median: {med_v:.2f} mm")
    
    # --- Final plot styling ---
    ax.legend()
    ax.set_xlabel("Beam FWHM (mm)")
    ax.set_ylabel("Number of Beams")
    ax.set_title(f"Cumulative Beam Width Distribution ({len(property_files)} Rotations)")
    # ax.set_xlim([0,5])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Save the figure ---
    out_path = os.path.join(args.out, "cumulative_beam_width_histogram.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved cumulative plot → {out_path}")

# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser(description="Generate a cumulative beam-width histogram from multiple property files.")
    ap.add_argument("--props-dir", required=True, help="Directory containing beam_properties_*.hdf5 files.")
    ap.add_argument("--out", required=True, help="Output directory for the PNG plot.")
    args = ap.parse_args()
    main(args)
