
"""
check_individual_mpxi.py
--------------------------------
* **Figure 1:** Histogram of *all* valid beam FWHM values (mm) with mean/median markers.
* **Figure 2:** Bar-chart showing **how many detector units have *k* beams** (k = 1, 2, 3 …).
* Prints the number of detectors that host at least one beam in the 2–5 mm “good-width” window.

Run e.g.
```bash
python check_individual_mpxi.py \
       --props  ../../../data/mph_hourglass_single_position_base_2mm_20pinholes_rotated_18custom_rot/outputs/beams_properties_configuration_00.hdf5 \
       --masks  ../../../data/mph_hourglass_single_position_base_2mm_20pinholes_rotated_18custom_rot/outputs/beams_masks_configuration_00.hdf5 \
       --out    ../../../data/mph_hourglass_single_position_base_2mm_20pinholes_rotated_18custom_rot/plots
```
"""

import argparse, os, h5py, numpy as np, matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------------ #

def read_columns(h5_path: str, desired: list[str]) -> dict[str, torch.Tensor]:
    """Return selected columns from beam_properties HDF5 as tensors."""
    with h5py.File(h5_path, "r") as f:
        # Handle case where header might be bytes
        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in f["beam_properties"].attrs["Header"]]
        data   = torch.from_numpy(f["beam_properties"][:])
    cols: dict[str, torch.Tensor] = {}
    for name in desired:
        if name not in header:
            raise RuntimeError(f"Column '{name}' not found in {h5_path}")
        cols[name] = data[:, header.index(name)]
    return cols

# ------------------------------------------------------------------ #

def main(args):
    # -- read FWHM and detector‑ID columns ---------------------------
    print(f"Reading properties from: {args.props}")
    cols   = read_columns(args.props, ["FWHM (mm)", "detector unit id"])
    fwhm_raw = cols["FWHM (mm)"].numpy()
    det_id_raw = cols["detector unit id"].to(torch.int64)

    # --- ADDED: Debugging print to show the raw data ---
    print(f"First 10 raw FWHM values read from file: {fwhm_raw[:10]}")
    # ----------------------------------------------------

    # --- IMPORTANT: Filter out NaN values from FWHM ---
    # This handles cases where FWHM calculation failed for a beam
    valid_fwhm_mask = ~np.isnan(fwhm_raw)
    fwhm = fwhm_raw[valid_fwhm_mask]
    det_id = det_id_raw[valid_fwhm_mask] # det_id is now a torch.Tensor
    
    print(f"Read {len(fwhm_raw)} total beam entries.")
    print(f"Found {len(fwhm)} beams with a valid FWHM value.")

    # -- detectors that have at least one 2‑5 mm beam ---------------
    if len(fwhm) > 0:
        good_width_mask = (fwhm >= 2.0) & (fwhm <= 5.0)
        # Apply the mask to the valid detector IDs before finding unique ones
        n_good_det = torch.unique(det_id[good_width_mask]).numel()
        print(f"Detectors with ≥1 beam in 2–5 mm window: {n_good_det}")
    else:
        print("No valid FWHM values to analyze.")
        n_good_det = 0

    # ------------------ Figure 1 – FWHM histogram ------------------
    os.makedirs(args.out, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(7,4), layout="constrained")
    
    # Plot histogram only with valid FWHM values
    if len(fwhm) > 0:
        ax1.hist(fwhm, bins=30, color="#4c72b0", alpha=0.85)
        mean_v, med_v = fwhm.mean(), np.median(fwhm)
        ax1.axvline(mean_v, color="red",   ls="--", lw=1.5, label=f"Mean {mean_v:.2f} mm")
        ax1.axvline(med_v,  color="green", ls=":",  lw=1.5, label=f"Median {med_v:.2f} mm")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No valid FWHM data to plot", ha='center', va='center')

    ax1.set_xlabel("Beam FWHM (mm)")
    ax1.set_ylabel("Number of beams")
    ax1.set_title("Distribution of Beam Widths (all valid beams)")
    
    out1 = os.path.join(args.out, "beam_width_histogram.png")
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"Saved → {out1}")

    # ------------- Figure 2 – beams‑per‑detector bar chart ----------
    if args.masks:
        print(f"Reading masks from: {args.masks}")
        with h5py.File(args.masks, "r") as f:
            masks = torch.from_numpy(f["beam_mask"][:])
        
        # Count beams per detector by finding unique non-zero IDs in each mask row
        # counts[i] = number of beams on detector 'i'
        counts = torch.tensor([(row.unique().numel() - 1) for row in masks])

        # How many detectors have k beams? (k = 1…max)
        if counts.numel() > 0 and counts.max() > 0:
            max_k  = int(counts.max().item())
            det_per_k = torch.bincount(counts, minlength=max_k+1)[1:]  # drop k=0 count
            ks = np.arange(1, max_k+1) # These are the k-values we will plot

            print("\nDetectors by beam count:")
            for k, c in zip(ks, det_per_k.tolist()):
                print(f"  k={k}: {c} detectors")

            fig2, ax2 = plt.subplots(figsize=(6,4), layout="constrained")
            bars = ax2.bar(ks, det_per_k.numpy(), color="#55a868", alpha=0.9)
            for bar, val in zip(bars, det_per_k.tolist()):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         str(val), ha="center", va="bottom", fontsize=9)
            ax2.set_xlabel("Number of beams per detector (k)")
            ax2.set_ylabel("Number of detectors")
            ax2.set_title("Beam Multiplicity Distribution")
            ax2.set_xticks(ks if max_k < 10 else np.arange(1, max_k + 1, 2)) # Adjust ticks for many bars

            out2 = os.path.join(args.out, "detector_beam_counts.png")
            fig2.savefig(out2, dpi=300)
            plt.close(fig2)
            print(f"Saved → {out2}")

            # --------------------------------------------------------------- #
            # --- START: NEW CODE FOR FIGURE 3 (FWHM Histograms by k) ---
            # --------------------------------------------------------------- #
            print("\nGenerating FWHM histograms by multiplexing degree (k)...")
            
            # We need to find the multiplexing degree (k) for each beam's host detector
            # 'fwhm' is a numpy array of valid FWHMs
            # 'det_id' is a torch tensor of the corresponding detector IDs
            # 'counts' is a torch tensor where counts[i] is the k-value for detector i
            
            beam_k_values_np = None
            try:
                # Use det_id to index into counts, getting the k-value for each beam
                beam_k_values = counts[det_id] 
                beam_k_values_np = beam_k_values.numpy()
            except IndexError:
                print(f"  ERROR: A detector ID from {args.props} (max: {det_id.max().item()})")
                print(f"  is out of bounds for the masks file (num detectors: {counts.numel()}).")
                print("  This can happen if props and masks files do not match.")
                print("  Skipping Figure 3.")

            if beam_k_values_np is not None:
                # ks = np.arange(1, max_k+1) is already defined above
                n_cols = max_k
                
                # Create the 1-row, n_cols-column grid
                fig3, axes3 = plt.subplots(
                    1, n_cols, 
                    figsize=(n_cols * 4, 4.5), # Width scales with number of plots
                    layout="constrained", 
                    sharey=True # Share Y-axis for easier comparison
                )
                
                # Handle case where n_cols=1 (max_k=1), plt.subplots returns a single Ax
                if n_cols == 1:
                    axes3 = [axes3]
                
                print(f"  Plotting {n_cols} histograms for k=1 to {max_k}...")

                # Iterate from k=1 to max_k
                for k, ax in zip(ks, axes3):
                    # Find all FWHM values for beams where the detector's k-value matches
                    fwhm_for_k = fwhm[beam_k_values_np == k]
                    n_beams_k = len(fwhm_for_k)
                    
                    ax.set_title(f"k = {k} (N={n_beams_k} beams)")
                    ax.set_xlabel("FWHM (mm)")
                    
                    if n_beams_k > 0:
                        ax.hist(fwhm_for_k, bins=20, color="#c44e52", alpha=0.85)
                        mean_v, med_v = fwhm_for_k.mean(), np.median(fwhm_for_k)
                        ax.axvline(mean_v, color="blue",  ls="--", lw=1.5, label=f"Mean {mean_v:.2f}")
                        ax.axvline(med_v,  color="black", ls=":",  lw=1.5, label=f"Median {med_v:.2f}")
                        ax.legend(fontsize="small")
                    else:
                        # Plot empty text if no beams found for this k
                        ax.text(0.5, 0.5, "No valid beams", 
                                ha='center', va='center', transform=ax.transAxes)
                
                # Set Y-label only on the first plot
                axes3[0].set_ylabel("Number of beams")
                fig3.suptitle("FWHM Distribution by Detector Multiplexing Degree (k)", fontsize=16)

                out3 = os.path.join(args.out, "fwhm_by_multiplexing_grid.png")
                fig3.savefig(out3, dpi=300)
                plt.close(fig3)
                print(f"Saved → {out3}")
            # --------------------------------------------------------------- #
            # --- END: NEW CODE FOR FIGURE 3 ---
            # --------------------------------------------------------------- #

        else:
            print("\nNo beams found in the masks file to generate multiplicity plot.")
    else:
        print("\n--masks file not provided, skipping Figure 2 and Figure 3.")

# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Beam-width histogram and detector beam-count plot.")
    ap.add_argument("--props", required=True, help="beam_properties_*.hdf5 file")
    ap.add_argument("--masks", default=None, help="beams_masks_*.hdf5 file (needed for Figure 2)")
    ap.add_argument("--out",   required=True, help="output directory for PNGs")
    args = ap.parse_args()
    main(args)
