import argparse
import os
import h5py
import numpy as np

def get_fwhm_stats(h5_path):
    with h5py.File(h5_path, "r") as f:
        header_raw = f["beam_properties"].attrs["Header"]
        # Decode header
        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in header_raw]
        
        if "FWHM (mm)" not in header:
            return np.nan
            
        col_index = header.index("FWHM (mm)")
        data = f["beam_properties"][:, col_index]
        
        # Filter NaNs and outliers (similar to your plot script)
        valid_data = data[~np.isnan(data)]
        # Optional: Filter for "Good" beams only? 
        # valid_data = valid_data[(valid_data >= 2.0) & (valid_data <= 5.0)] 
        
        if len(valid_data) == 0:
            return np.nan
            
        return np.mean(valid_data)

def get_asci_score(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "asci_score_scalar" in f.attrs:
            return float(f.attrs["asci_score_scalar"])
        else:
            # Fallback if attribute wasn't saved: count non-zeros
            hist = f["asci_histogram"][:]
            return float(np.sum(hist > 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Optional single run_id (backwards compatible)
    #parser.add_argument("--run_id", help="Single run_id (for backward compatibility)")
    # New: multiple run_ids
    parser.add_argument(
        "--run_ids",
        nargs="+",
        help="One or more run_id strings, e.g. d0p0disp0p0 d0p0disp10p0 ..."
    )
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--layout_idx", type=int, default=0)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    # Determine which run_ids to process
    if args.run_ids is not None:
        run_ids = args.run_ids
    elif args.run_id is not None:
        run_ids = [args.run_id]
    else:
        raise ValueError("You must provide either --run_id or --run_ids.")

    # Decide whether to append or create new file
    csv_exists = os.path.exists(args.out_csv)
    mode = "a" if csv_exists else "w"

    with open(args.out_csv, mode) as f_out:
        # Write header only if creating a new file
        if not csv_exists:
            f_out.write("run_id,layout_idx,fwhm,asci\n")

        for run_id in run_ids:
            # 1. Get FWHM
            prop_file = os.path.join(
                args.work_dir,
                f"beams_properties_{run_id}_layout_{args.layout_idx:03d}.hdf5"
            )
            if not os.path.exists(prop_file):
                print(f"[WARN] Missing beam properties file for run_id={run_id}: {prop_file}")
                fwhm_val = np.nan
            else:
                fwhm_val = get_fwhm_stats(prop_file)

            # 2. Get ASCI
            asci_file = os.path.join(
                args.work_dir,
                f"asci_histogram_{run_id}_layout_{args.layout_idx:03d}.hdf5"
            )
            if not os.path.exists(asci_file):
                print(f"[WARN] Missing ASCI file for run_id={run_id}: {asci_file}")
                asci_val = np.nan
            else:
                asci_val = get_asci_score(asci_file)

            # 3. Append a line for this run_id
            f_out.write(
                f"{run_id},{args.layout_idx},{fwhm_val},{asci_val}\n"
            )

            print(f"Metrics Calculated [{run_id}]: FWHM={fwhm_val:.4f}, ASCI={asci_val}")
