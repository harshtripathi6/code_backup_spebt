import torch
import h5py
import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layout_idx", type=int)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    args = parser.parse_args()

    layout_idx = args.layout_idx
    run_id = args.run_id
    work_dir = args.work_dir

    print(f"--- Starting ASCI histogram generation for {run_id} (Layout {layout_idx}) ---")

    # --- Configuration ---
    n_bins = 360
    angular_bin_boundaries = torch.arange(n_bins + 1) / 180 * torch.pi

    # --- Data Loading ---
    # Construct filenames based on RUN_ID
    prop_filename = f"beams_properties_{run_id}_layout_{layout_idx:03d}.hdf5"
    mask_filename = f"beams_masks_{run_id}_layout_{layout_idx:03d}.hdf5"
    
    prop_path = os.path.join(work_dir, prop_filename)
    mask_path = os.path.join(work_dir, mask_filename)

    if not os.path.exists(prop_path) or not os.path.exists(mask_path):
        print(f"Error: Missing properties or masks for {run_id}.")
        print(f"Looked for: {prop_path}")
        sys.exit(1)

    with h5py.File(prop_path, "r") as f:
        layout_beams_properties = torch.from_numpy(f["beam_properties"][:])

    with h5py.File(mask_path, "r") as f:
        beams_masks = torch.from_numpy(f["beam_mask"][:])

    # --- Data Processing ---
    # Digitize angles (Column 3 is usually angle)
    digitized_angles = torch.bucketize(
        layout_beams_properties[:, 3], angular_bin_boundaries, right=False
    )
    
    layout_beams_properties = torch.cat(
        (layout_beams_properties, (digitized_angles - 1).unsqueeze(1).float()),
        dim=1,
    )
    
    # Filter NaNs
    layout_beams_properties_filtered = layout_beams_properties[
        ~torch.isnan(layout_beams_properties[:, 3])
    ]

    # Filter by sensitivity
    if layout_beams_properties_filtered[:, 7].numel() > 0:
        beams_sensitivity_max = layout_beams_properties_filtered[:, 7].max()
        layout_beams_properties_filtered = layout_beams_properties_filtered[
            layout_beams_properties_filtered[:, 7] > beams_sensitivity_max * 0.01
        ]

    # Initialize histogram
    asci_histogram = torch.zeros((280 * 280, n_bins), dtype=torch.int32)

    for beam_props in layout_beams_properties_filtered:
        detector_idx = int(beam_props[1])
        beam_idx = int(beam_props[2])
        angle_bin_idx = int(beam_props[-1]) 
        
        if 0 <= angle_bin_idx < n_bins:
            # Check bounds for detector mask access
            if detector_idx < len(beams_masks):
                 asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

    # --- Save Output ---
    # Filename includes run_id
    out_filename = f"asci_histogram_{run_id}_layout_{layout_idx:03d}.hdf5"
    out_path = os.path.join(work_dir, out_filename)
    
    print(f"Saving histogram to: {out_path}")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram.numpy())
        # Calculate a simple scalar score for the CSV (e.g., total filled bins)
        score = (asci_histogram > 0).sum().item()
        f.attrs["asci_score_scalar"] = score

    print(f"Done. ASCI Score (bins filled): {score}")