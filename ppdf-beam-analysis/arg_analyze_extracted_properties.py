import torch
import h5py
import os
import sys

if __name__ == "__main__":
    # --- MODIFICATION: Get layout_idx from a command-line argument ---
    if len(sys.argv) != 2:
        print("Usage: python generate_asci_histogram_task.py <layout_idx>")
        sys.exit(1)
    try:
        layout_idx = int(sys.argv[1])
    except ValueError:
        print(f"Error: <layout_idx> must be an integer. Received: {sys.argv[1]}")
        sys.exit(1)

    print(f"--- Starting ASCI histogram generation for layout index: {layout_idx} ---")

    # --- Configuration ---
    n_bins = 360
    angular_bin_boundaries = torch.arange(n_bins + 1) / 180 * torch.pi
    input_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/outputs_3mm_aperture_displaced_24ph"

    # --- Data Loading ---
    print("Loading beams properties and masks...")
    try:
        # Load the beams properties
        beams_properties_hdf5_filename = f"beams_properties_configuration_{layout_idx:03d}.hdf5"
        with h5py.File(os.path.join(input_dir, beams_properties_hdf5_filename), "r") as f:
            layout_beams_properties = torch.from_numpy(f["beam_properties"][:])
            beam_properties_header = f["beam_properties"].attrs["Header"]

        # Load the beams masks for the layout
        beams_masks_hdf5_filename = f"beams_masks_configuration_{layout_idx:03d}.hdf5"
        with h5py.File(os.path.join(input_dir, beams_masks_hdf5_filename), "r") as beams_masks_hdf5:
            beams_masks = torch.from_numpy(beams_masks_hdf5["beam_mask"][:])
    except FileNotFoundError as e:
        print(f"Error: Input file not found for layout {layout_idx}. Make sure previous steps ran successfully.")
        print(e)
        sys.exit(1)

    # --- Data Processing ---
    print("Processing data and building histogram...")
    
    # Digitize the angles
    digitized_angles = torch.bucketize(
        layout_beams_properties[:, 3], angular_bin_boundaries, right=False
    )
    # Note: The original script had this block duplicated. One is sufficient.
    layout_beams_properties = torch.cat(
        (layout_beams_properties, (digitized_angles - 1).unsqueeze(1).float()),
        dim=1,
    )
    
    # Filter out beams with NaN angles
    layout_beams_properties_filtered = layout_beams_properties[
        torch.isnan(layout_beams_properties[:, 3]) == False
    ]

    # Filter by sensitivity (optional, kept from original script)
    if layout_beams_properties_filtered[:, 7].numel() > 0:
        beams_sensitivity_max = layout_beams_properties_filtered[:, 7].max()
    else:
        # Handle the empty case: set a default value, log a warning, etc.
        beams_sensitivity_max = 0.0  # Or float('-inf'), None, etc.
        print("Warning: No beams matched the filter criteria.")
        
    layout_beams_properties_filtered = layout_beams_properties_filtered[
        layout_beams_properties_filtered[:, 7] > beams_sensitivity_max * 0.01
    ]
    
    print(f"Found {layout_beams_properties_filtered.shape[0]} valid beams to process.")

    # Initialize the histogram for ASCI map
    asci_histogram = torch.zeros((280 * 280, n_bins), dtype=torch.int32)

    # Loop through the beams and populate the histogram
    for beam_props in layout_beams_properties_filtered:
        detector_idx = int(beam_props[1])
        beam_idx = int(beam_props[2])
        # The angle bin index is now the last column (index 11 after cat)
        angle_bin_idx = int(beam_props[-1]) 
        
        # Ensure angle_bin_idx is valid before using it
        if 0 <= angle_bin_idx < n_bins:
            asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

    # --- Save the Output ---
    asci_histogram_filename = os.path.join(input_dir, f"asci_histogram_{layout_idx:03d}.hdf5")
    print(f"Saving histogram to: {asci_histogram_filename}")
    with h5py.File(asci_histogram_filename, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram.numpy())

    print(f"--- Finished ASCI histogram generation for layout index: {layout_idx} ---")