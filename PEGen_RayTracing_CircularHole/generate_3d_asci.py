import numpy as np
import h5py
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_3d_asci.py <rot_idx>")
        sys.exit(1)
    try:
        rot_idx = int(sys.argv[1])
    except ValueError:
        print(f"Error: <rot_idx> must be an integer. Received: {sys.argv[1]}")
        sys.exit(1)

    print(f"--- Starting 3D ASCI histogram generation for rotation index: {rot_idx} ---")

    # --- Configuration ---
    input_dir = "beam_outputs"
    output_dir = "metrics"
    os.makedirs(output_dir, exist_ok=True)

    # 3D Angular Binning (1-degree resolution)
    deg_res = 1.0
    n_theta_bins = int(180 / deg_res)  # Polar (0 to 180) -> 180 bins
    n_phi_bins = int(360 / deg_res)    # Azimuthal (-180 to 180) -> 360 bins
    n_total_bins = n_theta_bins * n_phi_bins

    # Load System size to know voxel count
    img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
    n_voxels = int(img_raw[0] * img_raw[1] * img_raw[2])

    # --- Data Loading ---
    print("Loading 3D beam properties and masks...")
    try:
        prop_file = os.path.join(input_dir, f"beams_properties_rot_{rot_idx:03d}.hdf5")
        mask_file = os.path.join(input_dir, f"beams_masks_rot_{rot_idx:03d}.hdf5")
        
        with h5py.File(prop_file, "r") as f_props:
            beam_props = f_props["beam_properties"][:]
            
        # We keep the mask file open to read individual detector masks on-the-fly to save RAM
        f_masks = h5py.File(mask_file, "r")
        beam_masks = f_masks["beam_mask"]
    except Exception as e:
        print(f"Error loading files for rotation {rot_idx}: {e}")
        sys.exit(1)

    # --- NEW: Create a mapping from absolute detector ID to HDF5 mask row ---
    # Because some detectors were skipped during extraction, row 500 in the 
    # HDF5 file might actually belong to detector #505.
    unique_det_ids = np.unique(beam_props[:, 1].astype(int))
    det_to_mask_row = {det_id: row_idx for row_idx, det_id in enumerate(unique_det_ids)}

    # --- Process Data ---
    print(f"Found {len(beam_props)} valid beams to process.")
    print(f"Total angular bins per voxel: {n_total_bins}")
    
    # Initialize 2D histogram (Flattened voxels x Flattened Spherical Bins)
    asci_histogram = np.zeros((n_voxels, n_total_bins), dtype=np.uint16)

    for prop in beam_props:
        det_idx = int(prop[1])
        beam_id = int(prop[2])
        theta = prop[3] # 0 to pi
        phi = prop[4]   # -pi to pi
        
        # Map angles to bin indices
        theta_bin = min(int((theta / np.pi) * n_theta_bins), n_theta_bins - 1)
        
        phi_mapped = (phi + 2 * np.pi) % (2 * np.pi) # Map to 0 -> 2pi
        phi_bin = min(int((phi_mapped / (2 * np.pi)) * n_phi_bins), n_phi_bins - 1)
        
        flat_bin_idx = (theta_bin * n_phi_bins) + phi_bin
        
        # Get the correct row in the mask file using our mapping
        mask_row = det_to_mask_row[det_idx]
        mask_1d = beam_masks[mask_row]
        
        # Find exactly which voxels this beam covers
        voxel_indices = np.where(mask_1d == beam_id)[0]
        
        # Increment the histogram for these voxels
        asci_histogram[voxel_indices, flat_bin_idx] += 1

    f_masks.close()

    # --- Save the Output ---
    out_filename = os.path.join(output_dir, f"asci_histogram_rot_{rot_idx:03d}.hdf5")
    print(f"Saving histogram to: {out_filename}")
    with h5py.File(out_filename, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram, compression="gzip")

    print(f"--- Finished ASCI generation for rotation: {rot_idx} ---")