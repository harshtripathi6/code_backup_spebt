'''
import os
import sys
import glob
import time
import h5py
import numpy as np
from scipy import ndimage

def load_system_geometry():
    """Dynamically loads scanner dimensions and coordinates from .dat files."""
    try:
        img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
        det_raw = np.fromfile("Params_Detector.dat", dtype=np.float32)
    except FileNotFoundError:
        print("ERROR: Parameter files missing. Run this in the GPUPTS output directory.")
        sys.exit(1)

    geom = {
        "nx": int(img_raw[0]), "ny": int(img_raw[1]), "nz": int(img_raw[2]),
        "dx": img_raw[3], "dy": img_raw[4], "dz": img_raw[5],
        "fov_dist": img_raw[11],
        "n_rot": int(img_raw[6]),
        "n_det": int(det_raw[0])
    }
    
    # Build 3D coordinate grids for Center of Mass and FWHM calculations
    # FOV is centered at X=0, Z=0, and Y=-fov_dist
    x_coords = (np.arange(geom["nx"]) - (geom["nx"] - 1) / 2.0) * geom["dx"]
    y_coords = (np.arange(geom["ny"]) - (geom["ny"] - 1) / 2.0) * geom["dy"] - geom["fov_dist"]
    z_coords = (np.arange(geom["nz"]) - (geom["nz"] - 1) / 2.0) * geom["dz"]
    
    # Create 3D meshgrids for the physical coordinates of every voxel
    Z_grid, Y_grid, X_grid = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    geom["coord_grids"] = np.stack((X_grid, Y_grid, Z_grid), axis=-1)
    
    # Extract detector centers (Shape: n_det x 3)
    det_data = det_raw[1:1 + geom["n_det"] * 12].reshape(-1, 12)
    geom["det_centers"] = det_data[:, 0:3] # [X, Y, Z]

    return geom

def initialize_hdf5_files(out_dir, n_det, n_voxels):
    """Creates the output HDF5 files for masks and properties."""
    os.makedirs(out_dir, exist_ok=True)
    
    masks_path = os.path.join(out_dir, "beams_masks_3d.hdf5")
    props_path = os.path.join(out_dir, "beams_properties_3d.hdf5")
    
    f_masks = h5py.File(masks_path, "w")
    # Store a single uint8 combined mask per detector (0=background, 1=beam1, 2=beam2...)
    dset_masks = f_masks.create_dataset(
        "combined_masks", shape=(n_det, n_voxels), 
        dtype="uint8", chunks=(10, n_voxels), compression="gzip"
    )
    dset_masks.attrs["Description"] = "Flattened 3D beam combined masks (0=bg, >0=beam_id)"

    f_props = h5py.File(props_path, "w")
    dset_props = f_props.create_dataset(
        "beam_properties", shape=(0, 12), maxshape=(None, 12), 
        dtype="float32", chunks=(500, 12)
    )
    header =[
        "detector_id", "beam_id", 
        "polar_angle_theta (rad)", "azimuthal_angle_phi (rad)", 
        "FWHM_major (mm)", "FWHM_minor (mm)", 
        "center_X", "center_Y", "center_Z", 
        "absolute_sensitivity", "relative_sensitivity", "voxel_count"
    ]
    dset_props.attrs["Header"] = np.array(header, dtype=h5py.string_dtype(encoding="utf-8"))
    
    return f_masks, dset_masks, f_props, dset_props

def extract_fwhm_3d_pca(half_max_voxels, beam_vector):
    """
    Computes elliptical FWHM of a 3D beam using PCA on the half-max cross-section.
    """
    if len(half_max_voxels) < 3:
        return 0.0, 0.0
        
    # Center the coordinates
    centered = half_max_voxels - np.mean(half_max_voxels, axis=0)
    
    # Project voxels onto the 2D plane orthogonal to the 3D beam propagation vector
    v = beam_vector.reshape(3, 1)
    projection_matrix = np.eye(3) - (v @ v.T)
    projected_coords = centered @ projection_matrix.T
    
    # Compute covariance matrix of the cross-section
    cov_matrix = np.cov(projected_coords, rowvar=False)
    
    # Eigenvalues represent the variance along the major and minor axes
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0))[::-1] # Sort descending, guard <0
    
    # FWHM = 2.355 * Standard Deviation
    fwhm_major = 2.355 * np.sqrt(eigenvalues[0])
    fwhm_minor = 2.355 * np.sqrt(eigenvalues[1])
    
    return fwhm_major, fwhm_minor

def process_detector(det_id, ppdf_3d, geom, threshold=0.01):
    """Identifies individual beams inside a 3D PPDF and extracts their properties."""
    max_val = np.max(ppdf_3d)
    if max_val <= 0:
        return None, None
        
    # 1. Connected Component Analysis (Find the isolated 3D blobs)
    rel_ppdf = ppdf_3d / max_val
    binary_mask = rel_ppdf > threshold
    
    # scipy.ndimage.label groups contiguous True voxels into unique integers
    labeled_volume, num_beams = ndimage.label(binary_mask)
    
    if num_beams == 0:
        return None, None

    props_list = []
    det_center = geom["det_centers"][det_id]
    
    # 2. Extract properties for each found beam
    for beam_id in range(1, num_beams + 1):
        beam_mask = (labeled_volume == beam_id)
        voxel_count = np.sum(beam_mask)
        
        # Sensitivities
        abs_sens = np.sum(ppdf_3d[beam_mask])
        rel_sens = np.mean(rel_ppdf[beam_mask])
        
        # Weighted Center of Mass (in 3D physical coordinates)
        weights = ppdf_3d[beam_mask]
        coords = geom["coord_grids"][beam_mask]
        weighted_center = np.average(coords, axis=0, weights=weights) # [X, Y, Z]
        
        # Beam Direction Vector & Angles
        beam_vector = weighted_center - det_center
        beam_vector_norm = beam_vector / np.linalg.norm(beam_vector)
        
        # Assuming Y is depth axis. Polar(theta) from Y-axis. Azimuthal(phi) on X-Z plane.
        polar_theta = np.arccos(beam_vector_norm[1]) 
        azimuthal_phi = np.arctan2(beam_vector_norm[2], beam_vector_norm[0])
        azimuthal_phi += 2 * np.pi * (azimuthal_phi < 0) # Map to 0-2PI
        
        # FWHM via 3D PCA
        # Get voxels that are >= 50% of THIS beam's maximum intensity
        beam_max = np.max(ppdf_3d[beam_mask])
        half_max_mask = beam_mask & (ppdf_3d >= 0.5 * beam_max)
        half_max_voxels = geom["coord_grids"][half_max_mask]
        
        fwhm_major, fwhm_minor = extract_fwhm_3d_pca(half_max_voxels, beam_vector_norm)
        
        # Append to properties list
        props_list.append([
            det_id, beam_id, polar_theta, azimuthal_phi, 
            fwhm_major, fwhm_minor, 
            weighted_center[0], weighted_center[1], weighted_center[2], 
            abs_sens, rel_sens, voxel_count
        ])
        
    flattened_mask = labeled_volume.flatten().astype(np.uint8)
    return flattened_mask, np.array(props_list, dtype=np.float32)

def main():
    print("--- Starting 3D Beam Extraction Task ---")
    start_time = time.time()
    
    geom = load_system_geometry()
    nx, ny, nz = geom["nx"], geom["ny"], geom["nz"]
    n_det, n_rot = geom["n_det"], geom["n_rot"]
    n_voxels = nx * ny * nz
    
    print(f"Geometry Loaded: Grid {nx}x{ny}x{nz}, {n_det} Detectors.")
    
    sysmat_files = glob.glob("*.sysmat")
    if not sysmat_files:
        print("ERROR: No .sysmat file found!")
        sys.exit(1)
        
    filename = sysmat_files[0]
    print(f"Streaming System Matrix: {filename}")
    
    # Memory mapping avoids loading the multi-GB file into RAM
    sysmat = np.memmap(filename, dtype='float32', mode='r', shape=(n_rot, n_det, n_voxels))
    
    out_dir = "beam_analysis_outputs"
    f_masks, dset_masks, f_props, dset_props = initialize_hdf5_files(out_dir, n_det, n_voxels)

    # Note: Assuming rotation 0 for analysis. Add a rotation loop if needed.
    rotation_idx = 0 
    
    for det_id in range(n_det):
        # Extract the 3D volume for this detector
        ppdf_3d = sysmat[rotation_idx, det_id, :].reshape(nx, ny, nz)
        
        # Find beams and calculate properties
        combined_mask, props_array = process_detector(det_id, ppdf_3d, geom)
        
        if combined_mask is not None:
            # Save Mask
            dset_masks[det_id, :] = combined_mask
            
            # Save Properties
            n_new_rows = props_array.shape[0]
            curr_len = dset_props.shape[0]
            dset_props.resize(curr_len + n_new_rows, axis=0)
            dset_props[curr_len:] = props_array
            
        if (det_id + 1) % 200 == 0:
            print(f"  ... processed {det_id + 1}/{n_det} detector units.")

    f_masks.close()
    f_props.close()
    
    duration = time.time() - start_time
    print(f"--- Finished 3D Beam Extraction in {duration:.2f} seconds ---")
    print(f"Outputs saved to directory: ./{out_dir}/")

if __name__ == "__main__":
    main()
'''

import sys
import os
import glob
import h5py
import numpy as np
from scipy.ndimage import label
from scipy.linalg import eigh

def load_system_parameters():
    """Dynamically loads scanner geometry from the .dat files."""
    img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
    det_raw = np.fromfile("Params_Detector.dat", dtype=np.float32)
    
    # Image/FOV Params
    nx, ny, nz = int(img_raw[0]), int(img_raw[1]), int(img_raw[2])
    dx, dy, dz = img_raw[3], img_raw[4], img_raw[5]
    num_rot = int(img_raw[6])
    
    # Calculate exact 3D coordinates for every voxel
    # Based on GPUPTS center coordinates logic
    shift_x, shift_y, shift_z = img_raw[8], img_raw[9], img_raw[10]
    fov_dist = img_raw[11]
    
    cx = (np.arange(nx) - (nx - 1) / 2.0) * dx + shift_x
    #cy = (np.arange(ny) - (ny - 1) / 2.0) * dy - fov_dist - (ny * dy) / 2.0 + shift_y
    cy = (np.arange(ny) - (ny - 1) / 2.0) * dy + shift_y
    cz = (np.arange(nz) - (nz - 1) / 2.0) * dz + shift_z
    
    # Create 3D coordinate grids
    Z_grid, Y_grid, X_grid = np.meshgrid(cz, cy, cx, indexing='ij')

    # Detector Params
    num_det = int(det_raw[0])
    det_data = det_raw[1:1 + num_det*12].reshape(-1, 12)
    det_centers = det_data[:, 0:3].copy() # X, Y, Z
    det_centers[:, 1] += img_raw[11]

    return (nx, ny, nz), (X_grid, Y_grid, Z_grid), num_rot, det_centers

def calculate_3d_fwhm(X, Y, Z, weights):
    """
    Calculates the transverse FWHM of a 3D blob using Principal Component Analysis.
    Returns the Major and Minor FWHM cross-sections of the beam.
    """
    if len(weights) < 4:
        # Too few voxels to form a reliable 3D covariance matrix
        return 0.0, 0.0
    
    # 1. Calculate Center of Mass
    sum_w = np.sum(weights)
    wx = np.sum(X * weights) / sum_w
    wy = np.sum(Y * weights) / sum_w
    wz = np.sum(Z * weights) / sum_w

    # 2. Build Spatial Covariance Matrix
    dx, dy, dz = X - wx, Y - wy, Z - wz
    cov = np.zeros((3, 3))
    cov[0, 0] = np.sum(weights * dx * dx) / sum_w
    cov[1, 1] = np.sum(weights * dy * dy) / sum_w
    cov[2, 2] = np.sum(weights * dz * dz) / sum_w
    cov[0, 1] = cov[1, 0] = np.sum(weights * dx * dy) / sum_w
    cov[0, 2] = cov[2, 0] = np.sum(weights * dx * dz) / sum_w
    cov[1, 2] = cov[2, 1] = np.sum(weights * dy * dz) / sum_w

    # 3. Eigenvalues represent the variance along the principal axes
    eigenvalues = eigh(cov, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1] # Sort descending
    
    # The largest eigenvalue is the longitudinal beam length. 
    # The two smaller eigenvalues are the transverse beam widths (cross section).
    # FWHM = 2.355 * standard_deviation = 2.355 * sqrt(variance)
    fwhm_major = 2.355 * np.sqrt(max(eigenvalues[1], 0))
    fwhm_minor = 2.355 * np.sqrt(max(eigenvalues[2], 0))
    
    return fwhm_major, fwhm_minor

def process_all_rotations():
    print("--- Starting 3D Beam Extraction Pipeline ---")
    
    out_dir = "beam_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Parameters and Coordinates
    grid_shape, (X_grid, Y_grid, Z_grid), num_rot, det_centers = load_system_parameters()
    nx, ny, nz = grid_shape
    num_det = det_centers.shape[0]
    n_pixels_total = nx * ny * nz
    
    print(f"Loaded System: {num_rot} Rotations, {num_det} Detectors.")
    print(f"FOV Grid: {nx}x{ny}x{nz} ({n_pixels_total} voxels)")

    # 2. Locate the SysMat file
    sysmat_files = glob.glob("*.sysmat")
    if not sysmat_files:
        print("Error: No .sysmat file found!")
        sys.exit(1)
    
    print(f"Loading System Matrix: {sysmat_files[0]}")
    data = np.memmap(sysmat_files[0], dtype='float32', mode='r', shape=(num_rot, num_det, n_pixels_total))

    # 3. Thresholds
    RELATIVE_THRESHOLD = 0.01  # 1% of max (Identifies the beam footprint)
    FWHM_THRESHOLD = 0.50      # 50% of max (Used to calculate beam width)
    ABSOLUTE_FLOOR = 1e-6      # Ignore noise detectors
    
    # 4. Loop over every Rotation (Layout)
    for rot_idx in range(num_rot):
        print(f"\n--- Processing Rotation {rot_idx+1}/{num_rot} ---")
        
        # Initialize HDF5 files for this rotation
        prop_file = h5py.File(os.path.join(out_dir, f"beams_properties_rot_{rot_idx:03d}.hdf5"), 'w')
        mask_file = h5py.File(os.path.join(out_dir, f"beams_masks_rot_{rot_idx:03d}.hdf5"), 'w')
        
        # Datasets (Using gzip compression to save massive amounts of disk space for masks)
        ds_props = prop_file.create_dataset("beam_properties", shape=(0, 13), maxshape=(None, 13), chunks=(1000, 13))
        ds_masks = mask_file.create_dataset("beam_mask", shape=(0, n_pixels_total), maxshape=(None, n_pixels_total), 
                                            chunks=(100, n_pixels_total), dtype='uint16', compression="gzip")
        
        ds_props.attrs["Header"] = np.array([
            "rotation_id", "detector_id", "beam_id", 
            "polar_angle_theta", "azimuthal_angle_phi", 
            "fwhm_major_mm", "fwhm_minor_mm", 
            "weight_center_x", "weight_center_y", "weight_center_z",
            "absolute_sensitivity", "relative_sensitivity", "num_voxels"
        ], dtype='S')

        all_props = []
        all_masks =[]

        # 5. Process every detector in this rotation
        for det_idx in range(num_det):
            # Extract 3D volume for this detector
            ppdf_1d = data[rot_idx, det_idx, :]
            max_val = np.max(ppdf_1d)
            
            if max_val < ABSOLUTE_FLOOR:
                continue # Dead detector / no signal

            ppdf_3d = ppdf_1d.reshape((nz, ny, nx))
            
            # Connected Components (Find isolated 3D blobs)
            binary_footprint = ppdf_3d > (max_val * RELATIVE_THRESHOLD)
            labeled_blobs, num_beams = label(binary_footprint)
            
            if num_beams == 0:
                continue

            # Create a combined mask array for this detector (0 = background, 1 = beam1, 2 = beam2...)
            combined_mask_1d = labeled_blobs.flatten()
            
            det_x, det_y, det_z = det_centers[det_idx]

            for beam_id in range(1, num_beams + 1):
                # Isolate specific beam
                beam_mask_3d = (labeled_blobs == beam_id)
                beam_voxels = ppdf_3d[beam_mask_3d]
                
                num_voxels = len(beam_voxels)
                if num_voxels < 3: 
                    continue # Skip tiny noise artifacts

                # Sensitivity
                beam_max = np.max(beam_voxels)
                abs_sens = np.sum(beam_voxels)
                rel_sens = np.mean(beam_voxels / max_val)
                
                # Center of Mass
                wx = np.average(X_grid[beam_mask_3d], weights=beam_voxels)
                wy = np.average(Y_grid[beam_mask_3d], weights=beam_voxels)
                wz = np.average(Z_grid[beam_mask_3d], weights=beam_voxels)
                
                # 3D Angles relative to detector face
                vec_x, vec_y, vec_z = wx - det_x, wy - det_y, wz - det_z
                r = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
                
                # Polar (theta): Angle from the Y depth-axis. Azimuthal (phi): Angle in X-Z transverse plane
                polar_theta = np.arccos(vec_y / r) if r > 0 else 0
                azimuthal_phi = np.arctan2(vec_z, vec_x)
                
                # FWHM (Using PCA on the 50% core of the beam)
                core_mask = beam_mask_3d & (ppdf_3d > (beam_max * FWHM_THRESHOLD))
                fwhm_major, fwhm_minor = calculate_3d_fwhm(X_grid[core_mask], Y_grid[core_mask], Z_grid[core_mask], ppdf_3d[core_mask])

                # Append Properties
                all_props.append([
                    rot_idx, det_idx, beam_id, 
                    polar_theta, azimuthal_phi, 
                    fwhm_major, fwhm_minor, 
                    wx, wy, wz, 
                    abs_sens, rel_sens, num_voxels
                ])

            # Append the full 1D mask for this detector
            all_masks.append(combined_mask_1d)
            
            # Periodically write to disk to save RAM
            # if len(all_masks) >= 200 or det_idx == num_det - 1:
            #     if all_props:
            #         ds_props.resize(ds_props.shape[0] + len(all_props), axis=0)
            #         ds_props[-len(all_props):] = np.array(all_props)
                    
            #         ds_masks.resize(ds_masks.shape[0] + len(all_masks), axis=0)
            #         ds_masks[-len(all_masks):] = np.vstack(all_masks)
                    
            #         all_props.clear()
            #         all_masks.clear()
            if len(all_masks) >= 200 or det_idx == num_det - 1:
                if all_masks:
                    ds_masks.resize(ds_masks.shape[0] + len(all_masks), axis=0)
                    ds_masks[-len(all_masks):] = np.vstack(all_masks)
                    all_masks.clear()
                if all_props:
                    ds_props.resize(ds_props.shape[0] + len(all_props), axis=0)
                    ds_props[-len(all_props):] = np.array(all_props)
                    all_props.clear()

            if (det_idx + 1) % 500 == 0:
                print(f"  ... processed {det_idx + 1}/{num_det} detectors.")

        prop_file.close()
        mask_file.close()
        print(f"Rotation {rot_idx+1} complete. Files saved in '{out_dir}/'.")

    print("\n--- All Rotations Processed Successfully ---")

if __name__ == "__main__":
    process_all_rotations()