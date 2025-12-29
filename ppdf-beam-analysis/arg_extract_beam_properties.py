import sys
import os
from torch import cat, tensor, arange

from beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,
    get_beams_angle_radian,
    get_beams_basic_properties,
    sample_ppdf_on_arc_2d_local,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from ppdf_io import load_ppdfs_data_from_hdf5
from beam_property_io import (
    initialize_beam_properties_hdf5,
    append_to_hdf5_dataset,
    stack_beams_properties,
)

if __name__ == "__main__":
    # --- MODIFICATION: Get layout_idx from a command-line argument ---
    if len(sys.argv) != 2:
        print("Usage: python extract_beam_properties_task.py <layout_idx>")
        sys.exit(1)
    try:
        layout_idx = int(sys.argv[1])
    except ValueError:
        print(f"Error: <layout_idx> must be an integer. Received: {sys.argv[1]}")
        sys.exit(1)

    print(f"--- Starting beam property extraction for layout index: {layout_idx} ---")

    # --- Define paths (ensure they are correct relative to execution directory) ---
    scanner_layouts_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon"
    scanner_layouts_filename = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/scanner_layouts_71e7a1c93e5874311d77fd36727cb194_rot15_trans1x1_step0x0.tensor"
    ppdfs_dataset_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/ppdfs_15rots_3mm_aperture_displaced_24ph"
    out_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/outputs_3mm_aperture_displaced_24ph"
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data and configuration ---
    scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
        scanner_layouts_dir, scanner_layouts_filename
    )
    fov_dict = fov_tensor_dict(
        n_pixels=(280, 280),
        mm_per_pixel=(0.25, 0.25),
        center_coordinates=(0.0, 0.0),
    )
    
    # --- Initialize HDF5 file for this specific layout --- 
    out_hdf5_filename = f"beams_properties_configuration_{layout_idx:03d}.hdf5"
    out_hdf5_file, beam_properties_dataset = initialize_beam_properties_hdf5(
        out_hdf5_filename, out_dir
    )
    print(f"Output will be saved to: {os.path.join(out_dir, out_hdf5_filename)}")

    # Load scanner geometry and PPDFs for the specific layout
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        int(layout_idx), scanner_layouts_data
    )
    ppdfs_hdf5_filename = f"scanner_layouts_step0x0_layout_{layout_idx:03d}_3x3_subvoxels.hdf5"
    ppdfs = load_ppdfs_data_from_hdf5(
        ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_dict
    )

    # --- Prepare for processing ---
    detector_unit_centers = detector_units_vertices.mean(dim=1)
    fov_corners = (
        tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        * fov_dict["size in mm"]
        * 0.5
    )
    hull_points_batch = cat(
        (
            fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
            detector_unit_centers.unsqueeze(1),
        ),
        dim=1,
    )
    hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

    n_detector_units = int(detector_units_vertices.shape[0])
    detector_units_sequence = arange(0, n_detector_units)
    print(f"Processing {n_detector_units} detector units for layout {layout_idx}...")

    # --- Main processing loop for detector units (the original inner loop) ---
    for detector_unit_idx in detector_units_sequence:
        ppdf_data_2d = ppdfs[detector_unit_idx].view(
            int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
        )
        hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

        (sampled_ppdf, sampling_rads, sampling_points) = sample_ppdf_on_arc_2d_local(
            ppdf_data_2d,
            detector_unit_centers[detector_unit_idx],
            hull_2d,
            fov_dict,
        )

        beams_boundaries = beams_boundaries_radians(
            sampled_ppdf, sampling_rads, threshold=0.01
        )

        fov_points_xy = pixels_coordinates(fov_dict)
        fov_points_rads = pixels_to_detector_unit_rads(
            fov_points_xy, detector_unit_centers[detector_unit_idx]
        )
        beams_masks = get_beams_masks(fov_points_rads, beams_boundaries)
        
        if beams_masks.shape[0] == 0: # Skip if no beams were found
            continue

        beams_weighted_centers = get_beams_weighted_center(
            beams_masks, fov_points_xy, ppdf_data_2d
        )
        (beams_fwhm, _, _, _) = get_beam_width(
            beams_weighted_centers,
            detector_unit_centers[detector_unit_idx],
            beams_masks,
            ppdf_data_2d,
            fov_dict,
        )
        beams_angle = get_beams_angle_radian(
            beams_weighted_centers, detector_unit_centers[detector_unit_idx]
        )
        (
            beams_sizes,
            beams_relative_sensitivity,
            beams_absolute_sensitivity,
        ) = get_beams_basic_properties(beams_masks, ppdf_data_2d, fov_points_xy)

        stacked_beams_properties = stack_beams_properties(
            int(layout_idx),
            int(detector_unit_idx),
            angles=beams_angle,
            fwhms=beams_fwhm,
            sizes=beams_sizes,
            relative_sensitivities=beams_relative_sensitivity,
            absolute_sensitivities=beams_absolute_sensitivity,
            weighted_centers=beams_weighted_centers,
        )

        if stacked_beams_properties.numel():
            append_to_hdf5_dataset(beam_properties_dataset, stacked_beams_properties)
        
        # Log progress periodically instead of using a progress bar
        if (detector_unit_idx + 1) % 200 == 0:
            print(f"  ... processed {detector_unit_idx + 1}/{n_detector_units} detector units.")

    # --- Finalize ---
    out_hdf5_file.close()
    print(f"\nBeam properties for layout {layout_idx} saved successfully.")
    print(f"--- Finished job for layout index: {layout_idx} ---")