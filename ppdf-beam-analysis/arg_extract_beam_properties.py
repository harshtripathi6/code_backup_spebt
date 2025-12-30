import sys
import os
import argparse
import h5py
from torch import tensor, arange, cat

# --- Imports matching your environment ---
from beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    # The following are specific to Property Extraction
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
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("layout_idx", type=int, help="Layout index (e.g. 0)")
    parser.add_argument("--run_id", type=str, required=True, help="e.g. d1p0_disp0p0")
    parser.add_argument("--work_dir", type=str, required=True, help="Directory containing tensor/hdf5 files")
    args = parser.parse_args()

    layout_idx = args.layout_idx
    run_id = args.run_id
    work_dir = args.work_dir

    print(f"--- Starting Property Extraction for {run_id} (Layout {layout_idx}) ---")

    # --- 1. Define Paths ---
    # Input: Scanner Geometry
    scanner_layouts_filename = f"scanner_layouts_{run_id}.tensor"
    scanner_layouts_path = os.path.join(work_dir, scanner_layouts_filename)

    # Input: PPDFs
    ppdfs_hdf5_filename = f"ppdf_{run_id}_layout_{layout_idx:03d}.hdf5"
    ppdfs_path = os.path.join(work_dir, ppdfs_hdf5_filename)

    # Output: Beam Properties
    out_hdf5_filename = f"beams_properties_{run_id}_layout_{layout_idx:03d}.hdf5"

    # Verify inputs
    if not os.path.exists(scanner_layouts_path):
        print(f"Error: Layout file missing: {scanner_layouts_path}")
        sys.exit(1)
    if not os.path.exists(ppdfs_path):
        print(f"Error: PPDF file missing: {ppdfs_path}")
        sys.exit(1)

    # --- 2. Load Data and Config ---
    # Load Layouts
    scanner_layouts_data, _ = load_scanner_layouts(
        work_dir, scanner_layouts_filename
    )
    
    # Define FOV (Must match PPDF generation)
    fov_dict = fov_tensor_dict(
        n_pixels=(280, 280),
        mm_per_pixel=(0.25, 0.25),
        center_coordinates=(0.0, 0.0),
    )

    # Initialize Output HDF5
    out_hdf5_file, beam_properties_dataset = initialize_beam_properties_hdf5(
        out_hdf5_filename, work_dir
    )
    print(f"Output will be saved to: {os.path.join(work_dir, out_hdf5_filename)}")

    # Load Geometry for this layout
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        int(layout_idx), scanner_layouts_data
    )

    # Load PPDFs
    ppdfs = load_ppdfs_data_from_hdf5(
        work_dir, ppdfs_hdf5_filename, fov_dict
    )

    # --- 3. Pre-computation (Hulls & Centers) ---
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

    n_detector_units = detector_units_vertices.shape[0]
    detector_units_sequence = arange(0, n_detector_units)
    
    print(f"Processing {n_detector_units} detector units...")

    # --- 4. Main Loop ---
    for detector_unit_idx in detector_units_sequence:
        # Prepare Data
        ppdf_data_2d = ppdfs[detector_unit_idx].view(
            int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
        )
        hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

        # A. Sample PPDF on Arc
        (sampled_ppdf, sampling_rads, _) = sample_ppdf_on_arc_2d_local(
            ppdf_data_2d,
            detector_unit_centers[detector_unit_idx],
            hull_2d,
            fov_dict,
        )

        # B. Identify Beam Boundaries
        beams_boundaries = beams_boundaries_radians(
            sampled_ppdf, sampling_rads, threshold=0.01
        )

        fov_points_xy = pixels_coordinates(fov_dict)
        fov_points_rads = pixels_to_detector_unit_rads(
            fov_points_xy,
            detector_unit_centers[detector_unit_idx],
        )

        # C. Get Binary Masks (internally used for calculations)
        # Note: We do NOT combine/save these masks in this script
        beams_masks = get_beams_masks(fov_points_rads, beams_boundaries)
        
        if beams_masks.shape[0] == 0:
            continue

        # D. Calculate Properties (FWHM, Angle, etc.)
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

        # E. Stack and Save Properties
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

        if (detector_unit_idx + 1) % 100 == 0:
            print(f"  ... processed {detector_unit_idx + 1}/{n_detector_units}")

    # --- Finalize ---
    out_hdf5_file.close()
    print(f"--- Finished Property Extraction for {run_id} ---")