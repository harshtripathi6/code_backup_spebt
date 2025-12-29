
import os
import torch
from torch import save as torch_save, cat, Tensor # Added cat, Tensor

# Assuming helper.py is in the same directory or accessible via PYTHONPATH
try:
    from helper import (
        generate_mph_spect_geometry, # This is for the "Professor's MPH" or your target MPH
        generate_md5_from_tensors,
        plot_polygons_from_vertices_2d_mpl,
        rotate_and_repeat_4gon, # Make sure this is still needed by generate_mph_spect_geometry
        # Transformation functions from your other script (now should be in helper.py)
        positions_parameters,
        transform_to_positions_2d_batch,
        OutDataDict # Type hint for the output dictionary
    )
except ImportError as e:
    print(f"Error importing from helper.py: {e}")
    print("Ensure helper.py contains all necessary functions, including transformation functions.")
    exit()

if __name__ == "__main__":
    # --- 1. Define Base MPH-SPECT Configuration Parameters ---
    cfg_pinhole_diameter_mm = 3.0
    cfg_n_pinholes = 24
    cfg_collimator_ring_radius_mm = 215.0
    cfg_pinhole_to_detector_distance_mm = 542.0
    cfg_scint_tangential_mm = 3.5
    cfg_scint_radial_thickness_mm = 6.0
    cfg_pinhole_channel_length_mm = 20.0
    cfg_detector_outward_ratio = 0.20          # 50% detectors displaced
    cfg_detector_outward_disp_mm = 10  # random per-detector displacement in [0,2] mm
    cfg_detector_outward_seed = 123           # reproducible random selection

    print(f"Generating base MPH-SPECT geometry...")
    try:
        # Generate the BASE (untransformed) scanner layout
        base_detector_units, base_plate_segments = generate_mph_spect_geometry( # Use the correct generation function
            pinhole_diameter_mm=cfg_pinhole_diameter_mm,
            n_pinholes=cfg_n_pinholes,
            collimator_ring_radius_mm=cfg_collimator_ring_radius_mm,
            pinhole_to_detector_distance_mm=cfg_pinhole_to_detector_distance_mm,
            scint_tangential_mm=cfg_scint_tangential_mm,
            scint_radial_thickness_mm=cfg_scint_radial_thickness_mm,
            pinhole_channel_length_mm=cfg_pinhole_channel_length_mm,
            detector_outward_ratio=cfg_detector_outward_ratio,
            detector_outward_displacement_mm=cfg_detector_outward_disp_mm,
            detector_outward_seed=cfg_detector_outward_seed,
        )
    except ValueError as e:
        print(f"Error during base geometry generation: {e}")
        exit()
        
    print(f"\nGenerated base detector units: {base_detector_units.shape}")
    print(f"Generated base plate segments: {base_plate_segments.shape}")
    base_scanner_md5 = generate_md5_from_tensors(base_detector_units, base_plate_segments)
    print(f"MD5 of base scanner: {base_scanner_md5}")

    # --- 2. Define Motion Parameters for Transformations ---
    # These should match the schemes you intend to simulate (e.g., from the papers)
    # Example: 24 rotational views, and a 1x1 translation grid (i.e., no extra translation beyond centering)
    # Or use parameters like R5 x T4 if you map them correctly.
    n_rotations_for_motion = 15  # Number of distinct angular positions for the scanner, 20 rotations to duplicate the observation for gibbs artifact as seen by sid/avantika
    n_shifts_grid_for_motion = [1, 1]  # e.g., [1,1] for rotation only, [2,2] for a 2x2 translation grid
    shift_step_mm_for_motion = [0, 0]  # Step size in mm for translations (0 if n_shifts is [1,1])


    # For this example, we'll use the existing `positions_parameters` with even angular steps.
    print(f"\nDefining motion: {n_rotations_for_motion} rotations, {n_shifts_grid_for_motion} translations, step {shift_step_mm_for_motion}mm")

    # --- 3. Generate Transformation Positions ---
    motion_positions = positions_parameters(
        n_rotations_for_motion,
        360.0 / n_rotations_for_motion,   # angle_step_deg
        #20.0/n_rotations_for_motion,
        n_shifts_grid_for_motion,
        shift_step_mm_for_motion
    )

    n_total_positions = motion_positions.shape[0]
    print(f"Total number of scanner positions to generate: {n_total_positions}")

    # --- 4. Transform Components for Each Position ---
    num_det_vertices = base_detector_units.shape[0] * base_detector_units.shape[1]
    num_plate_vertices = base_plate_segments.shape[0] * base_plate_segments.shape[1]

    all_base_vertices = cat(
        (base_detector_units.reshape(-1, 2),
         base_plate_segments.reshape(-1, 2)),
        dim=0
    )

    transformed_all_vertices_batch = transform_to_positions_2d_batch(
        motion_positions,
        all_base_vertices
    )

    # --- 5. Prepare Data for Saving ---
    out_data: OutDataDict = { # type: ignore
        "scanner MD5": base_scanner_md5, # MD5 of the *base* untransformed scanner
        "motion_parameters": {
            "n_rotational_steps_defined": n_rotations_for_motion,
            "n_translational_shifts_grid": n_shifts_grid_for_motion,
            "translational_step_size_mm": shift_step_mm_for_motion,
            "generated_n_positions": n_total_positions
        },
        "layouts": {}, # This will store each transformed layout
        "design_perturbation": {
            "detector_outward_ratio": cfg_detector_outward_ratio,
            "detector_outward_displacement_mm": cfg_detector_outward_disp_mm,
            "detector_outward_seed": cfg_detector_outward_seed
        }
    }
    layouts_dict = {} # Temporary dictionary for populating out_data["layouts"]

    for i in range(n_total_positions):
        current_pos_transformed_vertices = transformed_all_vertices_batch[i]
        
        transformed_detector_units = current_pos_transformed_vertices[:num_det_vertices, :].reshape(
            base_detector_units.shape[0], base_detector_units.shape[1], 2
        )
        transformed_plate_segments = current_pos_transformed_vertices[num_det_vertices:, :].reshape(
            base_plate_segments.shape[0], base_plate_segments.shape[1], 2
        )
        
        # The key "position" inside the layout dict stored the [angle, x, y] parameters
        # The key for the layout itself was f"position {i:03d}"
        layout_key = f"position {i:03d}" # Matching your transformation script's output format
        layouts_dict[layout_key] = {
            "position": motion_positions[i], # Store the [angle_rad, x_mm, y_mm]
            "detector units": transformed_detector_units,
            "plate segments": transformed_plate_segments,
        }
    out_data["layouts"] = layouts_dict


    # --- 6. Save All Transformed Layouts ---
    # Let's create an ID based on the base scanner and motion parameters.
    motion_id_str_parts = [
        f"rot{n_rotations_for_motion}",
        f"trans{n_shifts_grid_for_motion[0]}x{n_shifts_grid_for_motion[1]}",
        f"step{shift_step_mm_for_motion[0]}x{shift_step_mm_for_motion[1]}"
    ]
    motion_descriptor = "_".join(str(p).replace('.', 'p') for p in motion_id_str_parts) # Make filenames friendlier
    
    # Create a unique ID for the entire dataset of layouts
    # For simplicity, using the base MD5 and motion descriptor.
    # A more robust approach might hash a sample of the transformed data or all of it if feasible.
    transformed_layouts_unique_id = f"{base_scanner_md5}_{motion_descriptor}"
    
    # Naming convention similar to your transformation script output
    out_file_name = f"scanner_layouts_{transformed_layouts_unique_id}.tensor"
    
    print(f"\nSaving {n_total_positions} transformed MPH-SPECT layouts to:\n  {out_file_name}")
    torch_save(out_data, out_file_name)
    print("\nTransformed MPH-SPECT layouts saved successfully.")

    # --- Optional: Visualization (only for the BASE layout for brevity) ---
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(figsize=(12, 12))
    if base_detector_units.numel() > 0: # Check if tensor is not empty
        plot_polygons_from_vertices_2d_mpl(base_detector_units, ax, facecolor='lightblue', edgecolor='blue', alpha=0.7, label="Base Detectors")
    if base_plate_segments.numel() > 0: # Check if tensor is not empty
        plot_polygons_from_vertices_2d_mpl(base_plate_segments, ax, facecolor='gray', edgecolor='black', label="Base Collimator")
    

     # =====================================================================
    # === CHANGE IS HERE: Define and draw the 70mm circular FOV        ===
    # =====================================================================
    CIRCULAR_FOV_DIAMETER_MM = 70.0 
    circular_fov_radius_mm = CIRCULAR_FOV_DIAMETER_MM / 2.0
    # Use plt.Circle to create the circular patch
    circular_fov_patch = Circle(
        (0, 0), # Centered at the origin
        circular_fov_radius_mm,
        edgecolor='red', 
        facecolor='none', 
        linestyle='--', 
        linewidth=2,
        label=f'Effective FOV (D={CIRCULAR_FOV_DIAMETER_MM}mm)'
    )
    ax.add_patch(circular_fov_patch)

    coll_ring_circle = plt.Circle((0,0), cfg_collimator_ring_radius_mm, color='darkgrey', fill=False, linestyle=':', label=f'Collimator Ring (R={cfg_collimator_ring_radius_mm}mm)')
    det_inner_radius = cfg_collimator_ring_radius_mm + cfg_pinhole_to_detector_distance_mm
    det_ring_circle_inner = plt.Circle((0,0), det_inner_radius, color='cyan', fill=False, linestyle=':', label=f'Detector Inner (R={det_inner_radius}mm)')
    ax.add_artist(coll_ring_circle)
    ax.add_artist(det_ring_circle_inner)

    ax.set_aspect('equal', adjustable='box')
    max_coord_det = 0
    if base_detector_units.numel() > 0: max_coord_det = torch.abs(base_detector_units).max().item()
    max_coord_coll = 0
    if base_plate_segments.numel() > 0: max_coord_coll = torch.abs(base_plate_segments).max().item()

    plot_limit = max(max_coord_det, max_coord_coll, det_inner_radius + 50, circular_fov_radius_mm + 50) * 1.05    
    #plot_limit = max(max_coord_det, max_coord_coll, det_inner_radius + 50, SQUARE_FOV_WIDTH_MM/2 + 50) * 1.05
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Base 2D MPH-SPECT Geometry (Pinhole Dia: {cfg_pinhole_diameter_mm}mm)")
    ax.legend(fontsize='small')
    plt.grid(True)
    plt.savefig("mph_base_layout_with_transforms_generated.png")
    print("\nSaved visualization of the base layout to mph_base_layout_with_transforms_generated.png")
    # plt.show() # Uncomment to display if running interactively