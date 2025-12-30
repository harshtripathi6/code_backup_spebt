import os
import sys
import time
import h5py
from torch import device, arange, tensor, get_num_threads
import argparse

# Keep your existing imports
from scanner_modeling._raytracer_2d._local_functions import (
    ppdf_2d_local,
    reduced_edges_2d_local,
    sfov_properties,
    subdivision_grid_rectangle,
)
from scanner_modeling.geometry_2d import (
    fov_tensor_dict,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)

def calculate_ppdf_for_layout(layout_idx: int, run_id: str, work_dir: str):
    """
    Calculates PPDF using dynamic paths based on run_id.
    """
    start_time = time.time()
    
    # --- 1. Dynamic Path Setup ---
    # Construct the expected input filename based on the run_id from Module 1
    input_filename = f"scanner_layouts_{run_id}.tensor"
    input_path = os.path.join(work_dir, input_filename)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"--- Loading Geometry from: {input_filename} ---")
    
    # Load the layouts
    # Note: load_scanner_layouts usually expects (dir, filename)
    scanner_layouts, layouts_md5 = load_scanner_layouts(work_dir, input_filename)
    
    # --- 2. Standard Setup ---
    default_device = device("cpu")
    
    # Check layout validity
    n_layouts_total = len(scanner_layouts)
    if not (0 <= layout_idx < n_layouts_total):
        print(f"Error: Invalid layout_idx {layout_idx}. Total layouts: {n_layouts_total}.")
        sys.exit(1)

    # Physics/Grid params (Keep as per your original script)
    mu_dict = tensor([3.5, 0.475], device=default_device)
    fov_dict = fov_tensor_dict((280, 280), (70, 70), (0.0, 0.0), (3, 3))
    crystal_n_subs = (3, 3)
    
    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())
    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )
    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    print(f"PyTorch using {get_num_threads()} threads.")

    # --- 3. Load Geometry for specific layout ---
    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals_total = crystal_objects_vertices.shape[0]
    crystal_idx_tensor = arange(n_crystals_total)
    n_crystals = int(crystal_idx_tensor.shape[0])
    
    # --- 4. Process & Save ---
    # Construct output filename using run_id so results are distinguishable
    out_filename = f"ppdf_{run_id}_layout_{layout_idx:03d}.hdf5"
    h5_file_path = os.path.join(work_dir, out_filename)
    
    print(f"Calculating PPDFs for {n_crystals} crystals...")
    
    with h5py.File(h5_file_path, "w") as h5file:
        ppdf_dataset = h5file.create_dataset("ppdfs", (n_crystals, fov_n_pxs), dtype="f")

        for dataset_idx, crystal_idx_tensor_val in enumerate(crystal_idx_tensor):
            crystal_idx = int(crystal_idx_tensor_val.item())
            
            reduced_crystal_edges_sfovs = []
            reduced_plate_edges_sfovs = []
            for sfov_idx in range(n_sfov):
                reduced_plate_edges, reduced_crystal_edges = reduced_edges_2d_local(
                    sfov_idx, crystal_idx, sfov_corners_batch,
                    plate_objects_vertices, plate_objects_edges,
                    crystal_objects_vertices, crystal_objects_edges,
                    default_device,
                )
                reduced_crystal_edges_sfovs.append(reduced_crystal_edges)
                reduced_plate_edges_sfovs.append(reduced_plate_edges)

            for sfov_idx in range(n_sfov):
                ppdf_slice = ppdf_2d_local(
                    sfov_idx, crystal_idx, sfov_pixels_batch,
                    crystal_objects_vertices, reduced_plate_edges_sfovs[sfov_idx],
                    reduced_crystal_edges_sfovs[sfov_idx], subdivision_grid,
                    mu_dict, default_device,
                )
                ppdf_dataset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

    end_time = time.time()
    print(f"--- Finished Layout {layout_idx} in {end_time - start_time:.2f}s ---")
    print(f"Results saved to {h5_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Positional arg for layout index (matches your subprocess call)
    parser.add_argument("layout_idx", type=int, help="Index of layout to process")
    # Named args for integration
    parser.add_argument("--run_id", type=str, required=True, help="Unique Run ID from driver")
    parser.add_argument("--work_dir", type=str, default=".", help="Directory for I/O")
    
    args = parser.parse_args()

    calculate_ppdf_for_layout(args.layout_idx, args.run_id, args.work_dir)