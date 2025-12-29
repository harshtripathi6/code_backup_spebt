import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

def find_asci_histogram_overlap_and_plot(
    file_path_A: str,
    file_path_B: str,
    output_dir: str = "./",
    output_filename_h5: str = "overlap_mask.hdf5",
    output_filename_plot: str = "overlap_visualization.png",
    dataset_name: str = "asci_histogram",
    overlap_dataset_name: str = "intersection_mask"
) -> None:
    """
    Loads two ASCI histogram HDF5 files, computes the intersection of
    their non-zero regions, saves the boolean mask, and plots the results.
    """

    # --- 1. Load Data ---
    print(f"Loading data from: {file_path_A}")
    with h5py.File(file_path_A, 'r') as f:
        map_A = f[dataset_name][:]
    
    print(f"Loading data from: {file_path_B}")
    with h5py.File(file_path_B, 'r') as f:
        map_B = f[dataset_name][:]

    if map_A.shape != map_B.shape:
        raise ValueError(
            f"HDF5 datasets must have the same shape for comparison. "
            f"Map A shape: {map_A.shape}, Map B shape: {map_B.shape}"
        )
    
    # --- 2. Find Overlap (Boolean AND) ---
    print("Computing non-zero region overlap...")

    mask_A = map_A > 0
    mask_B = map_B > 0
    intersection_mask = mask_A & mask_B
    
    # --- 3. Create Masked Maps ---
    masked_map_A = map_A.copy()
    masked_map_B = map_B.copy()
    masked_map_A[~intersection_mask] = 0
    masked_map_B[~intersection_mask] = 0

    # --- 4. Save Mask and Data (HDF5) ---
    os.makedirs(output_dir, exist_ok=True)
    output_path_h5 = os.path.join(output_dir, output_filename_h5)

    print(f"Saving intersection mask and masked data to: {output_path_h5}")
    with h5py.File(output_path_h5, 'w') as f:
        f.create_dataset(overlap_dataset_name, data=intersection_mask, dtype='bool')
        f.create_dataset('masked_map_A', data=masked_map_A, dtype=map_A.dtype)
        f.create_dataset('masked_map_B', data=masked_map_B, dtype=map_B.dtype)

    # --- 5. Visualization (1x3 Grid) ---
    print("Creating visualization...")
    
    # Determine global max for synchronized color scale
    # Add a check in case both maps are all zero
    vmax_A = np.max(masked_map_A)
    vmax_B = np.max(masked_map_B)
    global_max_value = max(vmax_A, vmax_B)
    
    # Prevent matplotlib error if max value is 0
    if global_max_value == 0:
        global_max_value = 1 
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # --- Plot 1: Overlap Mask ---
    # 👇 MODIFICATION: Added aspect='auto'
    axes[0].imshow(intersection_mask, cmap='gray', aspect='auto')
    axes[0].set_title("1. Overlap Mask (Non-Zero in Both)")
    axes[0].set_xlabel("Angle Bin")
    axes[0].set_ylabel("Pixel Index")
    # This text line is already fixed (no $ chars)
    axes[0].text(0.5, -0.15, 'Boolean AND', 
                 transform=axes[0].transAxes, ha='center', fontsize=10)
    
    # --- Plot 2 & 3: Masked Magnitudes (Synchronized Scale) ---
    
    # Image 2: Values of Map A in the overlap region
    # 👇 MODIFICATION: Added aspect='auto'
    im2 = axes[1].imshow(masked_map_A, cmap='viridis', vmin=0, vmax=global_max_value, aspect='auto')
    axes[1].set_title(f"2. Values of {os.path.basename(file_path_A)}")
    axes[1].set_xlabel("Angle Bin")
    axes[1].set_ylabel("Pixel Index")

    # Image 3: Values of Map B in the overlap region
    # 👇 MODIFICATION: Added aspect='auto'
    im3 = axes[2].imshow(masked_map_B, cmap='viridis', vmin=0, vmax=global_max_value, aspect='auto')
    axes[2].set_title(f"3. Values of {os.path.basename(file_path_B)}")
    axes[2].set_xlabel("Angle Bin")
    axes[2].set_ylabel("Pixel Index")
    
    # Add a single color bar for both Image 2 and 3
    cbar = fig.colorbar(im3, ax=axes[1:], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Histogram Count (Synchronized Scale)')
    
    # Save the figure
    output_path_plot = os.path.join(output_dir, output_filename_plot)
    plt.savefig(output_path_plot, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to: {output_path_plot}")

    # --- 6. Report Summary (Optional) ---
    total_pixels = map_A.size
    overlap_count = np.sum(intersection_mask)
    overlap_percent = (overlap_count / total_pixels) * 100
    
    print("\n--- Summary ---")
    print(f"Total entries (pixels x angle bins): {total_pixels}")
    print(f"Non-zero entries in Map A: {np.sum(mask_A)}")
    print(f"Non-zero entries in Map B: {np.sum(mask_B)}")
    print(f"Overlapping (non-zero in both) entries: {overlap_count}")
    print(f"Overlap represents: {overlap_percent:.4f}% of the total map size.")
    print("-----------------")


if __name__ == '__main__':
    # --- USER CONFIGURATION ---
    DATA_DIR = '../../../data/sc_mph_36det_3mm_base_layout_rotated_10custom_rot/filtered_outputs/'
    
    # These paths look correct based on your last message
    FILE_A = DATA_DIR + 'mpxi_1/asci_histogram_00.hdf5' 
    FILE_B = DATA_DIR + 'mpxi_2/asci_histogram_00.hdf5' 
    
    OUTPUT_FOLDER = '../../../data/sc_mph_36det_3mm_base_layout_rotated_10custom_rot/intersection_results'
    OUTPUT_H5_FILE = 'M2_AND_M3_overlap.hdf5'
    OUTPUT_PLOT_FILE = 'M2_AND_M3_overlap_plot.png'
    # --------------------------

    try:
        find_asci_histogram_overlap_and_plot(
            file_path_A=FILE_A,
            file_path_B=FILE_B,
            output_dir=OUTPUT_FOLDER,
            output_filename_h5=OUTPUT_H5_FILE,
            output_filename_plot=OUTPUT_PLOT_FILE,
            dataset_name='asci_histogram'
        )
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find an input file. Please check your file paths.")
        print(e)
    except ValueError as e:
        print(f"\nERROR: Data shape mismatch.")
        print(e)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")