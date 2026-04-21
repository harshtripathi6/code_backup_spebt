import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def analyze_3d_final():
    # 1. DYNAMICALLY LOAD PARAMETERS
    try:
        img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
        det_raw = np.fromfile("Params_Detector.dat", dtype=np.float32)
    except FileNotFoundError:
        print("Error: Parameter files not found. Run this in the same directory as your .dat files.")
        return

    nx, ny, nz = int(img_raw[0]), int(img_raw[1]), int(img_raw[2])
    dx, dy, dz = img_raw[3], img_raw[4], img_raw[5]
    num_rot = int(img_raw[6])
    num_det = int(det_raw[0])

    fov_wx, fov_wy, fov_wz = nx * dx, ny * dy, nz * dz

    print(f"Loaded System Specs: Grid={nx}x{ny}x{nz}, Voxel Size={dx}x{dy}x{dz}mm")
    print(f"Rotations: {num_rot}, Detectors: {num_det}")

    # 2. AUTOMATICALLY FIND THE SYSMAT FILE
    sysmat_files = glob.glob("*.sysmat")
    if not sysmat_files:
        print("Error: No .sysmat file found in the current directory.")
        return
    filename = sysmat_files[0]
    print(f"Reading System Matrix: {filename}")

    # 3. OPEN DATA (Memory Mapping for Efficiency)
    try:
        data = np.memmap(filename, dtype='float32', mode='r', 
                         shape=(num_rot, num_det, nx * ny * nz))
    except ValueError as e:
        print(f"Shape mismatch error: {e}")
        print("This usually means the generated .sysmat file does not match the Params_*.dat files.")
        return

    # 4. CREATE FIGURE WITH CONSTRAINED LAYOUT
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)

    # Automatically calculate spatial extents
    extent_xz =[-fov_wx/2, fov_wx/2, -fov_wz/2, fov_wz/2]

    # Select a detector in the middle of the High-Res Array (Layer 4)
    # Layer 4 starts after 3 * 32 * 16 = 1536. 
    target_det = min(3584, num_det - 1) 

    # Extract 3D PPDF for the target detector (at Rotation 0)
    ppdf_1d = data[0, target_det, :]
    # Note: If image looks rotated/scrambled, C++ might use (nz, ny, nx) memory order.
    ppdf_3d = ppdf_1d.reshape((nx, ny, nz)) 

    # ---------------------------------------------------------
    # PANEL 1: PPDF (Maximum Intensity Projection - X-Z Plane)
    # ---------------------------------------------------------
    # Project along Y (depth) to see the X-Z face
    ppdf_mip = np.max(ppdf_3d, axis=1) 
    
    im1 = axes[0].imshow(ppdf_mip.T, cmap='viridis', origin='lower', extent=extent_xz)
    axes[0].set_title(f"Detector {target_det}: PPDF (X-Z MIP)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("X Position (mm)")
    axes[0].set_ylabel("Z Position (mm)")
    fig.colorbar(im1, ax=axes[0], label="Probability")

    # ---------------------------------------------------------
    # PANEL 2: PPDF 2D SLICE (For Sim2Real Transfer Learning Context)
    # ---------------------------------------------------------
    # The paper uses slices like this to train their neural network
    target_slice_y = ny // 2  # Middle depth slice
    ppdf_slice = ppdf_3d[:, target_slice_y, :]
    
    im2 = axes[1].imshow(ppdf_slice.T, cmap='magma', origin='lower', extent=extent_xz)
    axes[1].set_title(f"Detector {target_det}: PPDF (Slice Y={target_slice_y})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("X Position (mm)")
    axes[1].set_ylabel("Z Position (mm)")
    fig.colorbar(im2, ax=axes[1], label="Probability")

    # ---------------------------------------------------------
    # PANEL 3: SENSITIVITY MAP (Total System View)
    # ---------------------------------------------------------
    print("Calculating Sensitivity Map from all detectors and rotations... (This might take a moment)")
    # Sum across all rotations (axis=0) and detectors (axis=1)
    sensitivity_1d = np.sum(data, axis=(0, 1))
    sensitivity_3d = sensitivity_1d.reshape(nx, ny, nz)
    sens_slice = sensitivity_3d[:, target_slice_y, :] 
    
    im3 = axes[2].imshow(sens_slice.T, cmap='inferno', origin='lower', extent=extent_xz)
    axes[2].set_title(f"Total System Sensitivity Map (Slice Y={target_slice_y})", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("X Position (mm)")
    axes[2].set_ylabel("Z Position (mm)")
    fig.colorbar(im3, ax=axes[2], label="Sensitivity")

    output_filename = f"verified_system_analysis_{nx}x{ny}x{nz}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Image saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_3d_final()