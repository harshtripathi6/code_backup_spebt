if __name__ == "__main__":
    import os
    import numpy as np

    # --- INPUT DIRECTORY ---
    # This is the full path to the folder where your .hdf5 files are located.
    topdir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/ppdfs/ppdfs_15rots_3mm_aperture_displaced_24ph"
    
    # --- FILENAME GENERATION ---
    
    file_idxs = np.arange(0, 15)
    fnames = ["scanner_layouts_step0x0_layout_{:03d}_3x3_subvoxels.hdf5".format(i) for i in file_idxs]
    
    # --- OUTPUT FILE ---
    # This is the full path and name for the .csv file you are creating.
    output_filepath = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/dataset_flists/dataset_flist18_3mm_displaced_24ph.csv"

    print(f"Reading HDF5 files from: {topdir}")
    print(f"Writing file list to:    {output_filepath}")

    with open(output_filepath, "w") as f:
        for fname in fnames:
            # This creates the full path for one HDF5 file, e.g.,
            # "/user/aguleria/ub_rutao/pymatcal/scanner_layouts_/position_000_ppdfs.hdf5"
            full_path_to_hdf5 = os.path.join(topdir, fname)
            
            # This writes that full path as a line in your .csv file
            f.write(full_path_to_hdf5 + "\n")

    print("File list generated successfully!")
