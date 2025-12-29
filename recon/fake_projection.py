import numpy as np
import time
import torch
import os
import h5py
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

# (get_flist function can remain the same)
def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [f.strip() for f in flist]
        return flist

# We don't need get_matrix anymore, as we'll load one file at a time.

if __name__ == "__main__":
    
    # --- Setup ---
    device = torch.device("cpu")
    data_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/dataset_flists"
    flist = get_flist(os.path.join(data_dir, "dataset_flist18_3mm_displaced_24ph.csv"))
    
    # These define the EXPECTED dimensions from the system matrix files
    sfov_expected = 280* 280
    sproj = 1366

    # --- Phantom Loading ---
    # ERROR 1 FIX: Use torch.load and the correct, direct path.
    phantom_filename = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/recon/hot_rods_phantom_70.0_mm_x_70.0_mm.pt"
    phantom_data = torch.load(phantom_filename)
    phantom_tensor = phantom_data["Phantom tensor"]

    # FIX: Flatten the 2D phantom into a 1D vector for matrix multiplication.
    # This is the line that was missing.
    phantom_flat = phantom_tensor.view(-1)
    

    # --- Batch Processing ---
    # MEMORY FIX: Process one system matrix file at a time instead of loading all at once.
    all_projs = []

    # Setup a nice progress bar
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Projecting...", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                # Load one chunk of the matrix
                matrix_chunk = torch.tensor(h5f["ppdfs"][:]).view(1, sproj, sfov_expected)
                
                # Perform matrix multiplication on just this chunk
                proj_chunk = torch.matmul(matrix_chunk, phantom_flat)
                all_projs.append(proj_chunk)
            
            progress.update(task, advance=1)
                

    # Combine the results from all the chunks
    final_projs = torch.cat(all_projs, dim=0)

    # --- Save the final result ---
    output_path = os.path.join(data_dir, "hotrod-projs-3mm-displaced-24ph.npy")
    np.save(output_path, final_projs.numpy())
    
    print("\nProjection complete!")
    print(f"Final projection shape: {final_projs.shape}")
    print(f"Saved projections to: {output_path}")