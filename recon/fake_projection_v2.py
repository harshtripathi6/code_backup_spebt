# fake_projection.py
import numpy as np
import time
import torch
import os
import h5py
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        return [ln.strip() for ln in f]

if __name__ == "__main__":
    # --- Setup ---
    device = torch.device("cpu")  # keep CPU (I/O bound anyway)
    data_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon"
    flist = get_flist(os.path.join(data_dir, "dataset_flist18.csv"))

    # Expected dims from system matrix files
    IMG_DIM = 512
    SFOV = IMG_DIM * IMG_DIM
    SPROJ = 1358

    # ---- inputs you will likely tweak ----
    phantom_filename = "/vscratch/grp-rutaoyao/sid/recon/ring_contrast_phantom_24rods_3.0mm.pt"
    out_npy = os.path.join(data_dir, "ring-projs18.npy")  # MLEM loads this path
    counts_per_view = 5e5         # total expected counts per view (angle)
    add_poisson = True            # set False if you want noise-free expectations
    torch.manual_seed(1234)       # reproducible noise (optional)

    # --- Phantom ---
    ph = torch.load(phantom_filename, map_location="cpu")["Phantom tensor"].float()   
    phantom_flat = ph.view(-1)  # (SFOV,)

    # --- Batch processing, one system-matrix file at a time ---
    rows = []  # will hold (SPROJ,) tensors

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Projecting + Poisson…", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                # system matrix chunk shape can be (SPROJ, SFOV) or flat
                m_np = h5f["ppdfs"][:]
            m_np = np.asarray(m_np)
            if m_np.ndim == 1 and m_np.size == SPROJ * SFOV:
                m_np = m_np.reshape(SPROJ, SFOV)
            elif m_np.ndim == 2 and m_np.shape == (SPROJ, SFOV):
                pass
            else:
                raise ValueError(f"{fname}: unexpected ppdfs shape {m_np.shape}, expected {(SPROJ, SFOV)}")

            m_chunk = torch.from_numpy(m_np).to(dtype=torch.float32)  # (SPROJ, SFOV)

            # forward projection for this file: (SPROJ,)
            proj = m_chunk @ phantom_flat
            proj = proj.clamp_min(1e-12)

            # normalize to desired counts/view and add Poisson if requested
            proj = proj / proj.sum() * counts_per_view
            if add_poisson:
                proj = torch.poisson(proj)

            rows.append(proj)  # (SPROJ,)

            progress.update(task, advance=1)

    # Stack to (N, SPROJ) exactly as mlem_torch_parallel.py expects
    final_projs = torch.stack(rows, dim=0).cpu().numpy()  # (N, SPROJ)

    # --- Save ---
    np.save(out_npy, final_projs)
    print("\nProjection complete!")
    print(f"Final projection shape: {final_projs.shape}")
    print(f"Saved projections to: {out_npy}")
