#!/usr/bin/env python3
import os
import sys
import time
import h5py
import numpy as np
import torch
import multiprocessing as mp
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

# ----------------------------
# Helpers / config
# ----------------------------
def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        return [ln.strip() for ln in f]

def is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def setup_threads_from_env(default_cap: int = 32):
    n_threads_env = os.environ.get("OMP_NUM_THREADS")
    if n_threads_env and n_threads_env.isdigit():
        n_threads = int(n_threads_env)
    else:
        n_threads = min(default_cap, mp.cpu_count() or 1)
    try:
        torch.set_num_threads(max(1, n_threads))
        torch.set_num_interop_threads(max(1, min(4, max(1, n_threads // 8))))
    except Exception:
        pass
    return n_threads

def fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # --- 1. Configuration ---
    base_dir = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon"
    flist_path = os.path.join(base_dir, "dataset_flists/dataset_flist18_3mm_displaced_24ph.csv")
    projs_path = os.path.join(base_dir, "hotrod-projs-3mm-displaced-24ph.npy")
    output_path = os.path.join(base_dir, "hotrod-projs-3mm-displaced-24ph.npz")

    IMG_DIM = 280
    SFOV = IMG_DIM * IMG_DIM
    SPROJ = 1366

    N_ITERATIONS = 150
    CONVERGENCE_TOLERANCE = 1e-4  # relative change threshold

    # Inner-loop logging cadence (files processed)
    PROGRESS_FILE_STEP = int(os.environ.get("MLEM_FILE_PROGRESS_STEP", "50"))

    # Threading + device
    used_threads = setup_threads_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device} | torch threads: {used_threads}", flush=True)

    # --- 2. Load Data ---
    print("Loading file list & projections…", flush=True)
    flist = get_flist(flist_path)

    # Memory-map projections for low RAM pressure
    pdata_np = np.load(projs_path, mmap_mode="r")
    if pdata_np.ndim != 2 or pdata_np.shape[1] != SPROJ:
        raise ValueError(f"Expected projections shape (N, {SPROJ}), got {pdata_np.shape}")

    # --- 3. Initialization ---
    estimate = torch.ones(SFOV, device=device, dtype=torch.float32)

    estimates_history = []
    times_history = []

    # --- 4. Main MLEM Reconstruction Loop ---
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        disable=not is_tty(),   # Rich off in Slurm logs
    )

    t0 = time.time()

    with progress:
        main_task = progress.add_task("[green]MLEM Iterations", total=N_ITERATIONS)

        for it in range(N_ITERATIONS):
            iter_start = time.time()
            estimate_prev = estimate.clone()

            back_projection = torch.zeros(SFOV, device=device, dtype=torch.float32)
            sensitivity_map = torch.zeros(SFOV, device=device, dtype=torch.float32)

            inner_task = progress.add_task(
                f"[cyan]  Iter {it+1}/{N_ITERATIONS}",
                total=len(flist),
                transient=True
            )

            # ---- process all files ----
            for i, fname in enumerate(flist):
                try:
                    with h5py.File(fname, "r") as h5f:
                        m_np = h5f["ppdfs"][:]
                except Exception as e:
                    print(f"[WARN] Failed reading {fname}: {e}", flush=True)
                    progress.update(inner_task, advance=1)
                    continue

                # Normalize shape to (SPROJ, SFOV)
                m_np = np.asarray(m_np)
                if m_np.ndim == 1 and m_np.size == SPROJ * SFOV:
                    m_np = m_np.reshape(SPROJ, SFOV)
                elif m_np.ndim == 2 and m_np.shape == (SPROJ, SFOV):
                    pass
                else:
                    raise ValueError(f"{fname}: unexpected ppdfs shape {m_np.shape}, expected {(SPROJ, SFOV)}")

                m_chunk = torch.from_numpy(m_np).to(device=device, dtype=torch.float32)  # (SPROJ, SFOV)
                p_row = torch.from_numpy(np.array(pdata_np[i], copy=False)).to(device=device, dtype=torch.float32)

                y = m_chunk @ estimate               # (SPROJ,)
                y = torch.where(y == 0, torch.ones_like(y), y)
                r = p_row / y                        # (SPROJ,)
                back_projection += (m_chunk.T @ r)   # (SFOV,)
                sensitivity_map += m_chunk.sum(dim=0)

                progress.update(inner_task, advance=1)

                # Optional inner-loop ping for Slurm logs
                if PROGRESS_FILE_STEP > 0 and ((i + 1) % PROGRESS_FILE_STEP == 0):
                    print(f"[iter {it+1}/{N_ITERATIONS}] files processed: {i+1}/{len(flist)}", flush=True)

            # ---- UPDATE STEP ----
            sensitivity_map = torch.where(sensitivity_map == 0, torch.ones_like(sensitivity_map), sensitivity_map)
            estimate = estimate * (back_projection / sensitivity_map)

            # ---- Logging & convergence ----
            iter_time = time.time() - iter_start
            times_history.append(iter_time)

            if it % 5 == 0:
                estimates_history.append(estimate.view(IMG_DIM, IMG_DIM).detach().cpu().numpy())

            den = torch.norm(estimate_prev)
            num = torch.norm(estimate - estimate_prev)
            diff = (num / den).item() if den.item() != 0 else float("inf")

            # --- Always print a single concise iteration line (goes to Slurm log) ---
            done = it + 1
            pct = 100.0 * done / N_ITERATIONS
            elapsed = time.time() - t0
            avg_it = elapsed / done
            eta = avg_it * (N_ITERATIONS - done)
            print(
                f"[{done:3d}/{N_ITERATIONS}] {pct:6.2f}%  "
                f"iter_time={iter_time:6.2f}s  elapsed={fmt_hms(elapsed)}  "
                f"eta~{fmt_hms(eta)}  diff={diff:.2e}",
                flush=True
            )

            if diff < CONVERGENCE_TOLERANCE:
                print(f"Convergence reached at iteration {done} (difference: {diff:.2e}). Stopping.", flush=True)
                progress.update(main_task, completed=N_ITERATIONS)
                break

            progress.update(main_task, advance=1, description=f"[green]MLEM Iterations (diff: {diff:.2e})")

    # --- 5. Save Final Results ---
    print("\nReconstruction complete. Saving results…", flush=True)
    np.savez_compressed(
        output_path,
        estimates=np.array(estimates_history, dtype=np.float32),
        times=np.array(times_history, dtype=np.float32),
    )
    print(f"Results saved to: {output_path}", flush=True)
