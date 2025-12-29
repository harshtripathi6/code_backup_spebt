#!/usr/bin/env python3
import os
import sys
import h5py
import math
import numpy as np
import torch
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        return [ln.strip() for ln in f]

def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def load_phantom_pt(phantom_pt: str):
    pack = torch.load(phantom_pt, map_location="cpu")
    img  = pack["Phantom tensor"].float()  # HxW
    meta = pack.get("Metadata", {})
    # Support both keys: "Rods" or "Lesions"
    rods = pack.get("Rods", [])
    if not rods and "Lesions" in pack:
        rods = []
        for li in pack["Lesions"]:
            cx, cy = li["center_px"]
            r  = li["radius_px"]
            rods.append({"center_px": (cx, cy), "radius_px": r})
    return img, meta, rods

def rebuild_rods_from_ring_meta(meta, H, W, mm_per_px=(0.25,0.25)):
    """If the PT lacks rods, rebuild them from ring params."""
    n_rods = int(meta.get("n_rods", 0))
    Rin = float(meta.get("ring_inner_radius_mm", 0.0))
    Rout = float(meta.get("ring_outer_radius_mm", 0.0))
    d_mm = float(meta.get("rod_diameter_mm", 0.0))
    if n_rods <= 0 or d_mm <= 0 or Rout <= 0:
        return []

    dx = float(mm_per_px[0])
    r_pix = max(1, int(round(0.5 * d_mm / dx)))
    Rmid_mm = 0.5 * (Rin + Rout)
    Rmid_px = Rmid_mm / dx
    cy0, cx0 = (H - 1) / 2.0, (W - 1) / 2.0  # (row, col)

    rods = []
    for k in range(n_rods):
        th = 2.0 * math.pi * k / n_rods
        # columns = x = cos, rows = y = sin
        col = int(round(cx0 + Rmid_px * math.cos(th)))
        row = int(round(cy0 + Rmid_px * math.sin(th)))
        rods.append({"center_px": (row, col), "radius_px": r_pix})
    return rods

def make_rod_masks(rods, H, W, shrink_px=0):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    masks = []
    for r in rods:
        row, col = r["center_px"]
        rad = max(int(r["radius_px"]) - int(shrink_px), 1)
        if row < 0 or col < 0 or row >= H or col >= W:
            masks.append(torch.zeros((H, W), dtype=torch.bool))
            continue
        m = (yy - row) ** 2 + (xx - col) ** 2 <= (rad ** 2)
        masks.append(m)
    return masks

if __name__ == "__main__":
    # ---- args (simple CLI) ----
    if len(sys.argv) < 6:
        print("Usage: python fake_projection_rates.py <flist.csv> <phantom.pt> <T_sec> <eHotVoxel> <eBkgVoxel> [out.npy]")
        print("Example:")
        print("  python fake_projection_rates.py /path/dataset_flist180.csv ring_contrast_phantom_24rods_3.0mm.pt 10 10 2 /path/hotrod-projs180.npy")
        sys.exit(1)

    flist_path   = sys.argv[1]
    phantom_path = sys.argv[2]
    T_sec        = float(sys.argv[3])
    e_hot        = float(sys.argv[4])
    e_bg         = float(sys.argv[5])
    out_npy      = sys.argv[6] if len(sys.argv) > 6 else os.path.join(os.path.dirname(flist_path), "hotrod-projs.npy")

    # ---- load file list ----
    flist = get_flist(flist_path)
    assert len(flist) > 0, "Empty file list."

    # ---- infer SPROJ/SFOV from first HDF5 ----
    with h5py.File(flist[0], "r") as h5f:
        m0 = np.asarray(h5f["ppdfs"][:])
    if m0.ndim == 1:
        # flat vector -> infer (SPROJ, SFOV)
        # We cannot know SPROJ uniquely from a flat vector; try to read attributes if any
        raise ValueError("First HDF5 'ppdfs' is 1D; please save 2D (SPROJ,SFOV) or adjust code to know SPROJ.")
    elif m0.ndim == 2:
        SPROJ, SFOV = m0.shape
    else:
        raise ValueError(f"Unexpected 'ppdfs' shape in first file: {m0.shape}")

    # ---- image dimension sanity ----
    img_dim_f = math.sqrt(SFOV)
    if abs(img_dim_f - round(img_dim_f)) > 1e-6:
        raise ValueError(f"SFOV={SFOV} is not a perfect square; got sqrt={img_dim_f}.")
    IMG_DIM = int(round(img_dim_f))

    # ---- load phantom; ensure size matches SFOV ----
    phantom_img, meta, rods = load_phantom_pt(phantom_path)  # HxW
    H, W = phantom_img.shape
    if H * W != SFOV:
        raise ValueError(f"Phantom size {H}x{W} != SFOV {SFOV} from system matrix.")

    # build rods if missing
    if not rods:
        # pull mm_per_px from metadata if present, else assume 0.25
        mmpp = meta.get("mm per pixel") or meta.get("mm_per_pixel") or (0.25, 0.25)
        mmpp = to_list(mmpp)
        rods = rebuild_rods_from_ring_meta(meta, H, W, mm_per_px=(float(mmpp[0]), float(mmpp[1])))

    # ---- override voxel counts by rates & time ----
    # voxel_counts_bg = T * eBkgVoxel
    # voxel_counts_hot = T * eHotVoxel
    phantom_counts = torch.full_like(phantom_img, fill_value=T_sec * e_bg)  # background everywhere
    if rods:
        hot_masks = make_rod_masks(rods, H, W, shrink_px=0)
        for m in hot_masks:
            phantom_counts[m] = T_sec * e_hot  # set hot-rod voxels

    # ---- forward project (one file at a time) + Poisson(q_i | p_i) ----
    rows = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Forward project + Poisson", total=len(flist))

        ph_flat = phantom_counts.view(-1).contiguous()  # (SFOV,)
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                m = np.asarray(h5f["ppdfs"][:])

            if m.ndim == 1 and m.size == SPROJ * SFOV:
                m = m.reshape(SPROJ, SFOV)
            elif m.ndim == 2 and m.shape == (SPROJ, SFOV):
                pass
            else:
                raise ValueError(f"{fname}: unexpected ppdfs shape {m.shape}, expected {(SPROJ, SFOV)}")

            m_t = torch.from_numpy(m).to(dtype=torch.float32)   # (SPROJ, SFOV)
            p = m_t @ ph_flat                                   # (SPROJ,) expected counts/bin
            p = p.clamp_min(1e-12)
            q = torch.poisson(p)                                # Poisson noise per bin
            rows.append(q)

            progress.update(task, advance=1)

    sino = torch.stack(rows, dim=0).cpu().numpy()  # (N_files, SPROJ)
    np.save(out_npy, sino)
    print(f"\nSaved noisy projections: {out_npy}")
    print(f"Shape: {sino.shape}  (N={sino.shape[0]}, SPROJ={sino.shape[1]})")
