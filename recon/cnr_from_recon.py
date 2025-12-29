# cnr_from_recon.py
import os, sys, torch, numpy as np

def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def make_rod_masks(rods, H, W, shrink_px=1):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    masks = []
    for r in rods:
        cx, cy = r["center_px"]
        rad = int(r["radius_px"]) - int(shrink_px)
        rad = max(rad, 1)
        m = (yy - cx)**2 + (xx - cy)**2 <= (rad**2)
        masks.append(m)
    return masks

def bg_mask_annulus(meta, H, W, erode_px=2):
    mm_per_px = meta.get("mm per pixel") or meta.get("mm_per_pixel") or meta.get("mm_per_px")
    Rin = float(meta.get("ring_inner_radius_mm", 0.0))
    Rout = float(meta.get("ring_outer_radius_mm", 1e9))
    px_per_mm = 1.0 / float(to_list(mm_per_px)[0])
    rin_px  = int(round(Rin  * px_per_mm)) + erode_px
    rout_px = int(round(Rout * px_per_mm)) - erode_px
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cx, cy = (H-1)/2.0, (W-1)/2.0
    rr2 = (yy - cx)**2 + (xx - cy)**2
    return (rr2 >= rin_px**2) & (rr2 <= rout_px**2)

def cnr(hot_vals, bg_vals):
    mu_h, mu_b = hot_vals.mean(), bg_vals.mean()
    var_h = hot_vals.var(unbiased=False) + 1e-12
    var_b = bg_vals.var(unbiased=False) + 1e-12
    return float((mu_h - mu_b).abs() / torch.sqrt(var_h + var_b))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cnr_from_recon.py <recon_mlem_torch_hotrod180.npz> <phantom.pt> [shrink_px 1] [erode_px 2]")
        sys.exit(1)

    recon_npz = sys.argv[1]
    phantom_pt = sys.argv[2]
    shrink_px = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    erode_px  = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    # load recon (take the last snapshot)
    nz = np.load(recon_npz)
    est_stack = nz["estimates"]  # shape [K, H, W]
    recon = torch.from_numpy(est_stack[-1])  # (H, W)
    H, W = recon.shape

    # load phantom metadata (for ROIs)
    ph = torch.load(phantom_pt, map_location="cpu")
    meta = ph["Metadata"]
    #rods = ph.get("Rods", [])  # list of dicts with center_px, radius_px
    if "Rods" in ph:
        rods = ph["Rods"]
    elif "Lesions" in ph:
        lesions = ph["Lesions"]
        rods = []
        for li in lesions:
            cx, cy = li["center_px"]
            r = li["radius_px"]
            rods.append({"center_px": (cx, cy), "radius_px": r})
    else:
        rods = []

    # build ROIs
    rod_masks = make_rod_masks(rods, H, W, shrink_px=shrink_px)
    bg_mask = bg_mask_annulus(meta, H, W, erode_px=erode_px)
    for m in rod_masks:
        bg_mask &= ~m  # exclude hot areas from background
    
    print("Recon finite:", torch.isfinite(recon).all().item())
    print("BG px:", int(bg_mask.sum()))
    for k, m in enumerate(rod_masks):
        print(f"Rod {k} px:", int(m.sum()))


    # compute CNR per rod
    bg_vals = recon[bg_mask]
    cnrs = []
    for m in rod_masks:
        hot_vals = recon[m]
        cnrs.append(cnr(hot_vals, bg_vals))
    cnrs = torch.tensor(cnrs)

    print(f"CNR per rod (N={len(cnrs)}): " + ", ".join(f"{x:.2f}" for x in cnrs.tolist()))
    print(f"Mean CNR: {cnrs.mean().item():.2f}   Median CNR: {cnrs.median().item():.2f}")
