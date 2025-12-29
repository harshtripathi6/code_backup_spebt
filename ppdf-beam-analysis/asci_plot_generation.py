import os, h5py, torch, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -------------------------------------------------------------------
INPUT_DIR  = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/outputs_3mm_aperture_displaced_24ph"
PLOT_DIR   = "/vscratch/grp-rutaoyao/Harsh/patch_based_recon/metrics/metrics_3mm_aperture_displaced_24ph"
LAYOUT_SEQ = range(15)            # range(24) after you have all histograms
N_BINS     = 360
FOV_SIDE   = 10                  # mm  (±16 mm)
# -------------------------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

asci_hist = torch.zeros(280*280, N_BINS, dtype=torch.int32)

for idx in LAYOUT_SEQ:
    with h5py.File(os.path.join(INPUT_DIR,
                                f"asci_histogram_{idx:03d}.hdf5"), "r") as f:
        asci_hist += torch.from_numpy(f["asci_histogram"][...])

asci_map = torch.count_nonzero(asci_hist, dim=1) / N_BINS   # 0‒1

# ---- plot -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,7), layout="constrained")

im = ax.imshow(asci_map.view(280,280).T,
               extent=(-FOV_SIDE/2, FOV_SIDE/2, -FOV_SIDE/2, FOV_SIDE/2),
               origin="lower", cmap="viridis")
cbar = fig.colorbar(im, ax=ax, label="ASCI")
cbar.formatter = PercentFormatter(xmax=1.0, decimals=1); cbar.update_ticks()

ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
ax.set_title(f"max {asci_map.max():.2%}, min {asci_map.min():.2%}")

fig.savefig(os.path.join(PLOT_DIR, "asci_map.png"), dpi=300)
plt.close(fig)
