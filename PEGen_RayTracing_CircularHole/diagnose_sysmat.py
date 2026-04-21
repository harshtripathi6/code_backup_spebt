"""
Diagnostic script to identify why the PE system matrix is all zeros.
Tests geometry alignment, data integrity, and simulates the kernel logic in Python.
"""
import numpy as np
import sys

def load_params(data_dir="."):
    col = np.fromfile(f"{data_dir}/Params_Collimator.dat", dtype=np.float32)
    det = np.fromfile(f"{data_dir}/Params_Detector.dat", dtype=np.float32)
    img = np.fromfile(f"{data_dir}/Params_Image.dat", dtype=np.float32)
    phy = np.fromfile(f"{data_dir}/Params_Physics.dat", dtype=np.float32)
    return col, det, img, phy


def diagnose(data_dir="."):
    col, det, img, phy = load_params(data_dir)

    # ===== 1. IMAGE PARAMS =====
    nx, ny, nz = int(img[0]), int(img[1]), int(img[2])
    dx, dy, dz = img[3], img[4], img[5]
    fov2col = img[11]
    shift_x, shift_y, shift_z = img[8], img[9], img[10]

    print("=" * 70)
    print("IMAGE PARAMETERS")
    print("=" * 70)
    print(f"  Voxels: {nx} x {ny} x {nz} = {nx*ny*nz:,}")
    print(f"  Voxel size: {dx} x {dy} x {dz} mm")
    print(f"  FOV extent X: [{-(nx/2)*dx + shift_x:.1f}, {(nx/2)*dx + shift_x:.1f}] mm")
    print(f"  FOV extent Y: [{-(ny/2)*dy + shift_y:.1f}, {(ny/2)*dy + shift_y:.1f}] mm")
    print(f"  FOV extent Z: [{-(nz/2)*dz + shift_z:.1f}, {(nz/2)*dz + shift_z:.1f}] mm")
    print(f"  FOV2Collimator: {fov2col} mm")
    print()

    # ===== 2. COLLIMATOR PARAMS =====
    n_layers = int(col[0])
    print("=" * 70)
    print("COLLIMATOR PARAMETERS")
    print("=" * 70)
    print(f"  Number of layers: {n_layers}")

    for layer in range(n_layers):
        base = (layer + 1) * 10
        n_holes = int(col[base + 0])
        width_x = col[base + 1]
        thick_y = col[base + 2]
        height_z = col[base + 3]
        offset_y = col[base + 4]
        coeff_tot = col[base + 5]

        # Physical Y position of collimator in world coords
        y1_col = -thick_y / 2.0 + fov2col + offset_y
        y2_col = thick_y / 2.0 + fov2col + offset_y

        print(f"  Layer {layer}:")
        print(f"    Holes: {n_holes}, Size: {width_x} x {thick_y} x {height_z} mm")
        print(f"    Y range in world coords: [{y1_col:.1f}, {y2_col:.1f}] mm")
        print(f"    Attenuation coeff (total): {coeff_tot}")
        print(f"    Full-thickness attenuation: exp(-{coeff_tot * thick_y:.1f}) = {np.exp(-coeff_tot * thick_y):.2e}")

    # Check holes
    if n_layers > 0:
        base = 10
        n_holes = int(col[base])
        print(f"\n  Hole data check (first 5 holes):")
        for i in range(min(5, n_holes)):
            bh = i * 9 + 100
            x, y1, y2, z, r = col[bh:bh+5]
            coeff = col[bh+5]
            print(f"    Hole {i}: center=({x:.2f}, {(y1+y2)/2+fov2col:.2f}, {z:.2f}), "
                  f"y_range=[{y1+fov2col:.1f},{y2+fov2col:.1f}], r={r:.2f}, coeff={coeff}")
    print()

    # ===== 3. DETECTOR PARAMS =====
    n_det = int(det[0])
    print("=" * 70)
    print("DETECTOR PARAMETERS")
    print("=" * 70)
    print(f"  Total detectors: {n_det}")

    # Check a few detectors from each layer
    y_values = []
    for i in range(n_det):
        base_d = i * 12 + 1
        y_center = det[base_d + 1]
        y_values.append(y_center)

    y_unique = np.unique(np.round(np.array(y_values), 1))
    print(f"  Unique Y centers (before FOV offset): {y_unique}")
    print(f"  Unique Y centers (in world coords):   {y_unique + fov2col}")

    for layer_idx, y_val in enumerate(y_unique):
        mask = np.abs(np.array(y_values) - y_val) < 0.5
        count = np.sum(mask)
        first_det = np.where(mask)[0][0]
        base_d = first_det * 12 + 1
        w, t, h = det[base_d+3], det[base_d+4], det[base_d+5]
        print(f"  Layer {layer_idx}: {count} detectors, Y_world={y_val+fov2col:.1f}, "
              f"size={w}x{t}x{h} mm")

    # Check specific detector
    for det_id in [0, 650, 1536, 5000]:
        if det_id >= n_det:
            continue
        base_d = det_id * 12 + 1
        x, y, z = det[base_d], det[base_d+1], det[base_d+2]
        w, t, h = det[base_d+3], det[base_d+4], det[base_d+5]
        coeff = det[base_d+6]
        rot = det[base_d+10]
        print(f"\n  Detector {det_id}: center=({x:.2f}, {y+fov2col:.2f}, {z:.2f}), "
              f"size=({w},{t},{h}), coeff_total={coeff}, rotation={rot}")
    print()

    # ===== 4. GEOMETRY ALIGNMENT CHECK =====
    print("=" * 70)
    print("GEOMETRY ALIGNMENT (Y-axis)")
    print("=" * 70)

    fov_y_min = -(ny/2) * dy + shift_y
    fov_y_max = (ny/2) * dy + shift_y

    base = 10
    thick_y = col[base + 2]
    offset_y = col[base + 4]
    col_y_min = -thick_y/2 + fov2col + offset_y
    col_y_max = thick_y/2 + fov2col + offset_y

    det_y_min = min(y_values) + fov2col
    det_y_max = max(y_values) + fov2col

    print(f"  FOV Y range:        [{fov_y_min:.1f}, {fov_y_max:.1f}] mm")
    print(f"  Collimator Y range: [{col_y_min:.1f}, {col_y_max:.1f}] mm")
    print(f"  Detector Y range:   [{det_y_min:.1f}, {det_y_max:.1f}] mm")
    print()

    if fov_y_max > col_y_min:
        overlap = fov_y_max - col_y_min
        n_overlap_voxels = int(np.ceil(overlap / dy))
        print(f"  ⚠️  WARNING: FOV overlaps collimator by {overlap:.1f} mm ({n_overlap_voxels} voxels)")
        print(f"     These voxels will have undefined behavior in length_box_ray!")
    else:
        gap = col_y_min - fov_y_max
        print(f"  ✓ Gap between FOV and collimator: {gap:.1f} mm")

    if col_y_max > det_y_min:
        print(f"  ⚠️  WARNING: Collimator overlaps detector!")
    else:
        gap = det_y_min - col_y_max
        print(f"  ✓ Gap between collimator and first detector: {gap:.1f} mm")
    print()

    # ===== 5. RAY TRACING SANITY CHECK =====
    print("=" * 70)
    print("RAY TRACING SIMULATION (Python)")
    print("=" * 70)

    # Pick a central voxel and a detector, trace the ray
    vox_ix, vox_iy, vox_iz = nx//2, ny//2, nz//2
    x_vox = (vox_ix - nx/2.0 + 0.5) * dx + shift_x
    y_vox = (vox_iy - ny/2.0 + 0.5) * dy + shift_y
    z_vox = (vox_iz - nz/2.0 + 0.5) * dz + shift_z

    print(f"  Test voxel ({vox_ix},{vox_iy},{vox_iz}): world=({x_vox:.2f}, {y_vox:.2f}, {z_vox:.2f})")

    # Find the detector closest to (0, y_det, 0) for a straight-through test
    det_centers = []
    for i in range(n_det):
        bd = i * 12 + 1
        det_centers.append([det[bd], det[bd+1] + fov2col, det[bd+2]])
    det_centers = np.array(det_centers)

    # Find detector closest to x=0, z=0
    lateral_dist = np.sqrt(det_centers[:, 0]**2 + det_centers[:, 2]**2)
    best_det = np.argmin(lateral_dist)
    x_det, y_det, z_det = det_centers[best_det]

    print(f"  Closest on-axis detector: {best_det}, at ({x_det:.2f}, {y_det:.2f}, {z_det:.2f})")

    # Check ray direction
    dy_ray = y_det - y_vox
    print(f"  Ray Y-travel: {y_vox:.2f} -> {y_det:.2f} (distance={dy_ray:.2f} mm)")

    if dy_ray <= 0:
        print(f"  ❌ FATAL: Ray goes in WRONG DIRECTION (voxel Y >= detector Y)")
        print(f"     This means the FOV is BEHIND the detectors!")
        return

    # Check if ray hits collimator bounding box
    base = 10
    width_x = col[base + 1]
    thick_y_c = col[base + 2]
    height_z = col[base + 3]
    offset_y_c = col[base + 4]

    x1_box = -width_x / 2.0
    x2_box = width_x / 2.0
    y1_box = -thick_y_c/2 + fov2col + offset_y_c
    y2_box = thick_y_c/2 + fov2col + offset_y_c
    z1_box = -height_z / 2.0
    z2_box = height_z / 2.0

    # Parametric ray intersection with collimator box
    ray_dir = np.array([x_det - x_vox, y_det - y_vox, z_det - z_vox])
    ray_len = np.linalg.norm(ray_dir)
    ray_dir_n = ray_dir / ray_len

    # Slab method
    box_min = np.array([x1_box, y1_box, z1_box])
    box_max = np.array([x2_box, y2_box, z2_box])
    origin = np.array([x_vox, y_vox, z_vox])

    with np.errstate(divide='ignore'):
        inv_dir = ray_len / ray_dir  # same as kernel's convention

    t_enter = np.where(inv_dir >= 0,
                       (box_min - origin) * inv_dir,
                       (box_max - origin) * inv_dir)
    t_exit = np.where(inv_dir >= 0,
                      (box_max - origin) * inv_dir,
                      (box_min - origin) * inv_dir)

    tmin = np.max(t_enter)
    tmax = np.min(t_exit)

    if tmin > tmax or tmax <= 0.001 or tmin >= ray_len:
        print(f"  Ray MISSES collimator box: tmin={tmin:.4f}, tmax={tmax:.4f}, ray_len={ray_len:.4f}")
        print(f"  ❌ This means the bounding box test kills the ray!")
    else:
        path_len = tmax - tmin
        print(f"  Ray HITS collimator box: path_length={path_len:.4f} mm")
        if path_len < 0.1:
            print(f"  ⚠️  Path length < 0.1 mm — kernel will add 100000 penalty!")
        else:
            print(f"  ✓ Path length passes the 0.1 mm threshold")

    # Check how many holes this ray could pass through
    n_holes_layer0 = int(col[10])
    cos_angle = ray_dir_n[1]  # Y component
    full_tungsten_path = thick_y_c / cos_angle if cos_angle > 0.01 else 999
    full_atten = col[base + 5] * full_tungsten_path
    print(f"\n  Full tungsten path (no holes): {full_tungsten_path:.2f} mm")
    print(f"  Full attenuation: exp(-{full_atten:.1f}) = {np.exp(-full_atten):.2e}")

    # Find holes near the ray's intersection with the collimator midplane
    col_midplane_y = fov2col + offset_y_c
    if cos_angle > 0.01:
        t_mid = (col_midplane_y - y_vox) / ray_dir_n[1] * ray_dir_n
        x_at_col = x_vox + t_mid[0]
        z_at_col = z_vox + t_mid[2]
    else:
        x_at_col, z_at_col = x_vox, z_vox

    print(f"  Ray at collimator midplane: ({x_at_col:.2f}, {z_at_col:.2f})")

    # Find nearest holes
    holes_data = col[100: 100 + n_holes_layer0 * 9].reshape(-1, 9)
    hole_x = holes_data[:, 0]
    hole_z = holes_data[:, 3]
    hole_r = holes_data[:, 4]

    distances = np.sqrt((hole_x - x_at_col)**2 + (hole_z - z_at_col)**2)
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]

    print(f"  Nearest hole: #{nearest_idx}, distance={nearest_dist:.3f} mm, radius={hole_r[nearest_idx]:.3f} mm")
    if nearest_dist <= hole_r[nearest_idx]:
        print(f"  ✓ Ray passes THROUGH a hole!")
    else:
        print(f"  ✗ Ray misses nearest hole by {nearest_dist - hole_r[nearest_idx]:.3f} mm")

    hits = np.sum(distances <= hole_r)
    print(f"  Total holes the ray passes through: {hits}")

    # ===== 6. STATISTICAL CHECK =====
    print()
    print("=" * 70)
    print("STATISTICAL COVERAGE CHECK")
    print("=" * 70)

    # For a grid of detectors, check how many have at least one hole aligned
    n_test_dets = min(100, n_det)
    test_dets = np.linspace(0, n_det-1, n_test_dets, dtype=int)
    n_with_holes = 0

    for det_id in test_dets:
        bd = det_id * 12 + 1
        xd, yd, zd = det[bd], det[bd+1] + fov2col, det[bd+2]
        # Ray from center voxel to this detector
        rd = np.array([xd - x_vox, yd - y_vox, zd - z_vox])
        rd_len = np.linalg.norm(rd)
        rd_n = rd / rd_len

        if rd_n[1] > 0.01:
            t_frac = (col_midplane_y - y_vox) / rd_n[1]
            x_at = x_vox + rd_n[0] * t_frac
            z_at = z_vox + rd_n[2] * t_frac
            dists = np.sqrt((hole_x - x_at)**2 + (hole_z - z_at)**2)
            if np.any(dists <= hole_r):
                n_with_holes += 1

    print(f"  Of {n_test_dets} sampled detectors, {n_with_holes} have a hole-aligned "
          f"ray from the center voxel ({100*n_with_holes/n_test_dets:.1f}%)")

    # ===== 7. SYSMAT FILE CHECK =====
    print()
    print("=" * 70)
    print("SYSMAT FILE CHECK")
    print("=" * 70)
    import glob
    sysmat_files = glob.glob(f"{data_dir}/*.sysmat")
    if not sysmat_files:
        print("  No .sysmat file found!")
        return

    for sf in sysmat_files:
        import os
        size = os.path.getsize(sf)
        expected = n_det * nx * ny * nz * 4  # float32
        print(f"  File: {sf}")
        print(f"    Size: {size:,} bytes ({size/1e9:.2f} GB)")
        print(f"    Expected: {expected:,} bytes ({expected/1e9:.2f} GB)")
        if size != expected:
            print(f"    ⚠️  SIZE MISMATCH! Ratio: {size/expected:.6f}")
            print(f"    This could be caused by int32 overflow in the C++ code!")
        else:
            print(f"    ✓ Size matches expected")

        # Check content
        data = np.memmap(sf, dtype=np.float32, mode='r')
        print(f"    Total elements: {len(data):,}")
        print(f"    Nonzero: {np.count_nonzero(data):,}")
        print(f"    Max: {np.max(data):.6e}")
        print(f"    Min (nonzero): {np.min(data[data > 0]):.6e}" if np.any(data > 0) else "    All zeros!")

        # Check for NaN/Inf
        n_nan = np.sum(np.isnan(data))
        n_inf = np.sum(np.isinf(data))
        if n_nan > 0 or n_inf > 0:
            print(f"    ⚠️  NaN: {n_nan}, Inf: {n_inf}")

    # ===== 8. INTEGER OVERFLOW CHECK =====
    print()
    print("=" * 70)
    print("INTEGER OVERFLOW CHECK (C++ code)")
    print("=" * 70)
    product_32bit = np.int32(n_det) * np.int32(nx * ny * nz)
    product_64bit = np.int64(n_det) * np.int64(nx * ny * nz)
    print(f"  numProjectionSingle * numImagebin:")
    print(f"    As int32: {product_32bit} (OVERFLOWED!)" if product_32bit != product_64bit
          else f"    As int32: {product_32bit} (OK)")
    print(f"    As int64: {product_64bit}")
    print(f"    Max int32: {np.iinfo(np.int32).max}")

    if product_64bit > np.iinfo(np.int32).max:
        print(f"  ❌ CRITICAL: Product overflows int32!")
        print(f"     cudaMemset size argument will be WRONG")
        print(f"     Host allocation 'new float[...]' may also overflow")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    diagnose(data_dir)