"""
Validate the optimized system matrix against a reference.
Run BOTH the original (fixed int overflow) and optimized kernels, then compare.

Usage:
    python validate_sysmat.py <reference.sysmat> <optimized.sysmat>
"""
import sys
import numpy as np
from pathlib import Path

# Process ~400MB at a time
CHUNK_SIZE = 100_000_000 

def load_sysmat(path, n_det=5632, nx=128, ny=128, nz=128, n_rot=1):
    expected = n_det * nx * ny * nz * n_rot
    # memmap keeps the data on disk, only loading parts into RAM when accessed
    data = np.memmap(path, dtype=np.float32, mode='r')
    print(f"\nFile: {path}")
    print(f"  Elements: {len(data):,} (expected {expected:,})")
    
    if len(data) != expected:
        print(f"  WARNING: Size mismatch!")
        return None
    
    return data


def inspect(data, label=""):
    print(f"  Inspecting {label.strip()} in chunks...")
    nz_total = 0
    max_val = -np.inf
    min_pos = np.inf
    sum_pos = 0.0 # using float64 internally to prevent sum overflow
    n_nan = 0
    n_inf = 0
    n_neg = 0

    # Iterate through the memmap in memory-safe chunks
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        
        nz_total += np.count_nonzero(chunk)
        max_val = max(max_val, np.max(chunk))
        
        pos_mask = chunk > 0
        if np.any(pos_mask):
            pos_vals = chunk[pos_mask]
            min_pos = min(min_pos, np.min(pos_vals))
            sum_pos += np.sum(pos_vals, dtype=np.float64)
        
        n_nan += np.sum(np.isnan(chunk))
        n_inf += np.sum(np.isinf(chunk))
        n_neg += np.sum(chunk < 0)

    print(f"  {label}Nonzero: {nz_total:,} ({100.0*nz_total/len(data):.4f}%)")
    print(f"  {label}Max: {max_val:.8e}")
    if nz_total > 0:
        print(f"  {label}Min (nonzero): {min_pos:.8e}")
        print(f"  {label}Mean (nonzero): {(sum_pos / nz_total):.8e}")
    
    if n_nan > 0: print(f"  {label}NaN: {n_nan}")
    if n_inf > 0: print(f"  {label}Inf: {n_inf}")
    if n_neg > 0: print(f"  {label}Negative: {n_neg}")


def compare(ref, opt):
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    both_nz_total = 0
    ref_only_total = 0
    opt_only_total = 0
    
    max_rel_err = 0.0
    sum_rel_err = 0.0
    
    max_lost_val = 0.0
    sum_lost_val = 0.0

    print("  Comparing arrays in chunks (this may take a minute)...")
    for i in range(0, len(ref), CHUNK_SIZE):
        r_chunk = ref[i : i + CHUNK_SIZE]
        o_chunk = opt[i : i + CHUNK_SIZE]
        
        both_nz = (r_chunk != 0) & (o_chunk != 0)
        ref_only = (r_chunk != 0) & (o_chunk == 0)
        opt_only = (r_chunk == 0) & (o_chunk != 0)
        
        both_nz_total += np.sum(both_nz)
        ref_only_total += np.sum(ref_only)
        opt_only_total += np.sum(opt_only)
        
        if np.any(both_nz):
            rb = r_chunk[both_nz]
            ob = o_chunk[both_nz]
            rel_err = np.abs(rb - ob) / np.abs(rb)
            
            max_rel_err = max(max_rel_err, np.max(rel_err))
            sum_rel_err += np.sum(rel_err, dtype=np.float64)
            
        if np.any(ref_only):
            lost = r_chunk[ref_only]
            max_lost_val = max(max_lost_val, np.max(lost))
            sum_lost_val += np.sum(lost, dtype=np.float64)

    if both_nz_total == 0 and ref_only_total == 0 and opt_only_total == 0:
        print("  Both are all zeros — no comparison possible.")
        return
        
    print(f"  Both nonzero: {both_nz_total:,}")
    print(f"  Ref-only nonzero: {ref_only_total:,}")
    print(f"  Opt-only nonzero: {opt_only_total:,}")
    
    if both_nz_total > 0:
        mean_rel_err = sum_rel_err / both_nz_total
        print(f"\n  Relative error (both-nonzero elements):")
        print(f"    Mean: {mean_rel_err:.6e}")
        print(f"    Max:  {max_rel_err:.6e}")
        print(f"    (Note: Median/Percentile omitted due to massive memory requirements)")
        
        if max_rel_err < 1e-4:
            print(f"  ✓ PASS: Max relative error < 1e-4 (float32 precision)")
        elif max_rel_err < 1e-2:
            print(f"  ~ MARGINAL: Max relative error < 1e-2")
        else:
            print(f"  ✗ FAIL: Significant differences detected")
            
    if ref_only_total > 0:
        mean_lost_val = sum_lost_val / ref_only_total
        print(f"\n  Elements nonzero in ref but zero in opt:")
        print(f"    Count: {ref_only_total:,}")
        print(f"    Max value: {max_lost_val:.6e}")
        print(f"    Mean value: {mean_lost_val:.6e}")
        if max_lost_val < 1e-10:
            print(f"    ✓ All negligible (< 1e-10)")
        else:
            print(f"    ⚠ Some significant values lost!")

    # Per-detector comparison
    # Slicing the memmap by detector is safe because 1 detector = ~2.1 million elements (8MB)
    n_det = 5632
    n_vox = len(ref) // n_det
    print(f"\n  Per-detector nonzero counts (sample):")
    for det_id in [0, 100, 500, 650, 1536, 3000, 5000]:
        if det_id >= n_det: continue
        ref_det = ref[det_id*n_vox:(det_id+1)*n_vox]
        opt_det = opt[det_id*n_vox:(det_id+1)*n_vox]
        print(f"    Det {det_id:5d}: ref={np.count_nonzero(ref_det):6d}, opt={np.count_nonzero(opt_det):6d}, "
              f"ref_max={np.max(ref_det):.4e}, opt_max={np.max(opt_det):.4e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_sysmat.py <reference.sysmat> [optimized.sysmat]")
        sys.exit(1)
    
    ref_data = load_sysmat(sys.argv[1])
    if ref_data is not None:
        inspect(ref_data, "REF ")
    
    if len(sys.argv) >= 3:
        opt_data = load_sysmat(sys.argv[2])
        if opt_data is not None:
            inspect(opt_data, "OPT ")
        
        if ref_data is not None and opt_data is not None:
            # Removed np.array()! Passed memmaps directly to avoid 94GB memory spike
            compare(ref_data, opt_data) 
    
    print("\nDone.")