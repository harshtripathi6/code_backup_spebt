#!/usr/bin/env python3
"""
usage:
  python pack_context.py --root /path/to/project --out context.txt \
    --exts .py .js .ts .tsx .json .md .html .css \
    --exclude-dir .git node_modules .venv __pycache__ dist build \
    --max-bytes 500000
"""
import argparse, os, sys, stat
from pathlib import Path

DEFAULT_EXCLUDE_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".idea", ".DS_Store", "tests_3d", "slurm_logs"}
DEFAULT_EXTS = set()  # empty means "all extensions"
DEFAULT_MAX_BYTES = 2_000_000  # skip very large files by default (2MB)

def is_binary_sample(b: bytes) -> bool:
    if not b:
        return False
    # Heuristic: NULL byte or high fraction of non-texty bytes
    if b"\x00" in b:
        return True
    # Count bytes outside common text range
    texty = sum(32 <= c <= 126 or c in (9,10,13) for c in b)
    return (texty / len(b)) < 0.75

def safe_read_text(path: Path, max_bytes: int) -> str | None:
    try:
        with path.open("rb") as fh:
            head = fh.read(min(4096, max_bytes))
            if is_binary_sample(head):
                return None
            rest = fh.read(max_bytes - len(head))
            data = head + rest
        # Try utf-8, then latin-1 as a fallback (lossless)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    except Exception:
        return None

def should_skip_file(p: Path, exts: set[str], max_bytes: int) -> bool:
    try:
        st = p.stat()
        # Skip sockets/devices, etc.
        if not stat.S_ISREG(st.st_mode):
            return True
        if st.st_size > max_bytes:
            return True
    except Exception:
        return True
    if exts and p.suffix not in exts:
        return True
    return False

def walk_files(root: Path, exclude_dirs: set[str]) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".DS_")]
        for name in filenames:
            files.append(Path(dirpath) / name)
    # stable order for reproducibility
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder to pack")
    ap.add_argument("--out", default="context.txt", help="Output text file")
    ap.add_argument("--exts", nargs="*", default=[], help="Whitelist of file extensions (e.g., .py .js). Empty = all")
    ap.add_argument("--exclude-dir", nargs="*", default=[], help="More directories to exclude by name")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Per-file size limit (bytes)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    exts = set(args.exts) if args.exts else DEFAULT_EXTS
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)

    files = walk_files(root, exclude_dirs)
    total_written = 0
    count = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for f in files:
            if should_skip_file(f, exts, args.max_bytes):
                continue
            content = safe_read_text(f, args.max_bytes)
            if content is None:
                continue
            header = f"\n===== BEGIN {f} =====\n"
            footer = f"\n===== END {f} =====\n"
            out.write(header)
            out.write(content)
            out.write(footer)
            total_written += len(header) + len(content) + len(footer)
            count += 1

    print(f"Packed {count} files into {out_path} ({total_written} bytes).")

if __name__ == "__main__":
    sys.exit(main())
