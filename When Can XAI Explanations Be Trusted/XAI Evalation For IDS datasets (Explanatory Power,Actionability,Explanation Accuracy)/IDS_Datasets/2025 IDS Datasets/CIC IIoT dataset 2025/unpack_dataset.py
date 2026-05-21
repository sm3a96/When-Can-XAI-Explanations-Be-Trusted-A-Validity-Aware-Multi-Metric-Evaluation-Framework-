#!/usr/bin/env python3
import argparse
import os
import lzma
import sys
import hashlib
import tarfile
from pathlib import Path

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"

def sha256_file(path, bufsize=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def read_expected_sha256(sha256_path):
    """
    Supports common formats like:
      <hash>  filename
      <hash> *filename
      <hash>
    Returns the hex digest (str) or None if unreadable.
    """
    try:
        text = Path(sha256_path).read_text(errors="ignore").strip()
        if not text:
            return None
        # take first 64 hex chars from the first token/line
        first_line = text.splitlines()[0].strip()
        token = first_line.split()[0]
        # in case the file contains just the hash with no spaces
        token = token.strip()
        # keep only hex chars
        token = "".join(c for c in token if c.lower() in "0123456789abcdef")
        return token if len(token) == 64 else None
    except Exception:
        return None

def is_within_directory(directory, target):
    """Prevent path traversal on extraction."""
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

def safe_extract(tar: tarfile.TarFile, path: str):
    for member in tar.getmembers():
        target_path = os.path.join(path, member.name)
        if not is_within_directory(path, target_path):
            raise Exception(f"Blocked path traversal attempt: {member.name}")
    tar.extractall(path)

def extract_tar_xz(archive_path: Path, dest_dir: Path):
    """
    Try to extract as .tar.xz (tar inside xz).
    If that fails with a tar header error, fall back to raw .xz (single file).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1) Try as tar.xz
    try:
        with tarfile.open(archive_path, mode="r:xz") as tf:
            # (keep your safe_extract if you use it)
            for m in tf.getmembers():
                target = dest_dir / m.name
                # basic path traversal guard
                if not str(target.resolve()).startswith(str(dest_dir.resolve()) + os.sep):
                    raise Exception(f"Blocked path traversal attempt: {m.name}")
            tf.extractall(dest_dir)
            return "tarxz"
    except tarfile.ReadError:
        # Not a tar after xz decompress → fall back to raw xz stream
        pass

    # 2) Raw .xz: decompress to a single file named after the archive
    # e.g., attack_samples_10sec.tar.xz → attack_samples_10sec (remove .xz; if also endswith .tar, strip that too)
    out_name = archive_path.name
    if out_name.endswith(".xz"):
        out_name = out_name[:-3]
    # If the name ends with ".tar" (because the file is actually not a tar), drop it for a clean filename
    if out_name.endswith(".tar"):
        out_name = out_name[:-4]

    raw_out = dest_dir / out_name
    with lzma.open(archive_path, "rb") as fin, open(raw_out, "wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    return "rawxz"

def process(src_dir: Path, dest_dir: Path):
    checksum_dir = src_dir / "checksums"
    has_checksums = checksum_dir.is_dir()

    if not has_checksums:
        print(f"{YELLOW}No checksum dir at {checksum_dir} → extract unconditionally{RESET}")

    archives = sorted(src_dir.glob("*.tar.xz"))
    if not archives:
        print(f"{RED}No .tar.xz files found in {src_dir}{RESET}")
        return 0

    total = 0
    ok = 0
    for arc in archives:
        total += 1
        arc_name = arc.name
        out_subdir = dest_dir  # extract directly into dest_dir (customize if you want subfolders)

        if has_checksums:
            sha_file = checksum_dir / f"{arc_name}.sha256"
            if not sha_file.exists():
                print(f"{YELLOW}[{arc_name}] checksum file missing: {sha_file.name}  → SKIP{RESET}")
                continue
            expected = read_expected_sha256(sha_file)
            if not expected:
                print(f"[{arc_name}] invalid checksum file format: {sha_file.name}  → SKIP")
                continue
            actual = sha256_file(arc)
            if actual.lower() != expected.lower():
                print(f"{RED}[{arc_name}] checksum FAILED (expected {expected}, got {actual})  → SKIP{RESET}")
                continue
            print(f"{GREEN}[{arc_name}] checksum passed{RESET}")  # <-- prints 'passed' per your requirement
            try:
                kind = extract_tar_xz(arc, out_subdir)
                if kind == "tarxz":
                    print(f"{GREEN}[{arc_name}]{RESET} extracted (tar.xz) to {out_subdir}")
                else:
                    print(f"{GREEN}[{arc_name}]{RESET} decompressed raw .xz → {out_subdir}")
                ok += 1
            except Exception as e:
                print(f"{RED}[{arc_name}] extract error: {e}  → SKIP{RESET}")
        else:
            # No checksum dir → extract unconditionally
            try:
                extract_tar_xz(arc, out_subdir)
                print(f"{GREEN}[{arc_name}]{RESET} extracted to {out_subdir}")
                ok += 1
            except Exception as e:
                print(f"{RED}[{arc_name}] extract error: {e}  → SKIP{RESET}")

    print(f"{GREEN}\nDone. Extracted {ok}/{total} archives.{RESET}")
    return 0 if ok == total else 1

def main():
    ap = argparse.ArgumentParser(description="Verify and extract .tar.xz archives with optional checksums.")
    ap.add_argument("src_dir", help="Directory containing .tar.xz files (and optional checksum/ folder)")
    ap.add_argument("dest_dir", nargs="?", default=None,
                    help="Destination directory (defaults to src_dir)")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src_dir)).resolve()
    if not src.is_dir():
        print(f"{RED}Error: src_dir is not a directory: {src}{RESET}", file=sys.stderr)
        sys.exit(2)

    dest = Path(os.path.expanduser(args.dest_dir)).resolve() if args.dest_dir else src
    dest.mkdir(parents=True, exist_ok=True)

    sys.exit(process(src, dest))

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # run with CLI args
        main()
    else:
        # === CONFIGS FOR IDE RUN ===
        src_dir = "/Users/amir/PycharmProjects/iiot-anomaly-detection/datasense/dataset_publish/results/csv/temp"  # change to your path
        dest_dir = "/Users/amir/PycharmProjects/iiot-anomaly-detection/datasense/dataset_publish/results/csv/temp"  # None → extracts into src_dir

        from pathlib import Path

        src = Path(src_dir).expanduser().resolve()
        dest = Path(dest_dir).expanduser().resolve() if dest_dir else src

        process(src, dest)