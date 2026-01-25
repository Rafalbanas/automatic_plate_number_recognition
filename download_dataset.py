from __future__ import annotations
from pathlib import Path
import argparse
import shutil
import os

import kagglehub


DATASET_ID = "piotrstefaskiue/poland-vehicle-license-plate-dataset"


def main():
    # Capture the project root explicitly before any operations that might change CWD
    project_root = Path(os.getcwd()).resolve()
    print(f"Debug: project_root = {project_root}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/raw/poland-plates",
        help="Docelowy folder w projekcie (default: data/raw/poland-plates)",
    )
    parser.add_argument(
        "--mode",
        choices=["use-cache", "copy", "symlink"],
        default="symlink",
        help="Jak podpiąć dane: use-cache (tylko wypisz ścieżkę), copy (skopiuj), symlink (utwórz dowiązanie)",
    )
    args = parser.parse_args()

    cache_dir = Path(kagglehub.dataset_download(DATASET_ID)).resolve()
    print(f"Debug: cache_dir (KaggleHub download location) = {cache_dir}")

    print(f"Debug: CWD before 'out' calculation = {Path.cwd()}")

    # Construct 'out' path relative to the explicitly captured project_root
    # Avoid .resolve() on the combined path to prevent unexpected canonicalization
    out = project_root / args.out
    print(f"Debug: out (intended project destination - direct join) = {out}")

    print("\u2705 KaggleHub cache dir:", cache_dir)

    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "use-cache":
        print("\u27a1\ufe0f U\u017cywaj tej \u015bcie\u017cki w kodzie:", cache_dir)
        return

    if out.exists():
        print("\u2139\ufe0f Folder docelowy ju\u017c istnieje:", out)
        return

    if args.mode == "copy":
        shutil.copytree(cache_dir, out)
        print("\u2705 Skopiowano do:", out)
        return

    # symlink
    out.symlink_to(cache_dir, target_is_directory=True)
    print("\u2705 Symlink utworzony:", out, "->", cache_dir)


if __name__ == "__main__":
    main()
