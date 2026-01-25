from __future__ import annotations
from pathlib import Path
import argparse
import shutil

import kagglehub


DATASET_ID = "piotrstefaskiue/poland-vehicle-license-plate-dataset"


def main():
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
    print("✅ KaggleHub cache dir:", cache_dir)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "use-cache":
        print("➡️ Używaj tej ścieżki w kodzie:", cache_dir)
        return

    if out.exists():
        print("ℹ️ Folder docelowy już istnieje:", out)
        return

    if args.mode == "copy":
        shutil.copytree(cache_dir, out)
        print("✅ Skopiowano do:", out)
        return

    # symlink
    out.symlink_to(cache_dir, target_is_directory=True)
    print("✅ Symlink utworzony:", out, "->", cache_dir)


if __name__ == "__main__":
    main()