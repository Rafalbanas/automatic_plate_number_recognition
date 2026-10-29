from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

SOURCE_FILES = (
    "webapp/__init__.py",
    "webapp/inference.py",
    "webapp/server.py",
    "webapp/static/index.html",
    "webapp/static/app.css",
    "webapp/static/app.js",
    "webapp/static/examples/test123.png",
    "webapp/static/examples/demo456.png",
    "requirements-web.txt",
)
REVISION_FILE = "webapp/revision.txt"
RELEASE_FILES = SOURCE_FILES + (REVISION_FILE,)

# The revision is the git commit SHA ($GITHUB_SHA in CI), not a hash computed
# here; it is validated and embedded in the archive only.
_REVISION_RE = re.compile(r"[a-f0-9]{40}")
# Fixed timestamps, file system, permission bits and entry order keep the
# archive byte-identical for a given tree and revision.
_ZIP_DATE_TIME = (2024, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o100644 << 16


def _validate_sources(root: Path) -> None:
    # Walk every path component without resolving, so a symlink at any
    # level (e.g. root/webapp -> another directory) is rejected.
    for rel in SOURCE_FILES:
        check = root
        for part in Path(rel).parts:
            check = check / part
            if check.is_symlink():
                raise ValueError(f"symlink in source path: {rel}")
        if not check.is_file():
            raise ValueError(f"missing source: {rel}")


def package(root: Path, output: Path, revision: str) -> None:
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError("invalid revision: expected 40 lowercase hex characters")
    _validate_sources(root)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in SOURCE_FILES:
            info = zipfile.ZipInfo(rel)
            info.date_time = _ZIP_DATE_TIME
            info.create_system = 3
            info.external_attr = _ZIP_EXTERNAL_ATTR
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (root / rel).read_bytes())
        # The revision file exists only inside the archive; nothing is
        # written to the working tree, the repository or elsewhere.
        info = zipfile.ZipInfo(REVISION_FILE)
        info.date_time = _ZIP_DATE_TIME
        info.create_system = 3
        info.external_attr = _ZIP_EXTERNAL_ATTR
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (revision + "\n").encode("ascii"))
    output.write_bytes(buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the release ZIP for a git commit SHA.")
    parser.add_argument("revision", help="Git commit SHA, 40 lowercase hex characters.")
    parser.add_argument("output", type=Path, help="Output ZIP path.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package(root, args.output, args.revision)


if __name__ == "__main__":
    main()
