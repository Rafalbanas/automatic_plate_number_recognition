from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import package as pkg
from package import REVISION_FILE, RELEASE_FILES, SOURCE_FILES


def make_root(root: Path) -> Path:
    """Copy the explicit application sources into a throwaway tree (no models, data or venv)."""
    target = root / "root"
    for rel in SOURCE_FILES:
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, destination)
    return target


class PackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="plates-release-")
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_deterministic_zip_with_exact_allowlist(self) -> None:
        root = make_root(self.tmpdir)
        first = self.tmpdir / "first.zip"
        second = self.tmpdir / "second.zip"
        sha = "a" * 40
        pkg.package(root, first, sha)
        pkg.package(root, second, sha)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(archive.namelist(), list(RELEASE_FILES))
            self.assertEqual(archive.read(REVISION_FILE), (sha + "\n").encode("ascii"))

        # No revision file is written to the working tree or repository.
        self.assertFalse((root / REVISION_FILE).is_file())
        self.assertFalse((REPO / REVISION_FILE).is_file())

    def test_invalid_revisions_are_rejected(self) -> None:
        root = make_root(self.tmpdir)
        out = self.tmpdir / "invalid.zip"
        for bad in ("abc", "A" * 40, "g" * 40, "a" * 39):
            with self.assertRaises(ValueError):
                pkg.package(root, out, bad)

    def test_source_and_parent_symlinks_are_rejected(self) -> None:
        root = make_root(self.tmpdir)
        out = self.tmpdir / "symlink.zip"
        sha = "a" * 40

        # Symlink on a parent directory component (root/webapp -> real dir).
        (root / "webapp").rename(root / "webapp-real")
        (root / "webapp").symlink_to(root / "webapp-real")
        with self.assertRaises(ValueError):
            pkg.package(root, out, sha)
        (root / "webapp").unlink()
        (root / "webapp-real").rename(root / "webapp")

        # Symlink on a file (source -> repository file).
        source = root / "webapp" / "inference.py"
        source.unlink()
        source.symlink_to(REPO / "webapp" / "inference.py")
        with self.assertRaises(ValueError):
            pkg.package(root, out, sha)

    def test_cli_takes_two_arguments_and_writes_no_local_revision(self) -> None:
        out = self.tmpdir / "cli.zip"
        sha = "a" * 40
        with patch.object(sys, "argv", ["package.py", sha, str(out)]):
            pkg.main()

        with zipfile.ZipFile(out) as archive:
            self.assertEqual(archive.namelist(), list(RELEASE_FILES))
            self.assertEqual(archive.read(REVISION_FILE), (sha + "\n").encode("ascii"))
        # main() derives root from the script location; the repository tree
        # must still contain no revision file afterwards.
        self.assertFalse((REPO / REVISION_FILE).is_file())


class HealthTest(unittest.TestCase):
    def test_healthz_stable_after_files_change_after_import(self) -> None:
        import webapp.inference

        engine = MagicMock(name="FakePlateEngine")
        with patch.object(webapp.inference, "PlateEngine", engine):
            import webapp.server as server

        facts = server.HEALTH_FACTS
        self.assertEqual(facts["revision"], "development")
        self.assertEqual(
            facts["index_sha256"],
            hashlib.sha256(
                (REPO / "webapp" / "static" / "index.html").read_bytes()
            ).hexdigest(),
        )

        server_srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        host, port = server_srv.server_address
        thread = threading.Thread(target=server_srv.serve_forever, daemon=True)
        thread.start()
        try:
            first = urllib.request.urlopen(f"http://{host}:{port}/healthz")
            self.assertEqual(first.getcode(), 200)
            first_body = json.loads(first.read())
            first.close()

            # Files "change" after import: any per-request re-read would fail,
            # while the once-captured startup facts must still serve 200.
            with patch.object(Path, "read_bytes", side_effect=OSError("files changed")), patch.object(
                Path, "read_text", side_effect=OSError("files changed")
            ):
                second = urllib.request.urlopen(f"http://{host}:{port}/healthz")
                self.assertEqual(second.getcode(), 200)
                second_body = json.loads(second.read())
                second.close()

            self.assertEqual(first_body, second_body)
            self.assertEqual(first_body["status"], "ready")
            self.assertEqual(first_body["model"], "loaded")
            self.assertEqual(first_body["revision"], facts["revision"])
            self.assertEqual(
                first_body["index_sha256"],
                hashlib.sha256(
                    (REPO / "webapp" / "static" / "index.html").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(first.headers.get("Cache-Control"), "no-store")
            self.assertEqual(second.headers.get("Cache-Control"), "no-store")
        finally:
            server_srv.shutdown()
            server_srv.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
