"""Receiver tests: all service control and HTTP are forbidden unless mocked."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deploy"))
sys.path.insert(0, str(REPO / "scripts"))
import deploy as receiver
import package as packager

OLD = "0" * 40
NEW = "a" * 40
REQUIREMENTS = (REPO / "requirements-web.txt").read_bytes()


def files(revision=NEW):
    return {
        name: ((revision + "\n").encode() if name == "webapp/revision.txt"
               else (REPO / name).read_bytes())
        for name in receiver.RELEASE_FILES
    }


def archive(contents=None, mode_overrides=None, extra=()):
    contents = files() if contents is None else contents
    output = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(output, "w") as z:
        warnings.simplefilter("ignore", UserWarning)
        for name, data in [*contents.items(), *extra]:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (mode_overrides or {}).get(name, 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    return output.getvalue()


class ReceiverTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="plates-deploy-test-")
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)
        self.old = self.state / "releases" / (OLD + "-" + "b" * 16)
        for name, data in files(OLD).items():
            target = self.old / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.current = self.state / "current"
        self.current.symlink_to(self.old)
        self.cfg = {
            "state_dir": str(self.state),
            "releases_dir": str(self.state / "releases"),
            "current_link": str(self.current),
            "health_url": "http://127.0.0.1:20145/healthz",
            "requirements_sha256": hashlib.sha256(REQUIREMENTS).hexdigest(),
            "max_archive_bytes": 16 * 1024 * 1024,
            "max_extracted_bytes": 16 * 1024 * 1024,
            "health_attempts": 1,
            "health_interval": 1,
        }
        # These patches are in place for EVERY test, including rejection tests.
        self.run = self.enterContext(patch.object(
            receiver.subprocess, "run", side_effect=AssertionError("Unexpected service control")
        ))
        self.http = self.enterContext(patch.object(
            receiver.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network")
        ))
        self.enterContext(patch.object(receiver.time, "sleep"))

    def deploy(self, payload=None, revision=NEW, digest=None):
        payload = archive() if payload is None else payload
        return receiver.deploy(self.cfg, revision, digest or hashlib.sha256(payload).hexdigest(), io.BytesIO(payload))

    def allow_restart(self):
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess(["systemctl"], 0)

    def healthy(self, revision=NEW, index=None):
        if index is None:
            index = hashlib.sha256(files(revision)["webapp/static/index.html"]).hexdigest()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "status": "ready", "model": "loaded", "revision": revision, "index_sha256": index,
        }).encode()
        return response

    def assert_unchanged(self):
        self.assertEqual(self.current.resolve(), self.old)
        self.run.assert_not_called()
        self.http.assert_not_called()
        self.assertFalse(list((self.state / "releases").glob("staging-*")))

    def test_allowlist_matches_packager(self):
        self.assertEqual(receiver.RELEASE_FILES, packager.RELEASE_FILES)

    def test_successful_deploy_and_idempotent_retry(self):
        self.allow_restart()
        self.http.side_effect = None
        self.http.return_value = self.healthy()
        payload = archive()
        result = self.deploy(payload)
        release = Path(result["release_dir"])
        self.assertEqual(self.current.resolve(), release)
        self.assertEqual(receiver.status(self.cfg), NEW)
        for name, data in files().items():
            self.assertEqual((release / name).read_bytes(), data)
            self.assertEqual((release / name).stat().st_mode & 0o777, 0o644)
        self.assertEqual((release / "webapp").stat().st_mode & 0o777, 0o755)
        metadata = (self.state / "rollback.json").read_bytes()
        self.assertEqual(json.loads(metadata)["previous_revision"], OLD)
        self.deploy(payload)
        self.assertEqual((self.state / "rollback.json").read_bytes(), metadata)
        self.run.assert_called_with(["systemctl", "restart", "plates-web.service"], check=True, timeout=45)

    def test_real_packager_is_accepted(self):
        payload = self.state / "release.zip"
        packager.package(REPO, payload, NEW)
        self.allow_restart()
        self.http.side_effect = None
        self.http.return_value = self.healthy()
        self.deploy(payload.read_bytes())
        self.assertEqual(receiver.status(self.cfg), NEW)

    def test_invalid_revision_digest_and_digest_mismatch(self):
        for kwargs in ({"revision": "A" * 40}, {"digest": "short"}, {"digest": "f" * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.deploy(**kwargs)
        self.assert_unchanged()

    def test_bounded_archive_read(self):
        self.cfg["max_archive_bytes"] = 100
        stream = MagicMock()
        stream.read.return_value = b"x" * 101
        with self.assertRaises(ValueError):
            receiver.deploy(self.cfg, NEW, "a" * 64, stream)
        stream.read.assert_called_once_with(101)
        self.assert_unchanged()

    def test_invalid_archive_entries_are_rejected_without_restart(self):
        source = files()
        candidates = [b"not zip", archive({k: v for k, v in source.items() if k != "webapp/server.py"})]
        candidates += [archive(extra=[(name, b"bad")]) for name in
                       ("../outside.py", "/absolute", "data/image.jpg", "weights.pt", "webapp/server.py")]
        candidates += [archive(mode_overrides={"webapp/server.py": mode}) for mode in (0o120777, 0o040755, 0o010600)]
        for payload in candidates:
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                self.deploy(payload)
        self.assert_unchanged()

    def test_content_validation(self):
        variants = [
            ("webapp/revision.txt", (OLD + "\n").encode()),
            ("webapp/static/index.html", b" \n"),
            ("webapp/server.py", b"def broken("),
            ("requirements-web.txt", b"unreviewed-dependency\n"),
        ]
        for name, data in variants:
            content = files()
            content[name] = data
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.deploy(archive(content))
        self.assert_unchanged()

    def test_expanded_size_limit(self):
        content = files()
        content["webapp/static/app.js"] = b"x" * 200000
        self.cfg["max_extracted_bytes"] = 100000
        with self.assertRaisesRegex(ValueError, "extracted limit"):
            self.deploy(archive(content))
        self.assert_unchanged()

    def test_missing_bootstrap_is_rejected(self):
        self.current.unlink()
        with self.assertRaisesRegex(ValueError, "current release link"):
            self.deploy()
        self.run.assert_not_called()

    def test_current_must_remain_inside_releases(self):
        outside = self.state / self.old.name
        self.old.rename(outside)
        self.current.unlink()
        self.current.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "inside"):
            self.deploy()
        self.run.assert_not_called()

    def test_lock_prevents_concurrent_deploy(self):
        with (self.state / "deploy.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.deploy()
        self.assert_unchanged()

    def test_health_failure_rolls_back_and_verifies_previous(self):
        self.allow_restart()
        self.http.side_effect = [self.healthy(OLD), self.healthy(OLD)]
        with self.assertRaisesRegex(RuntimeError, "rolled back"):
            self.deploy()
        self.assertEqual(self.current.resolve(), self.old)
        self.assertEqual(self.run.call_count, 2)
        self.assertEqual(self.http.call_count, 2)

    def test_restart_failure_rolls_back(self):
        self.run.side_effect = [subprocess.CalledProcessError(1, "systemctl"), None]
        self.http.side_effect = [self.healthy(OLD)]
        with self.assertRaisesRegex(RuntimeError, "rolled back"):
            self.deploy()
        self.assertEqual(self.current.resolve(), self.old)
        self.assertEqual(self.run.call_count, 2)

    def test_failed_rollback_is_explicit(self):
        self.allow_restart()
        self.http.side_effect = [self.healthy(OLD), self.healthy(NEW)]
        with self.assertRaisesRegex(RuntimeError, "rolled-back release is unhealthy"):
            self.deploy()
        self.assertEqual(self.current.resolve(), self.old)

    def test_metadata_failure_rolls_back(self):
        self.allow_restart()
        self.http.side_effect = [self.healthy(), self.healthy(OLD)]
        with patch.object(receiver, "_write_atomic", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(RuntimeError, "rolled back"):
                self.deploy()
        self.assertEqual(self.current.resolve(), self.old)

    def test_stale_digest_revision_and_oversized_health_rejected(self):
        cfg = receiver.load_config(self.cfg)
        for response in (self.healthy(OLD), self.healthy(index="f" * 64)):
            self.http.side_effect = None
            self.http.return_value = response
            with self.assertRaises(RuntimeError):
                receiver._check_health(cfg, NEW, hashlib.sha256(files()["webapp/static/index.html"]).hexdigest())
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"x" * 4097
        self.http.return_value = response
        with self.assertRaises(RuntimeError):
            receiver._check_health(cfg, NEW, "a" * 64)

    def test_existing_release_symlink_or_tamper_rejected(self):
        payload = archive()
        release = self.state / "releases" / (NEW + "-" + hashlib.sha256(payload).hexdigest()[:16])
        release.symlink_to(self.old)
        with self.assertRaises(ValueError):
            self.deploy(payload)
        release.unlink()
        release.mkdir()
        with self.assertRaises(ValueError):
            self.deploy(payload)
        self.assert_unchanged()

    def test_status_cli_prints_only_sha_and_ignores_stdin(self):
        config = self.state / "config.json"
        config.write_text(json.dumps(self.cfg))
        output = io.StringIO()
        with patch.object(receiver, "CONFIG_PATH", config), patch.object(sys, "argv", ["receiver", "status"]):
            with patch.object(sys, "stdout", output):
                receiver.main()
        self.assertEqual(output.getvalue(), OLD + "\n")
        with patch.object(receiver, "CONFIG_PATH", config), patch.object(sys, "argv", ["receiver"]):
            with patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                receiver.main()
        self.assert_unchanged()

    def test_cli_deploy_passes_bounded_stream_directly(self):
        config = self.state / "config.json"
        config.write_text(json.dumps(self.cfg))
        self.allow_restart()
        self.http.side_effect = None
        self.http.return_value = self.healthy()
        payload = archive()
        digest = hashlib.sha256(payload).hexdigest()
        stream = MagicMock()
        stream.buffer = io.BytesIO(payload)
        with patch.object(receiver, "CONFIG_PATH", config), patch.object(sys, "argv", ["receiver", NEW, digest]):
            with patch.object(sys, "stdin", stream), patch.object(sys, "stdout", io.StringIO()):
                receiver.main()
        self.assertEqual(receiver.status(self.cfg), NEW)

    def test_config_rejects_unbounded_limits_and_custom_switch_paths(self):
        for key, value in (("max_archive_bytes", 1 << 30), ("max_extracted_bytes", 0),
                           ("current_link", str(self.state / "other")),
                           ("releases_dir", str(self.state / "other"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                receiver.load_config({**self.cfg, key: value})


class WrapperTest(unittest.TestCase):
    def test_restricted_dispatch_without_real_sudo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_sudo = root / "sudo"
            fake_sudo.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake_sudo.chmod(0o755)
            for command in ("status", f"deploy {NEW} {'b' * 64}"):
                result = subprocess.run(["/bin/bash", str(REPO / "deploy/ssh-command.sh")],
                    env={"PATH": str(root), "SSH_ORIGINAL_COMMAND": command}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0)
                args = result.stdout.splitlines()
                self.assertEqual(args[:2], ["-n", "/usr/local/sbin/plates-deploy"])
                self.assertEqual(args[2:], ["status"] if command == "status" else [NEW, "b" * 64])
            for command in ("", "bash", "status\n", "status; id", "deploy a b", f"deploy\n{NEW} {'b' * 64}"):
                result = subprocess.run(["/bin/bash", str(REPO / "deploy/ssh-command.sh")],
                    env={"PATH": str(root), "SSH_ORIGINAL_COMMAND": command}, capture_output=True)
                self.assertEqual(result.returncode, 2)
