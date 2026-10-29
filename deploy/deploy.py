#!/usr/bin/python3
"""Root-owned receiver; uploaded code runs only in plates-web.service."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

CONFIG_PATH = Path("/etc/plates/deploy.json")
SERVICE_UNIT = "plates-web.service"
HEALTH_BODY_LIMIT = 4096
MAX_HEALTH_ATTEMPTS = 20
MAX_HEALTH_INTERVAL = 30
MAX_LIMIT_BYTES = 16 * 1024 * 1024  # hard upper bound for both configured limits

REVISION_RE = re.compile(r"[a-f0-9]{40}")
DIGEST_RE = re.compile(r"[a-f0-9]{64}")
RELEASE_DIR_RE = re.compile(r"([a-f0-9]{40})-([a-f0-9]{16})")

# The entry allowlist is identical to RELEASE_FILES in scripts/package.py.
RELEASE_FILES = (
    "webapp/__init__.py",
    "webapp/inference.py",
    "webapp/server.py",
    "webapp/static/index.html",
    "webapp/static/app.css",
    "webapp/static/app.js",
    "webapp/static/examples/test123.png",
    "webapp/static/examples/demo456.png",
    "requirements-web.txt",
    "webapp/revision.txt",
)
_PY_FILES = ("webapp/__init__.py", "webapp/inference.py", "webapp/server.py")
_INDEX_FILE = "webapp/static/index.html"
_REVISION_FILE = "webapp/revision.txt"
_REQUIREMENTS_FILE = "requirements-web.txt"

REQUIRED_CONFIG_KEYS = (
    "state_dir",
    "releases_dir",
    "current_link",
    "health_url",
    "requirements_sha256",
    "max_archive_bytes",
    "max_extracted_bytes",
    "health_attempts",
    "health_interval",
)


def load_config(config: dict) -> dict:
    """Validate the config and return resolved internal paths."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    for key in ("state_dir", "releases_dir", "current_link"):
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path string")
    state_dir = Path(config["state_dir"])
    releases_dir = Path(config["releases_dir"])
    if releases_dir != state_dir / "releases":
        raise ValueError("releases_dir must be state_dir/releases")
    current_link = Path(config["current_link"])
    if current_link != state_dir / "current":
        raise ValueError("current_link must be state_dir/current")
    if releases_dir.is_symlink():
        raise ValueError("releases_dir must not be a symlink")
    cursor = current_link.parent
    while cursor != Path("") and cursor != Path("/"):
        if not cursor.is_dir():
            raise ValueError(f"missing parent directory for current_link: {cursor}")
        if cursor.is_symlink():
            raise ValueError(f"symlink in current_link parent chain: {cursor}")
        cursor = cursor.parent
    if not isinstance(config["health_url"], str) or not config["health_url"].startswith(("http://", "https://")):
        raise ValueError("health_url must be an http(s) URL")
    if not isinstance(config["requirements_sha256"], str) or not DIGEST_RE.fullmatch(config["requirements_sha256"]):
        raise ValueError("requirements_sha256 must be 64 lowercase hex characters")
    for key in ("max_archive_bytes", "max_extracted_bytes", "health_attempts", "health_interval"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if config["max_archive_bytes"] <= 0 or config["max_extracted_bytes"] <= 0:
        raise ValueError("archive size limits must be positive")
    if config["max_archive_bytes"] > MAX_LIMIT_BYTES or config["max_extracted_bytes"] > MAX_LIMIT_BYTES:
        raise ValueError(f"archive/extracted limits must not exceed {MAX_LIMIT_BYTES} bytes (16 MiB)")
    if not 1 <= config["health_attempts"] <= MAX_HEALTH_ATTEMPTS:
        raise ValueError(f"health_attempts must be between 1 and {MAX_HEALTH_ATTEMPTS}")
    if not 1 <= config["health_interval"] <= MAX_HEALTH_INTERVAL:
        raise ValueError(f"health_interval must be between 1 and {MAX_HEALTH_INTERVAL}")
    return {
        "state_dir": state_dir,
        "releases_dir": releases_dir,
        "current_link": current_link,
        "health_url": config["health_url"],
        "requirements_sha256": config["requirements_sha256"],
        "max_archive_bytes": config["max_archive_bytes"],
        "max_extracted_bytes": config["max_extracted_bytes"],
        "health_attempts": config["health_attempts"],
        "health_interval": config["health_interval"],
        "lock_file": state_dir / "deploy.lock",
        "rollback_file": state_dir / "rollback.json",
    }


def _acquire_lock(cfg: dict) -> int:
    """Take the deploy lock nonblocking; fail if another deploy holds it."""
    fd = os.open(str(cfg["lock_file"]), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError("another deploy is already running") from exc
    return fd


def _extract_archive(data: bytes, cfg: dict, staging: Path) -> None:
    """Extract the archive into staging, enforcing the allowlist, file metadata,
    create system, and per-file/cumulative limits on the real bytes written."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {exc}") from exc
    with archive:
        infos = archive.infolist()
        if sorted(info.filename for info in infos) != sorted(RELEASE_FILES):
            raise ValueError("archive entries must match the release allowlist exactly")
        limit = cfg["max_extracted_bytes"]
        if sum(info.file_size for info in infos) > limit:
            raise ValueError("archive exceeds the total extracted limit")
        done = 0
        for info in infos:
            name = info.filename
            if info.create_system != 3 or not stat.S_ISREG(info.external_attr >> 16):
                raise ValueError(f"archive entry is not a regular Unix file: {name}")
            if info.flag_bits & 1:
                raise ValueError("encrypted release entries are not permitted")
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as target:
                while True:
                    chunk = source.read(min(65536, limit - done + 1))
                    if not chunk:
                        break
                    done += len(chunk)
                    if done > limit:
                        raise ValueError("extracted bytes exceed the total limit")
                    target.write(chunk)
            destination.chmod(0o644)
    for directory in staging.rglob("*"):
        if directory.is_dir():
            directory.chmod(0o755)


def _validate_release_content(staging: Path, cfg: dict, revision: str) -> None:
    """Check revision.txt, non-empty index, python sources, requirements hash."""
    revision_text = (staging / _REVISION_FILE).read_text(encoding="ascii").strip()
    if not REVISION_RE.fullmatch(revision_text) or revision_text != revision:
        raise ValueError("revision.txt in the archive does not match the given revision")
    if not (staging / _INDEX_FILE).read_bytes().strip():
        raise ValueError("index.html in the archive is empty")
    for name in _PY_FILES:
        try:
            ast.parse((staging / name).read_bytes())
        except SyntaxError as exc:
            raise ValueError(f"python file in the archive does not parse: {name} ({exc})") from exc
    requirements_sha = hashlib.sha256((staging / _REQUIREMENTS_FILE).read_bytes()).hexdigest()
    if requirements_sha != cfg["requirements_sha256"]:
        raise ValueError("requirements-web.txt SHA-256 does not match the config")


def _current_release(cfg: dict) -> tuple[Path, str, str]:
    """Resolve the existing current symlink to a valid previous release dir."""
    link = cfg["current_link"]
    if not link.is_symlink():
        raise ValueError("the current release link is missing or not a symlink")
    target = link.resolve()
    if target.parent != cfg["releases_dir"].resolve():
        raise ValueError("current release must be inside state_dir/releases")
    if not target.is_dir():
        raise ValueError("the current link does not resolve to a directory")
    match = RELEASE_DIR_RE.fullmatch(target.name)
    if not match:
        raise ValueError(f"the current release directory is not named <revision>-<digest16>: {target}")
    revision, digest16 = match.groups()
    if not (target / _REVISION_FILE).is_file():
        raise ValueError("the current release has no webapp/revision.txt")
    if (target / _REVISION_FILE).read_text(encoding="ascii").strip() != revision:
        raise ValueError("the current release revision does not match its directory name")
    return target, revision, digest16


def _same_release(release_dir: Path, staging: Path) -> bool:
    """True when an existing release dir is byte-identical to the staged extraction."""
    if release_dir.is_symlink() or not release_dir.is_dir():
        return False
    entries = list(release_dir.rglob("*"))
    if any(path.is_symlink() for path in entries):
        return False
    files = {path.relative_to(release_dir).as_posix() for path in entries if path.is_file()}
    return files == set(RELEASE_FILES) and all(
        stat.S_ISREG((release_dir / name).stat().st_mode)
        and (release_dir / name).read_bytes() == (staging / name).read_bytes()
        for name in RELEASE_FILES
    )


def _replace_symlink(cfg: dict, target: Path) -> None:
    """Atomically swap the current symlink onto a fixed temporary name."""
    current = cfg["current_link"]
    temporary = Path(str(current) + ".next")
    if os.path.lexists(str(temporary)):
        os.unlink(str(temporary))
    os.symlink(str(target), str(temporary))
    os.replace(str(temporary), str(current))


def _restart_service() -> None:
    subprocess.run(["systemctl", "restart", SERVICE_UNIT], check=True, timeout=45)


def _check_health(cfg: dict, revision: str, index_sha256: str) -> None:
    """Poll healthz; fail if the response is too large or inconsistent."""
    attempts = cfg["health_attempts"]
    interval = cfg["health_interval"]
    failure: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(cfg["health_url"], timeout=5) as response:
                body = response.read(HEALTH_BODY_LIMIT + 1)
            if len(body) > HEALTH_BODY_LIMIT:
                raise ValueError("health response exceeds 4096 bytes")
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get("status") != "ready" or payload.get("model") != "loaded":
                raise ValueError("health response is not ready/loaded")
            if payload.get("revision") != revision:
                raise ValueError("health revision does not match the deployed release")
            if payload.get("index_sha256") != index_sha256:
                raise ValueError("health index SHA-256 does not match the deployed release")
            return
        except Exception as exc:
            failure = exc
            time.sleep(interval)
    raise RuntimeError(f"health check failed after {attempts} attempts: {failure}")


def _write_atomic(path: Path, payload: str) -> None:
    """Write a file atomically (temp file + rename), always 644, with safe cleanup."""
    fd, tmp_name = tempfile.mkstemp(suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def deploy(config: dict, revision: str, digest: str, stream) -> dict:
    """Deploy a validated archive to the new release directory and switch the
    current symlink atomically; roll back only after a successful switch."""
    cfg = load_config(config)
    if not REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be 40 lowercase hex characters")
    if not DIGEST_RE.fullmatch(digest):
        raise ValueError("digest must be 64 lowercase hex characters (SHA-256)")
    limit = cfg["max_archive_bytes"]
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("archive is larger than the configured limit")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("archive SHA-256 does not match the given digest")

    state_dir = cfg["state_dir"]
    releases_dir = cfg["releases_dir"]
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(state_dir), 0o755)
    releases_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(releases_dir), 0o755)

    lock_fd = _acquire_lock(cfg)
    staging = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f"staging-{revision}-", dir=str(releases_dir)))
        # Phase 1: complete pre-switch validation before anything is moved, so
        # a rollback is always possible; validation failures raise without a restart.
        previous_dir, previous_revision, previous_digest16 = _current_release(cfg)
        previous_index = hashlib.sha256((previous_dir / _INDEX_FILE).read_bytes()).hexdigest()
        _extract_archive(data, cfg, staging)
        _validate_release_content(staging, cfg, revision)
        release_dir = releases_dir / f"{revision}-{digest[:16]}"
        reused = os.path.lexists(release_dir)
        if reused and not _same_release(release_dir, staging):
            raise ValueError(f"release directory already exists with different contents: {release_dir}")
        if not reused:
            os.chmod(str(staging), 0o755)
            os.replace(str(staging), str(release_dir))

        # Phase 2: the switch and everything after it; any failure rolls back.
        _replace_symlink(cfg, release_dir)
        try:
            _restart_service()
            index_sha = hashlib.sha256((release_dir / _INDEX_FILE).read_bytes()).hexdigest()
            _check_health(cfg, revision, index_sha)
            if previous_dir != release_dir:
                _write_atomic(cfg["rollback_file"], json.dumps(
                    {"previous_revision": previous_revision, "previous_digest16": previous_digest16},
                    separators=(",", ":"),
                ))
        except BaseException as error:
            try:
                _replace_symlink(cfg, previous_dir)
                _restart_service()
                _check_health(cfg, previous_revision, previous_index)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"deploy failed and the rolled-back release is unhealthy: {error}; {rollback_error}"
                ) from rollback_error
            raise RuntimeError(
                f"deploy failed; the service was rolled back to the previous release: {error}"
            ) from error
    finally:
        # Clean only this deploy's staging directory; the current and previous
        # releases are never removed here.
        if staging is not None and staging.exists():
            shutil.rmtree(str(staging))
        os.close(lock_fd)
    return {
        "revision": revision,
        "digest16": digest[:16],
        "release_dir": str(release_dir),
        "previous_revision": previous_revision,
        "previous_digest16": previous_digest16,
    }


def status(config: dict) -> str:
    """Return the revision of the release the current link points at."""
    _, revision, _digest16 = _current_release(load_config(config))
    return revision


def main() -> None:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot load {CONFIG_PATH}: {error}", file=sys.stderr)
        sys.exit(1)
    tokens = sys.argv[1:]
    try:
        if tokens == ["status"]:
            print(status(config))
        elif len(tokens) == 2:
            result = deploy(config, tokens[0], tokens[1], sys.stdin.buffer)
            print(json.dumps(result, sort_keys=True))
        else:
            raise ValueError("expected 'status' or '<revision> <digest>' arguments")
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
