from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .inference import InputError, MAX_FILE_BYTES, PlateEngine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = os.getenv("PLATES_HOST", "127.0.0.1")
PORT = int(os.getenv("PLATES_PORT", "20145"))
API_TIMEOUT_SECONDS = int(os.getenv("PLATES_API_TIMEOUT", "15"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("plates")
ENGINE = PlateEngine()
CAPACITY = threading.BoundedSemaphore(value=1)

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "PlatesDemo/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        # Deliberately omit URLs, file names, payloads and recognition results.
        return

    def _headers(self, status: int, content_type: str, length: int, cache: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.end_headers()

    def _json(self, status: int, body: dict[str, object]) -> None:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self._headers(status, "application/json; charset=utf-8", len(payload), "no-store")
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ready", "model": "loaded"})
            return
        if path == "/":
            path = "/index.html"
        relative = path.lstrip("/")
        candidate = (STATIC / relative).resolve()
        if STATIC.resolve() not in candidate.parents or not candidate.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        payload = candidate.read_bytes()
        cache = "no-store" if candidate.suffix == ".html" else "public, max-age=3600"
        self._headers(HTTPStatus.OK, MIME.get(candidate.suffix, "application/octet-stream"), len(payload), cache)
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/analyze":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        length_raw = self.headers.get("Content-Length")
        try:
            length = int(length_raw or "")
        except ValueError:
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "length_required"})
            return
        if length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "empty"})
            return
        if length > MAX_FILE_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "file_too_large"})
            return
        if not CAPACITY.acquire(blocking=False):
            self.close_connection = True
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "busy", "retry_after": 3})
            return

        request_id = secrets.token_hex(4)
        payload = b""

        def expire_connection() -> None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(API_TIMEOUT_SECONDS, expire_connection)
        try:
            payload = self.rfile.read(length)
            timer.start()
            result = ENGINE.analyze(payload, self.headers.get("Content-Type", ""))
            timer.cancel()
            self._json(
                HTTPStatus.OK,
                {
                    "plates": result.plates,
                    "image": {"width": result.width, "height": result.height},
                    "elapsed_ms": result.elapsed_ms,
                },
            )
            LOG.info("analysis request=%s status=ok duration_ms=%d", request_id, result.elapsed_ms)
        except InputError as exc:
            timer.cancel()
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            LOG.info("analysis request=%s status=rejected reason=%s", request_id, str(exc))
        except (BrokenPipeError, ConnectionError):
            LOG.warning("analysis request=%s status=client_disconnected", request_id)
        except Exception:
            timer.cancel()
            # Never include exception strings: upstream libraries could echo input-derived data.
            LOG.error("analysis request=%s status=failed", request_id)
            try:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "processing_failed"})
            except (BrokenPipeError, ConnectionError):
                pass
        finally:
            timer.cancel()
            CAPACITY.release()
            del payload



def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOG.info("server ready host=%s port=%d", HOST, PORT)
    server.serve_forever(poll_interval=0.25)
    server.server_close()


if __name__ == "__main__":
    main()
