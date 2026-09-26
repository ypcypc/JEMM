import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 16_000_000


def make_handler(model):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status, body):
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                self._send(200, {"status": "READY", "model": model.name})
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self):
            if not self.path.startswith("/v1/systemone"):
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                with lock:
                    response = model.respond(body)
            except (ValueError, KeyError, TypeError) as exc:
                self._send(422, {"error": type(exc).__name__, "detail": str(exc)[:200]})
                return
            except Exception as exc:
                self._send(500, {"error": type(exc).__name__, "detail": str(exc)[:200]})
                raise
            self._send(200, response)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Serve POST /v1/systemone.")
    ap.add_argument("--adapter", required=True, help="Hugging Face repo id or local directory")
    ap.add_argument("--base", help="base model; defaults to the one in adapter_config.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from .model import DecisionModel
    model = DecisionModel.from_pretrained(a.adapter, base=a.base, device=a.device)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(model))
    server.daemon_threads = True
    print(json.dumps({"listening": f"http://{a.host}:{a.port}", "model": model.name}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
