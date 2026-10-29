"""personakit-runner: lets the PersonaKit Studio page (https://personakit.denniskoshta.com) run Qwen3.5 on your compute,
on this machine's GPU (--local) or on your own Modal account (--modal, with your own `modal` login).

    uvx --from "personakit[local] @ git+https://github.com/DennisKoshta/personakit" personakit-runner --local
    uvx --from "personakit[modal] @ git+https://github.com/DennisKoshta/personakit" personakit-runner --modal

It listens on 127.0.0.1 only, answers only that page (its Origin), and only once you've typed the pairing code it
prints there. The page's server still works out the steering vectors and clamp lists (CPU work); your prompts and the
replies go between the page and this program, and your keys never leave this machine.

The page talks to it over HTTP, with server-sent events for anything streamed:
  GET  /hello                    {"runner": "personakit", "version", "mode"} (no code needed)
  GET  /status                   {"state": off|starting|live, "model", "info", "error", "mode"}
  POST /start {"model"}          status events while it loads, then {"type": "ready", "info"} or an error
  POST /chat {"rows", "max_new", "temperature", "seed"}
                                 the replies streamed (personakit.live.chat's events, typed as delta, done, error)
  POST /concept {"about", "layer"}
                                 {"mean": [d floats]}: the concept's mean residual leaving that layer
  POST /stop                     stop the model (on Modal, the app and its GPU)
Every request but /hello carries the code as X-Pair.
"""
import argparse
import json
import os
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORIGIN = "https://personakit.denniskoshta.com"
PORT = 8796
VERSION = 1
MODELS = ("4b", "9b")
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
MAX_BODY = 4 * 2 ** 20
MAX_TRIES = 20  # wrong codes before it refuses everything until restarted
INSTALL = 'uvx --from "personakit[{}] @ git+https://github.com/DennisKoshta/personakit" personakit-runner --{}'


class Runner:
    """One model at a time, on this machine's GPU or in an ephemeral Modal app on the user's account."""

    def __init__(self, mode: str, idle_minutes: float):
        self.mode, self.idle = mode, idle_minutes * 60
        self.state, self.model, self.info, self.error = "off", None, None, None
        self.lock = threading.Lock()       # starts and stops
        self.gpu = threading.Lock()        # one generation at a time
        self.ready = threading.Event()
        self.halt = threading.Event()
        self.thread = None
        self.engine = None                 # local: (model, tok); modal: the class instance
        self.means = {}                    # (model, about) -> [L, d], so each concept is derived once
        self.last_used = time.time()

    def status(self) -> dict:
        return {"state": self.state, "model": self.model, "info": self.info, "error": self.error, "mode": self.mode}

    def start(self, model: str):
        """Start `model` (in the background), unless it's already starting or live."""
        with self.lock:
            self.last_used = time.time()
            if self.model == model and self.state in ("starting", "live"):
                return
            self._stop()
            self.state, self.model, self.info, self.error = "starting", model, None, None
            self.ready, self.halt = threading.Event(), threading.Event()
            self.thread = threading.Thread(target=self._run, args=(model, self.ready, self.halt), daemon=True)
            self.thread.start()

    def _run(self, model, ready, halt):
        try:
            if self.mode == "local":
                from . import engine as E
                from . import live
                print(f"[runner] loading Qwen3.5-{model.upper()} (the first time downloads its weights)", flush=True)
                m, tok, self.info = live.load(model)
                self.engine, self.state = (m, tok), "live"
                print(f"[runner] Qwen3.5-{model.upper()} is ready on {self.info['gpu']}", flush=True)
                ready.set()
                halt.wait()
                self.engine = None
                del m, tok
                E.free()
            else:
                import modal
                from . import live_modal
                print(f"[runner] starting Qwen3.5-{model.upper()} on your Modal account (the first time builds the "
                      "image and downloads the weights there: a few minutes)", flush=True)
                with modal.enable_output(), live_modal.app.run():
                    eng = live_modal.ENGINES[model]()
                    self.info = eng.info.remote()
                    self.engine, self.state = eng, "live"
                    print(f"[runner] Qwen3.5-{model.upper()} is ready on {self.info['gpu']} (your Modal account; "
                          f"it stops after {self.idle / 60:g} idle minutes)", flush=True)
                    ready.set()
                    halt.wait()
                    self.engine = None
                print("[runner] stopped the Modal app", flush=True)
        except Exception as e:  # noqa: BLE001 (shown on the page)
            self.error = f"{type(e).__name__}: {e}"[:500]
            print(f"[runner] couldn't start Qwen3.5-{model.upper()}: {self.error}", flush=True)
        finally:
            self.engine = None
            if self.halt is halt:
                self.state = "off"
            ready.set()

    def _stop(self):
        if self.thread and self.thread.is_alive():
            self.halt.set()
            self.thread.join(timeout=120)
        self.state, self.engine = "off", None

    def stop(self):
        with self.lock:
            self._stop()

    def wait(self, model: str):
        """Status events until `model` is up; the last is ready or an error."""
        t0 = time.time()
        self.start(model)
        ready = self.ready
        while not ready.wait(2):
            yield {"type": "status", "stage": "runner", "elapsed": round(time.time() - t0), "runner": self.status()}
        if self.state == "live" and self.model == model:
            yield {"type": "ready", "info": self.info}
        else:
            yield {"type": "error", "message": self.error or "The model stopped before it was ready."}

    def chat(self, rows, max_new, temperature, seed):
        if self.state != "live":
            yield {"type": "error", "message": "The model isn't running on your runner."}
            return
        with self.gpu:
            self.last_used = time.time()
            evs = None
            try:
                if self.mode == "local":
                    from . import live
                    evs = live.chat(*self.engine, rows, max_new, temperature, seed)
                else:
                    evs = self.engine.chat.remote_gen(rows, max_new, temperature, seed)
                for ev in evs:
                    if "row" in ev:
                        yield {"type": "delta", **ev}
                    elif "done" in ev:
                        yield {"type": "done", **ev}
                    elif "error" in ev:
                        yield {"type": "error", "message": ev["error"]}
            except Exception as e:  # noqa: BLE001 (shown on the page)
                yield {"type": "error", "message": f"{type(e).__name__}: {e}"[:500]}
            finally:
                if hasattr(evs, "close"):  # the page went: stop generating
                    evs.close()
                self.last_used = time.time()

    def concept(self, about: str, layer: int):
        if self.state != "live":
            raise RuntimeError("The model isn't running on your runner.")
        k = (self.model, about)
        if k not in self.means:
            with self.gpu:
                self.last_used = time.time()
                if self.mode == "local":
                    from . import live
                    self.means[k] = live.concept_means(*self.engine, about).tolist()
                else:
                    self.means[k] = self.engine.concept_means.remote(about)
                self.last_used = time.time()
        return self.means[k][layer]

    def reaper(self):
        """On Modal the GPU bills while it's up: stop it after the idle minutes without a request."""
        while True:
            time.sleep(20)
            if self.mode == "modal" and self.state == "live" and not self.gpu.locked() \
                    and time.time() - self.last_used > self.idle:
                print(f"[runner] no request for {self.idle / 60:g} minutes: stopping", flush=True)
                self.stop()


def code_of(text: str) -> str:
    return "".join(c for c in str(text).upper() if c in ALPHABET)


def handler(runner: Runner, code: str, origins: set[str], port: int):
    tries = [0]
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "personakit-runner"

        def log_message(self, *args):  # quiet: no request lines, which would show what's asked
            pass

        def cors(self):
            origin = self.headers.get("Origin")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def reply(self, status: int, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self, paired: bool = True) -> bool:
            """Only the page (its Origin), on this machine's own address (not a rebound name), with the code."""
            if self.headers.get("Host") not in hosts:
                self.reply(403, {"error": "wrong host"})
                return False
            if self.headers.get("Origin") not in origins:
                self.reply(403, {"error": f"this runner only answers {', '.join(sorted(origins))}"})
                return False
            if not paired:
                return True
            if tries[0] >= MAX_TRIES:
                self.reply(429, {"error": "Too many wrong codes. Restart the runner for a new one."})
                return False
            if not secrets.compare_digest(code_of(self.headers.get("X-Pair", "")), code):
                tries[0] += 1
                self.reply(401, {"error": "That isn't the code the runner shows."})
                return False
            return True

        def body(self) -> dict | None:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self.reply(413, {"error": "request too large"})
                return None
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self.reply(400, {"error": "not JSON"})
                return None

        def events(self, evs):
            self.send_response(200)
            self.cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for ev in evs:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page went away; closing the generator stops the generation
            finally:
                if hasattr(evs, "close"):
                    evs.close()

        def do_OPTIONS(self):
            self.send_response(204)
            if self.headers.get("Origin") in origins and self.headers.get("Host") in hosts:
                self.cors()
                self.send_header("Access-Control-Allow-Methods", "GET, POST")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Pair")
                self.send_header("Access-Control-Max-Age", "600")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path == "/hello":
                if self.allowed(paired=False):
                    self.reply(200, {"runner": "personakit", "version": VERSION, "mode": runner.mode})
            elif self.path == "/status":
                if self.allowed():
                    self.reply(200, runner.status())
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/start", "/chat", "/concept", "/stop"):
                return self.reply(404, {"error": "not found"})
            if not self.allowed() or (b := self.body()) is None:
                return
            if self.path == "/start":
                model = b.get("model")
                if model not in MODELS:
                    return self.reply(400, {"error": f"no model {model!r}"})
                self.events(runner.wait(model))
            elif self.path == "/chat":
                try:
                    rows = b["rows"][:4]
                    args = int(min(b.get("max_new", 200), 600)), float(b.get("temperature", 0)), int(b.get("seed", 0))
                except (KeyError, TypeError, ValueError):
                    return self.reply(400, {"error": "bad chat request"})
                self.events(runner.chat(rows, *args))
            elif self.path == "/concept":
                try:
                    self.reply(200, {"mean": runner.concept(str(b["about"])[:200], int(b["layer"]))})
                except (KeyError, ValueError, IndexError) as e:
                    self.reply(400, {"error": f"bad concept request: {e}"})
                except Exception as e:  # noqa: BLE001
                    self.reply(500, {"error": f"{type(e).__name__}: {e}"[:500]})
            else:
                runner.stop()
                self.reply(200, runner.status())

    return Handler


def check(mode: str):
    """Exit with what to install, or what's missing, before anything starts."""
    if mode == "local":
        try:
            import torch
            import transformers  # noqa: F401
        except ImportError as e:
            sys.exit(f"personakit-runner --local needs {e.name}: {INSTALL.format('local', 'local')}")
        if not torch.cuda.is_available():
            print("[runner] no NVIDIA GPU found: the model would run on this machine's "
                  f"{'Apple GPU' if torch.backends.mps.is_available() else 'CPU'}, which is far slower. "
                  "--modal runs it on your Modal account instead.", flush=True)
    else:
        try:
            import modal  # noqa: F401
        except ImportError:
            sys.exit(f"personakit-runner --modal needs modal: {INSTALL.format('modal', 'modal')}")
        config = os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")
        if not (os.environ.get("MODAL_TOKEN_ID") or os.path.exists(config)):
            sys.exit("personakit-runner --modal runs on your own Modal account: log in first with `uvx modal setup` "
                     "(or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET).")


def main():
    ap = argparse.ArgumentParser(prog="personakit-runner", description=__doc__.split("\n\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--local", action="store_true", help="run on this machine's GPU")
    where.add_argument("--modal", action="store_true", help="run on your own Modal account (your `modal` login)")
    ap.add_argument("--port", type=int, default=PORT, help=f"port on 127.0.0.1 (default {PORT})")
    ap.add_argument("--origin", action="append", default=[], help=f"another page allowed to use it besides {ORIGIN} "
                    "(for testing a Studio of your own)")
    ap.add_argument("--idle-minutes", type=float, default=10, help="--modal: stop the GPU after this long unused")
    args = ap.parse_args()
    mode = "local" if args.local else "modal"
    check(mode)

    runner = Runner(mode, args.idle_minutes)
    code = "".join(secrets.choice(ALPHABET) for _ in range(8))
    origins = {ORIGIN, *args.origin}
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(runner, code, origins, args.port))
    except OSError as e:
        sys.exit(f"personakit-runner: can't listen on 127.0.0.1:{args.port} ({e}). Is it already running? "
                 "--port picks another port (tell the page the same one).")
    server.daemon_threads = True
    threading.Thread(target=runner.reaper, daemon=True).start()
    where = "this machine's GPU" if mode == "local" else "your Modal account"
    print(f"\npersonakit-runner: Qwen3.5 on {where}\n\n"
          f"  Open {ORIGIN}, then Run on your own compute → Your own GPU,\n"
          f"  and enter this code:  {code[:4]}-{code[4:]}\n\n"
          f"  Listening on http://127.0.0.1:{args.port}, for that page only. The model starts with your first "
          "message.\n  Ctrl-C stops it all.\n", flush=True)
    def quit_(*_):
        raise KeyboardInterrupt
    for sig in ("SIGTERM", "SIGHUP"):  # a closed terminal or a kill stops the model too, as Ctrl-C does
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), quit_)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[runner] stopping", flush=True)
    finally:
        server.server_close()
        runner.stop()


if __name__ == "__main__":
    main()
