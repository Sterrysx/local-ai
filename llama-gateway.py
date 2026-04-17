#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import atexit
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Issue F: Dynamic Portability
HOME = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME, ".config/locai/config.json")
STATE_DIR = os.path.join(HOME, ".local/share/llama-services")
LOG_DIR = os.path.join(STATE_DIR, "logs")
import secrets
API_KEY = os.getenv("LOCAI_API_KEY")
if not API_KEY:
    API_KEY = secrets.token_hex(16)
    print(f"WARNING: LOCAI_API_KEY not set. Generated random key: {API_KEY}")

class ModelOrchestrator:
    def __init__(self, config):
        self.config = config
        self.models = config["models"]
        self.settings = config["settings"]
        self.state_lock = threading.RLock()
        self.load_lock = threading.Lock()
        self.request_cv = threading.Condition(self.state_lock)
        self.backend_process = None
        self.current_model = None
        self.pending_model = None
        self.state = "idle"
        self.active_requests = 0
        self.last_request_at = int(time.time())

    def cleanup_orphans(self):
        """Issue C: Kill any llama-server processes left over."""
        try:
            port = self.settings.get("backend_port", 18081)
            result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
            for pid in result.stdout.strip().split():
                os.kill(int(pid), signal.SIGKILL)
        except Exception as e: logging.error(e)

    _gpu_vram_cache = {"time": 0, "data": []}
    def get_gpu_vram(self):
        if time.time() - self._gpu_vram_cache["time"] < 1.0: return self._gpu_vram_cache["data"]
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            self._gpu_vram_cache["data"] = [{"idx": int(l.split(",")[0]), "used": int(l.split(",")[1]), "total": int(l.split(",")[2])} for l in res.stdout.strip().splitlines()]
            self._gpu_vram_cache["time"] = time.time()
            return self._gpu_vram_cache["data"]
        except Exception as e:
            logging.error(e)
            return []

    def check_vram_preflight(self, model_name):
        """Issue D: Per-GPU Fit logic."""
        gpus = self.get_gpu_vram()
        if not gpus: return True, None
        required_mib = 1024 * (40 if "80B" in model_name else (35 if "70B" in model_name else 18))
        per_gpu_req = required_mib / len(gpus)
        for g in gpus:
            if (g["total"] - g["used"]) < per_gpu_req:
                return False, f"GPU {g['idx']} low VRAM ({per_gpu_req/1024:.1f}GB needed)"
        return True, None

    def stop_backend(self, force=False):
        with self.state_lock:
            if self.active_requests > 0 and not force: return
            if self.backend_process:
                try:
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGTERM)
                    self.backend_process.wait(timeout=2)
                except Exception as e: logging.error(e)
            self.backend_process = None
            self.current_model = None
            self.state = "idle"

    def ensure_model(self, model_name):
        # 1. Non-blocking state check
        with self.state_lock:
            if self.current_model == model_name and self.state == "ready":
                self.active_requests += 1
                return True
            if self.current_model and self.current_model != model_name:
                self.pending_model = model_name
                while self.active_requests > 0: self.request_cv.wait(timeout=1)
                self.stop_backend(force=True)
        
        # 2. Serialized Loading
        with self.load_lock:
            with self.state_lock:
                if self.current_model == model_name:
                    self.active_requests += 1
                    return True
            ok, err = self.check_vram_preflight(model_name)
            if not ok: raise RuntimeError(err)
            
            launcher = os.path.expanduser(self.models[model_name]["launcher"])
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, f"{model_name}.log"), "ab") as log:
                self.backend_process = subprocess.Popen([launcher], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.current_model = model_name
                self.state = "loading"
            
            # Issue B: Non-blocking health loop
            for _ in range(120):
                try:
                    with urllib.request.urlopen(f"{self.settings['backend_base']}/health", timeout=1) as r:
                        if r.status == 200:
                            with self.state_lock:
                                self.state = "ready"
                                self.active_requests += 1
                                self.pending_model = None
                            return True
                except: time.sleep(2)
            self.stop_backend(force=True)
            raise RuntimeError(f"Timeout loading {model_name}")

    def get_state_json(self):
        """Regression Fix: Restore telemetry state."""
        with self.state_lock:
            pid = self.backend_process.pid if self.backend_process else None
            vram_used = 0
            if pid:
                try:
                    res = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                    for line in res.stdout.strip().splitlines():
                        p, m = line.split(",")
                        if int(p.strip()) == pid: vram_used += int(m.strip())
                except Exception as e: logging.error(e)
                
            return {
                "active_model": self.current_model, "pending_model": self.pending_model,
                "backend_state": self.state, "active_requests": self.active_requests,
                "backend_vram_mib": vram_used,
                "available_models": list(self.models.keys()),
                "profiles": {m: {"ctx_size": 8192 if "80B" in m else 16384, "vram_weight_gb": 40 if "80B" in m else (38 if "70B" in m else (19 if "32B" in m else 5))} for m in self.models},
                "updated_at": int(time.time())
            }

ORCHESTRATOR = None

class SecureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    
    def check_auth(self):
        """Issue E: Bearer Token protection."""
        if self.headers.get("Authorization") != f"Bearer {API_KEY}":
            self.send_response(401); self.end_headers()
            self.wfile.write(b'{"error": "Unauthorized"}')
            return False
        return True

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.check_auth(): return
        body_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(body_len)
        try: payload = json.loads(body.decode())
        except: return self._json(400, {"error": "Invalid JSON"})
        
        model = payload.get("model")
        if self.path.startswith("/v1/chat"):
            try: ORCHESTRATOR.ensure_model(model)
            except Exception as e: return self._json(503, {"error": "Internal Server Error"})
            try:
                req = urllib.request.Request(f"{ORCHESTRATOR.settings['backend_base']}{self.path}", data=body, method="POST")
                for k,v in self.headers.items():
                    if k.lower() not in ["host", "authorization", "content-length"]: req.add_header(k, v)
                with urllib.request.urlopen(req, timeout=3600) as resp:
                    self.send_response(resp.status)
                    is_sse = resp.headers.get("Content-Type", "").startswith("text/event-stream")
                    for k,v in resp.headers.items():
                        if k.lower() not in ["transfer-encoding", "content-length", "connection"]: self.send_header(k, v)
                    if is_sse: self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    if is_sse:
                        while True:
                            line = resp.readline()
                            if not line: break
                            self.wfile.write(hex(len(line))[2:].encode() + b"\r\n" + line + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                    else: self.wfile.write(resp.read())
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                for k,v in e.headers.items():
                    if k.lower() not in ["transfer-encoding", "content-length", "connection"]: self.send_header(k, v)
                err_body = e.read()
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
            except Exception as e:
                self._json(500, {"error": "Internal Server Error"})
            finally:
                with ORCHESTRATOR.state_lock:
                    ORCHESTRATOR.active_requests -= 1
                    ORCHESTRATOR.request_cv.notify_all()
        elif self.path == "/internal/load":
            try: 
                ORCHESTRATOR.ensure_model(model)
                with ORCHESTRATOR.state_lock:
                    ORCHESTRATOR.active_requests -= 1
                    ORCHESTRATOR.request_cv.notify_all()
                self._json(200, {"status": "loaded"})
            except Exception as e: self._json(503, {"error": "Internal Server Error"})

    def do_GET(self):
        if not self.check_auth(): return
        if self.path == "/internal/status": self._json(200, ORCHESTRATOR.get_state_json())
        elif self.path == "/v1/models": self._json(200, {"data": [{"id": m} for m in ORCHESTRATOR.models]})

class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True

def main():
    global ORCHESTRATOR
    with open(CONFIG_PATH, "r") as f: config = json.load(f)
    ORCHESTRATOR = ModelOrchestrator(config)
    ORCHESTRATOR.cleanup_orphans()
    atexit.register(lambda: ORCHESTRATOR.stop_backend(force=True))
    
    # Dual-Server Threading
    def serve_public(): ReusableServer((config["settings"]["public_host"], config["settings"]["public_port"]), SecureHandler).serve_forever()
    def serve_admin(): ReusableServer((config["settings"]["admin_host"], config["settings"]["admin_port"]), SecureHandler).serve_forever()
    
    threading.Thread(target=serve_public, daemon=True).start()
    print(f"Gateway running on {config['settings']['public_host']}:{config['settings']['public_port']} (Public) and {config['settings']['admin_host']}:{config['settings']['admin_port']} (Admin)")
    serve_admin()

if __name__ == "__main__":
    main()
