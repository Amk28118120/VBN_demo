# web_stream.py — ultra-light MJPEG server for headless preview
import threading, time, socket
from http.server import BaseHTTPRequestHandler, HTTPServer

class _FrameStore:
    def __init__(self): 
        self._buf = None
        self._cv = threading.Condition()
        self._running = True
    def update(self, jpg_bytes: bytes):
        with self._cv:
            self._buf = jpg_bytes
            self._cv.notify_all()
    def get(self, wait=True, timeout=1.0):
        with self._cv:
            if wait and self._buf is None:
                self._cv.wait(timeout=timeout)
            return self._buf
    def stop(self):
        with self._cv: 
            self._running = False
            self._cv.notify_all()

class _Handler(BaseHTTPRequestHandler):
    server_version = "MJPEG/0.1"
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (b"<html><body><h2>VBN Stream</h2>"
                    b'<img src="/stream" style="max-width:100%"></body></html>')
            self.send_response(200); self.send_header("Content-Type","text/html")
            self.send_header("Content-Length", str(len(html))); self.end_headers()
            self.wfile.write(html); return
        if self.path == "/snapshot":
            frame = self.server.store.get(wait=True)
            if not frame:
                self.send_error(503, "no frame"); return
            self.send_response(200); self.send_header("Content-Type","image/jpeg")
            self.send_header("Content-Length", str(len(frame))); self.end_headers()
            self.wfile.write(frame); return
        if self.path == "/stream":
            boundary = "frameboundary"
            self.send_response(200)
            self.send_header("Age","0"); self.send_header("Cache-Control","no-cache, private")
            self.send_header("Pragma","no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.end_headers()
            while self.server.store._running:
                frame = self.server.store.get(wait=True, timeout=1.0)
                if not frame: continue
                try:
                    self.wfile.write(bytes(f"--{boundary}\r\n", "ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(bytes(f"Content-Length: {len(frame)}\r\n\r\n","ascii"))
                    self.wfile.write(frame); self.wfile.write(b"\r\n")
                except BrokenPipeError:
                    break
            return
        self.send_error(404, "not found")

class MJPEGServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.store = _FrameStore()
        self.httpd = HTTPServer((host, port), _Handler)
        self.httpd.store = self.store
        self._t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._t.start()
        # Get local IP for user-friendly message
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "YOUR_PI_IP"
        print(f"[stream] serving on http://{local_ip}:{port}/  (/stream, /snapshot)")
    def update(self, jpg_bytes: bytes):
        self.store.update(jpg_bytes)
    def stop(self):
        self.store.stop()
        try: self.httpd.shutdown()
        except: pass
