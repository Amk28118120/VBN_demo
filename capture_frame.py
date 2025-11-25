# capture_frame.py
# Cross-platform capture:
# - Raspberry Pi (libcamera sensors like OV9281): Picamera2 with short exposure
# - Elsewhere: OpenCV VideoCapture (best-effort low exposure)
# Returns frames as BGR (OpenCV-ready)

import time, platform, cv2, numpy as np

# ===== minimal knobs =====
SOURCE        = 0          # OpenCV device index or "rtsp://..." URL (ignored by Picamera2)
WIDTH, HEIGHT = None, None # if None on Pi, we'll pick a sensible default per sensor
FPS           = 30
EXPOSURE_MS   = 0.2        # ~2–10 ms works well for LEDs
USE_PICAMERA2 = True       # set False to force OpenCV even on a Pi

# =========================


def decode_raw_shifted(raw_frame, shift_bits):
    """
    raw_frame: (H, W*2) uint8
    shift_bits: how many bits original value was shifted left
    """
    # interpret as little-endian uint16
    raw16 = raw_frame.view(np.uint16)

    # shift back down
    gray = (raw16 >> shift_bits).astype(np.uint8)

    return gray

def _is_pi():
    try:
        return platform.system() == "Linux" and platform.machine() in ("aarch64", "armv7l", "armv6l")
    except Exception:
        return False

class FrameSource:
    def __init__(self):
        self.COUNTER= 0
        self.picam2 = None
        self.cap = None
        self._using_picam2 = False

        # Prefer Picamera2 on Pi for libcamera sensors (e.g., OV9281)
        if USE_PICAMERA2 and _is_pi() and isinstance(SOURCE, int):
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()

                # Detect sensor model to pick a sane default size
                props = getattr(self.picam2, "camera_properties", {}) or {}
                model = str(props.get("Model", "")).lower()

                if WIDTH and HEIGHT:
                    size = (int(WIDTH), int(HEIGHT))
                else:
                    # OV9281 native is 1280x800; otherwise use 1280x720
                    size = (1280, 800) if "ov9281" in model else (1280, 720)

                cfg = self.picam2.create_video_configuration(
                    main={"size": size, "format": "RGB888"},
                    raw={'size': size,'format': 'R8'}     # or 'R8' depending on mode
                )
                self.picam2.configure(cfg)

                # Start with auto exposure briefly so the pipe settles
                self._set_controls_safe({"AeEnable": True})
                self.picam2.start()
                time.sleep(0.25)

                # Then lock short exposure & low gain for stable LED blobs
                us = int(max(1, float(EXPOSURE_MS) * 1000.0))  # microseconds
                self._set_controls_safe({
                    "AeEnable": False,
                    "ExposureTime": us,
                    "AnalogueGain": 1.0,
                })
                # Try to set a fixed frame duration (some sensors may ignore)
                if FPS:
                    frame_us = int(max(1, 1_000_000 // int(FPS)))
                    self._set_controls_safe({"FrameDurationLimits": (frame_us, frame_us)})

                # Disable AWB only if the control exists (OV9281 is mono; no AWB)
                self._set_controls_safe({"AwbEnable": False})

                self._using_picam2 = True
                print(f"[capture] Picamera2 active ({model or 'unknown sensor'}), size={size}, exp={us}us")
                return
            except Exception as e:
                print(f"[capture] Picamera2 not available/failed: {e}\n[capture] Falling back to OpenCV.")

        # OpenCV fallback (PC or Pi without Picamera2)
        self.cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW if platform.system()=="Windows" else 0)
        if not self.cap.isOpened():
            raise RuntimeError("OpenCV VideoCapture failed to open source.")
        if WIDTH:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(WIDTH))
        if HEIGHT: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(HEIGHT))
        if FPS:    self.cap.set(cv2.CAP_PROP_FPS,          int(FPS))
        # Best-effort manual exposure (some cameras/drivers ignore; harmless if so)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # V4L2 manual-ish
            # DShow expects negative log2(seconds); V4L2 sometimes accepts raw values
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            if hasattr(cv2, "CAP_PROP_AUTO_WB"):
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass
        print("[capture] OpenCV VideoCapture active.")

    # Set a control only if the driver advertises it (avoids "not advertised" errors)
    def _set_controls_safe(self, d):
        try:
            ctrl_map = getattr(self.picam2, "camera_controls", {}) or {}
            safe = {k: v for k, v in d.items() if k in ctrl_map}
            if safe:
                self.picam2.set_controls(safe)
        except Exception:
            pass

    def read(self):
        if self._using_picam2:
            print(self.COUNTER)
            arr = self.picam2.capture_array("raw")  
            colour_arr = self.picam2.capture_array("main") # RGB
            arr = decode_raw_shifted(arr, 8)
            if arr is None or arr.size == 0:
                return False, None
            return True, arr, cv2.cvtColor(colour_arr, cv2.COLOR_RGB2BGR)
        else:
            self.COUNTER = self.COUNTER+1
            ok, frame = self.cap.read()
            return ok, frame, colour_arr

    def release(self):
        if self._using_picam2 and self.picam2:
            try: self.picam2.stop()
            except Exception: pass
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass

# Optional generator if you like for-loops:
def capture_frame():
    src = FrameSource()
    try:
        while True:
            ok, frame = src.read()
            if not ok: break
            yield frame
    finally:
        src.release()
