# main.py — VBN demo with headless MJPEG streaming
import os
import time
import cv2
import numpy as np

from capture_frame import FrameSource
from led_detection import detect_leds
from led_detection_man import detect_leds_python
from pose_estimation import identify_and_order, estimate_pose
from display import draw_raw_points, draw_pattern, draw_reprojections, draw_hud
from web_stream import MJPEGServer  # <-- tiny HTTP MJPEG server you copied
import platform

# If user sets HEADLESS explicitly, respect it. Otherwise, auto:
# - On Linux: headless when DISPLAY is empty (typical over SSH)
# - On Windows/macOS: default to GUI (headless=False)
_headless_env = os.environ.get("HEADLESS", "").strip()
if _headless_env != "":
    HEADLESS = _headless_env == "1"
else:
    HEADLESS = (platform.system() == "Linux" and os.environ.get("DISPLAY", "") == "")

STREAM = os.environ.get("HEADLESS_STREAM", "0") == "1"
SNAP_EVERY_N = int(os.environ.get("SNAP_EVERY_N", "0"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

# ---------- Pattern geometry (meters) ----------
R = 0.010  # radius from center to each of the 4 outer LEDs

# ---------- Camera intrinsics & distortion ----------
camMatrix = np.array([[908.62425565  , 0, 643.93436085],
                    [0, 908.92570486, 393.45889572],
                    [0, 0, 1.0]], dtype=np.float64)

distCoeff = np.array([ 2.18984921e-01, -5.80493965e-01, 1.15200278e-04,
                      -2.04177566e-03, 4.48611005e-01], dtype=np.float64)

def _los_from_center(center_uv, K):
    """Compute LOS azimuth/elevation (deg) from pattern center and intrinsics."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = float(center_uv[0]), float(center_uv[1])
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    az  = np.degrees(np.arctan2(x_n, 1.0))    # Az (ψ)
    el  = np.degrees(np.arctan2(-y_n, 1.0))   # El (θ)
    return az, el

def main():
    cap = FrameSource()
    streamer = MJPEGServer(port=8080) if STREAM else None

    t0 = time.time(); frames = 0
    print(f"[run] HEADLESS={HEADLESS} STREAM={STREAM} (browse http://<pi-ip>:8080/ if streaming)")
    priors = { "lock": False,"range" : 0, "centre" : [], "area_const" : 3236.41,"length_const" : 8600}
    try:
        while True:
            ok, frame, frame_colour = cap.read()
            #print(frame)
            print(frame.shape,frame.dtype)
            if not ok or frame is None:
                print("\n[capture] no frame from camera. Edit SOURCE in capture_frame.py or check camera.")
                time.sleep(0.02)
                continue

            frames += 1
            now = time.time()
            fps = frames / max(1e-6, (now - t0))

            # --- detect up to 5 bright blobs ---
            pts = detect_leds_python(frame, priors,20,5,max_pts=5)
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2) if pts is not None else np.empty((0,2), np.float32)

            vis = frame_colour.copy()
            draw_raw_points(vis, pts)  # cyan dots for all detections

            hud = []
            console_msg = ""
            
            if len(pts) == 5:
                hud += ["Lock:",f"{priors['lock']}",]
                res = identify_and_order(pts)
                if res["ok"]:
                    C        = res["center_uv"]
                    off      = res["offset_uv"]
                    outs     = res["outer_uv"]
                    ordered  = res["ordered_pts2d"]
                    priors["centre"] = res["center_uv"]
                    draw_pattern(vis, C, off, outs)
                    priors["lock"] = True

                    # --- hybrid pose: analytic + PnP ---
                    try:
                        pose = estimate_pose(ordered, camMatrix, distCoeff, R)
                    except Exception as e:
                        pose = {"analytic": {"ok": False, "reason": str(e)},
                                "pnp":      {"ok": False, "reason": "exception"}}

                    # ---- Analytic: RPY + LOS + Range ----
                    
                    az_a, el_a = _los_from_center(C, camMatrix)
                    ap = pose.get("analytic", {})
                    if ap.get("ok", False):
                        φa, θa, ψa = ap["rpy321_deg"]
                        Ra = ap["range_m"]
                        priors["range"] = Ra * 1000
                        
                        hud += [
                            "Analytic:",
                            f"R{φa:6.1f}deg  P{θa:6.1f}deg  Y{ψa:6.1f}deg",
                            f"LOS: Az{az_a:6.1f}deg  El{el_a:6.1f}deg   R{Ra:6.3f} m",
                        ]
                    else:
                        hud += ["Analytic: unavailable"]

                    # ---- PnP: RPY + LOS + Range ----
                    pnp = pose.get("pnp", {"ok": False})
                    if pnp.get("ok", False):
                        φp, θp, ψp = pnp["rpy321_deg"]
                        Rp = pnp["range_m"]
                        # Prefer PnP-derived LOS if available; else reuse center-based
                        if "AzEl_deg" in pnp:
                            az_p, el_p = pnp["AzEl_deg"]
                        else:
                            az_p, el_p = _los_from_center(C, camMatrix)

                        hud += [
                            "PnP:",
                            f"R{φp:6.1f}deg  P{θp:6.1f}deg  Y{ψp:6.1f}deg",
                            f"LOS: Az{az_p:6.1f}deg  El{el_p:6.1f}deg R{Rp:6.3f} m",
                        ]
                        # Safe reprojection overlay
                        try:
                            draw_reprojections(vis, pnp.get("reproj", None), pnp.get("z_cam", None))
                        except Exception:
                            pass
                    else:
                        hud += [f"PnP: {pnp.get('reason', pnp.get('error', 'unavailable'))}"]

                    console_msg = "Pattern OK; LEDs (px): " + np.array2string(pts, precision=1, floatmode='fixed')
                else:
                    hud += [res.get("reason", "pattern id failed")]
                    console_msg = res.get("reason", "pattern id failed")
            else:
                hud += ["Lock:",f"{priors['lock']}",]
                hud += [f"Need 5 LEDs (got {len(pts)})"]
                console_msg = f"Need 5 LEDs (got {len(pts)})"
                priors["lock"] = False

            draw_hud(vis, hud, fps=fps)

            # ---- Output: window / stream / snapshots ----
            if not HEADLESS and not STREAM:
                cv2.imshow("VBN | RPY + LOS + Range", vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                if cv2.waitKey(1) & 0xFF in (ord('s'), ord(' ')):
                    cv2.imwrite(f"vbn_{int(time.time())}.png", vis)
            else:
                # Encode once; stream or save
                ok_jpg, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok_jpg:
                    if streamer:
                        streamer.update(jpg.tobytes())
                    elif SNAP_EVERY_N > 0 and (frames % SNAP_EVERY_N == 0):
                        cv2.imwrite(f"vbn_{int(time.time())}.jpg", vis)

            # ---- Console status line (SSH-friendly) ----
            print(f"\rFPS: {fps:5.1f} | LEDs: {len(pts):d} | {console_msg:>s}   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        if streamer:
            try: streamer.stop()
            except Exception: pass
        cap.release()
        try: cv2.destroyAllWindows()
        except Exception: pass
        print("\n[run] shutdown complete.")

if __name__ == "__main__":
    main()
