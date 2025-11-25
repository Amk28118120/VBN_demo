#!/usr/bin/env python3
"""
kalman_service.py
KF daemon: receives measurements from VBN via /tmp/kf_socket (UNIX DGRAM).
Reads latest camera JPEG frame from /tmp/vbn_last.jpg (written atomically by VBN).
Overlays HUD (KF state / analytic / PnP) on top of latest frame and streams with MJPEGServer.
"""

import numpy as np
import time
import socket, os, json, signal, sys
import cv2
from web_stream import MJPEGServer
from display import draw_hud, draw_pattern, draw_reprojections, draw_raw_points
# --- paste or import your EKF functions here (the ones you previously had) ---
# We'll assume the propagate() and measurement_update() functions are defined
# along with euler_to_R, R_to_euler, wrap, etc.
# (Copy the EKF functions you already had into this file or import them.)

# For brevity in this message I assume the EKF helper functions (propagate,
# measurement_update, prop_velocity, prop_ang_velocity, etc.) are present
# exactly as in your prior file; copy them into this module.



# ------------------------- EKF FUNCTIONS (unchanged logic) -------------------------

def euler_to_R(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z @ (R_y @ R_x)

def R_to_euler(R):
    pitch = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    roll  = np.arctan2(-R[1, 2], R[2, 2])
    yaw   = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([roll, pitch, yaw])

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def B_matrix(phi, th):
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(th), np.cos(th)
    ct = ct if abs(ct) > 1e-8 else 1e-8
    B = np.zeros((3,3))
    B[0,0] = 1.0
    B[0,1] = sp * (st/ct)
    B[0,2] = cp * (st/ct)
    B[1,1] = cp
    B[1,2] = -sp
    B[2,1] = sp/ct
    B[2,2] = cp/ct
    return B

def rodrigues_update(R, omega, dt):
    wx, wy, wz = omega
    w = np.linalg.norm(omega)
    if w < 1e-12:
        K = np.array([[0, -wz, wy],
                      [wz, 0, -wx],
                      [-wy, wx, 0]])
        return R @ (np.eye(3) + K * dt)
    kx, ky, kz = omega / w
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])
    angle = w * dt
    R_inc = (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K))
    return R @ R_inc

def prop_state(x, dt):
    px, py, pz = x[0:3]
    phi, th, psi = x[3:6]
    vx, vy, vz = x[6:9]
    wx, wy, wz = x[9:12]
    B = B_matrix(phi, th)
    omega = np.array([wx, wy, wz])
    _ = B @ omega
    R = euler_to_R(phi, th, psi)
    Rnew = rodrigues_update(R, omega, dt)
    phi2, th2, psi2 = R_to_euler(Rnew)
    px2 = px + dt * vx
    py2 = py + dt * vy
    pz2 = pz + dt * vz
    return np.array([
        px2, py2, pz2,
        phi2, th2, psi2,
        vx, vy, vz,
        wx, wy, wz
    ])

def prop_velocity(state, prev_state, dt):
    px1, py1, pz1 = state[0:3]
    px2, py2, pz2 = prev_state[0:3]
    dt_eff = dt if abs(dt) > 1e-8 else 1e-8
    vx = (px2 - px1) / dt_eff
    vy = (py2 - py1) / dt_eff
    vz = (pz2 - pz1) / dt_eff
    state_final = state.copy()
    state_final[6:9] = np.array([vx, vy, vz])
    return state_final

def prop_ang_velocity(state, prev_state, dt):
    phi1, th1, psi1 = state[3:6]
    phi2, th2, psi2 = prev_state[3:6]
    dt_eff = dt if abs(dt) > 1e-8 else 1e-8
    dphi = wrap(phi2 - phi1)
    dth  = wrap(th2 - th1)
    dpsi = wrap(psi2 - psi1)
    wx = dphi / dt_eff
    wy = dth  / dt_eff
    wz = dpsi / dt_eff
    state_final = state.copy()
    state_final[9:12] = np.array([wx, wy, wz])
    return state_final

def numerical_jacobian(x, dt):
    n = len(x)
    J = np.zeros((n, n))
    f0 = prop_state(x, dt)
    dx = 1e-6
    for i in range(n):
        x_tmp = np.copy(x)
        x_tmp[i] += dx
        f1 = prop_state(x_tmp, dt)
        J[:, i] = (f1 - f0) / dx
    return J

# Globals (tune as needed)
Q = np.eye(12) * 0.01
Rcov = np.eye(6) * 0.1
P_init = np.eye(12) * 0.1

def propagate(state, covariance, dt):
    state_final = prop_state(state, dt)
    jacobian = numerical_jacobian(state, dt)
    covariance_final = jacobian @ covariance @ jacobian.T + Q
    return state_final, covariance_final

def measurement_update(x_pred, P_pred, z, dt):
    H = np.hstack([np.eye(6), np.zeros((6,6))])  # (6x12)
    z_pred = H @ x_pred
    y = z - z_pred
    y[3:6] = wrap(y[3:6])
    S = H @ P_pred @ H.T + Rcov
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_post = x_pred + K @ y
    P_post = (np.eye(12) - K @ H) @ P_pred
    x_post = prop_velocity(x_post, x_pred, dt)
    x_post = prop_ang_velocity(x_post, x_pred, dt)
    return x_post, P_post

# --- Configuration ---
SOCKET_PATH = "/tmp/kf_socket"
FRAME_PATH = "/tmp/vbn_last.jpg"
PROP_HZ = 200.0       # KF propagation frequency (Hz); tune as desired
MJPEG_PORT = 8080

# --- setup socket ---
if os.path.exists(SOCKET_PATH):
    try: os.remove(SOCKET_PATH)
    except Exception: pass

srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
srv.bind(SOCKET_PATH)
srv.setblocking(False)

# MJPEG server (local stream)
streamer = MJPEGServer(host="0.0.0.0", port=MJPEG_PORT)

# cleanup on exit
def cleanup(signum=None, frame=None):
    try:
        srv.close()
    except:
        pass
    try:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
    except:
        pass
    try:
        streamer.stop()
    except:
        pass
    print("[kf_service] exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ---------- EKF globals (use your tuned values) ----------
Q = np.eye(12) * 0.01
Rcov = np.eye(6) * 0.1
P_init = np.eye(12) * 0.1

# === (You must copy your EKF helper functions here) ===
# For brevity, paste the implementations of:
# euler_to_R, R_to_euler, wrap, B_matrix, rodrigues_update,
# prop_state, prop_velocity, prop_ang_velocity,
# numerical_jacobian, propagate, measurement_update
# (These are identical to what you already had and should be copied into this file.)
# ------------------------------------------------------------------------------

def parse_measurement_packet(data_bytes):
    try:
        j = json.loads(data_bytes.decode("utf-8"))
    except Exception:
        raise ValueError("invalid json")
    if isinstance(j, dict) and "meas" in j:
        arr = j["meas"]
    elif isinstance(j, list) and len(j) == 6:
        arr = j
    else:
        raise ValueError("no meas field or not 6-element")
    arr = np.array(arr, dtype=float).reshape(6,)
    # incoming RPY are expected in degrees; convert to radians for EKF internals
    arr[3:6] = np.deg2rad(arr[3:6])
    return arr

def load_latest_frame():
    """Return BGR image (uint8) or None if not present."""
    if not os.path.exists(FRAME_PATH):
        return None
    try:
        img = cv2.imread(FRAME_PATH, cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception:
        return None

def overlay_and_stream(frame_bgr, hud_lines, kf_state=None, other_draws=None):
    """Draw HUD lines and any other overlays on a copy and stream via MJPEG server."""
    img = frame_bgr.copy() if frame_bgr is not None else np.zeros((480,640,3), dtype=np.uint8)
    # draw HUD using your draw_hud helper (lines are strings)
    draw_hud(img, hud_lines)
    # any extra overlays done externally (e.g., reproject points), if provided
    # encode to JPEG and update streamer
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        streamer.update(jpg.tobytes())

def main_loop(prop_freq=PROP_HZ):
    dt_target = 1.0 / float(prop_freq)
    x = np.zeros(12)
    P = P_init.copy()

    # init: wait for two measurements
    first_z = None; second_z = None
    first_t = None; second_t = None
    meas_count = 0
    ekf_started = False
    last_prop_time = time.time()
    last_meas_time = None
    prev_state = x.copy()

    print("[kf_service] waiting for first two measurements...")
    while meas_count < 2:
        try:
            data, _ = srv.recvfrom(4096)
        except BlockingIOError:
            time.sleep(0.01); continue
        except Exception as e:
            print("[kf_service] socket error during init:", e); time.sleep(0.1); continue
        tnow = time.time()
        try:
            z6 = parse_measurement_packet(data)
        except Exception as e:
            print("[kf_service] bad init packet:", e); continue
        meas_count += 1
        if meas_count == 1:
            first_z = z6.copy(); first_t = tnow
            print("[kf_service] got 1st meas:", first_z)
        else:
            second_z = z6.copy(); second_t = tnow
            print("[kf_service] got 2nd meas:", second_z)
            dt_meas = max(1e-6, second_t - first_t)
            s1 = np.zeros(12); s1[0:6] = first_z
            s2 = np.zeros(12); s2[0:6] = second_z
            s2 = prop_velocity(s2, s1, dt_meas)
            s2 = prop_ang_velocity(s2, s1, dt_meas)
            x = s2.copy()
            P = P_init.copy()
            ekf_started = True
            last_prop_time = time.time()
            last_meas_time = second_t
            print("[kf_service] initialized state.")

    print("[kf_service] entering main loop")
    last_frame_mtime = 0.0
    vbn_fps = 0.0
    # Frequency trackers
    kf_prop_count = 0
    kf_prop_last = time.time()
    
    kf_meas_count = 0
    kf_meas_last = time.time()
    
    cam_frame_count = 0
    cam_frame_last = time.time()
    
        
    while True:
        tloop = time.time()
        dt = tloop - last_prop_time
        last_prop_time = tloop
        if dt <= 0: dt = 1e-6

        if ekf_started:
            prev_state = x.copy()
            x, P = propagate(x, P, dt)
            kf_prop_count += 1
            now = time.time()
            if now - kf_prop_last >= 1.0:
                kf_prop_freq = kf_prop_count / (now - kf_prop_last)
                kf_prop_last = now
                kf_prop_count = 0

        # check for measurement (non-blocking)
        z_new = None
        try:
            data, _ = srv.recvfrom(4096)
            try:
                z_new = parse_measurement_packet(data)
            except Exception as e:
                print("[kf_service] malformed meas:", e)
                z_new = None
        except BlockingIOError:
            z_new = None
        except Exception as e:
            print("[kf_service] socket recv err:", e); z_new = None

        if z_new is not None:
            kf_meas_count += 1
            now = time.time()
            if now - kf_meas_last >= 1.0:
                kf_meas_freq = kf_meas_count / (now - kf_meas_last)
                kf_meas_last = now
                kf_meas_count = 0

            tmeas = time.time()
            dt_meas = tmeas - (last_meas_time if last_meas_time else tmeas)
            last_meas_time = tmeas
            x, P = measurement_update(x, P, z_new, dt_meas)
            # convert back to degrees for display
            display_rpy_deg = np.rad2deg(x[3:6].copy())
            print("[MEAS UPDATE] pos:", np.round(x[0:3],4), "rpy_deg:", np.round(display_rpy_deg,3))

        # check if latest frame file changed; if so, load it
        frame_img = None
        try:
            st = os.stat(FRAME_PATH)
            mtime = st.st_mtime
            if mtime != last_frame_mtime:
                img = load_latest_frame()
                if img is not None:
                    frame_img = img
                    cam_frame_count += 1
                    now = time.time()
                    if now - cam_frame_last >= 1.0:
                        cam_fps = cam_frame_count / (now - cam_frame_last)
                        cam_frame_last = now
                        cam_frame_count = 0
                    last_frame_mtime = mtime
                    # compute vbn FPS if measurement contains ts? we can approximate
                    # store timestamp of last frame - not necessary here
        except FileNotFoundError:
            frame_img = None
        except Exception:
            frame_img = None

        # Build HUD lines
        hud = []
        # KF state lines (display RPY in degrees)
        hud += [
            "KF State:",
            f"Pos: x={x[0]:.3f} y={x[1]:.3f} z={x[2]:.3f}",
            f"RPY: R={np.rad2deg(x[3]):.2f} P={np.rad2deg(x[4]):.2f} Y={np.rad2deg(x[5]):.2f}",
            f"Vel: vx={x[6]:.3f} vy={x[7]:.3f} vz={x[8]:.3f}",
        ]
        # Add frequencies to HUD
        hud += [
            f"KF Prop Freq: {kf_prop_freq if 'kf_prop_freq' in locals() else 0:.1f} Hz",
            f"Meas Freq: {kf_meas_freq if 'kf_meas_freq' in locals() else 0:.1f} Hz",
            f"Camera FPS: {cam_fps if 'cam_fps' in locals() else 0:.1f} Hz",
        ]

        # analytic + PnP info: optional if you want to display last received analytic values
        # You can extend hud with last analytic/pnp values if VBN sends them in JSON.

        # Draw HUD onto chosen base image (frame_img if present else black canvas)
        base_img = frame_img if frame_img is not None else np.zeros((480,640,3), dtype=np.uint8)
        overlay_and_stream(base_img, hud)

        # pace loop to prop_freq
        elapsed = time.time() - tloop
        to_sleep = dt_target - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

if __name__ == "__main__":
    try:
        main_loop(prop_freq=PROP_HZ)
    except Exception as e:
        print("[kf_service] fatal:", e)
    finally:
        cleanup()

