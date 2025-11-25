import numpy as np
import time
from queue import Queue, Empty

# ------------------------- EKF FUNCTIONS -------------------------

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
    # keep arcsin domain-safe
    pitch = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    roll  = np.arctan2(-R[1, 2], R[2, 2])
    yaw   = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([roll, pitch, yaw])

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def B_matrix(phi, th):
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(th), np.cos(th)

    # avoid divide by zero
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

    R_inc = (
        np.eye(3)
        + np.sin(angle) * K
        + (1 - np.cos(angle)) * (K @ K)
    )

    return R @ R_inc

def prop_state(x, dt):
    px, py, pz = x[0:3]
    phi, th, psi = x[3:6]
    vx, vy, vz = x[6:9]
    wx, wy, wz = x[9:12]

    # Euler rate from B * omega (kept for compatibility)
    B = B_matrix(phi, th)
    omega = np.array([wx, wy, wz])
    _ = B @ omega

    # propagate orientation
    R = euler_to_R(phi, th, psi)
    Rnew = rodrigues_update(R, omega, dt)
    phi2, th2, psi2 = R_to_euler(Rnew)

    # propagate position
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
R = np.eye(6) * 0.1
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
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_post = x_pred + K @ y
    P_post = (np.eye(12) - K @ H) @ P_pred

    # fill velocity components using finite-diff between corrected x_post and predicted x_pred
    x_post = prop_velocity(x_post, x_pred, dt)
    x_post = prop_ang_velocity(x_post, x_pred, dt)

    return x_post, P_post

# ------------------------- MAIN LOOP (implements your 6 steps) -------------------------

measurement_queue = Queue()

def to_state12(z6):
    """Convert 6x1 measurement to 12x1 state (top 6 = measurement, bottom 6 = zeros)"""
    s = np.zeros(12)
    s[0:6] = z6.reshape(6,)
    return s

def main_loop(prop_freq=20.0):
    dt_target = 1.0 / float(prop_freq)

    # state + covariance
    x = np.zeros(12)
    P = P_init.copy()

    # storage for first two measurements
    first_z6 = None
    second_z6 = None
    first_t = None
    second_t = None
    meas_count = 0
    ekf_started = False

    # prev_state only used for velocity finite difference while propagating
    prev_state = x.copy()
    last_prop_time = time.time()
    last_meas_time = None

    print("Waiting for first two measurements to initialize...")

    # ---- BLOCK until first two measurements are received ----
    while meas_count < 2:
        try:
            z = measurement_queue.get(timeout=0.1)  # block with timeout so program stays responsive
        except Empty:
            continue
        t_now = time.time()
        meas_count += 1
        if meas_count == 1:
            first_z6 = z.copy()
            first_t = t_now
            print("Received 1st measurement:", first_z6)
        elif meas_count == 2:
            second_z6 = z.copy()
            second_t = t_now
            print("Received 2nd measurement:", second_z6)

            dt_meas = second_t - first_t if abs(second_t - first_t) > 1e-8 else 1e-8

            # convert both to 12x1 states
            s1 = to_state12(first_z6)
            s2 = to_state12(second_z6)

            # compute velocities using your prop_velocity/prop_ang_velocity
            s2_with_v = prop_velocity(s2, s1, dt_meas)
            s2_with_vw = prop_ang_velocity(s2_with_v, s1, dt_meas)

            # initialize filter state with second measurement + computed v/omega
            x = s2_with_vw.copy()

            # initialize covariance (you can tune)
            P = np.eye(12) * 0.1

            ekf_started = True
            last_prop_time = time.time()
            last_meas_time = second_t

            print("Initialized state from first two measurements.")
            print("pos:", x[0:3], "euler:", x[3:6], "v:", x[6:9], "w:", x[9:12])

    # ---- main propagate / update loop ----
    while True:
        t_loop_start = time.time()
        dt = t_loop_start - last_prop_time
        last_prop_time = t_loop_start

        # 4) propagate until measurement arrives
        if ekf_started:
            prev_state_copy = prev_state.copy()
            prev_state = x.copy()
            x, P = propagate(x, P, dt)
            # printing the propagated state (case: no measurement)
            print("PROPAGATED @ {:.3f}s -> pos: {}, euler: {}, v: {}, w: {}".format(
                time.time(), x[0:3], x[3:6], x[6:9], x[9:12]
            ))

        # 5) check for measurement arrival (non-blocking)
        try:
            z_new = measurement_queue.get_nowait()
        except Empty:
            z_new = None

        if z_new is not None:
            t_meas = time.time()
            dt_meas = t_meas - last_meas_time if last_meas_time is not None else dt
            last_meas_time = t_meas

            # 1) convert incoming 6x1 -> 12x1
            z12 = to_state12(z_new)

            # 5) stop propagation and perform measurement update
            # measurement_update expects (x_pred, P_pred, z (6,), dt)
            x, P = measurement_update(x, P, z_new, dt_meas)

            # after update print updated state
            print("MEAS UPDATE @ {:.3f}s -> pos: {}, euler: {}, v: {}, w: {}".format(
                time.time(), x[0:3], x[3:6], x[6:9], x[9:12]
            ))

            # continue looping (propagation resumes next iteration)

        # maintain propagation frequency
        elapsed = time.time() - t_loop_start
        sleep_t = dt_target - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


if __name__ == "__main__":
    # NOTE: No fake sensor here. Feed measurements from your sensor code as:
    # measurement_queue.put(np.array([px, py, pz, roll, pitch, yaw]))
    main_loop(prop_freq=20.0)
