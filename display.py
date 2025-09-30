import cv2
import numpy as np

# Use tilted cross if available; fall back to plain cross.
_MARKER_TYPE = getattr(cv2, "MARKER_TILTED_CROSS", cv2.MARKER_CROSS)

def _to_int_pair(p):
    """Robust cast of a 2-vector to Python int tuple."""
    u = int(round(float(p[0])))
    v = int(round(float(p[1])))
    return (u, v)

def draw_raw_points(img, pts, color=(255, 255, 0), radius=5):
    """
    Draw all detected bright points (for debugging).
    pts: Nx2 pixel coords
    """
    if pts is None:
        return img
    for p in np.asarray(pts).reshape(-1, 2):
        if not np.all(np.isfinite(p)): 
            continue
        cv2.circle(img, _to_int_pair(p), radius, color, -1)
    return img

def draw_pattern(img, center_uv, offset_uv, outer_uv):
    """
    Draw the identified 5-LED pattern:
      - center (blue), offset (red), outers (yellow),
      - line center→offset (red).
    """

    if center_uv is not None and np.all(np.isfinite(center_uv)):
        cv2.circle(img, _to_int_pair(center_uv), 4, (255, 0, 0), -1)  # blue
    if offset_uv is not None and np.all(np.isfinite(offset_uv)):
        cv2.circle(img, _to_int_pair(offset_uv), 7, (0, 0, 255), -1)  # red
    if center_uv is not None and offset_uv is not None:
        if np.all(np.isfinite(center_uv)) and np.all(np.isfinite(offset_uv)):
            cv2.line(img, _to_int_pair(center_uv), _to_int_pair(offset_uv), (0, 0, 255), 2)

    if outer_uv is not None:
        for q in np.asarray(outer_uv).reshape(-1, 2):
            if not np.all(np.isfinite(q)): 
                continue
            cv2.circle(img, _to_int_pair(q), 7, (0, 255, 255), -1)     # yellow

    return img

def draw_reprojections(img, reproj, z_cam=None, color=(0, 255, 0), thickness=2):
    """
    Draw reprojected model points. If z_cam is provided, only draw with z>0 and finite.
    Robust to dtype=object / NaNs / offscreen values.
    """
    if reproj is None:
        return img

    # Force to numeric array; object dtypes cause OpenCV grief.
    R = np.asarray(reproj, dtype=np.float64).reshape(-1, 2)

    Z = None
    if z_cam is not None:
        Z = np.asarray(z_cam, dtype=np.float64).reshape(-1)

    h, w = img.shape[:2]
    for i in range(R.shape[0]):
        q = R[i]
        if not np.all(np.isfinite(q)):
            continue
        if Z is not None:
            z = Z[i]
            if (not np.isfinite(z)) or (z <= 1e-6):
                continue

        # round → Python int; skip points way off the image
        u = int(round(float(q[0])))
        v = int(round(float(q[1])))
        if (u < -1000) or (u > w + 1000) or (v < -1000) or (v > h + 1000):
            continue

        cv2.drawMarker(img, (u, v), color, markerType=_MARKER_TYPE, thickness=thickness)

    return img

def draw_hud(img, lines, fps=None, origin=(10, 30), line_start_y=60, line_gap=22):
    """
    Draw FPS and a list of HUD lines.
    """
    h, w = img.shape[:2]
    img_center = (w // 2, h // 2)
    # --- Reference cross (green) ---
    cv2.line(img, (img_center[0], 0), (img_center[0], h), (0, 255, 0), 1)  # vertical
    cv2.line(img, (0, img_center[1]), (w, img_center[1]), (0, 255, 0), 1)  # horizontal

    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.1f}", origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    y = line_start_y
    for i, line in enumerate(lines or []):
        cv2.putText(img, line, (origin[0], y + i*line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img
