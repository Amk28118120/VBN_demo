# Minimal bright-blob detector. Returns up to 5 centroid points in pixels.
import cv2
import numpy as np

def detect_leds(frame_bgr, max_pts=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive bright threshold: mean + k*std
    m, s = cv2.meanStdDev(blur)
    thr = float(m + 3*s)  # tweak 2.0–3.5 if needed
    _, mask = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Parameters
    min_area = 25       # minimum contour area in pixels
    margin_top = 150       # pixels to ignore from top
    margin_bottom = 150    # pixels to ignore from bottom
    margin_left = 150      # pixels to ignore from left
    margin_right = 150     # pixels to ignore from right

    h, w = mask.shape[:2]  # image size

    # Now use filtered_contours instead of contours
    # Rank by brightness to prefer true LEDs
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)

        # Check if contour is inside central region
        if (x > margin_left and y > margin_top and x + cw < w - margin_right and y + ch < h - margin_bottom):
            if min_area < area < 5000:
                M = cv2.moments(c)
                if M["m00"] > 1e-6:
                    u = M["m10"]/M["m00"]; v = M["m01"]/M["m00"]
                    msk = np.zeros_like(mask); cv2.drawContours(msk, [c], -1, 255, -1)
                    mean_val = cv2.mean(gray, mask=msk)[0]
                    blobs.append((mean_val, (u, v)))

    blobs.sort(reverse=True)
    pts = np.array([p for _, p in blobs[:max_pts]], dtype=np.float32)
    print(pts)
    return pts
