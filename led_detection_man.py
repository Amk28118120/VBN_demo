import numpy as np
import cv2

# ========== Helper data structures ==========
class FeaturePoint2D:
    def __init__(self, y, z, size=0):
        self.y = float(y)  # like C++ FeaturePoint2D.y
        self.z = float(z)  # like C++ FeaturePoint2D.z
        self.size = float(size)

class FeatureFrame:
    def __init__(self):
        self.points = []  # list of FeaturePoint2D

# ========== Core functions ==========

def dist(p1: FeaturePoint2D, p2: FeaturePoint2D):
    dx = p1.y - p2.y
    dy = p1.z - p2.z
    return np.sqrt(dx*dx + dy*dy)

def threshold_image(img_gray, THRESHOLD):
    mask = np.where(img_gray >= THRESHOLD, img_gray, 0).astype(np.uint8)
    return mask

def find_contours(img_bin,priors,min_area=10, skips = 4, margin=100):
    """
    Flood-fill style contour extraction like in your C++ code.
    """
    h, w = img_bin.shape
    visited = np.zeros_like(img_bin, dtype=bool)
    contours = []

    # 8-connected neighbors
    dx = [-1, 0, 1, 1, 1, 0, -1, -1]
    dy = [-1, -1, -1, 0, 1, 1, 1, 0]
    x_start, x_end = margin, w - margin
    y_start, y_end = margin, h - margin
    if priors["lock"] == True:
        # Use prior center to limit search area
        l = priors["length_const"] 
        range_prev = priors["range"]
        box= int(2*(l/range_prev))
        # Taking centre of pattern as centre for ROI box
        C = priors["centre"]
        cx, cy = int(C[0]), int(C[1])
        x_start = max(margin, cx - box)
        x_end = min(w - margin, cx + box)
        y_start = max(margin, cy - box)
        y_end = min(h - margin, cy + box)

    for y in range(y_start, y_end,skips):
        for x in range(x_start, x_end,skips):
            if img_bin[y, x] != 0 and not visited[y, x]:
                stack = [(x, y)]
                contour_x, contour_y = [], []

                while stack:
                    cx, cy = stack.pop()
                    if visited[cy, cx]:
                        continue
                    visited[cy, cx] = True

                    contour_x.append(cx)
                    contour_y.append(cy)

                    for d in range(8):
                        nx, ny = cx + dx[d], cy + dy[d]
                        if 0 <= nx < w and 0 <= ny < h:
                            if img_bin[ny, nx] != 0 and not visited[ny, nx]:
                                stack.append((nx, ny))

                if len(contour_x) > min_area:
                    contours.append((contour_x, contour_y))

    return contours

def calculate_moments(contour, img_gray):
    xs, ys = contour
    bright_vals = img_gray[ys, xs]
    bright_denom = float(np.sum(bright_vals))

    if bright_denom == 0:
        return (0, 0, 0)

    # spatial moments
    M00 = len(xs)
    M10 = np.sum(np.array(xs) * bright_vals)
    M01 = np.sum(np.array(ys) * bright_vals)

    cx = M10 / bright_denom
    cy = M01 / bright_denom
    return (M00, cx, cy)

def mergeCloseLEDs(leds: FeatureFrame, threshold=3.0):
    merged = []
    used = [False] * len(leds.points)
    thresh2 = threshold * threshold

    for i, p in enumerate(leds.points):
        if used[i]:
            continue
        sumX, sumY, count = p.y, p.z, 1
        used[i] = True
        for j in range(i+1, len(leds.points)):
            if used[j]:
                continue
            q = leds.points[j]
            dx, dy = q.y - p.y, q.z - p.z
            if dx*dx + dy*dy <= thresh2:
                sumX += q.y
                sumY += q.z
                count += 1
                used[j] = True
        merged.append(FeaturePoint2D(sumX/count, sumY/count))
    leds.points = merged

def exchange(leds: FeatureFrame, i, j):
    leds.points[i], leds.points[j] = leds.points[j], leds.points[i]

def arrange_3(leds: FeatureFrame):
    if len(leds.points) != 3:
        return
    d01 = dist(leds.points[0], leds.points[1])
    d02 = dist(leds.points[0], leds.points[2])
    d12 = dist(leds.points[1], leds.points[2])
    sum0, sum1, sum2 = d01+d02, d01+d12, d02+d12
    central = 0 if sum0 < sum1 and sum0 < sum2 else (1 if sum1 < sum2 else 2)
    if central != 2:
        exchange(leds, central, 2)
    if leds.points[0].y > leds.points[1].y:
        exchange(leds, 0, 1)

def arrange_5(leds: FeatureFrame):
    if len(leds.points) != 5:
        return
    # mimic your C++ ordering
    maxy = max(range(5), key=lambda i: leds.points[i].y)
    exchange(leds, 0, maxy)
    maxz = max(range(1,5), key=lambda i: leds.points[i].z)
    exchange(leds, 1, maxz)
    miny = min(range(2,5), key=lambda i: leds.points[i].y)
    exchange(leds, 2, miny)
    minz = min(range(3,5), key=lambda i: leds.points[i].z)
    exchange(leds, 3, minz)

def best_comb_five(leds: FeatureFrame):
    """
    Choose the best 5-LED set by matching the cross-pattern geometry.
    Only modifies leds.points.
    """

    pts = leds.points
    n = len(pts)
    if n < 5:
        leds.points = []
        return

    import itertools
    best_score = 1e18
    best_set = None

    def ang(p, q, r):
        """angle p–q–r at q"""
        v1 = np.array([p.y - q.y, p.z - q.z])
        v2 = np.array([r.y - q.y, r.z - q.z])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    for comb in itertools.combinations(range(n), 5):

        # Extract the 5 points
        subset = [pts[i] for i in comb]

        # Copy into a temporary FeatureFrame object
        temp = FeatureFrame()
        temp.points = [FeaturePoint2D(p.y, p.z, p.size) for p in subset]

        # ---- Important: arrange using your fixed rule ----
        arrange_5(temp)

        # After arrange_5:
        # temp.points[0] = LED1 (right / max y)
        # temp.points[1] = LED2 (bottom / max z)
        # temp.points[2] = LED3 (left / min y)
        # temp.points[3] = LED4 (top / min z)
        # temp.points[4] = center LED

        L1, L2, L3, L4, C = temp.points
        d = [dist(C,p) for p in (L1, L2, L3, L4)]

        # Length-ratio score (all ratios expected near 1)
        ratio_score = 0
        for i in range(4):
            for j in range(i+1, 4):
                r = d[i]/(d[j] + 1e-4)
                ratio_score += (r - 1)**2       

        # Compute α1 = angle between opposite LEDs: L1 — C — L3
        alpha1 = ang(L1, C, L3)

        # Compute α2 = angle between opposite LEDs: L2 — C — L4
        alpha2 = ang(L2, C, L4)

        # β = angle between arms d1-d3 vs d2-d4
        beta  = ang(L1, C, L2)

        # Expected values:
        # α1 ≈ 180°, α2 ≈ 180° (collinear opposites)
        # β ≈ 90°     (cross shape)

        angle_score = ((alpha1 - 180)**2 +
                 (alpha2 - 180)**2 +
                 (beta   - 90 )**2)

        score = ratio_score + 0.1 * angle_score
        # Best match
        if score < best_score:
            best_score = score
            best_set = subset

    if best_set is None:
        leds.points = []
    else:
        leds.points = best_set

def extract_leds(leds: FeatureFrame, mode):
    mergeCloseLEDs(leds, 2)
    if mode == 5:
        best_comb_five(leds)
        if len(leds.points) != mode:
            leds.points = []
        arrange_5(leds)
    elif mode == 3:
        arrange_3(leds)

# ========== Main detection entry ==========

def detect_leds_python(gray, priors, THRESHOLD=20, mode=5, max_pts=5): # Merging 2 close blobs tuning is left
    #print (frame_bgr.shape,frame_bgr.dtype)
    #gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_bin = threshold_image(gray, THRESHOLD)
    h, w = gray.shape

    if priors["lock"] == True:
        k = priors["area_const"] 
        range_prev = priors["range"]
        min_area = 0.7 * (k/range_prev)**2
        print("min_area = ",min_area)
        skips = int(((min_area)**0.5)//2)
        contours = find_contours(img_bin,priors,min_area,skips)
    else :
        print("Lock was false")
        contours = find_contours(img_bin,priors)
        print(len(contours))
    leds = FeatureFrame()
    
    for contour in contours:
        M00, cx, cy = calculate_moments(contour, gray)
        if M00 > 0:
            leds.points.append(FeaturePoint2D(cx - w/2.0, cy - h/2.0, M00))

    extract_leds(leds, mode)
    if leds.points == []:
        return []
    # return Nx2 numpy array like your original detect_leds
    pts = np.array([[p.y, p.z] for p in leds.points[:max_pts]], dtype=np.float32)
    #h, w = frame_bgr.shape[:2]
    pts = pts + np.array([w/2, h/2])
    print("pts - ",pts)
    return pts
