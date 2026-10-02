import os
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from shapely.geometry import Point, Polygon

import sys

# --- DYNAMICKÝ VSTUP A VYTVOŘENÍ VÝSTUPNÍ SLOŽKY ---
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    video_path = sys.argv[1]
else:
    # Záložní cesta pro samostatné testování bez GUI / parametrů
    video_path = '../data/ex11.mp4'

# Získání adresáře a názvu videa bez přípon (např. 'C:/videa/pokus1.mp4' -> directory='C:/videa', video_name='pokus1')
video_dir = os.path.dirname(os.path.abspath(video_path))
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Vytvoření dynamické složky s příponou _outputs
OUTPUT_DIR = os.path.join(video_dir, f"{video_name}_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"-> Vstupní video: {video_path}")
print(f"-> Výstupy se uloží do: {OUTPUT_DIR}")

# Načtení videa
cap = cv2.VideoCapture(video_path)

DISPLAY_SCALE = 0.5  # Měřítko pro náhled v okně (0.5 = 50 %)

# --- HSV ROZSAHY PRO DETEKCI 11 BAREVNÝCH KOLEČEK ---
COLOR_RANGES = {
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Blue": [
        (np.array([100, 120, 70]), np.array([140, 255, 255]))
    ],
    "Yellow": [
        (np.array([20, 120, 70]), np.array([35, 255, 255]))
    ],
    "Green": [
        (np.array([36, 120, 70]), np.array([85, 255, 255]))
    ]
}

def detect_colored_circles(frame, roi_polygon_pts=None):
    """
    Detekuje barevná kolečka v obraze a vrací jejich středy, poloměry a ID.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_circles = []
    circle_id = 0

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if area < 150 or perimeter == 0:
                continue

            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.60:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                center = (int(cx), int(cy))
                radius = max(int(radius), 15)  # Minimální poloměr dotykové zóny

                if roi_polygon_pts is not None:
                    if cv2.pointPolygonTest(roi_polygon_pts, center, False) < 0:
                        continue

                detected_circles.append({
                    "id": circle_id,
                    "color": color_name,
                    "center": center,
                    "radius": radius,
                    "visited": False
                })
                circle_id += 1

    return detected_circles

# --- 1€ FILTER PRO ODSTRANĚNÍ ŠUMU ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t, x):
        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        edx = self.alpha(self.d_cutoff, dt) * dx + (1 - self.alpha(self.d_cutoff, dt)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self.alpha(cutoff, dt)
        x_filtered = a * x + (1 - a) * self.x_prev
        self.x_prev = x_filtered
        self.dx_prev = edx
        self.t_prev = t
        return x_filtered

# --- HLAVNÍ SKRIPT ---
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 1. ArUco Detektor
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 2. MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

zone_calibrated = False
cached_polygon = None
cached_hull_np = None
cached_centers = {}
cached_circles = []

has_entered_zone = False

# VideoWriter pro anotované video
video_trim_path = os.path.join(OUTPUT_DIR, "annotated_trim.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_trim_path, fourcc, fps, (width, height))

raw_frames = []
cv2.namedWindow('Tracking & Detekce Kolecek', cv2.WINDOW_NORMAL)

frame_index = 0

print("-> Zahajuji zpracování: Sleduji výhradně UKAZOVÁČEK (8) a PROSTŘEDNÍČEK (12)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # 1. Kalibrace ArUco zóny a 11 barevných koleček (jednorázově)
    if not zone_calibrated:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for idx, tag_id in enumerate(ids.flatten()):
                if tag_id in [0, 1, 2, 3]:
                    pts = corners[idx][0]
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    cached_centers[int(tag_id)] = (cx, cy)

        if len(cached_centers) == 4:
            raw_pts = np.array([cached_centers[i] for i in range(4)], dtype=np.int32)
            cached_hull_np = cv2.convexHull(raw_pts)
            poly_pts = [tuple(pt[0]) for pt in cached_hull_np]
            cached_polygon = Polygon(poly_pts)

            # Detekce barevných koleček uvnitř herní zóny
            cached_circles = detect_colored_circles(frame, roi_polygon_pts=cached_hull_np)

            zone_calibrated = True
            print(f"-> Kalibrace dokončena. Nalezeno {len(cached_circles)} barevných koleček.")

    # 2. Tracking: Ukazováček (8) + Prostředníček (12)[cite: 3]
    hands_res = hands.process(frame_rgb)

    frame_record = {
        "frame_id": frame_index,
        "timestamp": round(timestamp, 4),
        "L_index_tip": None,
        "L_middle_tip": None,
        "R_index_tip": None,
        "R_middle_tip": None,
        "raw_frame": frame.copy()
    }

    all_tracked_pts = []

    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label  # 'Left' nebo 'Right'[cite: 3]
            
            # Špička ukazováčku (8)[cite: 3]
            idx_lm = hand_landmarks.landmark[8]
            idx_pt = (int(idx_lm.x * width), int(idx_lm.y * height))
            all_tracked_pts.append(idx_pt)

            # Špička prostředníčku (12)[cite: 3]
            mid_lm = hand_landmarks.landmark[12]
            mid_pt = (int(mid_lm.x * width), int(mid_lm.y * height))
            all_tracked_pts.append(mid_pt)

            if label == 'Left':
                frame_record["L_index_tip"] = idx_pt
                frame_record["L_middle_tip"] = mid_pt
            else:
                frame_record["R_index_tip"] = idx_pt
                frame_record["R_middle_tip"] = mid_pt

    # 3. Kontrola vstupu/výstupu ze zóny a detekce návštěvy koleček
    currently_in_zone = False
    if zone_calibrated and len(all_tracked_pts) > 0:
        if any([cached_polygon.contains(Point(pt)) for pt in all_tracked_pts]):
            currently_in_zone = True

    if currently_in_zone:
        has_entered_zone = True
        raw_frames.append(frame_record)

        # KONTROLA NÁVŠTĚVY KOLEČEK: Ukazovák (8) nebo Prostředník (12)
        for circle in cached_circles:
            if not circle["visited"]:
                c_center = circle["center"]
                c_radius = circle["radius"]
                for pt in all_tracked_pts:
                    dist = np.sqrt((pt[0] - c_center[0])**2 + (pt[1] - c_center[1])**2)
                    if dist <= c_radius:
                        circle["visited"] = True
                        print(f"-> Kolečko ID {circle['id']} ({circle['color']}) navštíveno!")
                        break

    elif has_entered_zone and not currently_in_zone:
        print(f"-> Ruce opustily zónu na snímku {frame_index}. Ukončuji zpracování...")
        break

    # Živý náhled v okně
    if zone_calibrated:
        cv2.polylines(frame, [cached_hull_np], True, (0, 255, 0) if currently_in_zone else (0, 0, 255), 2)
        for circle in cached_circles:
            c_color = (0, 255, 0) if circle["visited"] else (0, 0, 255)
            cv2.circle(frame, circle["center"], circle["radius"], c_color, 2)

    disp = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
    cv2.imshow('Tracking & Detekce Kolecek', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# =========================================================================
# 4. FILTRACE A ULOŽENÍ ANOTOVANÉHO VIDEA + OBRÁZKU
# =========================================================================
print(f"-> Zpracovávám {len(raw_frames)} aktivních snímků...")

if len(raw_frames) > 0:
    df = pd.DataFrame(raw_frames)

    df['L_idx_x'] = df['L_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['L_idx_y'] = df['L_index_tip'].apply(lambda p: p[1] if p else np.nan)
    df['R_idx_x'] = df['R_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['R_idx_y'] = df['R_index_tip'].apply(lambda p: p[1] if p else np.nan)

    # Interpolace výpadků
    df['L_idx_x'] = df['L_idx_x'].interpolate(method='linear').bfill().ffill()
    df['L_idx_y'] = df['L_idx_y'].interpolate(method='linear').bfill().ffill()
    df['R_idx_x'] = df['R_idx_x'].interpolate(method='linear').bfill().ffill()
    df['R_idx_y'] = df['R_idx_y'].interpolate(method='linear').bfill().ffill()

    # 1€ Filter
    filter_Lx = OneEuroFilter(df['timestamp'].iloc[0], df['L_idx_x'].iloc[0])
    filter_Ly = OneEuroFilter(df['timestamp'].iloc[0], df['L_idx_y'].iloc[0])
    filter_Rx = OneEuroFilter(df['timestamp'].iloc[0], df['R_idx_x'].iloc[0])
    filter_Ry = OneEuroFilter(df['timestamp'].iloc[0], df['R_idx_y'].iloc[0])

    filtered_Lx, filtered_Ly, filtered_Rx, filtered_Ry = [], [], [], []

    for idx, row in df.iterrows():
        t = row['timestamp']
        filtered_Lx.append(filter_Lx.filter(t, row['L_idx_x']))
        filtered_Ly.append(filter_Ly.filter(t, row['L_idx_y']))
        filtered_Rx.append(filter_Rx.filter(t, row['R_idx_x']))
        filtered_Ry.append(filter_Ry.filter(t, row['R_idx_y']))

    df['L_x_filtered'] = filtered_Lx
    df['L_y_filtered'] = filtered_Ly
    df['R_x_filtered'] = filtered_Rx
    df['R_y_filtered'] = filtered_Ry

    # A. ULOŽENÍ ANOTOVANÉHO VIDEA
    filtered_frames_export = []
    for idx, row in df.iterrows():
        annotated_frame = row['raw_frame'].copy()
        if cached_hull_np is not None:
            cv2.polylines(annotated_frame, [cached_hull_np], True, (0, 255, 0), 3)

        for circle in cached_circles:
            col = (0, 255, 0) if circle["visited"] else (0, 0, 255)
            cv2.circle(annotated_frame, circle["center"], circle["radius"], col, 2)

        L_pt = (int(row['L_x_filtered']), int(row['L_y_filtered']))
        R_pt = (int(row['R_x_filtered']), int(row['R_y_filtered']))

        cv2.circle(annotated_frame, L_pt, 8, (255, 0, 0), -1)   # Modrá = Levá
        cv2.circle(annotated_frame, R_pt, 8, (0, 255, 255), -1) # Žlutá = Pravá

        video_writer.write(annotated_frame)

        filtered_frames_export.append({
            "frame_id": int(row['frame_id']),
            "timestamp": float(row['timestamp']),
            "L_index_tip": [round(float(row['L_x_filtered']), 2), round(float(row['L_y_filtered']), 2)],
            "R_index_tip": [round(float(row['R_x_filtered']), 2), round(float(row['R_y_filtered']), 2)]
        })

    video_writer.release()

    # B. ULOŽENÍ STATICKÉHO OBRÁZKU TRAJEKTORIÍ
    bg_image = raw_frames[0]['raw_frame'].copy()
    if cached_hull_np is not None:
        cv2.polylines(bg_image, [cached_hull_np], True, (0, 255, 0), 3)

    for circle in cached_circles:
        col = (0, 255, 0) if circle["visited"] else (0, 0, 255)
        cv2.circle(bg_image, circle["center"], circle["radius"], col, 2)

    L_pts = np.column_stack((df['L_x_filtered'].values, df['L_y_filtered'].values)).astype(np.int32)
    R_pts = np.column_stack((df['R_x_filtered'].values, df['R_y_filtered'].values)).astype(np.int32)

    cv2.polylines(bg_image, [L_pts], isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.polylines(bg_image, [R_pts], isClosed=False, color=(0, 255, 255), thickness=3)

    img_output_path = os.path.join(OUTPUT_DIR, "trajectory_plot.jpg")
    cv2.imwrite(img_output_path, bg_image)

    # C. VYHODNOCENÍ NAVŠTÍVENÍ VŠECH KOLEČEK (BOOLEAN)
    total_circles = len(cached_circles)
    visited_circles_count = sum(1 for c in cached_circles if c["visited"])
    all_circles_visited = (visited_circles_count == total_circles) if total_circles > 0 else False

    # Uložení motion_data.json
    json_path = os.path.join(OUTPUT_DIR, "motion_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"metadata": {"fps": fps}, "frames": filtered_frames_export}, f, indent=4)

    # D. VÝPOČET FINÁLNÍCH METRİK
    dt = df['timestamp'].diff().mean() or (1.0 / fps)
    total_duration = round(float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]), 3)

    df['L_vx'] = np.gradient(df['L_x_filtered'], dt)
    df['L_vy'] = np.gradient(df['L_y_filtered'], dt)
    df['L_speed'] = np.sqrt(df['L_vx']**2 + df['L_vy']**2)

    df['R_vx'] = np.gradient(df['R_x_filtered'], dt)
    df['R_vy'] = np.gradient(df['R_y_filtered'], dt)
    df['R_speed'] = np.sqrt(df['R_vx']**2 + df['R_vy']**2)

    df['L_ax'] = np.gradient(df['L_vx'], dt)
    df['L_ay'] = np.gradient(df['L_vy'], dt)
    df['L_jx'] = np.gradient(df['L_ax'], dt)
    df['L_jy'] = np.gradient(df['L_ay'], dt)
    L_jerk_squared_sum = np.sum(df['L_jx']**2 + df['L_jy']**2) * dt

    df['R_ax'] = np.gradient(df['R_vx'], dt)
    df['R_ay'] = np.gradient(df['R_vy'], dt)
    df['R_jx'] = np.gradient(df['R_ax'], dt)
    df['R_jy'] = np.gradient(df['R_ay'], dt)
    R_jerk_squared_sum = np.sum(df['R_jx']**2 + df['R_jy']**2) * dt

    L_path_length = np.sum(df['L_speed']) * dt
    R_path_length = np.sum(df['R_speed']) * dt

    L_norm_jerk = np.sqrt(0.5 * (total_duration**5) * L_jerk_squared_sum) / (L_path_length**2 + 1e-6)
    R_norm_jerk = np.sqrt(0.5 * (total_duration**5) * R_jerk_squared_sum) / (R_path_length**2 + 1e-6)

    sync_correlation = df['L_speed'].corr(df['R_speed'])
    if np.isnan(sync_correlation):
        sync_correlation = 0.0

    L_path_length = np.sum(df['L_speed']) * dt
    R_path_length = np.sum(df['R_speed']) * dt
    path_difference = abs(L_path_length - R_path_length)

    output_metrics = {
        "doba_pohybu_sekundy": total_duration,
        "navstivena_vsechna_kolecka": all_circles_visited,
        "pocet_detekovanych_kolecek": total_circles,
        "pocet_navstivenych_kolecek": visited_circles_count,
        "casova_synchronizace_L_a_P": round(float(sync_correlation), 4),
        "plynulost_pohybu_jerk": {
            "L_ukazovak_norm_jerk": round(float(L_norm_jerk), 4),
            "R_ukazovak_norm_jerk": round(float(R_norm_jerk), 4),
            "jednotka": "dimensionless_jerk"
        },
        "delka_drah_px": {
            "L_ukazovak_delka_px": round(float(L_path_length), 2),
            "R_ukazovak_delka_px": round(float(R_path_length), 2),
            "rozdil_drah_px": round(float(path_difference), 2)
        }
    }

    output_json_path = os.path.join(OUTPUT_DIR, "output_data.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False)

    print("\n=================== VÝSLEDKY MĚŘENÍ ===================")
    print(json.dumps(output_metrics, indent=4, ensure_ascii=False))
else:
    print("V herní zóně nebyl zachycen žádný pohyb.")
    video_writer.release()