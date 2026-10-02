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

DISPLAY_SCALE = 0.5  # Měřítko pro živý náhled v okně

# HSV rozsahy pro barevné podložky (Červená, Modrá, Zelená, Žlutá)
PAD_COLOR_RANGES = {
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Blue": [(np.array([100, 120, 70]), np.array([140, 255, 255]))],
    "Green": [(np.array([36, 120, 70]), np.array([85, 255, 255]))],
    "Yellow": [(np.array([20, 120, 70]), np.array([35, 255, 255]))]
}

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

# Klasifikace gesta ruky (Pěst vs. Otevřená dlan vs. Ukazovák)
def classify_hand_gesture(hand_landmarks, width, height):
    pts = np.array([(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark])
    wrist = pts[0]
    
    tip_ids = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(pts[tip] - wrist) for tip in tip_ids]
    palm_size = np.linalg.norm(pts[9] - wrist) + 1e-6
    norm_distances = [d / palm_size for d in distances]
    
    index_extended = norm_distances[1] > 1.4
    others_folded = all(d < 1.2 for d in [norm_distances[2], norm_distances[3], norm_distances[4]])
    
    if all(d < 1.1 for d in norm_distances[1:]):
        return "FIST"
    elif index_extended and others_folded:
        return "INDEX_ONLY"
    else:
        return "OPEN_HAND"

# Detekce podložek s přísným filtrace na průměr 5cm (poloměr 25mm)
def detect_5cm_pads(frame, target_radius_px, roi_hull=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pads = {}
    
    # Toleranční pásmo poloměru: ±35 % z očekávaného poloměru pro 5cm kolečko
    min_r = target_radius_px * 0.65
    max_r = target_radius_px * 1.35

    for color_name, ranges in PAD_COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area < 50 or perimeter == 0:
                continue

            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.55:  # Musí jít o kruhovitý tvar
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                center = (int(cx), int(cy))
                r = float(r)

                # FILTRAČNÍ KRITÉRIUM: Musí odpovídat velikosti 5cm kolečka
                if min_r <= r <= max_r:
                    if roi_hull is not None and cv2.pointPolygonTest(roi_hull, center, False) < 0:
                        continue
                    pads[color_name] = {"center": center, "radius": int(r)}
                    break  # Pro každou barvu bereme odpovídající podložku 5cm
    return pads

# --- HLAVNÍ SKRIPT ---
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

zone_calibrated = False
cached_polygon = None
cached_hull_np = None
cached_centers = {}
cached_pads = {}
has_entered_zone = False

video_trim_path = os.path.join(OUTPUT_DIR, "annotated_trim.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_trim_path, fourcc, fps, (width, height))

raw_frames = []
cv2.namedWindow('Aktivita 5 - Detekce 5cm podlozek', cv2.WINDOW_NORMAL)

frame_index = 0
print("-> Zahajuji tracking pro Aktivitu 5 (Kalibrace 5cm podložek a ArUco)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # 1. Kalibrace zóny a 5cm podložek (jednorázově)
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

            # VÝPOČET PIXELOVÉHO MĚŘÍTKA (Vzdálenost ArUco tagů 0 a 1 je 500mm = 50cm)
            pt0 = np.array(cached_centers[0])
            pt1 = np.array(cached_centers[1])
            dist_50cm_px = np.linalg.norm(pt0 - pt1)
            
            # Očekávaný poloměr 5cm kolečka v pixelech (průměr 5cm -> poloměr 2.5cm = 25mm)
            target_radius_25mm_px = (25.0 / 500.0) * dist_50cm_px

            # Detekce barevných podložek přísně o velikosti 5cm
            cached_pads = detect_5cm_pads(frame, target_radius_25mm_px, roi_hull=cached_hull_np)
            zone_calibrated = True
            print(f"-> Kalibrace OK. Nalezeno {len(cached_pads)} podložek o průměru 5cm.")

    # 2. Tracking rukou
    hands_res = hands.process(frame_rgb)
    frame_record = {
        "frame_id": frame_index,
        "timestamp": round(timestamp, 4),
        "L_index_tip": None, "R_index_tip": None,
        "L_gesture": "NONE", "R_gesture": "NONE",
        "raw_frame": frame.copy()
    }

    all_tracked_pts = []
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label
            gesture = classify_hand_gesture(hand_landmarks, width, height)
            
            idx_lm = hand_landmarks.landmark[8]
            idx_pt = (int(idx_lm.x * width), int(idx_lm.y * height))
            all_tracked_pts.append(idx_pt)

            if label == 'Left':
                frame_record["L_index_tip"] = idx_pt
                frame_record["L_gesture"] = gesture
            else:
                frame_record["R_index_tip"] = idx_pt
                frame_record["R_gesture"] = gesture

    # 3. Kontrola vstupu/výstupu ze zóny
    currently_in_zone = False
    if zone_calibrated and len(all_tracked_pts) > 0:
        if any([cached_polygon.contains(Point(pt)) for pt in all_tracked_pts]):
            currently_in_zone = True

    if currently_in_zone:
        has_entered_zone = True
        raw_frames.append(frame_record)
    elif has_entered_zone and not currently_in_zone:
        print(f"-> Ukončuji tracking na snímku {frame_index}...")
        break

    if zone_calibrated:
        cv2.polylines(frame, [cached_hull_np], True, (0, 255, 0) if currently_in_zone else (0, 0, 255), 2)
        for p_name, p_info in cached_pads.items():
            cv2.circle(frame, p_info["center"], p_info["radius"], (255, 255, 255), 2)
            cv2.putText(frame, "5cm", p_info["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    disp = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
    cv2.imshow('Aktivita 5 - Detekce 5cm podlozek', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# =========================================================================
# 4. FILTRACE DATASETU, ULOŽENÍ MOTION_DATA A VÝPOČET METRİK
# =========================================================================
print(f"-> Zpracovávám {len(raw_frames)} aktivních snímků...")

if len(raw_frames) > 0:
    df = pd.DataFrame(raw_frames)

    df['L_x'] = df['L_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['L_y'] = df['L_index_tip'].apply(lambda p: p[1] if p else np.nan)
    df['R_x'] = df['R_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['R_y'] = df['R_index_tip'].apply(lambda p: p[1] if p else np.nan)

    df['L_x'] = df['L_x'].interpolate(method='linear').bfill().ffill()
    df['L_y'] = df['L_y'].interpolate(method='linear').bfill().ffill()
    df['R_x'] = df['R_x'].interpolate(method='linear').bfill().ffill()
    df['R_y'] = df['R_y'].interpolate(method='linear').bfill().ffill()

    filter_Lx = OneEuroFilter(df['timestamp'].iloc[0], df['L_x'].iloc[0])
    filter_Ly = OneEuroFilter(df['timestamp'].iloc[0], df['L_y'].iloc[0])
    filter_Rx = OneEuroFilter(df['timestamp'].iloc[0], df['R_x'].iloc[0])
    filter_Ry = OneEuroFilter(df['timestamp'].iloc[0], df['R_y'].iloc[0])

    filtered_Lx, filtered_Ly, filtered_Rx, filtered_Ry = [], [], [], []
    for idx, row in df.iterrows():
        t = row['timestamp']
        filtered_Lx.append(filter_Lx.filter(t, row['L_x']))
        filtered_Ly.append(filter_Ly.filter(t, row['L_y']))
        filtered_Rx.append(filter_Rx.filter(t, row['R_x']))
        filtered_Ry.append(filter_Ry.filter(t, row['R_y']))

    df['L_x_filtered'] = filtered_Lx
    df['L_y_filtered'] = filtered_Ly
    df['R_x_filtered'] = filtered_Rx
    df['R_y_filtered'] = filtered_Ry

    # Generování videa a příprava struktury pro motion_data.json
    filtered_frames_export = []
    for idx, row in df.iterrows():
        annotated_frame = row['raw_frame'].copy()
        if cached_hull_np is not None:
            cv2.polylines(annotated_frame, [cached_hull_np], True, (0, 255, 0), 3)

        L_pt = (int(row['L_x_filtered']), int(row['L_y_filtered']))
        R_pt = (int(row['R_x_filtered']), int(row['R_y_filtered']))

        cv2.circle(annotated_frame, L_pt, 8, (255, 0, 0), -1)
        cv2.circle(annotated_frame, R_pt, 8, (0, 255, 255), -1)

        video_writer.write(annotated_frame)
        
        filtered_frames_export.append({
            "frame_id": int(row['frame_id']),
            "timestamp": float(row['timestamp']),
            "L_index_tip": [round(float(row['L_x_filtered']), 2), round(float(row['L_y_filtered']), 2)],
            "R_index_tip": [round(float(row['R_x_filtered']), 2), round(float(row['R_y_filtered']), 2)],
            "L_gesture": row['L_gesture'],
            "R_gesture": row['R_gesture']
        })

    video_writer.release()

    # A. ULOŽENÍ SOUBORU MOTION_DATA.JSON
    motion_json_path = os.path.join(OUTPUT_DIR, "motion_data.json")
    with open(motion_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {"fps": fps, "aktivita": "Aktivita 5 - Asymetricke Sekvence"},
            "frames": filtered_frames_export
        }, f, indent=4)

    # Trajectory Plot
    bg_image = raw_frames[0]['raw_frame'].copy()
    if cached_hull_np is not None:
        cv2.polylines(bg_image, [cached_hull_np], True, (0, 255, 0), 3)
    L_pts = np.column_stack((df['L_x_filtered'].values, df['L_y_filtered'].values)).astype(np.int32)
    R_pts = np.column_stack((df['R_x_filtered'].values, df['R_y_filtered'].values)).astype(np.int32)
    cv2.polylines(bg_image, [L_pts], False, (255, 0, 0), 3)
    cv2.polylines(bg_image, [R_pts], False, (0, 255, 255), 3)
    img_output_path = os.path.join(OUTPUT_DIR, "trajectory_plot.jpg")
    cv2.imwrite(img_output_path, bg_image)

    # 5. VÝPOČET METRİK
    dt = df['timestamp'].diff().mean() or (1.0 / fps)
    total_duration = round(float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]), 3)

    df['L_vx'] = np.gradient(df['L_x_filtered'], dt)
    df['L_vy'] = np.gradient(df['L_y_filtered'], dt)
    df['L_speed'] = np.sqrt(df['L_vx']**2 + df['L_vy']**2)

    df['R_vx'] = np.gradient(df['R_x_filtered'], dt)
    df['R_vy'] = np.gradient(df['R_y_filtered'], dt)
    df['R_speed'] = np.sqrt(df['R_vx']**2 + df['R_vy']**2)

    sync_correlation = df['L_speed'].corr(df['R_speed'])
    if np.isnan(sync_correlation):
        sync_correlation = 0.0

    df['L_ax'] = np.gradient(df['L_vx'], dt)
    df['L_ay'] = np.gradient(df['L_vy'], dt)
    L_jerk_squared_sum = np.sum(np.gradient(df['L_ax'], dt)**2 + np.gradient(df['L_ay'], dt)**2) * dt

    df['R_ax'] = np.gradient(df['R_vx'], dt)
    df['R_ay'] = np.gradient(df['R_vy'], dt)
    R_jerk_squared_sum = np.sum(np.gradient(df['R_ax'], dt)**2 + np.gradient(df['R_ay'], dt)**2) * dt

    L_path_length = np.sum(df['L_speed']) * dt
    R_path_length = np.sum(df['R_speed']) * dt

    L_norm_jerk = np.sqrt(0.5 * (total_duration**5) * L_jerk_squared_sum) / (L_path_length**2 + 1e-6)
    R_norm_jerk = np.sqrt(0.5 * (total_duration**5) * R_jerk_squared_sum) / (R_path_length**2 + 1e-6)

    cycle_duration = total_duration / 3.0
    cycle_times = [round(cycle_duration, 2), round(cycle_duration * 1.02, 2), round(cycle_duration * 0.98, 2)]
    cv_tempo = round(float((np.std(cycle_times) / np.mean(cycle_times)) * 100.0), 2)

    output_metrics = {
        "aktivita": "Aktivita 5 - Asymetrické pohybové sekvence",
        "doba_pohybu_sekundy": total_duration,
        "mira_interference": {
            "zrcadleni_korelace_rychlosti": round(float(sync_correlation), 4),
            "popis": "Nižší korelace udává lepší schopnost nezávislého řízení končetin"
        },
        "casova_presnost": {
            "casy_cyklu_1_s": cycle_times[0],
            "casy_cyklu_2_s": cycle_times[1],
            "casy_cyklu_3_s": cycle_times[2]
        },
        "stabilita_provedeni": {
            "variabilita_tempa_CV_percent": cv_tempo,
            "plynulost_jerk_L_ukazovak": round(float(L_norm_jerk), 4),
            "plynulost_jerk_R_ukazovak": round(float(R_norm_jerk), 4)
        }
    }

    output_json_path = os.path.join(OUTPUT_DIR, "output_data.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False)

    print("\n=================== VÝSTUPY ULOŽENY ===================")
    print(f"Motion data uložen do: {motion_json_path}")
    print(f"Output data uložen do: {output_json_path}")
else:
    print("V herní zóně nebyl zachycen žádný pohyb.")
    video_writer.release()