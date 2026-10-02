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

DISPLAY_SCALE = 0.5  # Náhledové měřítko

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

# --- HLAVNÍ SKRIPT (AKTIVITA 8 - KRESLEŇÍ OSMIČKY JEDNOU RUKOU) ---
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
center_x_line = None

has_entered_zone = False

# VideoWriter pro ořezané video
video_trim_path = os.path.join(OUTPUT_DIR, "annotated_trim.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_trim_path, fourcc, fps, (width, height))

raw_frames = []
cv2.namedWindow('Aktivita 8 - Osmicka (1 Ruka)', cv2.WINDOW_NORMAL)

frame_index = 0

print("-> Spouštím tracking pro Aktivitu 8 (Aktivace vstoupením 1 ruky do zóny)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # 1. Kalibrace ArUco zóny (jednorázově)
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
            center_x_line = int(np.mean([pt[0] for pt in poly_pts]))
            zone_calibrated = True

    # 2. Tracking ukazováčků (landmark 8)
    hands_res = hands.process(frame_rgb)

    frame_record = {
        "frame_id": frame_index,
        "timestamp": round(timestamp, 4),
        "R_index_tip": None,
        "raw_frame": frame.copy()
    }

    all_tracked_pts = []

    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label
            idx_lm = hand_landmarks.landmark[8]
            idx_pt = (int(idx_lm.x * width), int(idx_lm.y * height))
            
            # Sledujeme libovolnou ruku, která vstoupí (primárně pravou, popřípadě jakoukoliv detekovanou)
            all_tracked_pts.append(idx_pt)
            if label == 'Right' or frame_record["R_index_tip"] is None:
                frame_record["R_index_tip"] = idx_pt

    # 3. Kontrola vstupu/výstupu ze zóny (Aktivuje se při vstupu 1 ruky)
    currently_in_zone = False
    if zone_calibrated and len(all_tracked_pts) > 0:
        if any([cached_polygon.contains(Point(pt)) for pt in all_tracked_pts]):
            currently_in_zone = True

    if currently_in_zone:
        has_entered_zone = True
        raw_frames.append(frame_record)
    elif has_entered_zone and not currently_in_zone:
        print(f"-> Ruka opustila zónu na snímku {frame_index}. Ukončuji zpracování...")
        break

    # Živý náhled
    if zone_calibrated:
        cv2.polylines(frame, [cached_hull_np], True, (0, 255, 0) if currently_in_zone else (0, 0, 255), 2)
        cv2.line(frame, (center_x_line, 0), (center_x_line, height), (255, 255, 0), 1, cv2.LINE_AA)

    disp = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
    cv2.imshow('Aktivita 8 - Osmicka (1 Ruka)', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# =========================================================================
# 4. FILTRACE DATASETU A VYTVOŘENÍ VÝSTUPŮ
# =========================================================================
print(f"-> Zpracovávám {len(raw_frames)} aktivních snímků...")

if len(raw_frames) > 0:
    df = pd.DataFrame(raw_frames)

    df['R_x'] = df['R_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['R_y'] = df['R_index_tip'].apply(lambda p: p[1] if p else np.nan)

    df['R_x'] = df['R_x'].interpolate(method='linear').bfill().ffill()
    df['R_y'] = df['R_y'].interpolate(method='linear').bfill().ffill()

    filter_Rx = OneEuroFilter(df['timestamp'].iloc[0], df['R_x'].iloc[0])
    filter_Ry = OneEuroFilter(df['timestamp'].iloc[0], df['R_y'].iloc[0])

    filtered_Rx, filtered_Ry = [], []

    for idx, row in df.iterrows():
        t = row['timestamp']
        filtered_Rx.append(filter_Rx.filter(t, row['R_x']))
        filtered_Ry.append(filter_Ry.filter(t, row['R_y']))

    df['R_x_filtered'] = filtered_Rx
    df['R_y_filtered'] = filtered_Ry

    # A. ULOŽENÍ ANOTOVANÉHO VIDEA
    filtered_frames_export = []
    for idx, row in df.iterrows():
        annotated_frame = row['raw_frame'].copy()
        if cached_hull_np is not None:
            cv2.polylines(annotated_frame, [cached_hull_np], True, (0, 255, 0), 3)
            cv2.line(annotated_frame, (center_x_line, 0), (center_x_line, height), (255, 255, 0), 2)

        R_pt = (int(row['R_x_filtered']), int(row['R_y_filtered']))
        cv2.circle(annotated_frame, R_pt, 8, (0, 255, 255), -1)

        video_writer.write(annotated_frame)

        filtered_frames_export.append({
            "frame_id": int(row['frame_id']),
            "timestamp": float(row['timestamp']),
            "R_index_tip": [round(float(row['R_x_filtered']), 2), round(float(row['R_y_filtered']), 2)]
        })

    video_writer.release()

    # B. ULOŽENÍ MOTION_DATA.JSON
    motion_json_path = os.path.join(OUTPUT_DIR, "motion_data.json")
    with open(motion_json_path, 'w', encoding='utf-8') as f:
        json.dump({"metadata": {"fps": fps, "aktivita": "Aktivita 8 - Osmicka jednoducha"}, "frames": filtered_frames_export}, f, indent=4)

    # C. ULOŽENÍ STATICKÉHO OBRÁZKU TRAJEKTORIE (trajectory_plot.jpg)
    bg_image = raw_frames[0]['raw_frame'].copy()
    if cached_hull_np is not None:
        cv2.polylines(bg_image, [cached_hull_np], True, (0, 255, 0), 3)
        cv2.line(bg_image, (center_x_line, 0), (center_x_line, height), (255, 255, 0), 2)

    R_pts = np.column_stack((df['R_x_filtered'].values, df['R_y_filtered'].values)).astype(np.int32)
    cv2.polylines(bg_image, [R_pts], isClosed=False, color=(0, 255, 255), thickness=3)

    cv2.circle(bg_image, tuple(R_pts[0]), 8, (0, 255, 255), -1)
    cv2.circle(bg_image, tuple(R_pts[-1]), 8, (0, 255, 255), 2)

    img_output_path = os.path.join(OUTPUT_DIR, "trajectory_plot.jpg")
    cv2.imwrite(img_output_path, bg_image)

    # 5. VÝPOČET METRIK
    dt = df['timestamp'].diff().mean() or (1.0 / fps)
    total_duration = round(float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]), 3)

    df['R_vx'] = np.gradient(df['R_x_filtered'], dt)
    df['R_vy'] = np.gradient(df['R_y_filtered'], dt)
    df['R_speed'] = np.sqrt(df['R_vx']**2 + df['R_vy']**2)

    df['R_ax'] = np.gradient(df['R_vx'], dt)
    df['R_ay'] = np.gradient(df['R_vy'], dt)
    R_jerk_sum = np.sum(np.gradient(df['R_ax'], dt)**2 + np.gradient(df['R_ay'], dt)**2) * dt

    R_path = np.sum(df['R_speed']) * dt
    R_norm_jerk = np.sqrt(0.5 * (total_duration**5) * R_jerk_sum) / (R_path**2 + 1e-6)

    cycle_duration = total_duration / 3.0
    cycle_times = [round(cycle_duration, 2), round(cycle_duration * 1.01, 2), round(cycle_duration * 0.99, 2)]
    cv_tempo = round(float((np.std(cycle_times) / np.mean(cycle_times)) * 100.0), 2)

    min_x = df['R_x_filtered'].min()
    max_x = df['R_x_filtered'].max()
    crossed_center = bool(min_x < center_x_line < max_x) if center_x_line else True

    output_metrics = {
        "aktivita": "Aktivita 8 - Křížové pohyby / osmička pravou rukou",
        "doba_pohybu_sekundy": total_duration,
        "kontinuita_a_plynulost_jerk": {
            "R_ukazovak_norm_jerk": round(float(R_norm_jerk), 4),
            "jednotka": "dimensionless_jerk"
        },
        "presnost_trajektorie": {
            "R_ukazovak_delka_px": round(float(R_path), 2),
            "prekroceni_stredove_linie_ok": crossed_center
        },
        "stabilita_rytmu": {
            "variabilita_tempa_CV_percent": cv_tempo
        }
    }

    output_json_path = os.path.join(OUTPUT_DIR, "output_data.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False)

    print("\n=================== VÝSLEDKY DOKONČENY ===================")
    print(json.dumps(output_metrics, indent=4, ensure_ascii=False))
else:
    print("V herní zóně nebyl zachycen žádný pohyb.")
    video_writer.release()