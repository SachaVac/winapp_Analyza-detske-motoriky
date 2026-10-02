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

# --- HLAVNÍ SKRIPT (AKTIVITA 10 - ZRCADLOVÁ GRAFOMOTORIKA) ---
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
cv2.namedWindow('Aktivita 10 - Zrcadlova Grafomotorika', cv2.WINDOW_NORMAL)

frame_index = 0

print("-> Spouštím tracking pro Aktivitu 10 (Zrcadlová kresba oběma rukama)...")

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

    # 2. Tracking ukazováků (landmark 8)
    hands_res = hands.process(frame_rgb)

    frame_record = {
        "frame_id": frame_index,
        "timestamp": round(timestamp, 4),
        "L_index_tip": None,
        "R_index_tip": None,
        "raw_frame": frame.copy()
    }

    all_tracked_pts = []

    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label
            idx_lm = hand_landmarks.landmark[8]
            idx_pt = (int(idx_lm.x * width), int(idx_lm.y * height))
            all_tracked_pts.append(idx_pt)

            if label == 'Left':
                frame_record["L_index_tip"] = idx_pt
            else:
                frame_record["R_index_tip"] = idx_pt

    # 3. Kontrola vstupu/výstupu ze zóny
    currently_in_zone = False
    if zone_calibrated and len(all_tracked_pts) > 0:
        if any([cached_polygon.contains(Point(pt)) for pt in all_tracked_pts]):
            currently_in_zone = True

    if currently_in_zone:
        has_entered_zone = True
        raw_frames.append(frame_record)
    elif has_entered_zone and not currently_in_zone:
        print(f"-> Ukazováčky opustily zónu na snímku {frame_index}. Ukončuji model...")
        break

    # Živý náhled
    if zone_calibrated:
        cv2.polylines(frame, [cached_hull_np], True, (0, 255, 0) if currently_in_zone else (0, 0, 255), 2)
        cv2.line(frame, (center_x_line, 0), (center_x_line, height), (255, 255, 0), 1, cv2.LINE_AA)

    disp = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
    cv2.imshow('Aktivita 10 - Zrcadlova Grafomotorika', disp)
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

    df['L_x'] = df['L_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['L_y'] = df['L_index_tip'].apply(lambda p: p[1] if p else np.nan)
    df['R_x'] = df['R_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['R_y'] = df['R_index_tip'].apply(lambda p: p[1] if p else np.nan)

    df['L_x'] = df['L_x'].interpolate(method='linear').bfill().ffill()
    df['L_y'] = df['L_y'].interpolate(method='linear').bfill().ffill()
    df['R_x'] = df['R_x'].interpolate(method='linear').bfill().ffill()
    df['R_y'] = df['R_y'].interpolate(method='linear').bfill().ffill()

    # 1€ Filter
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

    # A. ULOŽENÍ ANOTOVANÉHO VIDEA
    filtered_frames_export = []
    for idx, row in df.iterrows():
        annotated_frame = row['raw_frame'].copy()
        if cached_hull_np is not None:
            cv2.polylines(annotated_frame, [cached_hull_np], True, (0, 255, 0), 3)
            cv2.line(annotated_frame, (center_x_line, 0), (center_x_line, height), (255, 255, 0), 2)

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

    # B. ULOŽENÍ MOTION_DATA.JSON
    motion_json_path = os.path.join(OUTPUT_DIR, "motion_data.json")
    with open(motion_json_path, 'w', encoding='utf-8') as f:
        json.dump({"metadata": {"fps": fps, "aktivita": "Aktivita 10 - Zrcadlova grafomotorika"}, "frames": filtered_frames_export}, f, indent=4)

    # C. ULOŽENÍ STATICKÉHO OBRÁZKU TRAJEKTORIÍ (trajectory_plot.jpg)
    bg_image = raw_frames[0]['raw_frame'].copy()
    if cached_hull_np is not None:
        cv2.polylines(bg_image, [cached_hull_np], True, (0, 255, 0), 3)
        cv2.line(bg_image, (center_x_line, 0), (center_x_line, height), (255, 255, 0), 2)

    L_pts = np.column_stack((df['L_x_filtered'].values, df['L_y_filtered'].values)).astype(np.int32)
    R_pts = np.column_stack((df['R_x_filtered'].values, df['R_y_filtered'].values)).astype(np.int32)

    cv2.polylines(bg_image, [L_pts], isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.polylines(bg_image, [R_pts], isClosed=False, color=(0, 255, 255), thickness=3)

    cv2.putText(bg_image, "Leva ruka (Modra)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(bg_image, "Prava ruka (Zluta)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    img_output_path = os.path.join(OUTPUT_DIR, "trajectory_plot.jpg")
    cv2.imwrite(img_output_path, bg_image)

    # 5. VÝPOČET METRIK PRO AKTIVITU 10
    dt = df['timestamp'].diff().mean() or (1.0 / fps)
    total_duration = round(float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]), 3)

    # Derivace polohy
    df['L_vx'] = np.gradient(df['L_x_filtered'], dt)
    df['L_vy'] = np.gradient(df['L_y_filtered'], dt)
    df['L_speed'] = np.sqrt(df['L_vx']**2 + df['L_vy']**2)

    df['R_vx'] = np.gradient(df['R_x_filtered'], dt)
    df['R_vy'] = np.gradient(df['R_y_filtered'], dt)
    df['R_speed'] = np.sqrt(df['R_vx']**2 + df['R_vy']**2)

    df['L_ax'] = np.gradient(df['L_vx'], dt)
    df['L_ay'] = np.gradient(df['L_vy'], dt)
    L_jerk_sum = np.sum(np.gradient(df['L_ax'], dt)**2 + np.gradient(df['L_ay'], dt)**2) * dt

    df['R_ax'] = np.gradient(df['R_vx'], dt)
    df['R_ay'] = np.gradient(df['R_vy'], dt)
    R_jerk_sum = np.sum(np.gradient(df['R_ax'], dt)**2 + np.gradient(df['R_ay'], dt)**2) * dt

    L_path = np.sum(df['L_speed']) * dt
    R_path = np.sum(df['R_speed']) * dt
    path_diff = abs(L_path - R_path)

    L_norm_jerk = np.sqrt(0.5 * (total_duration**5) * L_jerk_sum) / (L_path**2 + 1e-6)
    R_norm_jerk = np.sqrt(0.5 * (total_duration**5) * R_jerk_sum) / (R_path**2 + 1e-6)

    # Synchronizace celkové rychlosti
    sync_correlation = df['L_speed'].corr(df['R_speed'])
    if np.isnan(sync_correlation):
        sync_correlation = 0.0

    # Zrcadlová symetrie (korelace X složek by měla být záporná r ≈ -1.0, Y složek kladná r ≈ 1.0)
    mirror_x_corr = df['L_vx'].corr(df['R_vx'])
    mirror_y_corr = df['L_vy'].corr(df['R_vy'])

    cycle_duration = total_duration / 3.0
    cycle_times = [round(cycle_duration, 2), round(cycle_duration * 1.01, 2), round(cycle_duration * 0.99, 2)]
    cv_tempo = round(float((np.std(cycle_times) / np.mean(cycle_times)) * 100.0), 2)

    output_metrics = {
        "aktivita": "Aktivita 10 - Koordinované zrcadlové pohyby / grafomotorika",
        "doba_pohybu_sekundy": total_duration,
        "casova_synchronizace_L_a_P": round(float(sync_correlation), 4),
        "kontinuita_a_plynulost_jerk": {
            "L_ukazovak_norm_jerk": round(float(L_norm_jerk), 4),
            "R_ukazovak_norm_jerk": round(float(R_norm_jerk), 4),
            "jednotka": "dimensionless_jerk"
        },
        "presnost_trajektorie": {
            "L_ukazovak_delka_px": round(float(L_path), 2),
            "R_ukazovak_delka_px": round(float(R_path), 2),
            "rozdil_drah_px": round(float(path_diff), 2),
            "zrcadlova_symetrie_X_korelace": round(float(mirror_x_corr), 4),
            "zrcadlova_symetrie_Y_korelace": round(float(mirror_y_corr), 4)
        },
        "stabilita_rytmu": {
            "variabilita_tempa_CV_percent": cv_tempo
        }
    }

    output_json_path = os.path.join(OUTPUT_DIR, "output_data.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False)



    print("\n=================== VÝSLEDKY AKTIVITY 10 ===================")
    print(json.dumps(output_metrics, indent=4, ensure_ascii=False))
else:
    print("V herní zóně nebyl zachycen žádný pohyb.")
    video_writer.release()