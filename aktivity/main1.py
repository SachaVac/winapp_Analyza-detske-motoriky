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

DISPLAY_SCALE = 0.5  # Zmenšení okna pro živý náhled (0.5 = 50 %)

# --- IMPLEMENTACE 1€ FILTERU PRO FILTRACI ŠUMU Z KAMERY ---
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

# ArUco Detektor
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# MediaPipe Pose & Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)

zone_calibrated = False
cached_polygon = None
cached_hull_np = None
cached_centers = {}

# Stavové proměnné pro ukončení běhu
has_entered_zone = False  # Zda ruce již vstoupily do zóny

# Inicializace VideoWriteru pro ořezané video
video_trim_path = os.path.join(OUTPUT_DIR, "annotated_trim.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_trim_path, fourcc, fps, (width, height))

raw_frames = []
cv2.namedWindow('Tracking & Anotace', cv2.WINDOW_NORMAL)

frame_index = 0

print("-> Spouštím zpracování. Proces bude ukončen ihned po opuštění zóny...")

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
            zone_calibrated = True

    # 2. Tracking Kloubů a Ukazováků
    pose_res = pose.process(frame_rgb)
    hands_res = hands.process(frame_rgb)

    frame_record = {
        "frame_id": frame_index,
        "timestamp": round(timestamp, 4),
        "L_index_tip": None,
        "R_index_tip": None,
        "L_elbow_angle": None,
        "R_elbow_angle": None,
        "raw_frame": frame.copy()
    }

    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        L_s, L_e, L_w = (int(lm[11].x * width), int(lm[11].y * height)), (int(lm[13].x * width), int(lm[13].y * height)), (int(lm[15].x * width), int(lm[15].y * height))
        R_s, R_e, R_w = (int(lm[12].x * width), int(lm[12].y * height)), (int(lm[14].x * width), int(lm[14].y * height)), (int(lm[16].x * width), int(lm[16].y * height))

        frame_record["L_elbow_angle"] = calculate_angle(L_s, L_e, L_w)
        frame_record["R_elbow_angle"] = calculate_angle(R_s, R_e, R_w)

    index_tips_in_frame = []
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label
            tip_lm = hand_landmarks.landmark[8]
            tip_pt = (int(tip_lm.x * width), int(tip_lm.y * height))
            index_tips_in_frame.append(tip_pt)

            if label == 'Left':
                frame_record["L_index_tip"] = tip_pt
            else:
                frame_record["R_index_tip"] = tip_pt

    # 3. Kontrola vstupu/výstupu ze zóny
    currently_in_zone = False
    if zone_calibrated and len(index_tips_in_frame) > 0:
        if any([cached_polygon.contains(Point(pt)) for pt in index_tips_in_frame]):
            currently_in_zone = True

    # A. Vstup do zóny (první zachycení)
    if currently_in_zone:
        has_entered_zone = True
        raw_frames.append(frame_record)

    # B. OPUŠTĚNÍ ZÓNY PO PŘEDCHOZÍM VSTUPU -> OKAMŽITÉ UKONČENÍ MODELU
    elif has_entered_zone and not currently_in_zone:
        print(f"-> Ruce opustily herní pole na snímku {frame_index}. Ukončuji model a přesouvám se k metrikám...")
        break

    # Živý náhled v okně
    if zone_calibrated:
        cv2.polylines(frame, [cached_hull_np], True, (0, 255, 0) if currently_in_zone else (0, 0, 255), 2)
    disp = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
    cv2.imshow('Tracking & Anotace', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# =========================================================================
# 4. FILTRACE DATASETU, GENERIK VIDEA A STATICKÉHO OBRÁZKU
# =========================================================================
print(f"-> Zpracovávám a filtruji {len(raw_frames)} aktivních snímků...")

if len(raw_frames) > 0:
    df = pd.DataFrame(raw_frames)

    df['L_x'] = df['L_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['L_y'] = df['L_index_tip'].apply(lambda p: p[1] if p else np.nan)
    df['R_x'] = df['R_index_tip'].apply(lambda p: p[0] if p else np.nan)
    df['R_y'] = df['R_index_tip'].apply(lambda p: p[1] if p else np.nan)

    # Interpolace výpadků detekce
    df['L_x'] = df['L_x'].interpolate(method='linear').bfill().ffill()
    df['L_y'] = df['L_y'].interpolate(method='linear').bfill().ffill()
    df['R_x'] = df['R_x'].interpolate(method='linear').bfill().ffill()
    df['R_y'] = df['R_y'].interpolate(method='linear').bfill().ffill()
    df['L_elbow_angle'] = df['L_elbow_angle'].interpolate(method='linear').bfill().ffill()
    df['R_elbow_angle'] = df['R_elbow_angle'].interpolate(method='linear').bfill().ffill()

    # 1€ Filter pro vyhlazení šumu z kamery
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

    # A. ZÁPIS OŘEZANÉHO ANOTOVANÉHO VIDEA (annotated_trim.mp4)
    filtered_frames_export = []
    for idx, row in df.iterrows():
        annotated_frame = row['raw_frame'].copy()
        if cached_hull_np is not None:
            cv2.polylines(annotated_frame, [cached_hull_np], True, (0, 255, 0), 3)

        L_pt = (int(row['L_x_filtered']), int(row['L_y_filtered']))
        R_pt = (int(row['R_x_filtered']), int(row['R_y_filtered']))

        cv2.circle(annotated_frame, L_pt, 8, (255, 0, 0), -1)   # Modrá = Levá
        cv2.circle(annotated_frame, R_pt, 8, (0, 255, 255), -1) # Žlutá = Pravá

        cv2.putText(annotated_frame, f"L Angle: {int(row['L_elbow_angle'])} deg", (L_pt[0] + 10, L_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"R Angle: {int(row['R_elbow_angle'])} deg", (R_pt[0] + 10, R_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        video_writer.write(annotated_frame)

        filtered_frames_export.append({
            "frame_id": int(row['frame_id']),
            "timestamp": float(row['timestamp']),
            "L_index_tip": [round(float(row['L_x_filtered']), 2), round(float(row['L_y_filtered']), 2)],
            "R_index_tip": [round(float(row['R_x_filtered']), 2), round(float(row['R_y_filtered']), 2)],
            "L_elbow_angle": round(float(row['L_elbow_angle']), 2),
            "R_elbow_angle": round(float(row['R_elbow_angle']), 2)
        })

    video_writer.release()

    # B. GENEROVÁNÍ OBRÁZKU TRAJEKTORIÍ (trajectory_plot.jpg)
    bg_image = raw_frames[0]['raw_frame'].copy()
    if cached_hull_np is not None:
        cv2.polylines(bg_image, [cached_hull_np], True, (0, 255, 0), 3)

    L_pts = np.column_stack((df['L_x_filtered'].values, df['L_y_filtered'].values)).astype(np.int32)
    R_pts = np.column_stack((df['R_x_filtered'].values, df['R_y_filtered'].values)).astype(np.int32)

    cv2.polylines(bg_image, [L_pts], isClosed=False, color=(255, 0, 0), thickness=3)   # Modrá = Levá
    cv2.polylines(bg_image, [R_pts], isClosed=False, color=(0, 255, 255), thickness=3) # Žlutá = Pravá

    cv2.circle(bg_image, tuple(L_pts[0]), 8, (255, 0, 0), -1)
    cv2.circle(bg_image, tuple(L_pts[-1]), 8, (255, 0, 0), 2)
    cv2.circle(bg_image, tuple(R_pts[0]), 8, (0, 255, 255), -1)
    cv2.circle(bg_image, tuple(R_pts[-1]), 8, (0, 255, 255), 2)

    cv2.putText(bg_image, "Leva ruka (Modra)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(bg_image, "Prava ruka (Zluta)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    img_output_path = os.path.join(OUTPUT_DIR, "trajectory_plot.jpg")
    cv2.imwrite(img_output_path, bg_image)

    # Uložení motion_data.json
    json_path = os.path.join(OUTPUT_DIR, "motion_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"metadata": {"fps": fps}, "frames": filtered_frames_export}, f, indent=4)

    # =========================================================================
    # 5. VÝPOČET METRIK A GENEROVÁNÍ OUTPUT_DATA.JSON
    # =========================================================================
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

    output_metrics = {
        "doba_pohybu_sekundy": total_duration,
        "casova_synchronizace_L_a_P": round(float(sync_correlation), 4),
        "plynulost_pohybu_jerk": {
            "L_ukazovak_norm_jerk": round(float(L_norm_jerk), 4),
            "R_ukazovak_norm_jerk": round(float(R_norm_jerk), 4),
            "jednotka": "dimensionless_jerk"
        },
        "uhlova_extenze": {
            "L_koncetina_max_deg": round(float(df['L_elbow_angle'].max()), 2),
            "R_koncetina_max_deg": round(float(df['R_elbow_angle'].max()), 2)
        },
        "rozsah_pohybu_ROM": {
            "L_koncetina_rom_deg": round(float(df['L_elbow_angle'].max() - df['L_elbow_angle'].min()), 2),
            "R_koncetina_rom_deg": round(float(df['R_elbow_angle'].max() - df['R_elbow_angle'].min()), 2)
        }
    }

    output_json_path = os.path.join(OUTPUT_DIR, "output_data.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False)

    print("\n=================== PROCES DOKONČEN ===================")
    print(f"Ořezané video uloženo do:      {video_trim_path}")
    print(f"Obrázek trajektorií uložen do: {img_output_path}")
    print(f"JSON data uložena do:          {output_json_path}")
else:
    print("V herní zóně nebyl zachycen žádný pohyb.")
    video_writer.release()