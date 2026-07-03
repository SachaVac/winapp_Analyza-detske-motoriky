import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# --- KONFIGURACE ---
VIDEO_PATH = "video/vstupni4.mp4"  # Cesta k vašemu videu
OUTPUT_PATH = "srovnani_yolo_hands.mp4"
# --------------------

# Inicializace YOLOv8-Pose (pro srovnání)
yolo_model = YOLO("yolov8n-pose.pt")

# Inicializace MediaPipe HANDS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_estimator = mp_hands.Hands(
    static_image_mode=False,     # False znamená, že zpracováváme video (pohyby)
    max_num_hands=2,             # Hledáme maximálně 2 ruce
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Otevření videa
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Chyba: Video {VIDEO_PATH} nelze otevřít.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width * 2, height))

print(f"Video otevřeno. Celkem snímků: {total_frames}")
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    print(f"Zpracovávám snímek: {frame_count} / {total_frames}", end="\r")

    frame_yolo = frame.copy()
    frame_mp = frame.copy()

    # --- 1. YOLOv8-POSE ---
    yolo_results = yolo_model(frame_yolo, verbose=False)[0]
    frame_yolo = yolo_results.plot(labels=False, boxes=False)

    # --- 2. MEDIAPIPE HANDS ---
    frame_rgb = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
    mp_results = hands_estimator.process(frame_rgb)
    
    if mp_results.multi_hand_landmarks:
        # Procházíme všechny nalezené ruce a jejich klasifikaci (Pravá/Levá)
        for hand_landmarks, hand_class in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
            
            # Získání informace, zda jde o levou nebo pravou ruku
            ruka_label = hand_class.classification[0].label  # "Left" nebo "Right"
            
            # Vykreslení kostry ruky
            mp_drawing.draw_landmarks(
                frame_mp,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Vypíšeme text "Left"/"Right" přímo nad zápěstí (bod 0)
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * width), int(wrist.y * height)
            color = (0, 255, 0) if ruka_label == "Right" else (255, 0, 0)
            cv2.putText(frame_mp, ruka_label, (cx, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- 3. SPOJENÍ A UKLÁDÁNÍ ---
    cv2.putText(frame_yolo, "YOLOv8-Pose (Telo)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame_mp, "MediaPipe Hands (Ruce)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    combined_frame = np.hstack((frame_yolo, frame_mp))
    out.write(combined_frame)
    
    preview = cv2.resize(combined_frame, (int(width), int(height / 2)))
    cv2.imshow("Srovnani: Telo vs Ruce", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nHatovo! Video uloženo do: {OUTPUT_PATH}")