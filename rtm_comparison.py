import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from rtmlib import Wholebody, draw_skeleton

# --- KONFIGURACE ---
VIDEO_PATH = "video/vstupni3.mp4"  # Cesta k vašemu videu
OUTPUT_PATH = "srovnani_yolo_hands_rtmpose.mp4"
# --------------------

# 1. Inicializace YOLOv8-Pose
yolo_model = YOLO("yolov8n-pose.pt")

# 2. Inicializace MediaPipe HANDS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_estimator = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. Inicializace RTMLib (Wholebody model - tělo + ruce dohromady)
# rtmlib si při prvním spuštění sama stáhne potřebný .onnx model
rtm_model = Wholebody(backend='onnxruntime', device='cpu')

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
# Výstupní video bude mít dvojnásobnou šířku i výšku kvůli mřížce 2x2
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width * 2, height * 2))

print(f"Video otevřeno. Celkem snímků: {total_frames}")
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    print(f"Zpracovávám snímek: {frame_count} / {total_frames}", end="\r")

    # Příprava kopií snímku pro jednotlivé modely
    frame_yolo = frame.copy()
    frame_mp = frame.copy()
    frame_rtm = frame.copy()
    frame_orig = frame.copy()

    # --- 1. YOLOv8-POSE ---
    yolo_results = yolo_model(frame_yolo, verbose=False)[0]
    frame_yolo = yolo_results.plot(labels=False, boxes=False)

    # --- 2. MEDIAPIPE HANDS ---
    frame_rgb = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
    mp_results = hands_estimator.process(frame_rgb)
    
    if mp_results.multi_hand_landmarks:
        for hand_landmarks, hand_class in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
            ruka_label = hand_class.classification[0].label  # "Left" nebo "Right"
            
            mp_drawing.draw_landmarks(
                frame_mp,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * width), int(wrist.y * height)
            color = (0, 255, 0) if ruka_label == "Right" else (255, 0, 0)
            cv2.putText(frame_mp, ruka_label, (cx, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- 3. RTMLib WHOLEBODY ---
    keypoints, scores = rtm_model(frame_rtm)
    frame_rtm = draw_skeleton(frame_rtm, keypoints, scores, openpose_skeleton=False)

    # --- 4. ANOTACE TEXTŮ A SPOJENÍ DO MŘÍŽKY 2x2 ---
    cv2.putText(frame_yolo, "1. YOLOv8-Pose (Pouze telo)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame_mp, "2. MediaPipe (Pouze ruce)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame_rtm, "3. RTMLib Wholebody (Vse v jednom)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame_orig, "4. Originalni video", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Poskládání mřížky: horní řádek (YOLO + MP), spodní řádek (RTM + Originál)
    top_row = np.hstack((frame_yolo, frame_mp))
    bottom_row = np.hstack((frame_rtm, frame_orig))
    combined_frame = np.vstack((top_row, bottom_row))

    # Zápis do výsledného souboru
    out.write(combined_frame)
    
    # Zmenšení náhledu pro plynulé zobrazení na obrazovce (přizpůsobí se původní velikosti videa)
    preview = cv2.resize(combined_frame, (int(width), int(height)))
    cv2.imshow("Velke srovnani motoriky (Mrizka 2x2)", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nHotovo! Srovnávací video uloženo do: {OUTPUT_PATH}")