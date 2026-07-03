import cv2
import numpy as np
from rtmlib import PoseTracker, Wholebody, Body, draw_skeleton

# --- KONFIGURACE ZRYCHLENÍ ---
VIDEO_PATH = "video/vstupni3.mp4"
OUTPUT_PATH = "rtm_turbo_analyza.mp4"

# ZMĚŇTE ZDE: 
# -> Wholebody (pokud chcete detaily prstů na rukou)
# -> Body (pokud vám stačí pouze ramena, lokty a zápěstí = NEJVYŠŠÍ RYCHLOST)
ZVOLENY_MODEL = Wholebody  
# ------------------------------

# Inicializace optimalizovaného trackeru
rtm_tracker = PoseTracker(
    ZVOLENY_MODEL,
    det_frequency=12,        # Detekce postavy běží jen 1x za 12 snímků, zbytek se jen bleskově trackuje
    mode='lightweight',      # Přepne na odlehčený Tiny/Nano model pro slabší CPU
    backend='onnxruntime',
    device='cpu'
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
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Startuji TURBO analýzu. Celkem snímků: {total_frames}")
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    print(f"Zpracovávám snímek: {frame_count} / {total_frames}", end="\r")

    # Spuštění ultra rychlého trackeru
    keypoints, scores = rtm_tracker(frame)
    
    # Vykreslení výsledků
    annotated_frame = frame.copy()
    annotated_frame = draw_skeleton(annotated_frame, keypoints, scores, openpose_skeleton=False)

    # Přidání textu o nastavení
    model_name = "Wholebody (S prsty)" if ZVOLENY_MODEL == Wholebody else "Body (Pouze klouby)"
    cv2.putText(annotated_frame, f"RTM Turbo: {model_name}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)

    # Ukládání a náhled
    out.write(annotated_frame)
    
    preview = cv2.resize(annotated_frame, (int(width / 2), int(height / 2)))
    cv2.imshow("RTM Turbo Preview", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nHatovo! Optimalizované video uloženo do: {OUTPUT_PATH}")