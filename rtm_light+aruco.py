import cv2
import numpy as np
from rtmlib import PoseTracker, Wholebody, Body, draw_skeleton

# --- KONFIGURACE ---
VIDEO_PATH = "video/vstupni3.mp4"
OUTPUT_PATH = "rtm_turbo_analyza.mp4"

# ZMĚŇTE ZDE: 
# -> Wholebody (pokud chcete detaily prstů na rukou)
# -> Body (pokud vám stačí pouze ramena, lokty a zápěstí = NEJVYŠŠÍ RYCHLOST)
ZVOLENY_MODEL = Wholebody  
# ------------------------------

# 1. Inicializace optimalizovaného trackeru lidského těla
rtm_tracker = PoseTracker(
    ZVOLENY_MODEL,
    det_frequency=12,        # Detekce postavy běží jen 1x za 12 snímků
    mode='lightweight',      # Přepne na odlehčený Tiny/Nano model pro slabší CPU
    backend='onnxruntime',
    device='cpu'
)

# 2. Inicializace ArUco detektoru
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- PROMĚNNÉ PRO UZAMČENÍ DESKY ---
board_polygon_locked = False
board_vertices = None
REQUIRED_IDS = {0, 1, 2, 3}
# ------------------------------------

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

    # --- KROK A: Detekce lidského těla (RTMLib) ---
    keypoints, scores = rtm_tracker(frame)
    
    # Vytvoření kopie pro kreslení anotací
    annotated_frame = frame.copy()
    annotated_frame = draw_skeleton(annotated_frame, keypoints, scores, openpose_skeleton=False)

    # --- KROK B: Detekce ArUco značek (běží, dokud se plocha neuzamkne) ---
    if not board_polygon_locked:
        corners, ids, rejected = aruco_detector.detectMarkers(frame)
        
        if ids is not None:
        # Bezpečné převedení NumPy pole na klasický plochý seznam čísel
            detected_ids = ids.flatten().tolist()
            
            # Zkontrolujeme, zda jsou v záběru VŠECHNY 4 požadované značky najednou
            if REQUIRED_IDS.issubset(set(detected_ids)):
                centers = []
                # Spočítáme středy pro ID 0, 1, 2, 3
                for req_id in [0, 1, 2, 3]:
                    idx = detected_ids.index(req_id)
                    # Střed značky získáme zprůměrováním jejích 4 rohů
                    center = np.mean(corners[idx][0], axis=0).astype(np.int32)
                    centers.append(center)
                
                # Proženeme body funkcí convexHull, aby se propojily správně po obvodu a nekřížily se
                board_vertices = cv2.convexHull(np.array(centers))
                board_polygon_locked = True
                print(f"\n[INFO] Deska úspěšně detekována a uzamčena na snímku {frame_count}. ArUco vypnuto.")

    # --- KROK C: Vykreslení statického průhledného polygonu desky ---
    if board_polygon_locked:
        # Vytvoříme transparentní vrstvu (overlay)
        overlay = annotated_frame.copy()
        
        # Vyplníme vnitřek polygonu jemnou barvou (zde příjemná světle modro-zelená azurová: B=220, G=220, R=100)
        cv2.fillPoly(overlay, [board_vertices], color=(220, 220, 100))
        
        # Vykreslíme pevný obrysový polygon do hlavního obrazu
        cv2.polylines(annotated_frame, [board_vertices], isClosed=True, color=(255, 165, 0), thickness=2)
        
        # Prolneme vrstvy (alfa 0.2 znamená 20% průhlednost masky desky)
        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)

    # --- KROK D: Dynamické popisky dlaní (L / R) ---
    if len(keypoints) > 0:
        osoba = keypoints[0]
        vsechny_scores = scores[0]
        
        # Index 9 = levé zápěstí/dlaň, Index 10 = pravé zápěstí/dlaň
        levy_bod = osoba[9]
        pravy_bod = osoba[10]
        
        # Popisek "L" pro levou dlaň (vykreslí se, pokud je jistota detekce nad 30 %)
        if vsechny_scores[9] > 0.3:
            cv2.putText(annotated_frame, "L", (int(levy_bod[0]) + 15, int(levy_bod[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
        # Popisek "R" pro pravou dlaň
        if vsechny_scores[10] > 0.3:
            cv2.putText(annotated_frame, "R", (int(pravy_bod[0]) + 15, int(pravy_bod[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # --- KROK E: Obecné textové informace ---
    model_name = "wholebody" if ZVOLENY_MODEL == Wholebody else "Body (Pouze klouby)"
    cv2.putText(annotated_frame, f"RTM: {model_name}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)
    
    status_text = "ARUCO found" if board_polygon_locked else "NO ARUCO"
    status_color = (0, 255, 0) if board_polygon_locked else (0, 0, 255)
    cv2.putText(annotated_frame, status_text, (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # Ukládání a náhled
    out.write(annotated_frame)
    
    preview = cv2.resize(annotated_frame, (int(width / 2), int(height / 2)))
    cv2.imshow("RTM Turbo + Locked Board Preview", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nHatovo! Skript dokončen. Video uloženo do: {OUTPUT_PATH}")