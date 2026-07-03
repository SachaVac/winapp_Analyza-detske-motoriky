import cv2
import numpy as np
import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from rtmlib import PoseTracker, Wholebody, Body, draw_skeleton

# --- KONFIGURACE MODELU ---
ZVOLENY_MODEL = Wholebody  # Můžete změnit na Body pro vyšší rychlost
# --------------------------

class VideoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analýza motoriky s živým náhledem")
        self.root.geometry("800x650")
        self.root.configure(bg="#f0f0f0")

        self.input_video_path = ""
        self.temp_output_path = "temp_analyza_output.mp4"
        self.is_running = False

        self.create_widgets()

    def create_widgets(self):
        # Horní panel pro ovládání
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        self.btn_select = ttk.Button(control_frame, text="1. Nahrát video", command=self.select_video)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.lbl_file = ttk.Label(control_frame, text="Vyberte video soubor...", font=("Helvetica", 9, "italic"))
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_start = ttk.Button(control_frame, text="2. Spustit analýzu", command=self.start_analysis_thread, state=tk.DISABLED)
        self.btn_start.pack(side=tk.RIGHT, padx=5)

        # Středový panel pro živé zobrazování videa
        self.video_frame = tk.Frame(self.root, bg="black", width=760, height=450)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.video_frame.pack_propagate(False)

        self.lbl_video = tk.Label(self.video_frame, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # Spodní stavový panel a Progress bar
        status_frame = ttk.Frame(self.root, padding=10)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = ttk.Label(status_frame, text="Připraveno.", font=("Helvetica", 10, "bold"))
        self.lbl_status.pack(anchor=tk.W)

        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Vyberte vstupní video",
            filetypes=[("Video soubory", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.input_video_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path), font=("Helvetica", 9, "normal"))
            self.btn_start.config(state=tk.NORMAL)
            self.lbl_status.config(text="Video úspěšně nahráno. Můžete spustit analýzu.")

    def start_analysis_thread(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.btn_select.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        
        # Spuštění výpočtu na pozadí, aby GUI nezamrzlo
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            self.lbl_status.config(text="Inicializace AI modelů (RTM + ArUco)...")
            
            # Vaše inicializační logika
            rtm_tracker = PoseTracker(
                ZVOLENY_MODEL,
                det_frequency=12,
                mode='lightweight',
                backend='onnxruntime',
                device='cpu'
            )

            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

            board_polygon_locked = False
            board_vertices = None
            REQUIRED_IDS = {0, 1, 2, 3}

            cap = cv2.VideoCapture(self.input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.temp_output_path, fourcc, fps, (width, height))

            frame_count = 0

            while cap.isOpened() and self.is_running:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                
                # Aktualizace Progress Baru a textu
                progress_val = int((frame_count / total_frames) * 100)
                self.progress['value'] = progress_val
                self.lbl_status.config(text=f"Zpracovávám: {frame_count} / {total_frames} snímků ({progress_val}%)")

                # --- VAŠE DETEKČNÍ LOGIKA ---
                keypoints, scores = rtm_tracker(frame)
                annotated_frame = frame.copy()
                annotated_frame = draw_skeleton(annotated_frame, keypoints, scores, openpose_skeleton=False)

                # ArUco detekce
                if not board_polygon_locked:
                    corners, ids, rejected = aruco_detector.detectMarkers(frame)
                    if ids is not None:
                        detected_ids = ids.flatten().tolist()
                        if REQUIRED_IDS.issubset(set(detected_ids)):
                            centers = []
                            for req_id in [0, 1, 2, 3]:
                                idx = detected_ids.index(req_id)
                                center = np.mean(corners[idx][0], axis=0).astype(np.int32)
                                centers.append(center)
                            board_vertices = cv2.convexHull(np.array(centers))
                            board_polygon_locked = True

                # Vykreslení desky
                if board_polygon_locked:
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [board_vertices], color=(220, 220, 100))
                    cv2.polylines(annotated_frame, [board_vertices], isClosed=True, color=(255, 165, 0), thickness=2)
                    cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)

                # Popisky dlaní
                if len(keypoints) > 0:
                    osoba = keypoints[0]
                    vsechny_scores = scores[0]
                    if vsechny_scores[9] > 0.3:
                        cv2.putText(annotated_frame, "L", (int(osoba[9][0]) + 15, int(osoba[9][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    if vsechny_scores[10] > 0.3:
                        cv2.putText(annotated_frame, "R", (int(osoba[10][0]) + 15, int(osoba[10][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

                # Obecné texty
                model_name = "Wholebody" if ZVOLENY_MODEL == Wholebody else "Body"
                cv2.putText(annotated_frame, f"RTM: {model_name}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)
                
                status_text = "ARUCO found" if board_polygon_locked else "Hledam desku..."
                status_color = (0, 255, 0) if board_polygon_locked else (0, 0, 255)
                cv2.putText(annotated_frame, status_text, (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

                # Zápis do souboru
                out.write(annotated_frame)

                # --- ŽIVÝ NÁHLED DO OKNA APLIKACE ---
                # Změna velikosti pro GUI okno
                preview_img = cv2.resize(annotated_frame, (760, 450))
                # Převod BGR na RGB (OpenCV používá BGR, Tkinter vyžaduje RGB)
                preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                
                # Převod na formát Pillow -> Tkinter PhotoImage
                img_pil = Image.fromarray(preview_img)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                # Bezpečná aktualizace obrázku v hlavním okně
                self.lbl_video.config(image=img_tk)
                self.lbl_video.image = img_tk
                
                self.root.update_idletasks()

            cap.release()
            out.release()

            # Vyzvání ke stažení / uložení videa
            self.lbl_status.config(text="Analýza úspěšně dokončena.")
            self.prompt_save_file()

        except Exception as e:
            messagebox.showerror("Chyba", f"Došlo k chybě při zpracování:\n{str(e)}")
            self.lbl_status.config(text="Zpracování selhalo.")
        
        finally:
            self.is_running = False
            self.btn_select.config(state=tk.NORMAL)
            self.progress['value'] = 0

    def prompt_save_file(self):
        # Dotaz na stažení souboru
        if messagebox.askyesno("Uložit video", "Analýza proběhla úspěšně!\nChcete nyní anotované video stáhnout / uložit do počítače?"):
            save_path = filedialog.asksaveasfilename(
                title="Vyberte kam uložit anotované video",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")]
            )
            if save_path:
                # Zkopíruje dočasné video na uživatelem zvolené místo
                shutil.copy(self.temp_output_path, save_path)
                messagebox.showinfo("Uloženo", f"Video bylo úspěšně staženo do:\n{save_path}")
        
        # Smazání dočasného souboru
        if os.path.exists(self.temp_output_path):
            os.remove(self.temp_output_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalysisApp(root)
    root.mainloop()