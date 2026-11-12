import os
import sys
import ctypes
import queue
import threading
import time
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

import pipeline


# Windows taskbar identity; harmless on other platforms.
try:
    myappid = "cognilab.motoranalysis.app.1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass


def resource_path(relative_path):
    """Resolve resources both in development and in a PyInstaller bundle."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        # Resolve assets relative to this script, not the caller's working directory.
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def fit_size(src_w, src_h, box_w, box_h):
    """Largest (w, h) with the aspect ratio of src that fits into box (never distorts)."""
    if src_w <= 0 or src_h <= 0 or box_w <= 0 or box_h <= 0:
        return 2, 2
    scale = min(box_w / float(src_w), box_h / float(src_h))
    return max(2, int(src_w * scale)), max(2, int(src_h * scale))


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

PREVIEW_MIN_INTERVAL_S = 0.05   # at most ~20 preview frames per second
UI_POLL_INTERVAL_MS = 30


class MotorAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CogniLab – Motor Analysis")

        # Initial size follows the screen (never larger than it); the window is freely
        # resizable and every area below is laid out with grid weights.
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(1100, int(screen_w * 0.85))
        height = min(900, int(screen_h * 0.85))
        pos_x = max(0, (screen_w - width) // 2)
        pos_y = max(0, (screen_h - height) // 3)
        self.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.minsize(720, 520)
        self.resizable(True, True)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.selected_video_path = None
        self.selected_folder_path = None
        self.is_processing = False
        self._closing = False
        self._fullscreen = False
        self.cancel_event = threading.Event()
        self._worker = None

        # Worker -> UI communication is thread safe: workers only write to this lock
        # protected slot / queue; the Tk main thread polls them (no widget access from
        # worker threads).
        self._ui_queue = queue.Queue()
        self._preview_lock = threading.Lock()
        self._latest_preview = None        # (rgb ndarray, progress) – only the newest frame is kept
        self._last_preview_update = 0.0
        self._preview_box = (480, 640)     # available size in pixels, updated on resize
        self._preview_photo = None         # keep a reference so Tk does not discard the image
        self._last_rgb = None

        self.activities = [
            {"id": 1, "name": "Aktivita 1\nSymetrické pohyby", "icon": resource_path("assets/1.png")},
            {"id": 2, "name": "Aktivita 2\nRozdílné tempo", "icon": resource_path("assets/2.png")},
            {"id": 3, "name": "Aktivita 3\nGrafomotorika", "icon": resource_path("assets/3.png")},
            {"id": 4, "name": "Aktivita 4\nKreslení tvarů", "icon": resource_path("assets/4.png")},
            {"id": 5, "name": "Aktivita 5\nAsymetrie a Gesta", "icon": resource_path("assets/5.png")},
            {"id": 6, "name": "Aktivita 6\nAsymetrie a Počítání", "icon": resource_path("assets/6.png")},
            {"id": 7, "name": "Aktivita 7\nKřížové souhyby", "icon": resource_path("assets/7.png")},
            {"id": 8, "name": "Aktivita 8\nOsmička (Pravá)", "icon": resource_path("assets/8.png")},
            {"id": 9, "name": "Aktivita 9\nOsmička (Levá)", "icon": resource_path("assets/9.png")},
            {"id": 10, "name": "Aktivita 10\nKoordinované pohyby", "icon": resource_path("assets/10.png")},
        ]
        self._build_ui()
        self.after(UI_POLL_INTERVAL_MS, self._poll_ui)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        # Root grid: row 2 (content) takes all spare space, the others keep their height.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Systém pro Automatickou Analýzu Motoriky",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.title_label.pack(pady=10)

        self.file_frame = ctk.CTkFrame(self, corner_radius=10)
        self.file_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        self.btn_select_file = ctk.CTkButton(
            self.file_frame, text="Vybrat video", command=self.select_video_file,
            font=ctk.CTkFont(size=14), width=150,
        )
        self.btn_select_file.pack(side="left", padx=(12, 6), pady=10)
        self.btn_select_folder = ctk.CTkButton(
            self.file_frame, text="Batch: vybrat složku", command=self.select_video_folder,
            font=ctk.CTkFont(size=14), width=180,
        )
        self.btn_select_folder.pack(side="left", padx=6, pady=10)
        self.btn_start_batch = ctk.CTkButton(
            self.file_frame, text="Spustit batch", command=self.start_batch_process,
            font=ctk.CTkFont(size=14, weight="bold"), width=130, state="disabled",
        )
        self.btn_start_batch.pack(side="left", padx=6, pady=10)
        self.lbl_file_path = ctk.CTkLabel(
            self.file_frame,
            text="Vyberte video nebo složku pro batch processing",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray", anchor="w",
        )
        self.lbl_file_path.pack(side="left", padx=8, fill="x", expand=True)

        # Content area: the activity menu and the live preview share the same cell.
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=2, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.grid_frame = ctk.CTkFrame(self.content, corner_radius=10)
        self.grid_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
        for col in range(5):
            self.grid_frame.grid_columnconfigure(col, weight=1, uniform="activity")
        for row in range(2):
            self.grid_frame.grid_rowconfigure(row, weight=1, uniform="activity")

        self.activity_buttons = []
        self._activity_images = []
        for index, activity in enumerate(self.activities):
            row, col = divmod(index, 5)
            img_icon = None
            if os.path.exists(activity["icon"]):
                try:
                    pil_img = Image.open(activity["icon"])
                    img_icon = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(50, 50))
                    self._activity_images.append(img_icon)
                except Exception:
                    img_icon = None
            button = ctk.CTkButton(
                self.grid_frame,
                text=activity["name"],
                image=img_icon,
                compound="top",
                corner_radius=12,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda selected=activity: self.start_activity_process(selected),
            )
            button.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            self.activity_buttons.append(button)

        self.preview_frame = ctk.CTkFrame(self.content, corner_radius=10)
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(1, weight=1)   # the video takes the spare space
        self.lbl_preview_title = ctk.CTkLabel(
            self.preview_frame, text="Průběh analýzy a živý náhled",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.lbl_preview_title.grid(row=0, column=0, pady=(10, 5))

        # The container does not adapt to its content (propagation off), so the picture
        # always fits the space the layout gives it instead of dictating the window size.
        self.preview_container = tk.Frame(self.preview_frame, bg="black", highlightthickness=0)
        self.preview_container.grid(row=1, column=0, sticky="nsew", padx=12, pady=5)
        self.preview_container.grid_propagate(False)
        self.preview_container.pack_propagate(False)
        self.video_label = tk.Label(self.preview_container, bg="black", bd=0, highlightthickness=0)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.preview_container.bind("<Configure>", self._on_preview_resize)

        self.progress_bar = ctk.CTkProgressBar(self.preview_frame)
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=40, pady=(10, 2))
        self.progress_bar.set(0.0)
        self.lbl_progress_pct = ctk.CTkLabel(
            self.preview_frame, text="Průběh: 0 %", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_progress_pct.grid(row=3, column=0, pady=(0, 6))
        self.btn_cancel = ctk.CTkButton(
            self.preview_frame, text="Přerušit analýzu", fg_color="red", hover_color="darkred",
            command=self.cancel_analysis, font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.btn_cancel.grid(row=4, column=0, pady=(4, 12))
        self.preview_frame.grid_remove()

        self.status_frame = ctk.CTkFrame(self, corner_radius=10)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(5, 15))
        self.lbl_status = ctk.CTkLabel(
            self.status_frame,
            text="Připraveno. Vyberte video pro ruční analýzu nebo složku pro batch processing.",
            font=ctk.CTkFont(size=13), wraplength=900, justify="center",
        )
        self.lbl_status.pack(pady=8, padx=10, fill="x")
        self.status_frame.bind(
            "<Configure>", lambda e: self.lbl_status.configure(wraplength=max(200, e.width - 40))
        )

    # ------------------------------------------------------------ window helpers
    def _toggle_fullscreen(self, _event=None):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self, _event=None):
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)

    def _on_close(self):
        if self.is_processing:
            if not messagebox.askyesno(
                "Ukončit aplikaci", "Probíhá zpracování. Opravdu chcete aplikaci ukončit?"
            ):
                return
            self.cancel_event.set()
        self._closing = True
        self._finish_close(time.monotonic())

    def _finish_close(self, started):
        # Give the worker a moment to stop at the next frame boundary, then close.
        worker_alive = self._worker is not None and self._worker.is_alive()
        if worker_alive and time.monotonic() - started < 5.0:
            self.after(100, lambda: self._finish_close(started))
            return
        self.destroy()

    # --------------------------------------------------------------- file select
    def select_video_file(self):
        if self.is_processing:
            return
        file_path = filedialog.askopenfilename(
            title="Vyberte video soubor",
            filetypes=[
                ("Video soubory", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"),
                ("Všechny soubory", "*.*"),
            ],
        )
        if file_path:
            self.selected_video_path = file_path
            self.selected_folder_path = None
            self.btn_start_batch.configure(state="disabled")
            self.lbl_file_path.configure(text=os.path.basename(file_path), text_color="white")
            self.lbl_status.configure(text=f"Vybráno video: {os.path.basename(file_path)}")

    def select_video_folder(self):
        if self.is_processing:
            return
        folder_path = filedialog.askdirectory(title="Vyberte složku s videi pro batch processing")
        if folder_path:
            self.selected_folder_path = folder_path
            self.selected_video_path = None
            self.lbl_file_path.configure(text=f"Batch složka: {folder_path}", text_color="white")
            self.lbl_status.configure(
                text="Složka vybrána. Klikněte na ‚Spustit batch‘ pro zpracování všech podporovaných videí."
            )
            self.btn_start_batch.configure(state="normal")

    # ------------------------------------------------------------- processing UI
    def _show_processing_view(self, status_text):
        self.is_processing = True
        self.cancel_event.clear()
        self._last_preview_update = 0.0
        with self._preview_lock:
            self._latest_preview = None
        self._last_rgb = None
        self._preview_photo = None
        self.video_label.configure(image="")
        self.toggle_buttons_state(False)
        self.btn_cancel.configure(state="normal", text="Přerušit analýzu")
        self.progress_bar.set(0.0)
        self.lbl_progress_pct.configure(text="Průběh: 0 %")
        self.grid_frame.grid_remove()
        self.preview_frame.grid()
        self.lbl_status.configure(text=status_text)

    def _start_worker(self, target, args):
        self._worker = threading.Thread(target=target, args=args, daemon=True)
        self._worker.start()

    def start_activity_process(self, activity_info):
        if not self.selected_video_path:
            messagebox.showwarning("Upozornění", "Nejprve vyberte vstupní video soubor!")
            return
        if self.is_processing:
            messagebox.showwarning("Upozornění", "Právě probíhá jiné zpracování.")
            return
        activity_name = activity_info["name"].replace("\n", " ")
        video_path = self.selected_video_path
        self._show_processing_view(f"Probíhá {activity_name}...")
        self._start_worker(self._run_analysis_thread, (activity_info["id"], video_path, activity_name))

    def start_batch_process(self):
        if not self.selected_folder_path:
            messagebox.showwarning("Upozornění", "Nejprve vyberte složku obsahující videa!")
            return
        if self.is_processing:
            messagebox.showwarning("Upozornění", "Právě probíhá jiné zpracování.")
            return
        folder_path = self.selected_folder_path
        self._show_processing_view("Spouští se batch processing...")
        self._start_worker(self._run_batch_thread, (folder_path,))

    # ----------------------------------------------------- live preview (worker side)
    def update_frame_preview(self, cv2_frame, progress_pct):
        """Called from the worker thread for every decoded frame.

        Throttled first (cheap), then the frame is scaled to the current preview box with
        its original aspect ratio (9:16 for the 1080 x 1920 phone recordings) and stored
        as the newest frame. No Tk call is made from this thread.
        """
        if not self.is_processing:
            return
        now = time.monotonic()
        if now - self._last_preview_update < PREVIEW_MIN_INTERVAL_S:
            return
        self._last_preview_update = now
        try:
            src_h, src_w = cv2_frame.shape[:2]
            box_w, box_h = self._preview_box
            new_w, new_h = fit_size(src_w, src_h, box_w, box_h)
            interpolation = cv2.INTER_AREA if new_w < src_w else cv2.INTER_LINEAR
            small = cv2.resize(cv2_frame, (new_w, new_h), interpolation=interpolation)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        except Exception:
            return
        with self._preview_lock:
            self._latest_preview = (rgb, float(progress_pct))

    def update_batch_progress(self, status_text, progress_pct):
        self._ui_queue.put(("progress", status_text, float(progress_pct)))

    # ------------------------------------------------------------- UI thread side
    def _on_preview_resize(self, event):
        self._preview_box = (max(64, event.width - 4), max(64, event.height - 4))
        if self._last_rgb is not None:
            self._render_rgb(self._last_rgb)   # refit the last frame immediately

    def _render_rgb(self, rgb):
        self._last_rgb = rgb
        pil_img = Image.fromarray(rgb)
        box_w, box_h = self._preview_box
        new_w, new_h = fit_size(pil_img.width, pil_img.height, box_w, box_h)
        if (new_w, new_h) != (pil_img.width, pil_img.height):
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        self._preview_photo = ImageTk.PhotoImage(pil_img)
        self.video_label.configure(image=self._preview_photo)

    def _set_progress(self, progress_pct):
        progress_pct = max(0.0, min(1.0, float(progress_pct)))
        self.progress_bar.set(progress_pct)
        self.lbl_progress_pct.configure(text=f"Průběh: {int(progress_pct * 100)} %")

    def _poll_ui(self):
        """Main-thread poll: newest preview frame + queued messages from the workers."""
        try:
            if self.is_processing:
                with self._preview_lock:
                    latest, self._latest_preview = self._latest_preview, None
                if latest is not None:
                    rgb, progress = latest
                    self._render_rgb(rgb)
                    self._set_progress(progress)
            while True:
                try:
                    message = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                kind = message[0]
                if kind == "progress" and self.is_processing:
                    self.lbl_status.configure(text=message[1])
                    self._set_progress(message[2])
                elif kind == "done":
                    self.on_process_complete(message[1], message[2], message[3])
        finally:
            if not self._closing:
                self.after(UI_POLL_INTERVAL_MS, self._poll_ui)

    def cancel_analysis(self):
        if not self.is_processing:
            return
        if messagebox.askyesno(
            "Přerušit",
            "Opravdu chcete přerušit probíhající zpracování? Batch režim uloží částečné CSV výsledky.",
        ):
            self.cancel_event.set()
            self.btn_cancel.configure(state="disabled", text="Zastavuji...")
            self.lbl_status.configure(text="Zastavuji zpracování. Počkejte na dokončení aktuálního snímku...")
            # Do not reset the UI here: the worker must finish before another job can start.

    # ------------------------------------------------------------------- workers
    def _run_analysis_thread(self, activity_id, video_path, activity_name):
        try:
            success, message, _ = pipeline.run_pipeline(
                activity_id=activity_id,
                video_path=video_path,
                frame_callback=self.update_frame_preview,
                cancel_event=self.cancel_event,
            )
            self._ui_queue.put(("done", success, activity_name, message))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._ui_queue.put(("done", False, activity_name, f"Neočekávaná chyba:\n{exc}"))

    def _run_batch_thread(self, folder_path):
        try:
            success, message, _ = pipeline.run_batch_pipeline(
                folder_path=folder_path,
                frame_callback=self.update_frame_preview,
                progress_callback=self.update_batch_progress,
                cancel_event=self.cancel_event,
            )
            self._ui_queue.put(("done", success, "Batch processing", message))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._ui_queue.put(("done", False, "Batch processing", f"Batch processing selhal:\n{exc}"))

    def on_process_complete(self, success, process_name, message):
        self.is_processing = False
        self.lbl_status.configure(
            text=f"{process_name}: {'Dokončeno.' if success else 'Zpracování nebylo dokončeno.'}"
        )
        if self._closing:
            return
        if self.cancel_event.is_set() and "přerušen" in message.lower():
            messagebox.showinfo("Zpracování přerušeno", message)
        elif success:
            messagebox.showinfo("Dokončeno", message)
        else:
            messagebox.showerror("Výsledek analýzy", message)
        self.reset_to_main_menu()

    def reset_to_main_menu(self):
        self.is_processing = False
        self.preview_frame.grid_remove()
        self.grid_frame.grid()
        self.toggle_buttons_state(True)
        self.btn_cancel.configure(state="normal", text="Přerušit analýzu")

    def toggle_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_select_file.configure(state=state)
        self.btn_select_folder.configure(state=state)
        batch_state = "normal" if enabled and self.selected_folder_path else "disabled"
        self.btn_start_batch.configure(state=batch_state)
        for button in self.activity_buttons:
            button.configure(state=state)


if __name__ == "__main__":
    app = MotorAnalysisApp()
    app.mainloop()
