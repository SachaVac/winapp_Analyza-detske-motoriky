import os
import sys
import subprocess
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

# Nastavení vzhledu
ctk.set_appearance_mode("Dark")  # Možnosti: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Možnosti: "blue", "green", "dark-blue"

class MotorAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Analýza Motorických Aktivita Dětí")
        self.geometry("950x750")
        self.minsize(900, 700)

        self.selected_video_path = None
        self.is_processing = False

        # --- SEZNAM AKTIVIT (Pro grid 2x5) ---
        self.activities = [
            {"id": 1, "name": "Aktivita 1\nSymetrické pohyby", "script": "main1.py", "icon": "assets/1.png"},
            {"id": 2, "name": "Aktivita 2\nRozdílné tempo", "script": "main2.py", "icon": "assets/2.png"},
            {"id": 3, "name": "Aktivita 3\nGrafomotorika", "script": "main3.py", "icon": "assets/3.png"},
            {"id": 4, "name": "Aktivita 4\nKreslení tvarů", "script": "main4.py", "icon": "assets/4.png"},
            {"id": 5, "name": "Aktivita 5\nAsymetrie a Gesta", "script": "main5.py", "icon": "assets/5.png"},
            {"id": 6, "name": "Aktivita 6\nAsymetrie a Počítání", "script": "main6.py", "icon": "assets/6.png"},
            {"id": 7, "name": "Aktivita 7\nKřížové souhyby", "script": "main7.py", "icon": "assets/7.png"},
            {"id": 8, "name": "Aktivita 8\nOsmička (Pravá)", "script": "main8.py", "icon": "assets/8.png"},
            {"id": 9, "name": "Aktivita 9\nOsmička (Levá)", "script": "main9.py", "icon": "assets/9.png"},
            {"id": 10, "name": "Aktivita 10\nKoordinované pohyby", "script": "main10.py", "icon": "assets/10.png"},
        ]

        self._build_ui()

    def _build_ui(self):
        # Header
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.pack(fill="x", padx=20, pady=(20, 10))

        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="Systém pro Automatickou Analýzu Motoriky", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=10)

        # Sekce pro výběr videa
        self.file_frame = ctk.CTkFrame(self, corner_radius=10)
        self.file_frame.pack(fill="x", padx=20, pady=10)

        self.btn_select_file = ctk.CTkButton(
            self.file_frame, 
            text="Vybrat Video Soubor (.mp4, .avi)", 
            command=self.select_video_file,
            font=ctk.CTkFont(size=14)
        )
        self.btn_select_file.pack(side="left", padx=15, pady=15)

        # SPRÁVNĚ:
        self.lbl_file_path = ctk.CTkLabel(
            self.file_frame, 
            text="Žádné video nebylo vybráno", 
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray"
        )
        self.lbl_file_path.pack(side="left", padx=10, fill="x", expand=True)

        # Mřížka procesů (2x5 Grid)
        self.grid_frame = ctk.CTkFrame(self, corner_radius=10)
        self.grid_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Konfigurace sloupců gridu (5 sloupců)
        for col in range(5):
            self.grid_frame.grid_columnconfigure(col, weight=1)
        for row in range(2):
            self.grid_frame.grid_rowconfigure(row, weight=1)

        # Generování tlačítek v Gridu 2x5
        self.activity_buttons = []
        for index, act in enumerate(self.activities):
            row = index // 5
            col = index % 5

            # Načtení ikonky, pokud existuje
            img_icon = None
            if os.path.exists(act["icon"]):
                pil_img = Image.open(act["icon"])
                img_icon = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(60, 60))

            btn = ctk.CTkButton(
                self.grid_frame,
                text=act["name"],
                image=img_icon,
                compound="top",
                corner_radius=12,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda a=act: self.start_activity_process(a)
            )
            btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            self.activity_buttons.append(btn)

        # Footer / Logovací lišta
        self.status_frame = ctk.CTkFrame(self, corner_radius=10)
        self.status_frame.pack(fill="x", padx=20, pady=(10, 20))

        self.lbl_status = ctk.CTkLabel(
            self.status_frame, 
            text="Připraven k použití. Vyberte video a klikněte na požadovanou aktivitu.", 
            font=ctk.CTkFont(size=13)
        )
        self.lbl_status.pack(pady=10)

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Vyberte video soubor",
            filetypes=[("Video soubory", "*.mp4 *.avi *.mov *.mkv"), ("Všechny soubory", "*.*")]
        )
        if file_path:
            self.selected_video_path = file_path
            self.lbl_file_path.configure(text=os.path.basename(file_path), text_color="white")
            self.lbl_status.configure(text=f"Vybráno video: {os.path.basename(file_path)}")

    def start_activity_process(self, activity_info):
        if not self.selected_video_path:
            messagebox.showwarning("Upozornění", "Nejprve vyberte vstupní video soubor!")
            return

        if self.is_processing:
            messagebox.showwarning("Upozornění", "Právě probíhá jiné zpracování. Počkejte na jeho dokončení.")
            return

        script_path = os.path.join("aktivity", activity_info["script"])
        if not os.path.exists(script_path):
            messagebox.showerror("Chyba", f"Skript neexistuje v cestě: {script_path}")
            return

        # Spuštění výpočtu v odděleném vlákně
        self.is_processing = True
        self.toggle_buttons_state(False)
        self.lbl_status.configure(text=f"Běží {activity_info['name'].replace('\n', ' ')}...")

        threading.Thread(
            target=self._run_script_thread, 
            args=(script_path, self.selected_video_path, activity_info["name"]),
            daemon=True
        ).start()

    def _run_script_thread(self, script_path, video_path, act_name):
        try:
            # Spuštění Python skriptu jako podprocesu
            cmd = [sys.executable, script_path, video_path]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.after(0, lambda: self.on_process_complete(True, act_name, "Zpracování úspěšně dokončeno! Výstupy uloženy v ./outputs"))
            else:
                self.after(0, lambda: self.on_process_complete(False, act_name, f"Chyba při běhu skriptu:\n{stderr}"))

        except Exception as e:
            self.after(0, lambda: self.on_process_complete(False, act_name, f"Neočekávaná výjimka: {str(e)}"))

    def on_process_complete(self, success, act_name, message):
        self.is_processing = False
        self.toggle_buttons_state(True)
        clean_name = act_name.replace('\n', ' ')

        if success:
            self.lbl_status.configure(text=f"{clean_name}: Dokončeno! Výstupy naleznete ve složce ./outputs")
            messagebox.showinfo("Dokončeno", f"{clean_name} byla úspěšně zpracována!\n\nVýstupy byly uloženy do adresáře ./outputs")
        else:
            self.lbl_status.configure(text=f"{clean_name}: Chyba při zpracování!")
            messagebox.showerror("Chyba zpracování", message)

    def toggle_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_select_file.configure(state=state)
        for btn in self.activity_buttons:
            btn.configure(state=state)

if __name__ == "__main__":
    app = MotorAnalysisApp()
    app.mainloop()