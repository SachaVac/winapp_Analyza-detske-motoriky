import cv2
import numpy as np
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# --- KONFIGURACE ---
DICTIONARY_TYPE = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]       # 4 značky pro 4 rohy
PIXEL_SIZE = 400                # Rozlišení černé části (interní pro čisté hrany)
WHITE_BORDER_PERC = 0.15        # Velikost bílého okraje (15 % z velikosti značky)
OUTPUT_DIR = "aruco_znacky"
PDF_OUTPUT_PATH = "aruco_znacky_k_tisku_A4.pdf"

# --- STRUKTURA PRO PŘESNÝ TISK (v milimetrech) ---
BLACK_SIZE_MM = 25  # Požadovaná fyzická velikost ČERNÉ části na papíře
# --------------------

# Výpočet celkové fyzické velikosti včetně bílého okraje
# (okraj je z obou stran, proto 2x)
total_size_mm = BLACK_SIZE_MM * (1 + 2 * WHITE_BORDER_PERC)

# Vytvoření složky pro dočasné obrázky
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializace ArUco slovníku
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY_TYPE)

print("1. Generuji zdrojové PNG obrázky...")
image_paths = []

for marker_id in MARKER_IDS:
    # Vygenerování základní černé značky
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, PIXEL_SIZE)
    
    # Přidání bílého okraje (Quiet Zone)
    border_thickness = int(PIXEL_SIZE * WHITE_BORDER_PERC)
    final_img = cv2.copyMakeBorder(
        marker_img, 
        top=border_thickness, 
        bottom=border_thickness, 
        left=border_thickness, 
        right=border_thickness, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255
    )
    
    filename = os.path.join(OUTPUT_DIR, f"aruco_4x4_id{marker_id}.png")
    cv2.imwrite(filename, final_img)
    image_paths.append(filename)

print("2. Skládám PDF dokument s přesnými rozměry...")

# Inicializace ReportLab canvasu nastaveného na formát A4
pdf = canvas.Canvas(PDF_OUTPUT_PATH, pagesize=A4)

# Definice souřadnic [X, Y] v mm pro 4 značky na stránce A4 (210x297 mm)
# Rozmístěno do mřížky 2x2 tak, aby se pohodlně stříhalo
positions = [
    (35, 180),  # ID 0 (Vlevo nahoře)
    (125, 180), # ID 1 (Vpravo nahoře)
    (35, 80),   # ID 2 (Vlevo dole)
    (125, 80)   # ID 3 (Vpravo dole)
]

for idx, marker_id in enumerate(MARKER_IDS):
    x_mm, y_mm = positions[idx]
    img_path = image_paths[idx]
    
    # Kreslení jemné šedé přerušované linky pro ořez (označuje vnější hranu bílého okraje)
    pdf.setStrokeColorRGB(0.7, 0.7, 0.7)
    pdf.setLineWidth(0.5)
    pdf.setDash(2, 2)
    pdf.rect(x_mm * mm, y_mm * mm, total_size_mm * mm, total_size_mm * mm, stroke=1, fill=0)
    
    # Vložení samotného ArUco obrázku přesně na milimetry
    pdf.drawImage(img_path, x_mm * mm, y_mm * mm, width=total_size_mm * mm, height=total_size_mm * mm)
    
    # Přidání textového popisku pod značku
    pdf.setFont("Helvetica", 9)
    pdf.setFillColorRGB(0.3, 0.3, 0.3)
    pdf.drawString(x_mm * mm, (y_mm - 5) * mm, f"ArUco ID: {marker_id} (Cerna cast: {BLACK_SIZE_MM} mm)")

# Uložení hotového PDF
pdf.save()

print(f"\nHatovo! Výsledné PDF připravené k tisku uloženo do: {PDF_OUTPUT_PATH}")
print(f"Celková velikost jednoho čtverce k vystřižení (včetně bílé zóny): {total_size_mm:.1f} mm")