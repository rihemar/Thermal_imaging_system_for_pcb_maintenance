#!/usr/bin/env python3
"""
Controle Thermique PCB -- Application UNIQUE (tout-en-un)
============================================================

Fusionne dans un seul fichier :
  - La lecture du capteur MLX90640 (thread de fond, remplace l'executable C++ separe)
  - Le calibrage par homographie (Etape 1) -- clic de points DANS des Canvas Tkinter
  - La detection + overlay thermique component-aware (Etapes 2/3/4), en LIVE
  - L'affichage de la heatmap brute, en LIVE

ARCHITECTURE FENETRE UNIQUE
----------------------------
Contrairement aux versions precedentes, cette version n'ouvre JAMAIS de
fenetre Toplevel secondaire. Tout se passe dans LA MEME fenetre racine
(tk.Tk) : un conteneur central affiche une "vue" a la fois (menu,
calibrage, detection live, heatmap live), et on bascule d'une vue a
l'autre en detruisant l'ancienne et en montant la nouvelle a la place
(App.show_view). Comme il n'existe jamais plus d'une fenetre reelle,
le gestionnaire de fenetres n'a plus jamais a arbitrer le focus entre
plusieurs fenetres -- le probleme de focus qui ne "s'accrochait" pas a
la bonne fenetre disparait structurellement.

AUCUN fichier intermediaire n'est lu ou ecrit : la matrice thermique, l'image
RGB, l'homographie et le resultat final restent des objets Python/numpy en
memoire, passes directement entre les fonctions.

Seule exception assumee : les blueprints (schemas du PCB, un ou plusieurs
fichiers .jpg dans le dossier BLUEPRINT_DIR) sont lus depuis le disque --
ce sont de vrais fichiers d'entree externes, pas des fichiers intermediaires
generes par le pipeline. Au lancement de la Detection (option 2), l'appli
scanne BLUEPRINT_DIR/*.jpg et affiche un bouton par fichier trouve pour
choisir sur quel blueprint aligner la heatmap.

Dependances :
    pip install opencv-python numpy pillow matplotlib
    pip install adafruit-circuitpython-mlx90640 adafruit-blinka   # optionnel (capteur reel)

IMPORTANT : ne renommez jamais ce fichier "tkinter.py" (conflit avec le
module standard).
"""

import os
import sys
import time
import glob
import threading

import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import board
    import busio
    import adafruit_mlx90640
    HARDWARE_AVAILABLE = True
except (ImportError, NotImplementedError):
    HARDWARE_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

RGB_STREAM_URL = "http://192.168.1.19:81/stream"   # capture RGB -- inchangee
BLUEPRINT_DIR = "./blueprints"                      # dossier contenant les blueprints .jpg (seuls fichiers reellement lus depuis le disque)

SRC_W, SRC_H = 32, 24          # resolution native MLX90640
UPSCALE = 3
DST_W, DST_H = SRC_W * UPSCALE, SRC_H * UPSCALE

# Palette "cute" de l'interface
COLOR_BG = "#FDF6EC"
COLOR_HEADER = "#4A4E69"
COLOR_BTN_1 = "#FFADAD"
COLOR_BTN_2 = "#A0C4FF"
COLOR_BTN_3 = "#CAFFBF"
COLOR_BTN_TEXT = "#22223B"
COLOR_STATUS_OK = "#2ECC71"
COLOR_STATUS_ERR = "#E74C3C"
COLOR_STATUS_INFO = "#4A4E69"
COLOR_LOG_BG = "#22223B"
COLOR_LOG_TEXT = "#CAFFBF"


# ============================================================================
# FONCTIONS DE TRAITEMENT (pures, aucune I/O disque, aucune fenetre)
# ============================================================================

def build_lut_iron():
    stops_t = np.array([0.00, 0.20, 0.45, 0.65, 0.80, 0.92, 1.00], dtype=np.float32)
    stops_rgb = np.array([
        [0, 0, 0], [30, 0, 80], [120, 0, 120], [200, 30, 0],
        [255, 120, 0], [255, 220, 0], [255, 255, 200],
    ], dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        lo = 0
        for s in range(len(stops_t) - 1):
            if t >= stops_t[s]:
                lo = s
        hi = min(lo + 1, len(stops_t) - 1)
        span = max(1e-6, stops_t[hi] - stops_t[lo])
        f = (t - stops_t[lo]) / span
        lut[i] = stops_rgb[lo] + (stops_rgb[hi] - stops_rgb[lo]) * f
    return lut


LUT_IRON = build_lut_iron()


def sliding_window_upscale(src):
    """Upscale x3 par moyenne ponderee avec les 8 voisins (identique a la version C++)."""
    padded = np.pad(src, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    nw, ne = padded[:-2, :-2], padded[:-2, 2:]
    sw, se = padded[2:, :-2], padded[2:, 2:]

    dst = np.zeros((SRC_H * UPSCALE, SRC_W * UPSCALE), dtype=np.float32)
    dst[0::3, 0::3] = (nw + north + west + center) / 4.0
    dst[0::3, 1::3] = (center + north) / 2.0
    dst[0::3, 2::3] = (center + north + east + ne) / 4.0
    dst[1::3, 0::3] = (center + west) / 2.0
    dst[1::3, 1::3] = center
    dst[1::3, 2::3] = (center + east) / 2.0
    dst[2::3, 0::3] = (center + south + west + sw) / 4.0
    dst[2::3, 1::3] = (center + south) / 2.0
    dst[2::3, 2::3] = (center + south + east + se) / 4.0
    return dst


def temp_matrix_to_bgr(matrix, vmin=None, vmax=None):
    """Colorise une matrice de temperatures avec la palette iron -> image BGR uint8."""
    vmin = float(matrix.min()) if vmin is None else vmin
    vmax = float(matrix.max()) if vmax is None else vmax
    rng = max(1e-6, vmax - vmin)
    idx = np.clip(((matrix - vmin) / rng) * 255.0, 0, 255).astype(np.uint8)
    rgb = LUT_IRON[idx]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_pcb_corners(rgb_img, canny_low=50, canny_high=150, min_area_ratio=0.05):
    """Retourne les 4 coins detectes automatiquement, ou None si echec (-> fallback manuel)."""
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = rgb_img.shape[0] * rgb_img.shape[1]
    contours = [c for c in contours if cv2.contourArea(c) > min_area_ratio * img_area]
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect).astype(np.float32)

    return order_points(corners)


def compute_alignment_homography(rgb_corners, blueprint_corners):
    return cv2.getPerspectiveTransform(order_points(rgb_corners), order_points(blueprint_corners))


def find_hotspot(temp_matrix):
    smoothed = cv2.GaussianBlur(temp_matrix, (5, 5), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(smoothed)
    return max_loc[0], max_loc[1], max_val


def project_point(x, y, H_total):
    pt = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt, H_total)
    return float(projected[0][0][0]), float(projected[0][0][1])


def detect_component_contours(rgb_blueprint_space, min_area_px=40, max_area_ratio=0.15,
                               canny_low=40, canny_high=120):
    gray = cv2.cvtColor(rgb_blueprint_space, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    img_area = rgb_blueprint_space.shape[0] * rgb_blueprint_space.shape[1]
    max_area_px = max_area_ratio * img_area
    return [c for c in contours if min_area_px < cv2.contourArea(c) < max_area_px]


def normalize_temp_to_uint8(temp_map, vmin, vmax):
    if vmax - vmin < 1e-6:
        return np.zeros_like(temp_map, dtype=np.uint8)
    normalized = (temp_map - vmin) / (vmax - vmin) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def build_component_aware_overlay(temp_warped, valid_mask, component_contours, colormap=cv2.COLORMAP_INFERNO):
    h, w = temp_warped.shape[:2]
    uniform_temp_map = temp_warped.copy()
    covered_mask = np.zeros((h, w), dtype=np.uint8)

    for contour in component_contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask_valid = cv2.bitwise_and(mask, valid_mask)
        if cv2.countNonZero(mask_valid) == 0:
            continue
        mean_temp = cv2.mean(temp_warped, mask=mask_valid)[0]
        uniform_temp_map[mask == 255] = mean_temp
        covered_mask = cv2.bitwise_or(covered_mask, mask)

    valid_values = temp_warped[valid_mask > 0]
    if valid_values.size == 0:
        raise RuntimeError("Aucun pixel thermique valide apres warp. Verifiez l'homographie.")
    vmin, vmax = float(valid_values.min()), float(valid_values.max())

    uint8_map = normalize_temp_to_uint8(uniform_temp_map, vmin, vmax)
    colorized = cv2.applyColorMap(uint8_map, colormap)

    n_covered = int(np.count_nonzero(covered_mask))
    n_valid = int(np.count_nonzero(valid_mask))
    pct = 100.0 * n_covered / max(n_valid, 1)
    return colorized, vmin, vmax, pct


def grab_rgb_frame():
    """Capture UNE frame depuis le flux IP -- meme methode qu'avant, inchangee."""
    cap = cv2.VideoCapture(RGB_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible de se connecter au flux : {RGB_STREAM_URL}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Echec de capture d'une frame RGB.")
    return frame


def list_blueprint_files():
    """Scanne BLUEPRINT_DIR et retourne la liste triee des fichiers .jpg trouves."""
    if not os.path.isdir(BLUEPRINT_DIR):
        return []
    patterns = ("*.jpg","*.png", "*.jpeg", "*.JPG", "*.JPEG")
    files = set()
    for pat in patterns:
        files.update(glob.glob(os.path.join(BLUEPRINT_DIR, pat)))
    return sorted(files)


def make_header(parent, text, back_command, back_text="Retour au menu", back_bg="#A0C4FF"):
    """Barre d'en-tete standard (titre + bouton retour), utilisee par toutes les vues."""
    bar = tk.Frame(parent, bg=COLOR_HEADER, height=60)
    bar.pack(fill="x", side="top")
    bar.pack_propagate(False)
    tk.Label(bar, text=text, font=("Arial", 16, "bold"), bg=COLOR_HEADER, fg="white").pack(side="left", padx=20)
    tk.Button(bar, text=back_text, font=("Arial", 12, "bold"), bg=back_bg, fg=COLOR_BTN_TEXT, bd=0,
              command=back_command, padx=12, pady=6).pack(side="right", padx=20, pady=10)
    return bar


# ============================================================================
# THREAD CAPTEUR (remplace l'executable C++ separe)
# ============================================================================

class SensorThread(threading.Thread):
    """
    Lit le MLX90640 en continu et garde la derniere matrice en memoire
    (protege par un verrou). Mode simulation si le materiel est absent.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._raw = np.full((SRC_H, SRC_W), 25.0, dtype=np.float32)  # 25C ambiant par defaut
        self._running = True
        self.simulation = not HARDWARE_AVAILABLE
        self._mlx = None
        self._sim_t = 0.0

        if HARDWARE_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA, frequency=1_000_000)
                self._mlx = adafruit_mlx90640.MLX90640(i2c)
                self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
                try:
                    self._mlx.emissivity = 0.95
                except AttributeError:
                    pass
            except Exception as e:
                print(f"[MLX90640 init failed] {e}", file=sys.stderr)
                self.simulation = True  # bascule en simulation si l'init materielle echoue

    def get_raw(self):
        with self._lock:
            return self._raw.copy()

    def get_scaled(self):
        return sliding_window_upscale(self.get_raw())

    def stop(self):
        self._running = False

    def run(self):
        frame_buf = [0.0] * (SRC_W * SRC_H)
        while self._running:
            if self.simulation:
                self._sim_t += 0.5
                base = np.full((SRC_H, SRC_W), 24.0, dtype=np.float32)
                cx = SRC_W / 2 + 6 * np.sin(self._sim_t * 0.3)
                cy = SRC_H / 2 + 4 * np.cos(self._sim_t * 0.2)
                yy, xx = np.mgrid[0:SRC_H, 0:SRC_W]
                hotspot = 35 * np.exp(-(((xx - cx) ** 2) / 40 + ((yy - cy) ** 2) / 25))
                new_raw = base + hotspot
                with self._lock:
                    self._raw = new_raw.astype(np.float32)
                time.sleep(0.5)
            else:
                try:
                    self._mlx.getFrame(frame_buf)
                    raw = np.array(frame_buf, dtype=np.float32).reshape(SRC_H, SRC_W)[::-1, :]
                    with self._lock:
                        self._raw = raw
                except Exception as e:
                    # glitch de lecture ou erreur I2C -- on ecrit sur stderr sans jamais
                    # tuer le thread (une exception non rattrapee ici arreterait
                    # silencieusement les mises a jour, et l'UI figerait sur la
                    # derniere frame sans que rien ne l'indique).
                    print(f"[SensorThread Error] {e}", file=sys.stderr)
                time.sleep(0.25)


# ============================================================================
# WIDGET : selection de points par clic, dans un Canvas Tkinter
# ============================================================================

class PointPickerCanvas(tk.Frame):
    """Affiche une image BGR dans un Canvas et laisse cliquer des points, dans l'ordre."""

    def __init__(self, master, bgr_image, canvas_w=760, canvas_h=560, **kwargs):
        super().__init__(master, bg="black", **kwargs)
        self.points = []  # coordonnees dans l'image ORIGINALE (pas le canvas)
        self.canvas_w, self.canvas_h = canvas_w, canvas_h

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]
        self.scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * self.scale), int(img_h * self.scale)
        self.ox, self.oy = (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2

        resized = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.canvas = tk.Canvas(self, width=canvas_w, height=canvas_h, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self.photo)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

    def _on_left_click(self, event):
        ix = (event.x - self.ox) / self.scale
        iy = (event.y - self.oy) / self.scale
        self.points.append((ix, iy))
        n = len(self.points)
        self.canvas.create_oval(event.x - 6, event.y - 6, event.x + 6, event.y + 6,
                                 outline="#00FF66", width=2, tags=f"pt{n}")
        self.canvas.create_text(event.x + 12, event.y - 12, text=str(n), fill="#00FF66",
                                 font=("Arial", 12, "bold"), tags=f"pt{n}")

    def _on_right_click(self, event):
        if self.points:
            n = len(self.points)
            self.canvas.delete(f"pt{n}")
            self.points.pop()

    def get_points(self):
        return list(self.points)


# ============================================================================
# VUE : Menu principal
# ============================================================================

class BigButton(tk.Frame):
    def __init__(self, master, title, subtitle, bg, command, **kwargs):
        super().__init__(master, bg=bg, cursor="hand2", **kwargs)
        self.bg = bg
        self.command = command
        self.title_lbl = tk.Label(self, text=title, font=("Arial", 22, "bold"), bg=bg, fg=COLOR_BTN_TEXT)
        self.title_lbl.pack(expand=True)
        for widget in (self, self.title_lbl):
            widget.bind("<Button-1>", lambda e: self.command())
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, event):
        self.configure(bg=self._lighten(self.bg))
        self.title_lbl.configure(bg=self._lighten(self.bg))

    def _on_leave(self, event):
        self.configure(bg=self.bg)
        self.title_lbl.configure(bg=self.bg)

    @staticmethod
    def _lighten(hex_color, factor=0.15):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"


class MenuView(tk.Frame):
    """Vue racine : les 3 gros boutons. Aucune fenetre secondaire n'est jamais ouverte --
    cliquer un bouton fait juste basculer app.show_view() vers une autre vue."""

    def __init__(self, master, app):
        super().__init__(master, bg=COLOR_BG)
        self.app = app

        tk.Label(self, text=" Controle Thermique PCB", font=("Arial", 26, "bold"),
                 bg=COLOR_BG, fg=COLOR_HEADER).pack(fill="x", pady=(20, 0), padx=10, anchor="w")

        btn_container = tk.Frame(self, bg=COLOR_BG)
        btn_container.pack(fill="both", expand=True)

        btn_stack = tk.Frame(btn_container, bg=COLOR_BG)
        btn_stack.place(relx=0.5, rely=0.5, anchor="center")

        b1 = BigButton(btn_stack, "Calibration", "Etape 1 : cliquer les points\nRGB / Thermique",
                       COLOR_BTN_1, lambda: app.show_view(CalibrationView), width=280, height=90)
        b2 = BigButton(btn_stack, "Detection", "Etapes 2-3-4 : trouver et\nlocaliser le point chaud (live)",
                       COLOR_BTN_2, self._start_detection, width=280, height=90)
        b3 = BigButton(btn_stack, "Heatmap brute", "Voir la matrice thermique\nsans aucune alteration (live)",
                       COLOR_BTN_3, lambda: app.show_view(HeatmapView), width=280, height=90)
        for b in (b1, b2, b3):
            b.pack_propagate(False)
            b.pack(pady=12)

    def _start_detection(self):
        if self.app.H1 is None:
            messagebox.showwarning("Calibrage requis", "Lancez d'abord la Calibration.", parent=self.app)
            return
        if not list_blueprint_files():
            messagebox.showerror("Aucun blueprint",
                                  f"Aucun fichier .jpg trouve dans {BLUEPRINT_DIR}/.\n"
                                  f"Ajoutez au moins un blueprint puis reessayez.",
                                  parent=self.app)
            return
        self.app.show_view(BlueprintPickerView)


# ============================================================================
# VUE : Choix du blueprint (un bouton par fichier .jpg dans BLUEPRINT_DIR)
# ============================================================================

class BlueprintButton(tk.Frame):
    """Bouton 'cute' affichant une miniature + le nom de fichier, meme esprit que BigButton."""

    def __init__(self, master, photo, filename, command, **kwargs):
        super().__init__(master, bg="white", cursor="hand2",
                          highlightthickness=1, highlightbackground="#DADADA", **kwargs)
        self.command = command
        self.photo = photo  # garder une reference, sinon le GC la recupere
        self.img_lbl = tk.Label(self, image=photo, bg="white")
        self.img_lbl.pack(padx=10, pady=(10, 6))
        self.name_lbl = tk.Label(self, text=filename, font=("Arial", 11, "bold"), bg="white",
                                  fg=COLOR_BTN_TEXT, wraplength=170, justify="center")
        self.name_lbl.pack(padx=10, pady=(0, 10))
        for w in (self, self.img_lbl, self.name_lbl):
            w.bind("<Button-1>", lambda e: self.command())
            w.bind("<Enter>", lambda e: self._set_bg("#F0F4FF"))
            w.bind("<Leave>", lambda e: self._set_bg("white"))

    def _set_bg(self, color):
        self.configure(bg=color)
        self.img_lbl.configure(bg=color)
        self.name_lbl.configure(bg=color)


class BlueprintPickerView(tk.Frame):
    """Scanne BLUEPRINT_DIR/*.jpg et affiche un bouton (miniature + nom) par fichier.
    Cliquer un bouton charge ce blueprint puis enchaine directement sur la Detection."""

    def __init__(self, master, app):
        super().__init__(master, bg=COLOR_BG)
        self.app = app
        self._photos = []  # references gardees pour eviter le garbage collection

        make_header(self, "Choisissez un blueprint (.jpg)",
                    lambda: app.show_view(MenuView), "Annuler", "#E74C3C")

        files = list_blueprint_files()
        if not files:
            tk.Label(self, text=f"Aucun fichier .jpg trouve dans {BLUEPRINT_DIR}/",
                     font=("Arial", 14), bg=COLOR_BG, fg=COLOR_STATUS_ERR).pack(expand=True)
            return

        app.log(f"{len(files)} blueprint(s) trouve(s) dans {BLUEPRINT_DIR}/")

        # Zone scrollable (grille de boutons) au cas ou il y a beaucoup de fichiers
        outer = tk.Frame(self, bg=COLOR_BG)
        outer.pack(fill="both", expand=True, padx=20, pady=20)
        canvas = tk.Canvas(outer, bg=COLOR_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        grid_frame = tk.Frame(canvas, bg=COLOR_BG)

        grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            delta = event.delta if event.delta else (120 if getattr(event, "num", None) == 4 else -120)
            canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)     # Windows / macOS
        canvas.bind("<Button-4>", _on_mousewheel)        # Linux molette haut
        canvas.bind("<Button-5>", _on_mousewheel)        # Linux molette bas

        cols = 4
        for i, path in enumerate(files):
            thumb = self._make_thumbnail(path)
            if thumb is None:
                continue
            self._photos.append(thumb)
            filename = os.path.basename(path)
            btn = BlueprintButton(grid_frame, thumb, filename, command=lambda p=path: self._select(p))
            r, c = divmod(len(self._photos) - 1, cols)
            btn.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")

    def _make_thumbnail(self, path, max_w=160, max_h=120):
        img = cv2.imread(path)
        if img is None:
            self.app.log(f"!! Impossible de lire {path}", level="err")
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(max_w / w, max_h / h)
        resized = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        return ImageTk.PhotoImage(image=Image.fromarray(resized))

    def _select(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de charger {path}", parent=self.app)
            return
        self.app.blueprint_img = img
        self.app.blueprint_path = path
        self.app.log(f"Blueprint selectionne : {os.path.basename(path)}", level="ok")
        self.app.show_view(DetectionView)


# ============================================================================
# VUE : Calibrage (clic thermique puis clic RGB) -- remplace CalibrationDialog
# ============================================================================

class CalibrationView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=COLOR_BG)
        self.app = app
        self.thermal_bgr = None
        self.rgb_bgr = None
        self.thermal_points = None
        self.picker = None

        make_header(self, "Calibrage - preparation...", lambda: app.show_view(MenuView), "Annuler", "#E74C3C")
        tk.Label(self, text="Capture d'une frame RGB en cours...", font=("Arial", 14),
                 bg=COLOR_BG).pack(expand=True)
        # laisse le temps a la fenetre de s'afficher avant l'appel bloquant grab_rgb_frame()
        self.after(50, self._start_capture)

    def stop(self):
        pass  # aucune boucle live a arreter dans cette vue

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _start_capture(self):
        try:
            self.app.log("Capture d'une frame RGB pour le calibrage...")
            self.rgb_bgr = grab_rgb_frame()
        except Exception as e:
            messagebox.showerror("Erreur camera", str(e), parent=self.app)
            self.app.show_view(MenuView)
            return
        self.thermal_bgr = temp_matrix_to_bgr(self.app.sensor.get_scaled())
        self._show_step_thermal()

    def _show_step_thermal(self):
        self._clear()
        make_header(self, "1/2 - Cliquez 4+ points sur la THERMIQUE (clic droit = annuler)",
                    lambda: self.app.show_view(MenuView), "Annuler", "#E74C3C")
        self.picker = PointPickerCanvas(self, self.thermal_bgr)
        self.picker.pack(expand=True, pady=10)
        tk.Button(self, text="Valider ->", font=("Arial", 14, "bold"), bg=COLOR_BTN_2, bd=0,
                  command=self._validate_thermal).pack(pady=10)

    def _validate_thermal(self):
        pts = self.picker.get_points()
        if len(pts) < 4:
            messagebox.showwarning("Points insuffisants", "Il faut au moins 4 points.", parent=self.app)
            return
        self.thermal_points = pts
        self._show_step_rgb()

    def _show_step_rgb(self):
        self._clear()
        make_header(self, f"2/2 - Cliquez {len(self.thermal_points)} points sur le RGB (MEME ORDRE)",
                    lambda: self.app.show_view(MenuView), "Annuler", "#E74C3C")
        self.picker = PointPickerCanvas(self, self.rgb_bgr)
        self.picker.pack(expand=True, pady=10)
        tk.Button(self, text="Calculer l'homographie", font=("Arial", 14, "bold"), bg=COLOR_BTN_1, bd=0,
                  command=self._validate_rgb).pack(pady=10)

    def _validate_rgb(self):
        pts = self.picker.get_points()
        if len(pts) != len(self.thermal_points):
            messagebox.showwarning(
                "Nombre de points different",
                f"{len(self.thermal_points)} points thermiques vs {len(pts)} points RGB. "
                f"Recommencez avec le meme nombre.",
                parent=self.app
            )
            return

        pts_thermal = np.array(self.thermal_points, dtype=np.float32)
        pts_rgb = np.array(pts, dtype=np.float32)
        H, mask = cv2.findHomography(pts_thermal, pts_rgb, cv2.RANSAC, ransacReprojThreshold=3.0)

        if H is None:
            messagebox.showerror("Echec", "Le calcul de l'homographie a echoue. Reessayez avec d'autres points.",
                                  parent=self.app)
            return

        inliers = int(mask.sum())
        pts_thermal_h = cv2.perspectiveTransform(pts_thermal.reshape(-1, 1, 2), H).reshape(-1, 2)
        erreurs = np.linalg.norm(pts_thermal_h - pts_rgb, axis=1)

        self.app.H1 = H
        self.app.log(f"Calibrage termine : {inliers}/{len(pts_thermal)} inliers, "
                      f"erreur moyenne {erreurs.mean():.2f}px", level="ok")
        self.app.show_view(MenuView)


# ============================================================================
# VUE : Detection + Overlay LIVE -- remplace FullscreenWindow.show_live_detection
# ============================================================================

class DetectionView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg="black")
        self.app = app
        self._job = None
        self.rgb_frame = None

        make_header(self, "Detection - preparation...", lambda: app.show_view(MenuView))
        tk.Label(self, text="Capture d'une frame RGB en cours...", font=("Arial", 14),
                 bg="black", fg="white").pack(expand=True)
        self.after(50, self._start_capture)

    def stop(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _start_capture(self):
        try:
            self.app.log("Capture d'une frame RGB pour la detection...")
            self.rgb_frame = grab_rgb_frame()
        except Exception as e:
            messagebox.showerror("Erreur camera", str(e), parent=self.app)
            self.app.show_view(MenuView)
            return

        try:
            self.app.log("Initialisation de la detection et overlay live...")
            self._setup_live(self.rgb_frame, manual_corners=None)
        except RuntimeError as e:
            self.app.log(f"{e} Basculement en selection manuelle des coins.", level="err")
            self._build_manual_corner_picker(self.rgb_frame)
        except Exception as e:
            messagebox.showerror("Erreur", str(e), parent=self.app)
            self.app.show_view(MenuView)

    def _build_manual_corner_picker(self, rgb_frame):
        self._clear()
        make_header(self, "Cliquez les 4 coins du PCB sur l'image RGB",
                    lambda: self.app.show_view(MenuView), "Annuler", "#E74C3C")
        picker = PointPickerCanvas(self, rgb_frame)
        picker.pack(expand=True)

        def validate():
            pts = picker.get_points()
            if len(pts) != 4:
                messagebox.showwarning("4 points requis", "Cliquez exactement 4 coins.", parent=self.app)
                return
            try:
                self._setup_live(rgb_frame, manual_corners=pts)
            except Exception as e:
                messagebox.showerror("Erreur", str(e), parent=self.app)
                self.app.show_view(MenuView)

        tk.Button(self, text="Valider", font=("Arial", 14, "bold"), bg=COLOR_BTN_2, bd=0,
                  command=validate).pack(pady=10)

    def _setup_live(self, rgb_frame, manual_corners, alpha=0.5, refresh_ms=500):
        """Calcule tout ce qui est constant (homographie, masque, contours) UNE fois,
        puis demarre la boucle live qui ne recalcule que la temperature et l'overlay."""
        blueprint_img = self.app.blueprint_img
        H1 = self.app.H1
        bh, bw = blueprint_img.shape[:2]

        if manual_corners is not None:
            rgb_corners = np.array(manual_corners, dtype=np.float32)
        else:
            rgb_corners = detect_pcb_corners(rgb_frame)
            if rgb_corners is None:
                raise RuntimeError("Detection automatique des coins PCB (RGB) echouee -- coins manuels requis.")

        blueprint_corners = np.array([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]], dtype=np.float32)
        H2 = compute_alignment_homography(rgb_corners, blueprint_corners)
        H_total = H2 @ H1

        coverage = np.full((SRC_H * UPSCALE, SRC_W * UPSCALE), 255, dtype=np.uint8)
        coverage_warped = cv2.warpPerspective(coverage, H_total, (bw, bh), flags=cv2.INTER_NEAREST, borderValue=0)
        valid_mask = (coverage_warped > 200).astype(np.uint8) * 255
        valid_mask_3c = cv2.merge([valid_mask] * 3)

        rgb_warped = cv2.warpPerspective(rgb_frame, H2, (bw, bh))
        component_contours = detect_component_contours(rgb_warped)

        self._clear()
        make_header(self, "Resultat - Point chaud detecte (live)", lambda: self.app.show_view(MenuView))
        content = tk.Frame(self, bg="black")
        content.pack(fill="both", expand=True)
        img_label = tk.Label(content, bg="black")
        img_label.pack(fill="both", expand=True)

        self._last_rgb = None

        def render(event=None):
            if self._last_rgb is None or not self.winfo_exists():
                return
            fw, fh = content.winfo_width(), content.winfo_height()
            if fw < 10 or fh < 10:
                return
            h, w = self._last_rgb.shape[:2]
            scale = min(fw / w, fh / h)
            resized = cv2.resize(self._last_rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
            photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
            img_label.configure(image=photo)
            img_label.image = photo

        content.bind("<Configure>", render)

        SENTINEL = -9999.0

        def _update():
            if not self.winfo_exists():
                return
            temp_matrix = self.app.sensor.get_scaled()
            xt, yt, hotspot_temp = find_hotspot(temp_matrix)
            xj, yj = project_point(xt, yt, H_total)

            temp_warped = cv2.warpPerspective(temp_matrix, H_total, (bw, bh),
                                               flags=cv2.INTER_LINEAR, borderValue=SENTINEL)
            try:
                overlay_colorized, vmin_c, vmax_c, pct_blocks = build_component_aware_overlay(
                    temp_warped, valid_mask, component_contours
                )
            except Exception as e:
                print(f"[LiveDetection Error] {e}", file=sys.stderr)
                self._job = self.after(refresh_ms, _update)
                return

            blend_zone = cv2.addWeighted(blueprint_img, 1 - alpha, overlay_colorized, alpha, 0)
            blended = np.where(valid_mask_3c > 0, blend_zone, blueprint_img)

            cv2.drawMarker(blended, (int(xj), int(yj)), (0, 0, 255), cv2.MARKER_CROSS, markerSize=30, thickness=3)
            cv2.circle(blended, (int(xj), int(yj)), 18, (0, 0, 255), 2)
            cv2.putText(blended, f"{hotspot_temp:.1f}C", (int(xj) + 22, int(yj) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self._last_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            render()
            self._job = self.after(refresh_ms, _update)

        _update()
        self.app.log("Detection live active.", level="ok")


# ============================================================================
# VUE : Heatmap brute LIVE -- remplace FullscreenWindow.show_live_heatmap
# ============================================================================

class HeatmapView(tk.Frame):
    def __init__(self, master, app, refresh_ms=500):
        super().__init__(master, bg="black")
        self.app = app
        self._job = None

        make_header(self, "Heatmap brute (live, sans alteration)", lambda: app.show_view(MenuView))
        content = tk.Frame(self, bg="black")
        content.pack(fill="both", expand=True)

        initial = app.sensor.get_scaled()
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(initial, cmap="inferno", interpolation="nearest",
                        vmin=float(initial.min()), vmax=float(initial.max()))
        title = ax.set_title("", fontsize=14)
        fig.colorbar(im, ax=ax, label="Temperature (C)")

        canvas = FigureCanvasTkAgg(fig, master=content)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def _update():
            if not self.winfo_exists():
                return
            matrix = app.sensor.get_scaled()
            im.set_data(matrix)
            im.set_clim(vmin=float(matrix.min()), vmax=float(matrix.max()))
            title.set_text(f"Temperatures brutes (live)  |  min={matrix.min():.1f}C   max={matrix.max():.1f}C")
            canvas.draw_idle()
            self._job = self.after(refresh_ms, _update)

        _update()
        app.log("Heatmap live demarree (rafraichissement continu).", level="ok")

    def stop(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None


# ============================================================================
# APPLICATION PRINCIPALE (fenetre unique)
# ============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controle Thermique PCB")
        self.configure(bg=COLOR_BG)
        self.attributes("-fullscreen", True)
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<Escape>", lambda e: self._on_escape())
        self.protocol("WM_DELETE_WINDOW", self._confirm_quit)

        self.H1 = None  # homographie thermique -> RGB, calculee par le calibrage (en memoire)
        self.blueprint_img = None   # choisi via BlueprintPickerView au moment de la Detection
        self.blueprint_path = None
        self.current_view = None

        self.sensor = SensorThread()
        self.sensor.start()

        self._log_expanded = False
        self._build_shell()

        n_blueprints = len(list_blueprint_files())
        if n_blueprints > 0:
            self.log(f"{n_blueprints} blueprint(s) .jpg trouve(s) dans {BLUEPRINT_DIR}/", level="ok")
        else:
            self.log(f"!! Aucun blueprint .jpg trouve dans {BLUEPRINT_DIR}/", level="err")

        if self.sensor.simulation:
            self.log("Capteur MLX90640 non detecte -- MODE SIMULATION active.", level="err")
        else:
            self.log("Capteur MLX90640 detecte, lecture en cours.", level="ok")

        self.show_view(MenuView)

    # ------------------------------------------------------------------
    # Navigation entre vues (remplace la creation de fenetres Toplevel)
    # ------------------------------------------------------------------
    def show_view(self, view_class):
        if self.current_view is not None:
            stop = getattr(self.current_view, "stop", None)
            if callable(stop):
                stop()
            self.current_view.destroy()
        self.current_view = view_class(self.container, self)
        self.current_view.pack(fill="both", expand=True)

    def _on_escape(self):
        if isinstance(self.current_view, MenuView):
            self._confirm_quit()
        else:
            self.show_view(MenuView)

    def _toggle_fullscreen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _confirm_quit(self):
        if messagebox.askyesno("Quitter", "Fermer l'application ?", parent=self):
            self.sensor.stop()
            self.destroy()

    # ------------------------------------------------------------------
    # Coquille persistante : conteneur de vues + barre de statut/logs
    # ------------------------------------------------------------------
    def _build_shell(self):
        self.status_bar = tk.Frame(self, bg=COLOR_LOG_BG, height=46)
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)
        self.status_label = tk.Label(self.status_bar, text="Pret.", font=("Consolas", 12),
                                      bg=COLOR_LOG_BG, fg=COLOR_LOG_TEXT, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True, padx=15)
        self.toggle_btn = tk.Button(self.status_bar, text="▲ Logs", font=("Arial", 11, "bold"), bg="#4A4E69",
                                     fg="white", bd=0, command=self._toggle_log)
        self.toggle_btn.pack(side="right", padx=10, pady=6)

        self.log_box = tk.Text(self, height=6, bg=COLOR_LOG_BG, fg=COLOR_LOG_TEXT,
                                font=("Consolas", 10), bd=0, state="disabled")

        # Le conteneur central accueille la vue active (MenuView, CalibrationView,
        # DetectionView, HeatmapView, ...) -- une seule a la fois, jamais de
        # fenetre Toplevel separee.
        self.container = tk.Frame(self, bg=COLOR_BG)
        self.container.pack(fill="both", expand=True, side="top")

    def _toggle_log(self):
        self._log_expanded = not self._log_expanded
        if self._log_expanded:
            self.log_box.pack(fill="x", side="bottom", before=self.status_bar)
            self.toggle_btn.configure(text="▼ Logs")
        else:
            self.log_box.pack_forget()
            self.toggle_btn.configure(text="▲ Logs")

    def log(self, message, level="info"):
        color = {"info": COLOR_STATUS_INFO, "ok": COLOR_STATUS_OK, "err": COLOR_STATUS_ERR}.get(level, "white")

        def _write():
            self.status_label.configure(text=message, fg=color if level != "info" else COLOR_LOG_TEXT)
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.after(0, _write)


if __name__ == "__main__":
    app = App()
    app.mainloop()