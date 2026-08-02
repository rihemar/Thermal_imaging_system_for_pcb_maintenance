#!/usr/bin/env python3
"""
Application Tkinter - Menu de controle Thermique / RGB / Blueprint (version tablette)
========================================================================================

IMPORTANT - ne renommez JAMAIS ce fichier "tkinter.py" : ca entre en conflit
avec le module standard "tkinter" et casse tous les imports (ImportError
circulaire). Gardez le nom "app_menu_tkinter.py" ou equivalent.

Boutons :
  1. Calibration (Etape 1)              -> lance etape1_calibrage_homographie.py
  2. Detection + Overlay (Etape 2/3/4)  -> lance etape2_3_4_alignement_overlay.py,
                                            puis affiche le resultat en PLEIN ECRAN
  3. Heatmap brute (sans alteration)    -> affiche ./data/CameraArrayScaled.txt tel quel,
                                            en PLEIN ECRAN

Au demarrage : execute STARTUP_BASH_COMMAND en arriere-plan (non bloquant).

Dependances :
    pip install opencv-python numpy matplotlib pillow
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


# ============================================================================
# CONFIGURATION - gardez vos valeurs ici, c'est le seul bloc a modifier
# ============================================================================

STARTUP_BASH_COMMAND = "./CameraThermique/examples/build/GUI "   # <-- REMPLACEZ PAR VOTRE VRAIE COMMANDE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_SCRIPT = os.path.join(SCRIPT_DIR,"CameraRGB", "calibrage_homographie.py")
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "CameraRGB", "alignement_overlay.py")

DEFAULT_THERMAL_MATRIX = "./data/CameraArrayScaled.txt"
DEFAULT_RGB = "./data/RGB_frame.jpg"  # non utilise par les scripts actuels (webcam directe), garde pour reference
DEFAULT_BLUEPRINT = "./data/blueprint.png"
DEFAULT_H1 = "./data/homography_default.npy"  # doit matcher le output_path hardcode dans calibrer_par_defaut()
DEFAULT_OUT = "./resultat_overlay.jpg"

# Palette "cute" -- modifiable librement
COLOR_BG = "#FDF6EC"
COLOR_HEADER = "#4A4E69"
COLOR_BTN_1 = "#FFADAD"   # calibration - rose
COLOR_BTN_2 = "#A0C4FF"   # detection - bleu
COLOR_BTN_3 = "#CAFFBF"   # heatmap - vert
COLOR_BTN_TEXT = "#22223B"
COLOR_STATUS_OK = "#2ECC71"
COLOR_STATUS_ERR = "#E74C3C"
COLOR_STATUS_INFO = "#4A4E69"
COLOR_LOG_BG = "#22223B"
COLOR_LOG_TEXT = "#CAFFBF"


# ============================================================================
# Widget bouton "cute" - grand, colore, coins arrondis simules par le padding
# ============================================================================

class BigButton(tk.Frame):
    def __init__(self, master, title, subtitle, bg, command, **kwargs):
        super().__init__(master, bg=bg, cursor="hand2", **kwargs)
        self.bg = bg
        self.command = command


        self.title_lbl = tk.Label(self, text=title, font=("Arial", 22, "bold"),
                                   bg=bg, fg=COLOR_BTN_TEXT)
        self.title_lbl.pack()

        self.subtitle_lbl = tk.Label(self, text=subtitle, font=("Arial", 12),
                                      bg=bg, fg=COLOR_BTN_TEXT, wraplength=260, justify="center")
        self.subtitle_lbl.pack(pady=(2, 20))

        for widget in (self, self.title_lbl, self.subtitle_lbl):
            widget.bind("<Button-1>", lambda e: self.command())
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, event):
        self.configure(bg=self._lighten(self.bg))
        for w in ( self.title_lbl, self.subtitle_lbl):
            w.configure(bg=self._lighten(self.bg))

    def _on_leave(self, event):
        self.configure(bg=self.bg)
        for w in ( self.title_lbl, self.subtitle_lbl):
            w.configure(bg=self.bg)

    @staticmethod
    def _lighten(hex_color, factor=0.15):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controle Thermique PCB")
        self.configure(bg=COLOR_BG)
        self.attributes("-fullscreen", True)
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<Escape>", lambda e: self._confirm_quit())

        self._log_expanded = False
        self._build_ui()
        self._run_startup_command()

    # ------------------------------------------------------------------
    def _toggle_fullscreen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _confirm_quit(self):
        if messagebox.askyesno("Quitter", "Fermer l'application ?"):
            self.destroy()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Barre du haut ----
        header = tk.Frame(self, bg=COLOR_HEADER, height=90)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text=" Controle Thermique PCB", font=("Arial", 26, "bold"),
                 bg=COLOR_HEADER, fg="white").pack(side="left", padx=30)

        tk.Button(header, text="✕", font=("Arial", 16, "bold"), bg="#E74C3C", fg="white",
                  bd=0, activebackground="#c0392b", command=self._confirm_quit,
                  width=3).pack(side="right", padx=20, pady=20)

        # ---- Grille des gros boutons ----
        btn_container = tk.Frame(self, bg=COLOR_BG)
        btn_container.pack(fill="both", expand=True, padx=40, pady=30)
        btn_container.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")
        btn_container.grid_rowconfigure(0, weight=1)

        b1 = BigButton(btn_container, "Calibration", "Etape 1 : cliquer les points\nRGB / Thermique",
                        COLOR_BTN_1, self.run_calibration)
        b2 = BigButton(btn_container, "Detection", "Etapes 2-3-4 : trouver et\nlocaliser le point chaud",
                        COLOR_BTN_2, self.run_main_detection)
        b3 = BigButton(btn_container, "Heatmap brute", "Voir la matrice thermique\nsans aucune alteration",
                        COLOR_BTN_3, self.show_raw_heatmap)

        for i, b in enumerate((b1, b2, b3)):
            b.grid(row=0, column=i, sticky="nsew", padx=15, pady=15)

        # ---- Bandeau de statut + log reduit/extensible ----
        self.status_bar = tk.Frame(self, bg=COLOR_LOG_BG, height=46)
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)

        self.status_label = tk.Label(self.status_bar, text="Pret.", font=("Consolas", 12),
                                      bg=COLOR_LOG_BG, fg=COLOR_LOG_TEXT, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True, padx=15)

        self.toggle_btn = tk.Button(self.status_bar, text="▲ Logs", font=("Arial", 11, "bold"),
                                     bg="#4A4E69", fg="white", bd=0, activebackground="#5c6088",
                                     command=self._toggle_log)
        self.toggle_btn.pack(side="right", padx=10, pady=6)

        # Petite boite de log, cachee par defaut (agrandie au clic sur "Logs")
        self.log_box = tk.Text(self, height=6, bg=COLOR_LOG_BG, fg=COLOR_LOG_TEXT,
                                font=("Consolas", 10), bd=0, state="disabled")
        # non pack() ici : affiche seulement quand on deplie

    def _toggle_log(self):
        self._log_expanded = not self._log_expanded
        if self._log_expanded:
            self.log_box.pack(fill="x", side="bottom", before=self.status_bar)
            self.toggle_btn.configure(text="▼ Logs")
        else:
            self.log_box.pack_forget()
            self.toggle_btn.configure(text="▲ Logs")

    # ------------------------------------------------------------------
    def log(self, message, level="info"):
        """Met a jour le statut court + ajoute au log complet. Thread-safe."""
        color = {"info": COLOR_STATUS_INFO, "ok": COLOR_STATUS_OK, "err": COLOR_STATUS_ERR}.get(level, "white")

        def _write():
            self.status_label.configure(text=message, fg=color if level != "info" else COLOR_LOG_TEXT)
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.after(0, _write)

    # ------------------------------------------------------------------
    # Commande bash au demarrage
    # ------------------------------------------------------------------
    def _run_startup_command(self):
        def task():
            self.log(f"[Startup] Execution : {STARTUP_BASH_COMMAND}")
            try:
                result = subprocess.run(
                    STARTUP_BASH_COMMAND, shell=True, capture_output=True, text=True, timeout=30
                )
                if result.stdout.strip():
                    self.log(result.stdout.strip())
                if result.stderr.strip():
                    self.log(f"[stderr] {result.stderr.strip()}", level="err")
                self.log(f"[Startup] Termine (code {result.returncode})",
                          level="ok" if result.returncode == 0 else "err")
            except Exception as e:
                self.log(f"[Startup] Erreur : {e}", level="err")

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------
    # Bouton 1 : Calibration
    # ------------------------------------------------------------------
    def run_calibration(self):
        if not os.path.exists(CALIBRATION_SCRIPT):
            messagebox.showerror("Erreur", f"Script introuvable : {CALIBRATION_SCRIPT}")
            return

        def task():
            self.log("Calibration en cours... (chemins/URL webcam geres en interne par le script)")
            # Pas d'arguments : l'argparse de ce script est desactive, il appelle
            # toujours calibrer_par_defaut() avec ses chemins/URL hardcodes.
            cmd = [sys.executable, CALIBRATION_SCRIPT]
            code = self._run_subprocess(cmd)
            if code == 0:
                self.log("Calibration terminee avec succes.", level="ok")
            else:
                self.log("Calibration annulee ou en erreur.", level="err")

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------
    # Bouton 2 : Detection + Overlay
    # ------------------------------------------------------------------
    def run_main_detection(self):
        if not os.path.exists(MAIN_SCRIPT):
            messagebox.showerror("Erreur", f"Script introuvable : {MAIN_SCRIPT}")
            return

        def task():
            self.log("Detection en cours...")
            cmd = [sys.executable, MAIN_SCRIPT,
                   "--thermal", DEFAULT_THERMAL_MATRIX, "--blueprint", DEFAULT_BLUEPRINT,
                   "--h1", DEFAULT_H1, "--out", DEFAULT_OUT]
            code = self._run_subprocess(cmd)

            if code == 0 and os.path.exists(DEFAULT_OUT):
                self.log("Detection terminee. Affichage du resultat...", level="ok")
                self.after(0, lambda: self._show_fullscreen_image(DEFAULT_OUT, "Resultat - Point chaud detecte"))
            else:
                self.log("Detection annulee ou en erreur.", level="err")

        threading.Thread(target=task, daemon=True).start()

    def _run_subprocess(self, cmd):
        """Execute une commande en flux, log chaque ligne. Retourne le code de sortie."""
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in process.stdout:
                self.log(line.rstrip())
            process.wait()
            return process.returncode
        except Exception as e:
            self.log(f"Erreur : {e}", level="err")
            return -1

    # ------------------------------------------------------------------
    # Bouton 3 : Heatmap brute (plein ecran)
    # ------------------------------------------------------------------
    def show_raw_heatmap(self):
        if not os.path.exists(DEFAULT_THERMAL_MATRIX):
            messagebox.showerror("Erreur", f"Fichier introuvable : {DEFAULT_THERMAL_MATRIX}")
            return
        try:
            matrix = np.loadtxt(DEFAULT_THERMAL_MATRIX)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger {DEFAULT_THERMAL_MATRIX} :\n{e}")
            return

        win = self._make_fullscreen_toplevel("Heatmap brute (sans alteration)")

        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix, cmap="inferno", interpolation="nearest")  # AUCUNE alteration
        ax.set_title(f"Temperatures brutes  |  min={matrix.min():.1f}C   max={matrix.max():.1f}C",
                     fontsize=14)
        fig.colorbar(im, ax=ax, label="Temperature (C)")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.log(f"Heatmap affichee (shape={matrix.shape}, {matrix.min():.1f}C a {matrix.max():.1f}C)", level="ok")

    # ------------------------------------------------------------------
    # Fenetre plein ecran generique (image resultat, heatmap, etc.)
    # ------------------------------------------------------------------
    def _make_fullscreen_toplevel(self, title):
        """Cree une Toplevel plein ecran avec une barre 'Retour au menu' + ECHAP pour fermer."""
        win = tk.Toplevel(self)
        win.attributes("-fullscreen", True)
        win.configure(bg="black")
        win.bind("<Escape>", lambda e: win.destroy())

        top_bar = tk.Frame(win, bg=COLOR_HEADER, height=60)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        tk.Label(top_bar, text=title, font=("Arial", 18, "bold"),
                 bg=COLOR_HEADER, fg="white").pack(side="left", padx=20)
        tk.Button(top_bar, text="⬅ Retour au menu", font=("Arial", 14, "bold"),
                  bg="#A0C4FF", fg=COLOR_BTN_TEXT, bd=0, activebackground="#89b4e8",
                  command=win.destroy, padx=15, pady=8).pack(side="right", padx=20, pady=8)

        return win

    def _show_fullscreen_image(self, image_path, title):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            messagebox.showerror("Erreur", f"Impossible de charger {image_path}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        win = self._make_fullscreen_toplevel(title)

        # Conteneur qui redimensionne l'image a la taille de l'ecran, en gardant le ratio
        img_frame = tk.Frame(win, bg="black")
        img_frame.pack(fill="both", expand=True)
        img_label = tk.Label(img_frame, bg="black")
        img_label.pack(fill="both", expand=True)

        def render(event=None):
            fw = img_frame.winfo_width()
            fh = img_frame.winfo_height()
            if fw < 10 or fh < 10:
                return
            h, w = img_rgb.shape[:2]
            scale = min(fw / w, fh / h)
            resized = cv2.resize(img_rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
            photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
            img_label.configure(image=photo)
            img_label.image = photo  # garder une reference

        img_frame.bind("<Configure>", render)
        win.after(100, render)


if __name__ == "__main__":
    app = App()
    app.mainloop()