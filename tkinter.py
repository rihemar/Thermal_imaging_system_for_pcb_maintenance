#!/usr/bin/env python3
"""
Application Tkinter - Menu de controle Thermique / RGB / Blueprint
=====================================================================

Boutons :
  1. Calibration (Etape 1)              -> lance etape1_calibrage_homographie.py
  2. Detection + Overlay (Etape 2/3/4)  -> lance etape2_3_4_alignement_overlay.py,
                                            puis affiche le resultat final dans une fenetre
  3. Heatmap brute (sans alteration)    -> affiche ./data/CameraArrayScaled.txt tel quel
                                            (colormap direct, AUCUN flou/lissage/composant)

Au demarrage de l'application : execute une commande bash configurable
(voir STARTUP_BASH_COMMAND ci-dessous) en arriere-plan, sans bloquer l'UI.

Dependances :
    pip install opencv-python numpy matplotlib pillow
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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
# CONFIGURATION - a adapter a votre environnement
# ============================================================================

# Commande executee automatiquement a l'ouverture de l'app (ex: lancer un service,
# verifier une connexion camera, monter un peripherique, etc.)
STARTUP_BASH_COMMAND = "./CameraThermique/examples/build/GUI "   # <-- REMPLACEZ PAR VOTRE VRAIE COMMANDE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_SCRIPT = os.path.join(SCRIPT_DIR, "calibrage_homographie.py")
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "alignement_overlay.py")

DEFAULT_THERMAL_MATRIX = "./data/CameraArrayScaled.txt"
# DEFAULT_RGB = "./data/RGB_frame.jpg"
DEFAULT_BLUEPRINT = "./data/blueprint.png"
DEFAULT_H1 = "./data/homography.npy"
DEFAULT_OUT = "./data/resultat_overlay.jpg"


# ============================================================================
# APPLICATION
# ============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controle Thermique PCB")
        self.geometry("900x650")

        self._build_ui()
        self._run_startup_command()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def _build_ui(self):
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Controle Thermique / RGB / Blueprint",
                  font=("Arial", 16, "bold")).pack()

        btn_frame = ttk.Frame(self, padding=10)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="1. Lancer la Calibration",
                   command=self.run_calibration).pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="2. Lancer Detection + Overlay ",
                   command=self.run_main_detection).pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="3. Afficher Heatmap brute (sans alteration)",
                   command=self.show_raw_heatmap).pack(fill="x", pady=5)

        ttk.Label(self, text="Journal :").pack(anchor="w", padx=10)
        self.log_widget = scrolledtext.ScrolledText(
            self, height=18, state="disabled", bg="black", fg="#00ff66", font=("Consolas", 10)
        )
        self.log_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def log(self, message):
        """Ecrit une ligne dans le journal (thread-safe via after)."""
        def _write():
            self.log_widget.configure(state="normal")
            self.log_widget.insert("end", message + "\n")
            self.log_widget.see("end")
            self.log_widget.configure(state="disabled")
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
                    self.log(f"[stderr] {result.stderr.strip()}")
                self.log(f"[Startup] Termine (code retour {result.returncode})")
            except Exception as e:
                self.log(f"[Startup] Erreur : {e}")

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------
    # Bouton 1 : Calibration
    # ------------------------------------------------------------------
    def run_calibration(self):
        if not os.path.exists(CALIBRATION_SCRIPT):
            messagebox.showerror("Erreur", f"Script introuvable : {CALIBRATION_SCRIPT}")
            return

        def task():
            self.log("\n[Calibration] Lancement (des fenetres OpenCV vont s'ouvrir)...")
            cmd = [
                sys.executable, CALIBRATION_SCRIPT,
                # "--thermal", DEFAULT_THERMAL_MATRIX,
                # "--rgb", DEFAULT_RGB,
                # "--out", DEFAULT_H1,
            ]
            self._run_subprocess(cmd)

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------
    # Bouton 2 : Detection + Overlay
    # ------------------------------------------------------------------
    def run_main_detection(self):
        if not os.path.exists(MAIN_SCRIPT):
            messagebox.showerror("Erreur", f"Script introuvable : {MAIN_SCRIPT}")
            return

        def task():
            self.log("\n[Detection] Lancement...")
            cmd = [
                sys.executable, MAIN_SCRIPT,
                "--thermal", DEFAULT_THERMAL_MATRIX,
                "--blueprint", DEFAULT_BLUEPRINT,
                "--h1", DEFAULT_H1,
                "--out", DEFAULT_OUT,
            ]
            self._run_subprocess(cmd)

            if os.path.exists(DEFAULT_OUT):
                self.after(0, lambda: self._show_image_window(DEFAULT_OUT, "Resultat Overlay"))
            else:
                self.log(f"[Detection] !! {DEFAULT_OUT} n'a pas ete genere.")

        threading.Thread(target=task, daemon=True).start()

    def _run_subprocess(self, cmd):
        """Execute une commande en flux, en logguant chaque ligne au fur et a mesure."""
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in process.stdout:
                self.log(line.rstrip())
            process.wait()
            self.log(f"[OK] Termine (code retour {process.returncode})")
        except Exception as e:
            self.log(f"[Erreur] {e}")

    # ------------------------------------------------------------------
    # Bouton 3 : Heatmap brute (sans alteration)
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

        win = tk.Toplevel(self)
        win.title(f"Heatmap brute - {os.path.basename(DEFAULT_THERMAL_MATRIX)}")

        fig = Figure(figsize=(7, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        # AUCUNE alteration : pas de flou gaussien, pas de bloc par composant,
        # pas de lissage bilineaire -- une valeur brute = un pixel affiche, point.
        im = ax.imshow(matrix, cmap="inferno", interpolation="nearest")
        ax.set_title(f"Temperatures brutes (min={matrix.min():.1f}C, max={matrix.max():.1f}C)")
        fig.colorbar(im, ax=ax, label="Temperature (C)")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.log(f"[Heatmap] {DEFAULT_THERMAL_MATRIX} affichee "
                  f"(shape={matrix.shape}, min={matrix.min():.1f}C, max={matrix.max():.1f}C)")

    # ------------------------------------------------------------------
    # Affichage d'une image resultat dans une fenetre Tkinter
    # ------------------------------------------------------------------
    def _show_image_window(self, image_path, title):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            messagebox.showerror("Erreur", f"Impossible de charger {image_path}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        max_dim = 900
        h, w = img_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

        win = tk.Toplevel(self)
        win.title(title)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        label = tk.Label(win, image=photo)
        label.image = photo  # garder une reference (sinon garbage-collected)
        label.pack()


if __name__ == "__main__":
    app = App()
    app.mainloop()