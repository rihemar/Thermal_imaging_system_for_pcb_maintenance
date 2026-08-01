"""
ETAPE 1 - Calibrage par Homographie (RGB <-> Thermique)
=========================================================

Ce script permet de :
1. Charger une image RGB et une image thermique du même PCB
2. Cliquer manuellement sur des points correspondants (minimum 4, idéalement 8-10)
   - d'abord sur l'image thermique
   - puis sur l'image RGB, dans le MEME ORDRE
3. Calculer la matrice d'homographie H avec RANSAC (robuste aux erreurs de clic)
4. Sauvegarder H dans un fichier .npy pour être réutilisée par les étapes 2/3/4

Dépendances :
    pip install opencv-python numpy

Utilisation :
    python etape1_calibrage_homographie.py --thermal thermal.jpg --rgb rgb.jpg --out homography.npy
"""

import cv2
import numpy as np
import argparse
import json
import os
from Array_proccessing import convert_and_save_image

class PointPicker:
    """Fenêtre interactive pour cliquer des points sur une image et les enregistrer dans l'ordre."""

    def __init__(self, image, window_name, max_zoom_display=1000):
        self.original = image.copy()
        self.window_name = window_name
        self.points = []

        # Redimensionnement pour affichage si l'image est trop grande
        h, w = image.shape[:2]
        self.scale = 1.0
        if max(h, w) > max_zoom_display:
            self.scale = max_zoom_display / max(h, w)
            self.display_img = cv2.resize(image, (int(w * self.scale), int(h * self.scale)))
        else:
            self.display_img = image.copy()

        self.canvas = self.display_img.copy()

    def _redraw(self):
        self.canvas = self.display_img.copy()
        for i, (x, y) in enumerate(self.points):
            disp_x, disp_y = int(x * self.scale), int(y * self.scale)
            cv2.circle(self.canvas, (disp_x, disp_y), 6, (0, 0, 255), -1)
            cv2.putText(self.canvas, str(i + 1), (disp_x + 8, disp_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(self.window_name, self.canvas)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Reconvertir en coordonnées de l'image originale (pas de l'affichage redimensionné)
            orig_x = x / self.scale
            orig_y = y / self.scale
            self.points.append((orig_x, orig_y))
            print(f"  Point {len(self.points)} enregistre : ({orig_x:.1f}, {orig_y:.1f})")
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic droit = annuler le dernier point
            if self.points:
                removed = self.points.pop()
                print(f"  Point annule : {removed}")
                self._redraw()

    def pick(self, min_points=4):
        print(f"\n>>> Fenetre '{self.window_name}'")
        print("    - Clic GAUCHE : ajouter un point")
        print("    - Clic DROIT  : annuler le dernier point")
        print(f"    - Touche 'q'  : valider (minimum {min_points} points requis)")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                if len(self.points) >= min_points:
                    break
                else:
                    print(f"    !! Il faut au moins {min_points} points ({len(self.points)} actuellement)")
            elif key == 27:  # ESC = quitter sans valider
                cv2.destroyWindow(self.window_name)
                raise KeyboardInterrupt("Selection annulee par l'utilisateur")

        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.float32)


def calibrer_homographie(thermal_path, rgb_path, output_path, min_points=4):
    """
    Pipeline complet de l'Etape 1 :
    clic points -> calcul homographie RANSAC -> sauvegarde
    """
    thermal_img = cv2.imread(thermal_path)
    rgb_img = rgb_path

    if thermal_img is None:
        raise FileNotFoundError(f"Image thermique introuvable : {thermal_path}")
    if rgb_img is None:
        raise FileNotFoundError(f"Image RGB introuvable : {rgb_path}")

    print("=" * 60)
    print("ETAPE 1 : CALIBRAGE PAR HOMOGRAPHIE")
    print("=" * 60)
    print("Cliquez sur les MEMES points physiques (ex: resistances,")
    print("coins de composants) dans le MEME ORDRE sur les deux images.")

    # 1. Points sur l'image thermique
    picker_thermal = PointPicker(thermal_img, "1/2 - Points sur image THERMIQUE (q pour valider)")
    pts_thermal = picker_thermal.pick(min_points=min_points)

    # 2. Points sur l'image RGB, dans le meme ordre
    picker_rgb = PointPicker(rgb_img, "2/2 - Points sur image RGB (MEME ORDRE, q pour valider)")
    pts_rgb = picker_rgb.pick(min_points=min_points)

    if len(pts_thermal) != len(pts_rgb):
        raise ValueError(
            f"Nombre de points different : {len(pts_thermal)} (thermique) "
            f"vs {len(pts_rgb)} (RGB). Recommencez avec le meme nombre de points."
        )

    # 3. Calcul de l'homographie : thermique -> RGB, avec RANSAC pour robustesse
    H, mask = cv2.findHomography(pts_thermal, pts_rgb, cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None:
        raise RuntimeError("Le calcul de l'homographie a echoue. Verifiez vos points (non alignes, etc.)")

    inliers = int(mask.sum())
    print(f"\nHomographie calculee avec {inliers}/{len(pts_thermal)} points consideres comme fiables (inliers).")

    # 4. Verification : erreur de reprojection moyenne
    pts_thermal_h = cv2.perspectiveTransform(pts_thermal.reshape(-1, 1, 2), H).reshape(-1, 2)
    erreurs = np.linalg.norm(pts_thermal_h - pts_rgb, axis=1)
    print(f"Erreur de reprojection moyenne : {erreurs.mean():.2f} px  (max : {erreurs.max():.2f} px)")
    if erreurs.mean() > 15:
        print("!! ATTENTION : erreur elevee. Recalibrez avec des points plus precis/nombreux.")

    # 5. Sauvegarde de la matrice + metadonnees
    np.save(output_path, H)
    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "thermal_image": thermal_path,
            "rgb_image": rgb_path,
            "points_thermal": pts_thermal.tolist(),
            "points_rgb": pts_rgb.tolist(),
            "inliers": inliers,
            "total_points": len(pts_thermal),
            "reprojection_error_mean_px": float(erreurs.mean()),
            "reprojection_error_max_px": float(erreurs.max()),
        }, f, indent=2)

    print(f"\nMatrice d'homographie sauvegardee : {output_path}")
    print(f"Metadonnees sauvegardees          : {meta_path}")
    print("\nMatrice H :")
    print(H)

    return H


def calibrer_par_defaut():
    url = "http://192.168.1.19:81/stream"

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Could not connect to IP Webcam")
        exit()
    ret , frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return 
    convert_and_save_image()  # Assurez-vous que CameraArray.txt est présent et correct
    calibrer_homographie(
        thermal_path="./data/thermal_default.jpg",
        rgb_path=frame,
        output_path="./data/homography_default.npy",
        min_points=4
    )

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Etape 1 - Calibrage homographie thermique <-> RGB")
    # parser.add_argument("--thermal", required=True, help="Chemin vers l'image thermique")
    # parser.add_argument("--rgb", required=True, help="Chemin vers l'image RGB")
    # parser.add_argument("--out", default="homography.npy", help="Fichier de sortie pour la matrice H")
    # parser.add_argument("--min-points", type=int, default=4, help="Nombre minimum de points (defaut: 4)")
    # args = parser.parse_args()

    # calibrer_homographie(args.thermal, args.rgb, args.out, args.min_points)
    calibrer_par_defaut()