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


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON tolerant aux types numpy (ndarray, float32, int64, etc.)
    Evite le TypeError 'Object of type ndarray is not JSON serializable'
    meme si un tableau numpy brut se glisse dans les donnees a serialiser."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def load_thermal_matrix(path):
    """
    Charge la matrice de temperatures brutes (en degres Celsius, un float par pixel).
    Formats supportes : .npy (numpy), .csv ou .txt (texte).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        matrix = np.load(path)
    elif ext in (".csv", ".txt"):
        matrix = np.loadtxt(path, delimiter=",") if ext == ".csv" else np.loadtxt(path)
    else:
        raise ValueError(f"Format non supporte : '{ext}'. Utilisez .npy, .csv ou .txt.")
    matrix = matrix.astype(np.float32)
    print(f"[Thermal] Matrice chargee : shape={matrix.shape}, "
          f"min={matrix.min():.1f}C, max={matrix.max():.1f}C")
    return matrix


def matrix_to_display_image(matrix, colormap=cv2.COLORMAP_INFERNO):
    """
    Convertit la matrice de temperatures en image BGR affichable/cliquable,
    A LA RESOLUTION NATIVE DU CAPTEUR (ex: 32x24).
    """
    vmin, vmax = float(matrix.min()), float(matrix.max())
    if vmax - vmin < 1e-6:
        norm = np.zeros_like(matrix, dtype=np.uint8)
    else:
        norm = np.clip((matrix - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)


def upscale_for_clicking(img, target_min_dim=600, interpolation=cv2.INTER_NEAREST):
    """
    Agrandit une petite image pour pouvoir cliquer confortablement, SANS toucher
    au comportement par defaut de PointPicker (qui ne fait que retrecir les
    images trop grandes). Retourne (image_agrandie, facteur_utilise) : il faudra
    diviser les points cliques par ce facteur pour revenir a la resolution native.
    """
    h, w = img.shape[:2]
    if min(h, w) >= target_min_dim:
        return img.copy(), 1.0
    factor = target_min_dim / min(h, w)
    img_up = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=interpolation)
    return img_up, factor


class PointPicker:
    """Fenêtre interactive pour cliquer des points sur une image et les enregistrer dans l'ordre."""

    def __init__(self, image, window_name, max_zoom_display=1000, interpolation=cv2.INTER_LINEAR):
        self.original = image.copy()
        self.window_name = window_name
        self.points = []

        # Redimensionnement pour affichage si l'image est trop grande
        h, w = image.shape[:2]
        self.scale = 1.0
        if max(h, w) > max_zoom_display:
            self.scale = max_zoom_display / max(h, w)
            self.display_img = cv2.resize(
                image, (int(w * self.scale), int(h * self.scale)), interpolation=interpolation
            )
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

    IMPORTANT : thermal_path pointe maintenant vers la matrice radiometrique
    (.npy / .csv / .txt), PAS une image JPG. On clique sur une VISUALISATION
    colorisee de cette matrice, a sa resolution NATIVE (ex: 32x24), pour que
    H1 soit calculee exactement dans le repere des donnees brutes utilisees
    plus tard (evite tout decalage d'echelle en Etape 4).
    """
    temp_matrix = load_thermal_matrix(thermal_path)
    thermal_img = matrix_to_display_image(temp_matrix)  # BGR, meme resolution que la matrice
    rgb_img = rgb_path if isinstance(rgb_path, np.ndarray) else cv2.imread(rgb_path)

    if rgb_img is None:
        raise FileNotFoundError(f"Image RGB introuvable : {rgb_path}")

    print("=" * 60)
    print("ETAPE 1 : CALIBRAGE PAR HOMOGRAPHIE")
    print("=" * 60)
    print("Cliquez sur les MEMES points physiques (ex: resistances,")
    print("coins de composants) dans le MEME ORDRE sur les deux images.")
    print(f"Resolution native de la matrice thermique : {temp_matrix.shape[1]}x{temp_matrix.shape[0]} "
          f"(l'affichage est agrandi pour faciliter le clic, sans changer les coordonnees enregistrees).")

    # 1. Points sur la visualisation de la matrice thermique (INTER_NEAREST = pixels bien distincts)
    picker_thermal = PointPicker(
        thermal_img, "1/2 - Points sur matrice THERMIQUE (q pour valider)", interpolation=cv2.INTER_NEAREST
    )
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
        }, f, indent=2, cls=NumpyEncoder)

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
    # convert_and_save_image()  # Assurez-vous que CameraArray.txt est présent et correct
    calibrer_homographie(
        thermal_path="./data/CameraArray.txt",  # Chemin vers la matrice radiometrique
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