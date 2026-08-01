"""
ETAPES 2, 3, 4 - Detection PCB, Alignement Blueprint, Localisation + Overlay thermique
========================================================================================

Ce script :
  ETAPE 2 : detecte automatiquement les 4 coins du PCB sur l'image RGB (Canny + contours)
  ETAPE 3 : calcule l'homographie RGB -> Blueprint JPG (registration / recalage)
  ETAPE 4 : - localise le pixel le plus chaud sur l'image thermique
            - le projette (Etape1 -> Etape3) jusque dans le repere du blueprint
            - dessine un reticule sur le blueprint
  BONUS   : genere une image "overlay" = thermique superpose au blueprint, MAIS avec
            une interpolation "component-aware" : au lieu d'un degrade continu classique,
            chaque composant (contour ferme detecte sur le PCB) recoit UNE SEULE couleur
            uniforme = la temperature moyenne mesuree a l'interieur de son propre contour.
            Cela evite qu'un pixel de bordure d'un composant "vole" une couleur intermediaire
            issue du composant voisin (probleme classique de l'interpolation bilineaire brute).

Dependances :
    pip install opencv-python numpy

Pre-requis :
    - avoir deja genere homography.npy via etape1_calibrage_homographie.py
      (matrice H1 : thermique -> RGB)

Utilisation typique :
    python etape2_3_4_alignement_overlay.py \
        --thermal thermal.jpg --rgb rgb.jpg --blueprint blueprint.jpg \
        --h1 homography.npy --out resultat_overlay.jpg
"""

import time

import cv2
import numpy as np
import argparse
import json
import os


# ----------------------------------------------------------------------------
# Utilitaires generaux
# ----------------------------------------------------------------------------

def order_points(pts):
    """Ordonne 4 points dans l'ordre : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]        # haut-gauche : somme x+y minimale
    rect[2] = pts[np.argmax(s)]        # bas-droit   : somme x+y maximale
    rect[1] = pts[np.argmin(diff)]     # haut-droit  : x-y minimale
    rect[3] = pts[np.argmax(diff)]     # bas-gauche  : x-y maximale
    return rect


class PointPicker:
    """Selection manuelle de points au clic (utilise comme fallback si la detection auto echoue)."""

    def __init__(self, image, window_name, max_display=1000):
        h, w = image.shape[:2]
        self.scale = min(1.0, max_display / max(h, w))
        self.display_img = cv2.resize(image, (int(w * self.scale), int(h * self.scale))) \
            if self.scale < 1.0 else image.copy()
        self.points = []
        self.window_name = window_name

    def _redraw(self):
        canvas = self.display_img.copy()
        for i, (x, y) in enumerate(self.points):
            dx, dy = int(x * self.scale), int(y * self.scale)
            cv2.circle(canvas, (dx, dy), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(self.window_name, canvas)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x / self.scale, y / self.scale))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self._redraw()

    def pick(self, n_points=4):
        print(f"\n>>> {self.window_name} : cliquez {n_points} points (clic droit = annuler, 'q' = valider)")
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') and len(self.points) >= n_points:
                break
        cv2.destroyWindow(self.window_name)
        return np.array(self.points[:n_points], dtype=np.float32)


# ----------------------------------------------------------------------------
# ETAPE 2 : Detection des 4 coins du PCB sur l'image RGB
# ----------------------------------------------------------------------------

def detect_pcb_corners(rgb_img, canny_low=50, canny_high=150, min_area_ratio=0.05, debug_out=None):
    """
    Detecte le contour du PCB (rectangle) sur fond de plan de travail
    et retourne ses 4 coins ordonnes en pixels RGB.
    Fallback automatique : si aucun contour a 4 cotes n'est trouve,
    utilise le rectangle englobant oriente (minAreaRect) du plus gros contour.
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Aucun contour detecte. Verifiez le contraste PCB / fond de travail.")

    img_area = rgb_img.shape[0] * rgb_img.shape[1]
    contours = [c for c in contours if cv2.contourArea(c) > min_area_ratio * img_area]
    if not contours:
        raise RuntimeError("Aucun contour assez grand. Reduisez --min-area-ratio ou ameliorez l'eclairage.")

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
        method = "approxPolyDP (4 cotes detectes)"
    else:
        rect = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect).astype(np.float32)
        method = f"minAreaRect (fallback, approx avait {len(approx)} points)"

    corners = order_points(corners)
    print(f"[Etape 2] Coins PCB detectes via : {method}")
    print(f"[Etape 2] Coins (RGB) : {corners.tolist()}")

    if debug_out:
        vis = rgb_img.copy()
        cv2.drawContours(vis, [largest], -1, (0, 255, 0), 2)
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(debug_out, vis)
        print(f"[Etape 2] Visualisation sauvegardee : {debug_out}")

    return corners


# ----------------------------------------------------------------------------
# ETAPE 3 : Alignement RGB -> Blueprint (registration)
# ----------------------------------------------------------------------------

def compute_alignment_homography(rgb_corners, blueprint_corners):
    """
    Calcule H2 telle que : p_blueprint = H2 . p_rgb
    Utilise getPerspectiveTransform (exact, 4 points) car les 4 coins
    physiques du PCB doivent correspondre exactement aux 4 coins du dessin.
    """
    rgb_corners = order_points(rgb_corners)
    blueprint_corners = order_points(blueprint_corners)
    H2 = cv2.getPerspectiveTransform(rgb_corners, blueprint_corners)
    return H2


# ----------------------------------------------------------------------------
# ETAPE 4a : Detection du point le plus chaud
# ----------------------------------------------------------------------------

def find_hotspot(thermal_img):
    """Retourne (x, y, valeur) du pixel le plus chaud sur l'image thermique (niveaux de gris)."""
    if len(thermal_img.shape) == 3:
        thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
    else:
        thermal_gray = thermal_img
    thermal_gray = cv2.GaussianBlur(thermal_gray, (5, 5), 0)  # evite un pixel bruite isole
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thermal_gray)
    return max_loc[0], max_loc[1], max_val


def project_point(x, y, H_total):
    """Projette un point (x, y) via une matrice d'homographie 3x3 (coordonnees homogenes)."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt, H_total)
    return float(projected[0][0][0]), float(projected[0][0][1])


# ----------------------------------------------------------------------------
# BONUS : Overlay thermique "component-aware"
# ----------------------------------------------------------------------------

def detect_component_contours(rgb_blueprint_space, min_area_px=40, max_area_ratio=0.15):
    """
    Detecte les contours FERMES correspondant probablement a des composants
    (footprints / silkscreen) sur l'image RGB deja projetee dans le repere blueprint.

    Principe : un composant est une forme fermee -> on peut lui assigner UNE
    temperature representative unique, evitant le "bavage" de couleur entre
    composants voisins qu'un flou/interpolation bilineaire classique provoquerait.
    """
    gray = cv2.cvtColor(rgb_blueprint_space, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    img_area = rgb_blueprint_space.shape[0] * rgb_blueprint_space.shape[1]
    max_area_px = max_area_ratio * img_area

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area_px < area < max_area_px:
            # on ne garde que des contours "raisonnablement fermes" (perimetre coherent avec l'aire)
            valid.append(c)

    print(f"[Overlay] {len(valid)} contours de composants retenus (sur {len(contours)} bruts).")
    return valid


def build_component_aware_overlay(thermal_warped_gray, rgb_blueprint_space, component_contours,
                                   colormap=cv2.COLORMAP_INFERNO):
    """
    Construit une carte de temperature "par blocs" :
      - a l'interieur de chaque contour de composant : valeur UNIFORME
        = moyenne des pixels thermiques (deja recales) contenus dans ce contour.
      - en dehors de tout contour (substrat / pistes / vide) : on garde
        l'interpolation continue brute (comportement standard).

    Retourne une image couleur (BGR) prete a etre fusionnee (alpha blend) avec le blueprint.
    """
    h, w = thermal_warped_gray.shape[:2]
    uniform_temp_map = thermal_warped_gray.copy().astype(np.float32)

    covered_mask = np.zeros((h, w), dtype=np.uint8)

    for contour in component_contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        mean_val = cv2.mean(thermal_warped_gray, mask=mask)[0]
        uniform_temp_map[mask == 255] = mean_val
        covered_mask = cv2.bitwise_or(covered_mask, mask)

    uniform_temp_map = np.clip(uniform_temp_map, 0, 255).astype(np.uint8)

    colorized = cv2.applyColorMap(uniform_temp_map, colormap)

    n_covered = int(np.count_nonzero(covered_mask))
    pct = 100.0 * n_covered / (h * w)
    print(f"[Overlay] {pct:.1f}% de la surface traitee en blocs uniformes par composant, "
          f"le reste garde un degrade continu.")

    return colorized


# ----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------------------

def run_pipeline(thermal_path, rgb_path, blueprint_path, h1_path, out_path,
                  debug_dir=None, alpha=0.5, manual_blueprint_corners=False, manual_rgb_corners=False):

    thermal_img = cv2.imread(thermal_path)
    rgb_img = rgb_path
    blueprint_img = cv2.imread(blueprint_path)
    H1 = np.load(h1_path)  # thermique -> RGB (Etape 1)

    if thermal_img is None or rgb_img is None or blueprint_img is None:
        raise FileNotFoundError("Une des images (thermal/rgb/blueprint) est introuvable.")

    os.makedirs(debug_dir, exist_ok=True) if debug_dir else None

    # ---------------- ETAPE 2 ----------------
    if manual_rgb_corners:
        rgb_corners = PointPicker(rgb_img, "Etape 2 (manuel) - 4 coins du PCB sur l'image RGB").pick(4)
    else:
        try:
            rgb_corners = detect_pcb_corners(
                rgb_img, debug_out=os.path.join(debug_dir, "etape2_coins_rgb.jpg") if debug_dir else None
            )
        except RuntimeError as e:
            print(f"[Etape 2] Detection automatique echouee ({e}). Bascule en mode manuel.")
            rgb_corners = PointPicker(rgb_img, "Etape 2 (manuel) - 4 coins du PCB sur l'image RGB").pick(4)

    # ---------------- ETAPE 3 ----------------
    if manual_blueprint_corners:
        blueprint_corners = PointPicker(
            blueprint_img, "Etape 3 (manuel) - 4 coins du PCB sur le BLUEPRINT (meme ordre qu'Etape 2)"
        ).pick(4)
    else:
        # Par defaut : on considere que le PCB occupe tout le blueprint (les 4 coins de l'image)
        bh, bw = blueprint_img.shape[:2]
        rh , rw = rgb_img.shape[:2]
        blueprint_corners = np.array([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]], dtype=np.float32)
        print("[Etape 3] Coins blueprint = coins de l'image entiere (utilisez --manual-blueprint-corners sinon).")

    H2 = compute_alignment_homography(rgb_corners, blueprint_corners)
    H_total = H2 @ H1  # thermique -> blueprint, en une seule matrice composee

    bh, bw = blueprint_img.shape[:2]

    # ---------------- ETAPE 4a : hotspot ----------------
    xt, yt, max_temp_val = find_hotspot(thermal_img)
    print(f"[Etape 4] Pixel le plus chaud (thermique) : ({xt}, {yt}), intensite = {max_temp_val:.1f}")

    xj, yj = project_point(xt, yt, H_total)
    print(f"[Etape 4] Point chaud projete sur le blueprint : ({xj:.1f}, {yj:.1f})")

    # ---------------- BONUS : overlay component-aware ----------------
    thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY) if len(thermal_img.shape) == 3 else thermal_img
    thermal_1 =cv2.warpPerspective(thermal_gray, H1, (rw, rh))
    cv2.imshow("Thermal warped to RGB", thermal_1)
    thermal_warped = cv2.warpPerspective(thermal_gray, H_total, (bw, bh))

    rgb_warped = cv2.warpPerspective(rgb_img, H2, (bw, bh))  # RGB projete dans le repere blueprint

    component_contours = detect_component_contours(rgb_warped)
    overlay_colorized = build_component_aware_overlay(thermal_warped, rgb_warped, component_contours)

    # Masque : ne fusionner que la ou le thermique a effectivement ete projete (evite un cadre noir)
    valid_mask = (thermal_warped > 0).astype(np.uint8) * 255
    valid_mask_3c = cv2.merge([valid_mask] * 3)

    blended = blueprint_img.copy()
    blend_zone = cv2.addWeighted(blueprint_img, 1 - alpha, overlay_colorized, alpha, 0)
    blended = np.where(valid_mask_3c > 0, blend_zone, blended)

    # Reticule sur le point chaud
    cv2.drawMarker(blended, (int(xj), int(yj)), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
    cv2.circle(blended, (int(xj), int(yj)), 18, (0, 0, 255), 2)
    cv2.putText(blended, f"Point chaud ({xj:.0f},{yj:.0f})", (int(xj) + 22, int(yj) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(out_path, blended)
    print(f"\n[OK] Resultat final sauvegarde : {out_path}")

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "etape3_rgb_warped.jpg"), rgb_warped)
        cv2.imwrite(os.path.join(debug_dir, "thermal_warped.jpg"), thermal_warped)
        cv2.imwrite(os.path.join(debug_dir, "overlay_colorized.jpg"), overlay_colorized)
        with open(os.path.join(debug_dir, "resultats.json"), "w") as f:
            json.dump({
                "hotspot_thermal_px": [xt, yt],
                "hotspot_blueprint_px": [xj, yj],
                "max_intensity": float(max_temp_val),
                "rgb_corners": rgb_corners.tolist(),
                "blueprint_corners": blueprint_corners.tolist(),
            }, f, indent=2)
        print(f"[Debug] Images intermediaires sauvegardees dans : {debug_dir}")

    return blended, (xj, yj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etapes 2/3/4 - Alignement PCB/Blueprint + Overlay thermique")
    parser.add_argument("--thermal", required=True, help="Image thermique")
    #parser.add_argument("--rgb", required=True, help="Image RGB du PCB")
    parser.add_argument("--blueprint", required=True, help="Image JPG du plan/blueprint")
    parser.add_argument("--h1", required=True, help="Fichier .npy de la matrice d'homographie de l'Etape 1")
    parser.add_argument("--out", default="resultat_overlay.jpg", help="Fichier image de sortie")
    parser.add_argument("--debug-dir", default="debug_output", help="Dossier pour les images intermediaires")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacite de l'overlay thermique (0-1)")
    parser.add_argument("--manual-rgb-corners", action="store_true",
                         help="Force la selection manuelle des coins PCB sur l'image RGB")
    parser.add_argument("--manual-blueprint-corners", action="store_true",
                         help="Force la selection manuelle des coins PCB sur le blueprint")
    args = parser.parse_args()

    while True:
        url = "http://192.168.1.19:81/stream"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print("Could not connect to IP Webcam")
            exit()
        ret , frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break 
        out , (xi , xj)=run_pipeline(
            thermal_path=args.thermal,
            rgb_path=frame,
            blueprint_path=args.blueprint,
            h1_path=args.h1,
            out_path=args.out,
            debug_dir=args.debug_dir,
            alpha=args.alpha,
            manual_blueprint_corners=args.manual_blueprint_corners,
            manual_rgb_corners=args.manual_rgb_corners,
        )
        cv2.imshow("Resultat Overlay", out)
        cv2.drawMarker(out, (int(xi), int(xj)), (0, 255, 0), cv2.MARKER_CROSS, markerSize=20, thickness=2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)  # Attendre 1 seconde avant de capturer la prochaine image