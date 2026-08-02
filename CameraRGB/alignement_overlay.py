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
def load_thermal_matrix(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        matrix = np.load(path)
    elif ext in (".csv", ".txt"):
        matrix = np.loadtxt(path, delimiter=",") if ext == ".csv" else np.loadtxt(path)
    else:
        raise ValueError(f"Format non supporte : '{ext}'. Utilisez .npy, .csv ou .txt.")
    matrix = matrix.astype(np.float32)
    print(f"[Thermal] Matrice chargee : shape={matrix.shape}, min={matrix.min():.1f}C, max={matrix.max():.1f}C")
    return matrix

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

def find_hotspot(temp_matrix):
    """Retourne (x, y, temperature_reelle_C) du pixel le plus chaud de la matrice radiometrique."""
    smoothed = cv2.GaussianBlur(temp_matrix, (5, 5), 0)  # evite un pixel bruite isole
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(smoothed)
    return max_loc[0], max_loc[1], max_val
 
 

def project_point(x, y, H_total):
    """Projette un point (x, y) via une matrice d'homographie 3x3 (coordonnees homogenes)."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt, H_total)
    return float(projected[0][0][0]), float(projected[0][0][1])


# ----------------------------------------------------------------------------
# BONUS : Overlay thermique "component-aware"
# ----------------------------------------------------------------------------
def detect_component_contours(rgb_blueprint_space, min_area_px=40, max_area_ratio=0.15,
                               canny_low=40, canny_high=120, debug_dir=None):
    gray = cv2.cvtColor(rgb_blueprint_space, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "overlay_canny_edges.jpg"), edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    img_area = rgb_blueprint_space.shape[0] * rgb_blueprint_space.shape[1]
    max_area_px = max_area_ratio * img_area

    valid = [c for c in contours if min_area_px < cv2.contourArea(c) < max_area_px]
    print(f"[Overlay] {len(valid)} contours de composants retenus (sur {len(contours)} bruts).")
    return valid
 
def normalize_temp_to_uint8(temp_map, vmin, vmax):
    """Convertit une carte de temperatures reelles (float, Celsius) en image 8-bit
    pour affichage/colorisation, en se basant sur un min/max COMMUN a toute l'image
    (indispensable pour que froid et chaud restent visuellement coherents sur toute
    la carte, et ne s'ecrasent pas l'un l'autre par un mauvais clipping)."""
    if vmax - vmin < 1e-6:
        return np.zeros_like(temp_map, dtype=np.uint8)
    normalized = (temp_map - vmin) / (vmax - vmin) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)
 
def build_component_aware_overlay(temp_warped, valid_mask, component_contours,colormap=cv2.COLORMAP_INFERNO):
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
        raise RuntimeError("Aucun pixel thermique valide apres warp. Verifiez H1/H2.")
    vmin, vmax = float(valid_values.min()), float(valid_values.max())

    uint8_map = normalize_temp_to_uint8(uniform_temp_map, vmin, vmax)
    colorized = cv2.applyColorMap(uint8_map, colormap)

    n_covered = int(np.count_nonzero(covered_mask))
    n_valid = int(np.count_nonzero(valid_mask))
    print(f"[Overlay] {100.0*n_covered/max(n_valid,1):.1f}% traite en blocs uniformes par composant.")
    print(f"[Overlay] Plage : {vmin:.1f}C a {vmax:.1f}C")

    return colorized, vmin, vmax

# ----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------------------
def run_pipeline(thermal_path, rgb_img, blueprint_path, h1_path, out_path,
                  debug_dir=None, alpha=0.5, manual_blueprint_corners=False, manual_rgb_corners=False,
                  min_area_px=40, max_area_ratio=0.15, canny_low=40, canny_high=120):

    temp_matrix = load_thermal_matrix(thermal_path)  # temperatures REELLES en Celsius
    blueprint_img = cv2.imread(blueprint_path)
    H1 = np.load(h1_path)  # thermique -> RGB (Etape 1)

    if rgb_img is None or blueprint_img is None:
        raise FileNotFoundError("L'image RGB (frame webcam) ou le blueprint est introuvable.")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    rh, rw = rgb_img.shape[:2]  # calcule une seule fois, avant les branches if/else

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
        bh, bw = blueprint_img.shape[:2]
        blueprint_corners = np.array([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]], dtype=np.float32)
        print("[Etape 3] Coins blueprint = coins de l'image entiere (utilisez --manual-blueprint-corners sinon).")

    H2 = compute_alignment_homography(rgb_corners, blueprint_corners)
    H_total = H2 @ H1

    bh, bw = blueprint_img.shape[:2]

    # ---------------- ETAPE 4a : hotspot ----------------
    xt, yt, hotspot_temp = find_hotspot(temp_matrix)
    print(f"[Etape 4] Pixel le plus chaud (thermique) : ({xt}, {yt}), temperature = {hotspot_temp:.1f}C")

    xj, yj = project_point(xt, yt, H_total)
    print(f"[Etape 4] Point chaud projete sur le blueprint : ({xj:.1f}, {yj:.1f})")

    # ---------------- BONUS : overlay component-aware ----------------
    SENTINEL = -9999.0
    temp_warped = cv2.warpPerspective(temp_matrix, H_total, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=SENTINEL)
    valid_mask = (temp_warped > SENTINEL + 1).astype(np.uint8) * 255
    valid_mask_3c = cv2.merge([valid_mask] * 3)

    rgb_warped = cv2.warpPerspective(rgb_img, H2, (bw, bh))

    component_contours = detect_component_contours(
        rgb_warped, min_area_px=min_area_px, max_area_ratio=max_area_ratio,
        canny_low=canny_low, canny_high=canny_high, debug_dir=debug_dir
    )
    overlay_colorized, vmin_c, vmax_c = build_component_aware_overlay(temp_warped, valid_mask, component_contours)

    blended = blueprint_img.copy()
    blend_zone = cv2.addWeighted(blueprint_img, 1 - alpha, overlay_colorized, alpha, 0)
    blended = np.where(valid_mask_3c > 0, blend_zone, blended)

    cv2.drawMarker(blended, (int(xj), int(yj)), (0, 0, 255), cv2.MARKER_CROSS, markerSize=30, thickness=3)
    cv2.circle(blended, (int(xj), int(yj)), 18, (0, 0, 255), 2)
    cv2.putText(blended, f"{hotspot_temp:.1f}C", (int(xj) + 22, int(yj) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(out_path, blended)
    print(f"\n[OK] Resultat final sauvegarde : {out_path}")

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "etape3_rgb_warped.jpg"), rgb_warped)
        cv2.imwrite(os.path.join(debug_dir, "thermal_warped_normalized.jpg"),
                    normalize_temp_to_uint8(temp_warped, vmin_c, vmax_c))
        cv2.imwrite(os.path.join(debug_dir, "overlay_colorized.jpg"), overlay_colorized)
        with open(os.path.join(debug_dir, "resultats.json"), "w") as f:
            json.dump({
                "hotspot_thermal_px": [xt, yt],
                "hotspot_temperature_C": float(hotspot_temp),
                "hotspot_blueprint_px": [xj, yj],
                "temperature_min_C": vmin_c,
                "temperature_max_C": vmax_c,
                "rgb_corners": rgb_corners.tolist(),
                "blueprint_corners": blueprint_corners.tolist(),
            }, f, indent=2, cls=NumpyEncoder)
        print(f"[Debug] Images intermediaires sauvegardees dans : {debug_dir}")

    return blended, (xj, yj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etapes 2/3/4 - Alignement PCB/Blueprint + Overlay thermique")
    parser.add_argument("--thermal", required=True, help="Matrice radiometrique (.npy/.csv/.txt)")
    parser.add_argument("--blueprint", required=True, help="Image JPG du plan/blueprint")
    parser.add_argument("--h1", required=True, help="Fichier .npy de la matrice d'homographie de l'Etape 1")
    parser.add_argument("--out", default="resultat_overlay.jpg", help="Fichier image de sortie")
    parser.add_argument("--debug-dir", default="debug_output", help="Dossier pour les images intermediaires")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacite de l'overlay thermique (0-1)")
    parser.add_argument("--manual-rgb-corners", action="store_true")
    parser.add_argument("--manual-blueprint-corners", action="store_true")
    args = parser.parse_args()

    url = "http://192.168.1.19:81/stream"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Could not connect to IP Webcam")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        exit()

    blended, (xj, yj) = run_pipeline(
        thermal_path=args.thermal,
        rgb_img=frame,
        blueprint_path=args.blueprint,
        h1_path=args.h1,
        out_path=args.out,
        debug_dir=args.debug_dir,
        alpha=args.alpha,
        manual_blueprint_corners=args.manual_blueprint_corners,
        manual_rgb_corners=args.manual_rgb_corners,
    )

    cap.release()
    print(f"[OK] Point chaud a ({xj:.1f}, {yj:.1f}) sur le blueprint. Resultat dans {args.out}.")