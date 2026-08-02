#!/usr/bin/env python3
"""
MLX90640 Thermal Viewer -- conversion Python de la version C++/SDL2/bcm2835.

Fonctionnalites reprises a l'identique :
  - Lecture du capteur MLX90640 en I2C
  - Palette de couleurs "iron" (degrade noir -> violet -> rouge -> jaune -> blanc)
  - Upscale x3 par "sliding window" (moyenne ponderee avec les 8 voisins)
  - Export a chaque frame :
      ./data/CameraArray.txt        (32x24, temperatures brutes)
      ./data/CameraArrayScaled.txt  (96x72, temperatures upscalees)
      ../CameraRGB/thermalCameraFrame.jpg (apercu colorise)
  - Fenetre avec reticule souris, barre de statut (MIN/MAX/AVG/FPS/curseur)
  - Touches : ECHAP/Q quitter, P pause, 1-5 palette, +/- zoom fenetre

Dependances :
    pip install adafruit-circuitpython-mlx90640 pygame numpy pillow

Materiel : capteur MLX90640 cable en I2C sur un Raspberry Pi (memes broches
que dans la version C++ -- board.SCL/board.SDA gerent ca automatiquement).

Note fidelite : dans le code C++ d'origine, buildLut() ignorait deja le
parametre "palette" et generait toujours le degrade "iron", quelle que soit
la touche 1-5 pressee (comportement conserve ici a l'identique).
"""

import os
import time

import numpy as np
import pygame
from PIL import Image

try:
    import board
    import busio
    import adafruit_mlx90640
    HARDWARE_AVAILABLE = True
except (ImportError, NotImplementedError):
    # Permet de lire/tester ce fichier (py_compile, revue de code, etc.)
    # sur une machine sans capteur/Blinka installe.
    HARDWARE_AVAILABLE = False


# ── constantes capteur ───────────────────────────────────────────────────────
SRC_W, SRC_H = 32, 24
SCALE = 3                      # upscale interne (garde le flou)
DST_W, DST_H = SRC_W * SCALE, SRC_H * SCALE   # 96 x 72

# ── constantes fenetre ───────────────────────────────────────────────────────
WIN_SCALE_DEFAULT = 5           # chaque pixel upscale -> WIN_SCALE x WIN_SCALE pixels ecran
STATUS_BAR_H = 28

PALETTE_NAMES = ["iron", "inferno", "hot", "cool", "gray"]

DATA_DIR = "./data"
RAW_TXT_PATH = os.path.join(DATA_DIR, "CameraArray.txt")
SCALED_TXT_PATH = os.path.join(DATA_DIR, "CameraArrayScaled.txt")
JPG_OUTPUT_PATH = "../CameraRGB/thermalCameraFrame.jpg"


# ── palette (LUT 256 entrees) ────────────────────────────────────────────────
def build_lut_iron():
    """Reproduit exactement les 7 points de controle du degrade 'iron' du C++."""
    stops_t = np.array([0.00, 0.20, 0.45, 0.65, 0.80, 0.92, 1.00], dtype=np.float32)
    stops_rgb = np.array([
        [0,   0,   0],
        [30,   0,  80],
        [120,   0, 120],
        [200,  30,   0],
        [255, 120,   0],
        [255, 220,   0],
        [255, 255, 200],
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


# ── upscale x3 par fenetre glissante (vectorise numpy) ───────────────────────
def sliding_window_upscale(src):
    """
    Reproduit le bloc 3x3 du C++ :
        coins  = moyenne du centre + 2 voisins orthogonaux + 1 voisin diagonal
        milieux = moyenne du centre + 1 voisin orthogonal
        centre  = valeur du pixel source, inchangee
    src : (SRC_H, SRC_W) float32 -> retourne (DST_H, DST_W) float32
    """
    # 'edge' = clamp-to-edge, identique au comportement rm/rp/cm/cp du C++
    padded = np.pad(src, 1, mode="edge")

    center = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    nw = padded[:-2, :-2]
    ne = padded[:-2, 2:]
    sw = padded[2:, :-2]
    se = padded[2:, 2:]

    top_left = (nw + north + west + center) / 4.0
    top_mid = (center + north) / 2.0
    top_right = (center + north + east + ne) / 4.0
    mid_left = (center + west) / 2.0
    mid_right = (center + east) / 2.0
    bot_left = (center + south + west + sw) / 4.0
    bot_mid = (center + south) / 2.0
    bot_right = (center + south + east + se) / 4.0

    dst = np.zeros((SRC_H * SCALE, SRC_W * SCALE), dtype=np.float32)
    dst[0::3, 0::3] = top_left
    dst[0::3, 1::3] = top_mid
    dst[0::3, 2::3] = top_right
    dst[1::3, 0::3] = mid_left
    dst[1::3, 1::3] = center
    dst[1::3, 2::3] = mid_right
    dst[2::3, 0::3] = bot_left
    dst[2::3, 1::3] = bot_mid
    dst[2::3, 2::3] = bot_right

    return dst


def save_matrix_txt(matrix, path):
    """Meme format que le C++ : valeurs separees par des espaces, une ligne par rangee."""
    np.savetxt(path, matrix, fmt="%.4f")


def draw_text(screen, font, text, x, y, color):
    if font is None:
        return
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JPG_OUTPUT_PATH), exist_ok=True)

    # ── init capteur ─────────────────────────────────────────────────────────
    if not HARDWARE_AVAILABLE:
        raise RuntimeError(
            "Bibliotheques materielles indisponibles (board/busio/adafruit_mlx90640). "
            "Installez-les sur le Raspberry Pi : "
            "pip install adafruit-circuitpython-mlx90640 adafruit-blinka"
        )

    i2c = busio.I2C(board.SCL, board.SDA, frequency=1_000_000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ  # == 0b010 dans le C++

    try:
        mlx.emissivity = 0.95  # equivalent a `float emissivity = 0.95f;` -- absent sur certaines versions de la lib
    except AttributeError:
        pass

    frame = [0.0] * (SRC_W * SRC_H)

    # ── init pygame ──────────────────────────────────────────────────────────
    pygame.init()
    win_scale = WIN_SCALE_DEFAULT
    win_w, win_h = DST_W * win_scale, DST_H * win_scale + STATUS_BAR_H + 2
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("MLX90640 Thermal Viewer")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("dejavusansmono", 14)
    except Exception:
        font = pygame.font.Font(None, 16)

    palette_idx = 0
    lut = build_lut_iron()

    running = True
    paused = False
    frame_n = 0
    min_t, max_t, avg_t = 20.0, 50.0, 35.0
    mouse_x, mouse_y = -1, -1
    fps = 0.0
    t_last = time.time()

    upscaled = np.zeros((DST_H, DST_W), dtype=np.float32)
    rgb_frame = np.zeros((DST_H, DST_W, 3), dtype=np.uint8)

    while running:
        # ── evenements ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                win_w, win_h = event.w, event.h
                screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    palette_idx = event.key - pygame.K_1
                    lut = build_lut_iron()  # cf. note en tete de fichier : le C++ ignorait deja "palette"
                    print(f"palette activated : {palette_idx}")
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    win_scale = min(16, win_scale + 1)
                    win_w, win_h = DST_W * win_scale, DST_H * win_scale + STATUS_BAR_H
                    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    win_scale = max(2, win_scale - 1)
                    win_w, win_h = DST_W * win_scale, DST_H * win_scale + STATUS_BAR_H
                    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

        if not running:
            break

        # ── lecture capteur (sautee en pause) ───────────────────────────────
        if not paused:
            try:
                mlx.getFrame(frame)
            except ValueError:
                # glitch de lecture occasionnel du MLX90640 -- on saute juste cette frame
                continue

            raw = np.array(frame, dtype=np.float32).reshape(SRC_H, SRC_W)
            oriented = raw[::-1, :]  # flip vertical, comme le C++

            save_matrix_txt(oriented, RAW_TXT_PATH)

            upscaled = sliding_window_upscale(oriented)
            save_matrix_txt(upscaled, SCALED_TXT_PATH)

            # ── stats ────────────────────────────────────────────────────────
            min_t, max_t = float(upscaled.min()), float(upscaled.max())
            avg_t = float(upscaled.mean())

            # ── mapping RGB via LUT ─────────────────────────────────────────
            rng = max(0.01, max_t - min_t)
            idx = np.clip(((upscaled - min_t) / rng) * 255.0, 0, 255).astype(np.uint8)
            rgb_frame = lut[idx]  # (DST_H, DST_W, 3)

            Image.fromarray(rgb_frame, mode="RGB").save(JPG_OUTPUT_PATH, quality=95)
            frame_n += 1

            t_now = time.time()
            dt = max(1e-6, t_now - t_last)
            t_last = t_now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # ── rendu ────────────────────────────────────────────────────────────
        img_h = win_h - STATUS_BAR_H
        screen.fill((10, 14, 26))

        # pygame attend (largeur, hauteur, 3) mais indexe (x, y) -> transpose
        surf_small = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surf_small, (win_w, img_h))
        screen.blit(scaled_surf, (0, 0))

        if mouse_x >= 0 and 0 <= mouse_y < img_h:
            pygame.draw.line(screen, (255, 255, 255), (mouse_x, 0), (mouse_x, img_h))
            pygame.draw.line(screen, (255, 255, 255), (0, mouse_y), (win_w, mouse_y))

        pygame.draw.rect(screen, (17, 24, 39), (0, img_h, win_w, STATUS_BAR_H))
        pygame.draw.line(screen, (40, 60, 80), (0, img_h), (win_w, img_h))

        draw_text(screen, font, f"MAX {max_t:.1f} C", 8, img_h + 6, (255, 120, 40))
        draw_text(screen, font, f"MIN {min_t:.1f} C", 130, img_h + 6, (80, 200, 255))
        draw_text(screen, font, f"AVG {avg_t:.1f} C", 250, img_h + 6, (220, 230, 240))
        draw_text(screen, font,
                  f"{fps:.1f} fps  frame {frame_n}  palette:{PALETTE_NAMES[palette_idx % 5]}",
                  390, img_h + 6, (100, 120, 140))

        if mouse_x >= 0 and 0 <= mouse_y < img_h and win_w > 0 and img_h > 0:
            px = int(mouse_x / win_w * DST_W)
            py = int(mouse_y / img_h * DST_H)
            px = max(0, min(DST_W - 1, px))
            py = max(0, min(DST_H - 1, py))
            val = upscaled[py, px]
            draw_text(screen, font, f"cursor {val:.1f} C ({px // SCALE},{py // SCALE})",
                      max(0, win_w - 220), img_h + 6, (220, 230, 240))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()