#!/usr/bin/env python3
"""
image_test.py : capture, warp, inference YOLO sur l'image de l'échiquier et annotation de la grille 8×8.
"""
import os
import sys
import random
import string
from math import pi

import cv2
import numpy as np
from ultralytics import YOLO
from pyniryo import NiryoRobot, PoseObject, uncompress_image, undistort_image, show_img

# On réutilise les utilitaires de vision de ChessUtils
essential_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(essential_path, "ChessUtils"))
from ChessUtils.chess_board import board as ChessBoard
from ChessUtils import (
    get_workspace_poses_ssh,
    extract_img_markers_with_margin,
    get_cell_boxes
)

# Configuration
target_ip   = '10.10.10.10'
MODEL_PATH  = 'best.pt'
IMAGE_DIR   = './Img_dataset'
WAIT_POSE   = 'wait_pose'

os.makedirs(IMAGE_DIR, exist_ok=True)


def main():
    # Initialisation robot et caméra
    robot = NiryoRobot(target_ip)
    robot.calibrate_auto()
    robot.set_brightness(0.75)
    robot.set_contrast(1.5)
    robot.set_saturation(1.8)
    mtx, dist = robot.get_camera_intrinsics()

    # Chargement du modèle YOLO
    yolo = YOLO(MODEL_PATH)

    # Pose d'attente au-dessus du plateau
    wait_pose = robot.get_pose_saved(WAIT_POSE)
    robot.move(wait_pose)

    # Calibration visuelle : détection des markers sur plateau VIDE
    print("→ Calibration : placez le plateau VIDE et appuyez sur Entrée…")
    input()
    img_und0 = uncompress_image(robot.get_img_compressed())
    show_img("rawImg", img_und0, wait_ms=3000)

    # Extraction de la vue warpée et des paramètres de calibration
    warp_empty, M, (W_tot, H_tot), corners, margin_px = \
        extract_img_markers_with_margin(img_und0, margin_cells=1.5)
    if warp_empty is None:
        raise RuntimeError("Échec de la détection des markers pour la calibration.")
    print("Calibration OK : markers détectés.")
    show_img("warp_empty", warp_empty, wait_ms=3000)

    # Génération de la grille 8×8 pour annotation des cases
    # player_plays_white=False assume perspective noir → renversement automatique des noms de cases
    cell_boxes = get_cell_boxes(warp_empty, margin_px, player_plays_white=False)

    # Boucle de capture et d'inférence
    print("→ Entrée pour capturer, 'q'+Entrée pour quitter, 's'+Entrée pour sauvegarder l'image warpée.")
    while True:
        cmd = input().lower().strip()
        if cmd == 'q':
            break

        # Acquisition et undistortion
        img_und = uncompress_image(robot.get_img_compressed())

        # Warp selon homographie calculée
        board_warp = cv2.warpPerspective(img_und, M, (W_tot, H_tot))

        # Annotation : dessiner le contour de chaque case de la grille
        img_grid = board_warp.copy()
        for (x1, y1, x2, y2) in cell_boxes.values():
            cv2.rectangle(img_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)
        show_img("Grille 8x8", img_grid, wait_ms=1000)

        # Inference YOLO sur l'image warpée et affichage
        result = yolo(board_warp)[0]
        img_ann = result.plot()
        show_img("YOLO Detection", img_ann, wait_ms=1000)

        # Sauvegarde optionnelle
        if cmd == 's':
            rand_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
            fname = f"warp_{rand_str}.jpg"
            path = os.path.join(IMAGE_DIR, fname)
            cv2.imwrite(path, board_warp)
            print(f"Image warpée sauvegardée : {path}")

    robot.close_connection()


if __name__ == '__main__':
    main()
