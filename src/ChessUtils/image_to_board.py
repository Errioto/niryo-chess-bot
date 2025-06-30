# ─────────────────────────────────────────────────────────────────────────────
# ChessUtils/image_to_board.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import cv2

# Valeur par défaut, côté « utile » du plateau (avant marge)
IM_EXTRACT_SMALL_SIDE_PIXELS = 250

# ─────────────────────────────────────────────────────────────────────────────
# 1) Détection des markers & calcul de la matrice d’homographie
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_dist_2_pts(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


class PotentialMarker:
    def __init__(self, center, radius, cnt):
        self.center = center
        self.x = center[0]
        self.y = center[1]
        self.radius = radius
        self.contour = cnt
        self.is_merged = False

    def get_center(self):
        return self.center


class Marker:
    def __init__(self, potential_marker: PotentialMarker):
        self.list_centers = [potential_marker.get_center()]
        self.list_radius  = [potential_marker.radius]
        self.cx, self.cy  = potential_marker.get_center()
        self.radius       = potential_marker.radius
        self.identifiant  = None

    def add_circle(self, other: PotentialMarker):
        self.list_centers.append(other.get_center())
        self.list_radius.append(other.radius)
        other.is_merged = True
        mx, my = np.mean(self.list_centers, axis=0)
        self.cx, self.cy = int(round(mx)), int(round(my))
        self.radius      = int(round(max(self.list_radius)))

    def nb_circles(self):
        return len(self.list_centers)

    def get_id_from_slice(self, img_thresh):
        x, y, w, h = self.cx - 1, self.cy - 1, 3, 3
        val = np.mean(img_thresh[y:y+h, x:x+w])
        self.identifiant = "A" if val > 200 else "B"
        return self.identifiant

    def get_center(self):
        return (self.cx, self.cy)

    def get_radius(self):
        return self.radius


def find_markers_from_img_thresh(
    img_thresh,
    max_dist_between_centers=3,
    min_radius_circle=4,
    max_radius_circle=35,
    min_radius_marker=7
):
    contours = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    pots = []
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if not (min_radius_circle < r < max_radius_circle):
            continue
        pots.append(PotentialMarker((int(round(x)), int(round(y))), int(round(r)), cnt))

    pots = sorted(pots, key=lambda m: m.x)
    markers = []
    for i, p in enumerate(pots):
        if p.is_merged:
            continue
        m = Marker(p)
        cx, cy = m.get_center()
        for q in pots[i+1:]:
            if q.is_merged:
                continue
            if q.x - cx > max_dist_between_centers:
                break
            if euclidean_dist_2_pts((cx, cy), q.get_center()) <= max_dist_between_centers:
                m.add_circle(q)
                cx, cy = m.get_center()
        if m.nb_circles() > 2 and m.get_radius() >= min_radius_marker:
            markers.append(m)
            m.get_id_from_slice(img_thresh)
    return markers


def sort_markers_detection(list_markers):
    ym = sorted(list_markers, key=lambda m: m.cy)
    top1, top2, bot1, bot2 = ym
    tl = top1 if top1.cx < top2.cx else top2
    tr = top2 if tl is top1 else top1
    bl = bot1 if bot1.cx < bot2.cx else bot2
    br = bot2 if bl is bot1 else bot1

    quad = [tl, tr, br, bl]
    ids  = [m.identifiant for m in quad]
    if ids.count("A") == 1:
        n = ids.index("A")
        return quad[n:] + quad[:n]
    if ids.count("B") == 1:
        n = ids.index("B")
        return quad[n:] + quad[:n]
    return quad


def complicated_sort_markers(list_markers, workspace_ratio):
    import itertools

    if workspace_ratio >= 1.0:
        tw = int(round(workspace_ratio * IM_EXTRACT_SMALL_SIDE_PIXELS))
        th = IM_EXTRACT_SMALL_SIDE_PIXELS
    else:
        thr = 1.0 / workspace_ratio
        th = int(round(thr * IM_EXTRACT_SMALL_SIDE_PIXELS))
        tw = IM_EXTRACT_SMALL_SIDE_PIXELS

    ids = [m.identifiant for m in list_markers]
    a_cnt = ids.count("A")
    b_cnt = ids.count("B")
    if a_cnt < 3 and b_cnt < 3:
        return None

    first, second = ("A", "B") if a_cnt >= b_cnt else ("B", "A")
    combs = []
    list1 = [m for m in list_markers if m.identifiant == first]
    list2 = [m for m in list_markers if m.identifiant == second]

    if list1:
        for m1 in list1:
            for trio in itertools.combinations(list2, 3):
                combs.append(sort_markers_detection([m1, *trio]))
    else:
        for quad in itertools.combinations(list2, 4):
            combs.append(sort_markers_detection(list(quad)))

    if not combs:
        return None

    final_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
    dets = []
    for quad in combs:
        src = np.array([[m.cx, m.cy] for m in quad], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, final_pts)
        dets.append(np.linalg.det(M))

    best = np.argmin(np.abs(np.array(dets) - 1))
    return combs[best]


def extract_img_markers_with_margin(
    img,
    workspace_ratio: float = 1.0,
    base_size: int = IM_EXTRACT_SMALL_SIDE_PIXELS,
    margin_cells: int = 1
):
    """
    Détecte 4 markers au centre des cases a1,a8,h1,h8 et renvoie :
      - warp (BGR) de taille (W_tot,H_tot) = (tw+2*margin_px, th+2*margin_px)
      - matrice M,
      - (W_tot, H_tot),
      - corners triés,
      - margin_px

    base_size    : taille en px du plateau sans marge (ex. 200).
    margin_cells : nb de cases de marge désiré autour (ex. 1 ou 2).
    """
    # 1) seuillage et détection des markers
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 25
    )
    marks = find_markers_from_img_thresh(img_thresh)
    if not marks or len(marks) > 6:
        return None, None, None, None, None

    # tri classique ou compliqué si >4 markers
    if len(marks) == 4:
        corners = sort_markers_detection(marks)
    else:
        corners = complicated_sort_markers(marks, workspace_ratio)
        if corners is None:
            return None, None, None, None, None

    # 2) calcul de la taille d'une case et de la marge en pixels
    cell_px   = base_size / 8.0
    margin_px = int(round(cell_px * margin_cells))

    # 3) dimensions du plateau utile avant marge
    if workspace_ratio >= 1.0:
        tw = th = base_size
    else:
        tw = int(round((1.0 / workspace_ratio) * base_size))
        th = tw

    # 4) dimensions totales avec marge
    W_tot = tw + 2 * margin_px
    H_tot = th + 2 * margin_px

    # 5) points source (centres détectés)
    src_pts = np.array([m.get_center() for m in corners], dtype=np.float32)

    # 6) points destination → centres des 4 coins du plateau utile
    half = cell_px / 2.0
    dst_pts = np.array([
        [margin_px + half,          margin_px + half],           # coin inférieur gauche
        [margin_px + tw - half - 1, margin_px + half],           # coin inférieur droit
        [margin_px + tw - half - 1, margin_px + th - half - 1],  # coin supérieur droit
        [margin_px + half,          margin_px + th - half - 1],  # coin supérieur gauche
    ], dtype=np.float32)

    # 7) homographie + warp
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img, M, (W_tot, H_tot))

    return warp, M, (W_tot, H_tot), corners, margin_px



# ─────────────────────────────────────────────────────────────────────────────
# 2) Génération des cases 8×8 au sein de l’image warpée (en ignorant la marge)
# ─────────────────────────────────────────────────────────────────────────────

def get_cell_boxes(board_img, margin_px, player_plays_white=True):
    """
    board_img  : image warpée de taille (W_tot, H_tot) = (base_size+2*margin, base_size+2*margin).
    margin_px  : nombre de pixels à ignorer tout autour.
    player_plays_white : booléen indiquant si le joueur a les pièces blanches.
                         Si False, on inverse la numérotation des cases (180°).

    → Retourne dict { 'a8': (x1,y1,x2,y2), …, 'h1': (…) }
      où la zone utile est le carré [margin_px : margin_px+base_size-1] en largeur ET hauteur.
      Si player_plays_white est False, les noms de cases sont renversés :
        - (i, j) → (7-i, 7-j) dans la grille 8×8.
    """
    H_tot, W_tot = board_img.shape[:2]
    # Base_size = W_tot - 2*margin_px
    base_size = W_tot - 2 * margin_px

    cell_w = base_size / 8.0
    cell_h = base_size / 8.0
    files = 'abcdefgh'
    ranks = '87654321'
    boxes = {}

    for i in range(8):
        for j in range(8):
            x1 = int(round(margin_px + j * cell_w))
            y1 = int(round(margin_px + i * cell_h))
            x2 = int(round(margin_px + (j + 1) * cell_w))
            y2 = int(round(margin_px + (i + 1) * cell_h))

            if player_plays_white:
                # Nom “standard” si le joueur est blanc :
                cell_name = f"{files[j]}{ranks[i]}"
            else:
                # Inversion 180° : (i, j) → (7-i, 7-j)
                flipped_file = files[7 - j]
                flipped_rank = ranks[7 - i]
                cell_name = f"{flipped_file}{flipped_rank}"

            boxes[cell_name] = (x1, y1, x2, y2)

    return boxes



# ─────────────────────────────────────────────────────────────────────────────
# 3) Intersection area 
# ─────────────────────────────────────────────────────────────────────────────

def intersect_area(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    w = max(0, xB - xA); h = max(0, yB - yA)
    return w * h


# ─────────────────────────────────────────────────────────────────────────────
# 4) Conversion d’un résultat YOLO en état « board_state »
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

def board_state_from_yolo(yolo_result, cell_boxes, player_plays_white=True, ignore_class="Hand"):
    """

    Arguments :
        yolo_result        : résultat de l’inférence YOLO sur l’image warpée
                             (yolo(board_img)[0]), avec :
                              - yolo_result.names : liste des noms de classes
                              - yolo_result.boxes.xyxy : Nx4 numpy array ([x1, y1, x2, y2])
                              - yolo_result.boxes.cls  : N indices (tensor) des classes
                              - yolo_result.boxes.conf : N confidences (tensor)
        cell_boxes         : dict { cell_name: (x1,y1,x2,y2) } tel que retourné par
                             get_cell_boxes(board_img, margin_px, player_plays_white)
        player_plays_white : booléen indiquant l’orientation. Si False, alors `cell_boxes`
                             a été construit avec les noms de cases “retournés” (flippés).
        ignore_class       : nom de la classe à ignorer (par exemple "Hand")

    → Retourne :
         matrix : liste de 8 listes de 8 strings, où matrix[0] est la rangée “8” (cases a8-h8)
                  et matrix[7] est la rangée “1” (cases a1-h1). Chaque élément vaut :
                  - ' ' si rien n’est détecté dans la case
                  - le nom exact de la classe YOLO (ex. 'P', 'k', 'Q', 'n', etc.)

    Notes :
      1. Si player_plays_white=False, les clés de `cell_boxes` seront du type 'h1','g1',...
         correspondant à la perspective noir. On “inverse” alors les noms pour retrouver
         l’appellation standard a1…h8, puis on place dans la bonne entrée de la matrice.
      2. On garde pour chaque case la détection de plus haute confiance (comme avant).
      3. On suppose que les classes des pièces noires sont en minuscules, et celles des blanches
         en majuscules (mais cela n’affecte pas ici, on se contente de recopier le nom tel quel).
    """

    # Listes “standard” pour fichiers/rangs
    files = ['a','b','c','d','e','f','g','h']
    ranks = ['1','2','3','4','5','6','7','8']

    # Initialisation de la matrice résultat (8×8) et d’un tableau de confiances pour chaque case
    matrix = [[' ' for _ in range(8)] for _ in range(8)]
    conf_matrix = [[0.0 for _ in range(8)] for _ in range(8)]

    # Extraction des infos YOLO
    names = yolo_result.names
    classes = yolo_result.boxes.cls.cpu().numpy().astype(int)
    confs = yolo_result.boxes.conf.cpu().numpy()
    boxes = yolo_result.boxes.xyxy.cpu().numpy()

    # Pour chaque détection, on recherche la “meilleure” case (celle où l’intersection est maxi)
    for box, cls_idx, conf in zip(boxes, classes, confs):
        class_name = names[cls_idx]

        # Si on capte une main, on retourne None (comme avant)
        if class_name == ignore_class:
            return None

        # Parcours de cell_boxes pour trouver la case ayant l'intersection max
        best_cell = None
        best_area = -1
        for cell_name, cell_box in cell_boxes.items():
            # intersection
            xA = max(box[0], cell_box[0])
            yA = max(box[1], cell_box[1])
            xB = min(box[2], cell_box[2])
            yB = min(box[3], cell_box[3])
            w = max(0, xB - xA)
            h = max(0, yB - yA)
            area = w * h
            if area > best_area:
                best_area = area
                best_cell = cell_name

        # Si aucune intersection, on ignore cette détection
        if best_area <= 0:
            continue

        # On ne conserve cette détection que si sa confiance est supérieure à celle déjà stockée
        # dans la case visée
        # → On doit déterminer les indices (row_idx, col_idx) DANS LA MATRICE 8×8 “standard”
        #   (avec rangs 8→1 et files a→h)
        # Mais best_cell est nommé en fonction de player_plays_white :
        # - si True  : best_cell est déjà dans l’ordre standard (ex. 'e4' → file 'e', rank '4')
        # - si False : best_cell a été “flippé”, ex. 'b2' (du point de vue noir) correspond
        #               à la vraie case 'g7' en nom standard, etc. Il faut reconvertir.

        if player_plays_white:
            # pas d'inversion nécessaire
            std_file = best_cell[0]
            std_rank = best_cell[1]
        else:
            # on inverse le flip qui a eu lieu dans get_cell_boxes
            flipped_file = best_cell[0]
            flipped_rank = best_cell[1]
            fi = files.index(flipped_file)      # indice 0..7
            ri = ranks.index(flipped_rank)      # indice 0..7
            std_file = files[7 - fi]
            std_rank = ranks[7 - ri]

        # Calcul des indices d’accès dans matrix :
        #   row_idx = 8 - int(std_rank)  → ex. rank '8' → row_idx = 0 ;   rank '1' → row_idx = 7
        #   col_idx = files.index(std_file)  → 'a'→0, 'b'→1, ..., 'h'→7
        row_idx = 8 - int(std_rank)
        col_idx = files.index(std_file)

        # Si la détection actuelle a plus de confiance que ce qu’on a déjà stocké :
        if conf > conf_matrix[row_idx][col_idx]:
            matrix[row_idx][col_idx] = class_name
            conf_matrix[row_idx][col_idx] = conf

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# FIN de image_to_board.py
# ─────────────────────────────────────────────────────────────────────────────
