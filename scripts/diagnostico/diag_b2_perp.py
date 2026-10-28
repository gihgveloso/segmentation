"""(b) v2 — Perfis perpendiculares reais às linhas Hough.

Uso: python3 scripts/diagnostico/diag_b2_perp.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import cv2

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
OUT  = os.path.join(BASE, 'debug_out')

# Segmentos fornecidos pela sessão anterior (calibração B, px originais)
SEGS_FIXOS = {
    'bitbots': [(77,333,314,228), (63,351,285,464), (281,428,462,265)],
    'eliza':   [(71,246,303,317), (413,359,635,426), (428,355,604,406)],
}

CALIB_B_W_LO = np.array([0,  0, 165], dtype=np.uint8)
CALIB_B_W_HI = np.array([179,60, 255], dtype=np.uint8)

IMG_PATHS = {
    'bitbots': os.path.join(BASE, 'images',
        'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(BASE, 'images', 'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
    'image_1': os.path.join(BASE, 'images', 'image_1.png'),
}


def load(path):
    img = cv2.imread(path)
    if img is None: sys.exit(f"ERRO: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def hough_segs(img):
    """Retorna ≥3 segmentos Hough longos no branco B, ordenados por comprimento desc."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wm  = cv2.inRange(hsv, CALIB_B_W_LO, CALIB_B_W_HI)
    edges = cv2.Canny(wm, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=40, maxLineGap=10)
    if segs is None:
        return []
    segs2 = segs.reshape(-1, 4)
    segs2 = sorted(segs2, key=lambda s: np.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)
    # Filtrar para pegar segmentos em direções distintas (≥20° de diferença entre si)
    chosen = []
    for s in segs2:
        ang = np.degrees(np.arctan2(s[3]-s[1], s[2]-s[0])) % 180
        if all(abs(((ang - np.degrees(np.arctan2(c[3]-c[1],
                    c[2]-c[0])) % 180) + 90) % 180 - 90) > 10
               for c in chosen):
            chosen.append(s)
        if len(chosen) >= 3:
            break
    # Se não há 3 com ângulos distintos, pega os 3 mais longos
    if len(chosen) < 3:
        chosen = segs2[:3]
    return [tuple(s) for s in chosen[:3]]


def sample_perpendicular(V_img, x1, y1, x2, y2, n=8, half=40):
    """Amostra n perfis perpendiculares ao segmento, ±half px.

    Retorna (profiles, sample_pts, normal) onde:
      profiles   : lista de n arrays (2*half+1,) de V float
      sample_pts : lista de (cx,cy) em px originais
      normal     : (nx, ny) unitário perpendicular ao segmento
    """
    dx, dy = x2 - x1, y2 - y1
    seg_len = np.hypot(dx, dy)
    if seg_len < 1: return [], [], (0,1)
    ux, uy = dx/seg_len, dy/seg_len
    nx, ny = -uy, ux                # normal 90° CCW

    ts = np.linspace(0.1, 0.9, n)
    H_img, W_img = V_img.shape
    ss = np.arange(-half, half + 1, dtype=float)   # 81 posições

    profiles = []
    pts = []
    for t in ts:
        cx = x1 + t * dx
        cy = y1 + t * dy
        xs = np.clip(cx + ss * nx, 0, W_img - 1.001)
        ys = np.clip(cy + ss * ny, 0, H_img - 1.001)

        # Interpolação bilinear
        x0 = np.floor(xs).astype(int); x1_ = np.minimum(x0+1, W_img-1)
        y0 = np.floor(ys).astype(int); y1_ = np.minimum(y0+1, H_img-1)
        fx = xs - x0; fy = ys - y0
        vals = (V_img[y0, x0] * (1-fx) * (1-fy)
              + V_img[y0, x1_] *   fx  * (1-fy)
              + V_img[y1_, x0] * (1-fx) *   fy
              + V_img[y1_, x1_] *  fx  *   fy)
        profiles.append(vals)
        pts.append((float(cx), float(cy)))

    return profiles, pts, (float(nx), float(ny))


def compute_fwhm(profile, min_contrast=40, edge_margin=10):
    """FWHM a partir de um perfil 1D de V.

    Retorna (width_px, valid, info_str).
    V_fundo = média dos primeiros e últimos edge_margin valores (extremos do perfil).
    half = V_fundo + (V_pico - V_fundo)/2.
    Descarta se contraste < min_contrast, pico nas bordas ou região saturada uniforme.
    """
    arr = np.asarray(profile, dtype=float)
    n = len(arr)
    p = int(np.argmax(arr))

    if p < edge_margin or p > n - edge_margin - 1:
        return None, False, f"pico na borda (pos={p})"

    bg = float(np.mean(np.concatenate([arr[:edge_margin], arr[-edge_margin:]])))
    pk = float(arr[p])
    contrast = pk - bg

    if contrast < min_contrast:
        return None, False, f"contraste baixo ({contrast:.1f}<{min_contrast})"

    if pk >= 252 and bg >= 245:
        return None, False, f"saturado/uniforme (pk={pk:.0f} bg={bg:.0f})"

    half = bg + contrast / 2.0

    # Borda esquerda: busca do pico para a esquerda
    left = p
    for i in range(p-1, -1, -1):
        if arr[i] < half:
            # Interpolação linear entre i e i+1
            left = i + (arr[i+1] - half) / max(arr[i+1] - arr[i], 1e-6)
            break

    # Borda direita
    right = p
    for i in range(p+1, n):
        if arr[i] < half:
            right = (i-1) + (arr[i-1] - half) / max(arr[i-1] - arr[i], 1e-6)
            break

    width = right - left
    return float(width), True, f"V_pk={pk:.0f} V_bg={bg:.0f} half={half:.0f}"


# ── Main ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("(b) PERFIS PERPENDICULARES AOS SEGMENTOS HOUGH  (px imagem ORIGINAL)")
print("=" * 70)

FATOR = 8   # fator fixo do pipeline
raw_to_print = {}   # 3 por imagem: um por segmento

for nome in ['bitbots', 'eliza', 'image_1']:
    img = load(IMG_PATHS[nome])
    H_img, W_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V_img = hsv[:,:,2].astype(float)

    if nome in SEGS_FIXOS:
        segs = SEGS_FIXOS[nome]
        print(f"\n{nome} ({W_img}x{H_img}px)  — segmentos fornecidos:")
    else:
        segs = hough_segs(img)
        print(f"\n{nome} ({W_img}x{H_img}px)  — segmentos Hough calculados:")
        for i, s in enumerate(segs):
            print(f"  seg{i+1}: ({s[0]},{s[1]})→({s[2]},{s[3]}) "
                  f"len={np.hypot(s[2]-s[0],s[3]-s[1]):.0f}px")

    ann = img.copy()
    raw_to_print[nome] = []

    print(f"  {'Seg':>4} {'yCentro':>8} {'válidos/8':>10} "
          f"{'FWHM_med':>9} {'FWHM_min':>9} {'FWHM_max':>9} "
          f"{'FWHM_grade':>11}")
    print("  " + "-" * 65)

    for seg_i, (x1,y1,x2,y2) in enumerate(segs):
        y_center = (y1 + y2) / 2.0
        dx, dy = x2-x1, y2-y1
        # Desenhar segmento na anotação
        cv2.line(ann, (x1,y1),(x2,y2),(0,255,255),2)
        # Rótulo do segmento
        mx, my = (x1+x2)//2, (y1+y2)//2
        cv2.putText(ann, f"S{seg_i+1}", (mx+3,my-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        profiles, pts, (nx, ny) = sample_perpendicular(
            V_img, x1, y1, x2, y2, n=8, half=40)

        fwhms = []
        first_raw_printed = False

        for j, (prof, (cx, cy)) in enumerate(zip(profiles, pts)):
            w, valid, info = compute_fwhm(prof, min_contrast=40)
            color = (0,200,0) if valid else (0,0,200)

            # Desenhar linha perpendicular de ±40px
            px_s = (int(cx - 40*nx), int(cy - 40*ny))
            px_e = (int(cx + 40*nx), int(cy + 40*ny))
            # Clipar às bordas
            px_s = (np.clip(px_s[0],0,W_img-1), np.clip(px_s[1],0,H_img-1))
            px_e = (np.clip(px_e[0],0,W_img-1), np.clip(px_e[1],0,H_img-1))
            cv2.line(ann, px_s, px_e, color, 1)
            cv2.circle(ann, (int(cx),int(cy)), 2, (255,255,0), -1)

            if valid:
                fwhms.append(w)
                if not first_raw_printed:
                    raw_to_print[nome].append((seg_i+1, j, list(prof), info))
                    first_raw_printed = True

        n_valid = len(fwhms)
        if fwhms:
            med = float(np.median(fwhms))
            mn, mx_fw = min(fwhms), max(fwhms)
            print(f"  seg{seg_i+1}  y={y_center:>6.0f}  "
                  f"{n_valid:>5}/8  "
                  f"{med:>8.1f}px  {mn:>8.1f}px  {mx_fw:>8.1f}px  "
                  f"{med/FATOR:>8.2f}px_grade")
        else:
            print(f"  seg{seg_i+1}  y={y_center:>6.0f}  "
                  f"  0/8   todos descartados")

    # Salvar imagem anotada
    cv2.putText(ann,
        "AMARELO=seg  CIANO=rot_seg  VERDE=perf_valido  VERMELHO=perf_invalido",
        (5, H_img-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    out = os.path.join(OUT, f"{nome}_b3_perp.png")
    cv2.imwrite(out, ann)
    print(f"  Anotação salva: {out}")

print()
print("=" * 70)
print("PERFIS BRUTOS (3 por imagem, um por segmento — array V de 81 posições)")
print("posição s=0 → extremo negativo (−40px da normal), s=80 → extremo (+40px)")
print("=" * 70)

for nome in ['bitbots', 'eliza', 'image_1']:
    print(f"\n{nome}:")
    for (seg_i, prof_j, vals, info) in raw_to_print.get(nome, []):
        arr = [f"{v:.0f}" for v in vals[::5]]   # a cada 5 posições = 17 valores
        print(f"  seg{seg_i} perfil{prof_j} ({info}):")
        print(f"    s= -40,-35,...,+40: {arr}")
        # Também pico e largura
        v_arr = np.array(vals)
        peak_pos = int(np.argmax(v_arr)) - 40
        print(f"    pico em s={peak_pos:+d}px  V_max={v_arr.max():.0f}  V_min={v_arr.min():.0f}")
