"""Diagnóstico Fase 1 — itens (c), (d), (e).

(d) Máscaras branco+verde com 3 conjuntos candidatos; salva side-by-side em debug_out/.
(c) Pipeline simulado (fator 8/4/2) com cada conjunto; report tabular.
(e) Branco relativo: top-hat e V/mediana_campo vs faixa fixa; só medição, sem implementar.

Uso: python3 scripts/diagnostico/diag_fase1_cde.py
"""
import sys, os
import cv2
import numpy as np
from scipy import ndimage

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
OUT  = os.path.join(BASE, 'debug_out')
os.makedirs(OUT, exist_ok=True)

IMAGES = {
    'bitbots': os.path.join(BASE, 'images', 'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(BASE, 'images', 'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
    'image_1': os.path.join(BASE, 'images', 'image_1.png'),
}

# 3 conjuntos candidatos de calibração
# Medido: branco linha real H=0-173,S=0-70,V=150-211; verde real H=30-110,V=25-180
CALIBS = {
    'A_liberal': {
        'white': dict(H=(0,180), S=(0,80), V=(150,255)),
        'green': dict(H=(30,110), S=(25,255), V=(25,200)),
    },
    'B_moderado': {
        'white': dict(H=(0,180), S=(0,60), V=(165,255)),
        'green': dict(H=(35,100), S=(40,255), V=(40,180)),
    },
    'C_verde_aperto': {
        'white': dict(H=(0,180), S=(0,70), V=(155,255)),
        'green': dict(H=(45,95), S=(70,255), V=(50,190)),
    },
}

FATORES = [8, 4, 2]
TROW  = 10000   # padrão main.py
TPIX  = 20
TWIN  = 3000
REF_W = 94      # grid_width_referencia (main.py linha 15)


def aplicar_mascara_hsv(hsv, cdef):
    lo = np.array([cdef['H'][0], cdef['S'][0], cdef['V'][0]], dtype=np.uint8)
    hi = np.array([cdef['H'][1], cdef['S'][1], cdef['V'][1]], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def skeletonizar(bin_img):
    """Replica _skeletonization_and_connect do main.py (detecção de cristas)."""
    relevo = cv2.GaussianBlur(bin_img.astype(np.uint8), (5, 5), 0).astype(np.float32)
    padded = np.pad(relevo, ((1,1),(1,1)), mode='constant', constant_values=0)
    vizinhos = np.stack([
        padded[:-2, 1:-1], padded[2:, 1:-1],
        padded[1:-1, 2:],  padded[1:-1, :-2],
        padded[:-2, 2:],   padded[2:, 2:],
        padded[:-2, :-2],  padded[2:, :-2],
    ])
    c_xy = np.sum(vizinhos >= relevo[np.newaxis], axis=0).astype(np.uint8)
    sk = (relevo > 0) & (c_xy < 3)
    return sk.astype(np.uint8) * 255


def simular_pipeline(img_bgr, calib, fator):
    """Simula as etapas relevantes do Pixel_Segment para um dado fator."""
    H, W = img_bgr.shape[:2]
    gH, gW = H // fator, W // fator
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    white_orig = aplicar_mascara_hsv(hsv, calib['white'])
    green_orig = aplicar_mascara_hsv(hsv, calib['green'])

    # Reproduz exatamente _masks_and_resize + _binarization do main.py
    white_res = cv2.resize(white_orig, (gW, gH), interpolation=cv2.INTER_AREA).astype(np.float32)
    green_res = cv2.resize(green_orig, (gW, gH), interpolation=cv2.INTER_AREA).astype(np.float32)

    # uniq_img = white*0.5 + green, clipped (main.py linha 87-98)
    uniq = np.clip(white_res * 0.5 + green_res, 0, 255).astype(np.uint8)

    # row_sums_1d = soma_linha + soma_linha_anterior (main.py linhas 106-117)
    trow_ef = TROW * gW / REF_W
    soma = uniq.sum(axis=1).astype(float)         # soma raw (uint8 values, NOT /255)
    row_sums_1d = soma + np.roll(soma, 1)
    row_sums_1d[0] = soma[0]
    campo = bool(row_sums_1d.max() >= trow_ef)

    # tpix gate (simplificado: só white)
    tpix_ok = bool((white_res > 0).sum() >= TPIX)

    sk_px = 0
    n_nodes = 0
    n_cross = 0

    if campo and tpix_ok:
        # field_line_mask: white dentro do campo (approx: threshold uniq para campo mask)
        _, campo_bin = cv2.threshold(uniq, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gW//4, gH//4))
        campo_conv = cv2.morphologyEx(campo_bin, cv2.MORPH_CLOSE, kernel)
        campo_conv = cv2.morphologyEx(campo_conv, cv2.MORPH_DILATE, kernel)
        campo_conv = cv2.dilate(campo_conv, np.ones((3,3),np.uint8), iterations=5)
        white_bin = (white_res > 127).astype(np.uint8) * 255
        flm = cv2.bitwise_and(white_bin, campo_conv)

        # skeleton
        try:
            sk = skeletonizar(flm)
            sk_px = int((sk > 0).sum())

            if sk_px > 0:
                # contar nós (pixel com >=3 vizinhos 8-conectados no esqueleto)
                sk_bool = sk > 0
                neigh = sum(
                    np.roll(np.roll(sk_bool, dy, 0), dx, 1)
                    for dy in [-1,0,1] for dx in [-1,0,1]
                    if (dy,dx) != (0,0)
                )
                n_nodes = int((neigh[sk_bool] >= 3).sum())

        except Exception as e:
            sk_px = -1
            n_nodes = -1

    return {
        'campo': campo,
        'tpix_ok': tpix_ok,
        'trow_ef': trow_ef,
        'row_max': float(row_sums_1d.max()),
        'white_px': int((white_res > 0).sum()),
        'green_px': int((green_res > 0).sum()),
        'sk_px': sk_px,
        'n_nodes': n_nodes,
        'gH': gH, 'gW': gW,
    }


# ── (d) Máscaras side-by-side ──────────────────────────────────────────────
print("=" * 60)
print("(d) Máscaras candidatas — 3 conjuntos × 2 imagens reais")
print("=" * 60)

for img_nome in ['bitbots', 'eliza', 'image_1']:
    img_path = IMAGES[img_nome]
    img = cv2.imread(img_path)
    if img is None:
        print(f"  SKIP {img_nome} (não encontrado)")
        continue
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = img.shape[:2]
    gH, gW = H // 8, W // 8

    panels = []
    labels = []

    for cal_nome, calib in CALIBS.items():
        white = aplicar_mascara_hsv(hsv, calib['white'])
        green = aplicar_mascara_hsv(hsv, calib['green'])

        # Visualização: RGB overlay sobre imagem original
        vis = img.copy()
        vis[white > 0] = (255, 255, 255)   # branco puro onde detectado
        vis[green > 0] = (0, 200, 0)        # verde onde detectado

        # Adicionar legenda no topo
        n_w = int((white > 0).sum())
        n_g = int((green > 0).sum())
        lbl = f"{cal_nome} | W={n_w} G={n_g}"
        cv2.putText(vis, lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        panels.append(vis)
        labels.append(cal_nome)
        print(f"  {img_nome} / {cal_nome}: white={n_w}px  green={n_g}px")

    # Concatenar horizontalmente
    row = np.concatenate(panels, axis=1)
    out_path = os.path.join(OUT, f"{img_nome}_d_calibs.png")
    cv2.imwrite(out_path, row)
    print(f"  → {out_path}")
    print()


# ── (c) Fatores de subamostragem ───────────────────────────────────────────
print("=" * 60)
print("(c) Fator 8/4/2 × calibrações × imagens reais")
print("    (fator 8 hardcoded em main.py:44-45 — simulado aqui)")
print("=" * 60)
print(f"{'Imagem':<10} {'Calib':<20} {'Fat':>3} {'gHxgW':>10} {'campo':>6} {'W_px':>7} {'G_px':>7} {'row_max/trow_ef':>17} {'sk_px':>7} {'nós':>5}")
print("-" * 100)

for img_nome in ['bitbots', 'eliza', 'image_1']:
    img_path = IMAGES[img_nome]
    img = cv2.imread(img_path)
    if img is None:
        continue
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    for cal_nome, calib in CALIBS.items():
        for fator in FATORES:
            r = simular_pipeline(img, calib, fator)
            ratio_str = f"{r['row_max']:.0f}/{r['trow_ef']:.0f}"
            print(f"{img_nome:<10} {cal_nome:<20} {fator:>3} {r['gH']:>4}x{r['gW']:<5} "
                  f"{'SIM' if r['campo'] else 'NAO':>6} "
                  f"{r['white_px']:>7} {r['green_px']:>7} "
                  f"{ratio_str:>17} "
                  f"{r['sk_px']:>7} {r['n_nodes']:>5}")
    print()


# ── (e) Branco relativo ────────────────────────────────────────────────────
print("=" * 60)
print("(e) Branco relativo: top-hat e V/mediana_campo")
print("    (só medição — linhas conhecidas: bitbots y=464, eliza y=450, image_1 várias)")
print("=" * 60)

# Para cada imagem: medir V_linha, V_mediana_campo_verde, top-hat no ponto
for img_nome, y_linha in [('bitbots', 464), ('eliza', 450), ('image_1', 392)]:
    img_path = IMAGES[img_nome]
    img = cv2.imread(img_path)
    if img is None:
        continue
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2].astype(float)

    # Máscara verde (set B como proxy do campo)
    calib_b = CALIBS['B_moderado']
    green_mask = aplicar_mascara_hsv(hsv, calib_b['green'])

    V_mediana_campo = float(np.median(V[green_mask > 0])) if (green_mask > 0).any() else 0.0

    # Valor V na linha
    row_V = V[y_linha, :]
    V_linha_max = float(row_V.max())
    V_linha_mean = float(row_V[row_V > 100].mean()) if (row_V > 100).any() else 0.0

    # Top-hat morfológico: remove estruturas grandes (campo), deixa linhas
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    V8 = V.astype(np.uint8)
    tophat = cv2.morphologyEx(V8, cv2.MORPH_TOPHAT, kernel_tophat)
    tophat_linha = float(tophat[y_linha, :].max())
    tophat_campo = float(np.median(tophat[green_mask > 0])) if (green_mask > 0).any() else 0.0

    # Razão V_linha / V_mediana_campo
    ratio = V_linha_max / V_mediana_campo if V_mediana_campo > 0 else float('inf')

    print(f"  {img_nome} (y={y_linha}):")
    print(f"    V_mediana_campo(verde): {V_mediana_campo:.1f}")
    print(f"    V_linha_max:            {V_linha_max:.1f}")
    print(f"    V_linha_max / V_campo:  {ratio:.2f}×")
    print(f"    top-hat@linha (max):    {tophat_linha:.1f}")
    print(f"    top-hat@campo (median): {tophat_campo:.1f}")
    print(f"    top-hat SNR:            {tophat_linha / (tophat_campo + 1):.2f}")
    print()

    # Salvar mapa top-hat
    tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    tophat_col = cv2.applyColorMap(tophat_norm, cv2.COLORMAP_HOT)
    cv2.line(tophat_col, (0, y_linha), (tophat_col.shape[1], y_linha), (0,255,0), 1)
    cv2.imwrite(os.path.join(OUT, f"{img_nome}_e_tophat.png"), tophat_col)

    # Comparar faixas: fixo vs relativo
    # Faixa fixa (atual): V>=213
    # Faixa relativa: V >= 1.5 × V_mediana_campo
    k = 1.5
    V_thr_relativo = V_mediana_campo * k
    mascara_fixo    = (V >= 213).astype(np.uint8)
    mascara_relativo = (V >= V_thr_relativo).astype(np.uint8)
    mascara_tophat  = (tophat >= 20).astype(np.uint8)

    # Medir na faixa de linha (±15px ao redor de y_linha)
    y0, y1 = max(0, y_linha-15), min(V.shape[0], y_linha+16)
    n_linha_px = int((V[y0:y1,:] > 150).sum())  # proxy pixels de linha

    n_fixo     = int(mascara_fixo[y0:y1,:].sum())
    n_relativo = int(mascara_relativo[y0:y1,:].sum())
    n_tophat_m = int(mascara_tophat[y0:y1,:].sum())
    n_total_campo_fixo = int(mascara_fixo.sum())
    n_total_campo_rel  = int(mascara_relativo.sum())
    n_total_campo_th   = int(mascara_tophat.sum())

    print(f"    Pixels na faixa ±15px y={y_linha} (linha):")
    print(f"      V>=213 (fixo):             {n_fixo:5d}  total={n_total_campo_fixo}")
    print(f"      V>={V_thr_relativo:.0f} (1.5×campo):     {n_relativo:5d}  total={n_total_campo_rel}")
    print(f"      top-hat>=20:               {n_tophat_m:5d}  total={n_total_campo_th}")
    print()

print("Arquivos novos em debug_out/:")
for f in sorted(os.listdir(OUT)):
    if '_d_calibs' in f or '_e_tophat' in f:
        print(f"  debug_out/{f}")
