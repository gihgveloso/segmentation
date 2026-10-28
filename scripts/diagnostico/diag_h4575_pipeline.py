"""Pipeline completo com calibração H 45-75 (real_provisoria_n2.json atualizado).

Reporta: frações verde, y_gauss, convexa, esqueleto, nós, cruzamentos, linhas,
cobertura do gabarito, avaliar_mascara_branco B∩convexa e classificação das linhas.
Gera sobreposições em debug_out/.

Uso: python3 scripts/diagnostico/diag_h4575_pipeline.py
"""
from __future__ import annotations
import sys, os, io, contextlib, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import cv2

BASE  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT   = os.path.join(BASE, 'debug_out')
os.makedirs(OUT, exist_ok=True)

ELIZA_PATH   = os.path.join(BASE, 'images', 'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg')
BITBOTS_PATH = os.path.join(BASE, 'images',
    'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg')
LUT_CSV       = os.path.join(BASE, 'green_pixels.csv')
GABARITO_PATH = os.path.join(BASE, 'tests', 'gabarito_linhas.json')

W_LOWER = np.array([  0,  0, 165], dtype=np.int32)
W_UPPER = np.array([179, 60, 255], dtype=np.int32)
H_MIN, H_MAX = 45, 75

with contextlib.redirect_stdout(io.StringIO()):
    import def_white_threshold as _bm
_bm.lower = W_LOWER.copy()
_bm.up    = W_UPPER.copy()
with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment

from features import extrair_features
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao


# ─── helpers ──────────────────────────────────────────────────────────────────

def build_lut(h_min=H_MIN, h_max=H_MAX, s_min=40, v_min=40, v_max=180):
    lut = np.zeros((180, 256, 256), dtype=np.uint8)
    for h in range(h_min, h_max + 1):
        lut[h, s_min:256, v_min:v_max + 1] = 255
    return lut


def apply_green(hsv, lut):
    return lut[hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]


BANDAS = [(0, 160), (160, 320), (320, 480), (480, 640)]


def fracao_por_faixa(gm):
    return [float((gm[y0:y1, :] > 0).sum()) / (640 * (y1 - y0))
            for y0, y1 in BANDAS]


def sample_poly(poly, step=1.0):
    pts = []
    for i in range(len(poly) - 1):
        x1, y1 = float(poly[i][0]),   float(poly[i][1])
        x2, y2 = float(poly[i+1][0]), float(poly[i+1][1])
        n = max(1, int(np.ceil(np.hypot(x2-x1, y2-y1) / step)))
        for j in range(n):
            t = j / n
            pts.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
    if poly:
        pts.append([float(poly[-1][0]), float(poly[-1][1])])
    return np.array(pts, np.float32) if pts else np.zeros((0, 2), np.float32)


def dist_to_nonzero(binary):
    if not (binary > 0).any():
        return np.full(binary.shape, np.inf, np.float32)
    return cv2.distanceTransform((binary == 0).astype(np.uint8),
                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def rasterize_polys(polys, H, W):
    cl = np.zeros((H, W), np.uint8)
    for poly in polys:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in poly]
        for i in range(len(pts) - 1):
            cv2.line(cl, pts[i], pts[i+1], 255, 1)
    return cl


def recall_cl(polys, dist_to_mask, tol, H, W):
    total = hit = 0
    per = []
    for poly in polys:
        pts = sample_poly(poly)
        if len(pts) == 0:
            per.append(0.0); continue
        xs = np.clip(pts[:, 0].round().astype(int), 0, W-1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, H-1)
        n_hit = int((dist_to_mask[ys, xs] <= tol).sum())
        per.append(n_hit / len(pts))
        total += len(pts); hit += n_hit
    return (hit / total if total > 0 else 0.0), per


def precision_d(mask, dist_to_poly, d):
    mb = mask > 0
    n = int(mb.sum())
    if n == 0:
        return 0.0
    return float(int((dist_to_poly[mb] <= d).sum()) / n)


def rasterize_lines(lines, H, W):
    img = np.zeros((H, W), np.uint8)
    for lf in lines:
        p1 = (int(round(lf.x1_orig)), int(round(lf.y1_orig)))
        p2 = (int(round(lf.x2_orig)), int(round(lf.y2_orig)))
        cv2.line(img, p1, p2, 255, 1)
    return img


def cobertura_gabarito(polys, lines, H, W, d_match=25):
    if not lines:
        return [0.0] * len(polys)
    dist_lines = dist_to_nonzero(rasterize_lines(lines, H, W))
    cob = []
    for poly in polys:
        pts = sample_poly(poly, step=1.0)
        if len(pts) == 0:
            cob.append(0.0); continue
        xs = np.clip(pts[:, 0].round().astype(int), 0, W-1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, H-1)
        cob.append(float((dist_lines[ys, xs] <= d_match).mean()))
    return cob


def classificar_linhas(lines, polys, H, W, d_match=25):
    if not polys:
        return [-1] * len(lines)
    dist_mats = [dist_to_nonzero(rasterize_polys([poly], H, W)) for poly in polys]
    assigns = []
    for lf in lines:
        pts = sample_poly([[lf.x1_orig, lf.y1_orig], [lf.x2_orig, lf.y2_orig]], step=2.0)
        xs = np.clip(pts[:, 0].round().astype(int), 0, W-1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, H-1)
        min_dists = [float(dm[ys, xs].mean()) for dm in dist_mats]
        best = int(np.argmin(min_dists))
        assigns.append(best if min_dists[best] <= d_match else -1)
    return assigns


# ─── carrega ──────────────────────────────────────────────────────────────────

img_e = cv2.imread(ELIZA_PATH)
img_b = cv2.imread(BITBOTS_PATH)
hsv_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2HSV)
hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)

with open(GABARITO_PATH, encoding='utf-8') as f:
    gabarito = json.load(f)
polys_e = gabarito['eliza']
polys_b = gabarito['bitbots']

cl_e = rasterize_polys(polys_e, 640, 640)
cl_b = rasterize_polys(polys_b, 640, 640)
dtp_e = dist_to_nonzero(cl_e)
dtp_b = dist_to_nonzero(cl_b)

B_LOWER = np.array([  0,  0, 165], np.uint8)
B_UPPER = np.array([179, 60, 255], np.uint8)
D_MATCH = 25
TOL_E, D_E = 3, 8
TOL_B, D_B = 3, 15

# ─── pipeline ─────────────────────────────────────────────────────────────────

lut = build_lut()
print(f'Calibração: H {H_MIN}–{H_MAX}, S≥40, V 40–180\n')

SCALE = 8  # fator grade→original

results = {}
for img_nome, img, hsv, polys, dtp, tol, d_prec in [
    ('eliza',   img_e, hsv_e, polys_e, dtp_e, TOL_E, D_E),
    ('bitbots', img_b, hsv_b, polys_b, dtp_b, TOL_B, D_B),
]:
    H, W = img.shape[:2]

    gm  = apply_green(hsv, lut)
    ps  = Pixel_Segment(LUT_CSV)
    ps.lut_verde = lut
    res = ps.processar(img)

    # Frações verdes
    fracs = fracao_por_faixa(gm)
    yg_min = float(res['y_gauss'].min())
    mc_pct = float((res['mascara_convexa'] > 0).mean()) * 100
    campo  = res['campo_detectado']
    skel_px = int(res['skeleton_img'].astype(bool).sum())

    # Grafo — nós pre e post simplificação
    skel   = res['skeleton_img'].astype(bool)
    relevo = res['mascara_relevo'].astype(np.float32)
    branco = res['white_mask_resized'].astype(np.uint8)

    g_pre = construir_grafo(skel, relevo, ConfigGrafo(), branco=branco)
    n_nos_pre = len(g_pre.nodes)
    g_post = construir_grafo(skel, relevo, ConfigGrafo(), branco=branco)
    simplificar(g_post, skel, relevo, ConfigSimplificacao())
    n_nos_post = len(g_post.nodes)

    # Features
    feat = extrair_features(res)
    n_L = sum(1 for c in feat.crossings if c.tipo == 'L')
    n_T = sum(1 for c in feat.crossings if c.tipo == 'T')
    n_X = sum(1 for c in feat.crossings if c.tipo == 'X')
    n_linhas = len(feat.lines)

    # Cobertura do gabarito
    cob = cobertura_gabarito(polys, feat.lines, H, W, D_MATCH)

    # Classificação das linhas por grupo
    assigns = classificar_linhas(feat.lines, polys, H, W, D_MATCH)
    grupos_cont = {i: 0 for i in range(-1, len(polys))}
    grupos_comp = {i: 0.0 for i in range(-1, len(polys))}
    for a, lf in zip(assigns, feat.lines):
        grupos_cont[a] = grupos_cont.get(a, 0) + 1
        grupos_comp[a] = grupos_comp.get(a, 0.0) + lf.distancia

    # B_moderado ∩ mascara_convexa
    b_mask  = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), B_LOWER, B_UPPER)
    mc_orig = cv2.resize(res['mascara_convexa'], (W, H), cv2.INTER_NEAREST)
    mask_mc = cv2.bitwise_and(b_mask, mc_orig)
    dtm_mc  = dist_to_nonzero(mask_mc)
    rec_mc, per_poly_mc = recall_cl(polys, dtm_mc, tol, H, W)
    prc_mc  = precision_d(mask_mc, dtp, d_prec)
    f1_mc   = 2*rec_mc*prc_mc / (rec_mc+prc_mc) if rec_mc+prc_mc > 0 else 0.0

    results[img_nome] = dict(
        fracs=fracs, yg_min=yg_min, mc_pct=mc_pct, campo=campo,
        skel_px=skel_px, n_nos_pre=n_nos_pre, n_nos_post=n_nos_post,
        n_L=n_L, n_T=n_T, n_X=n_X, n_linhas=n_linhas,
        cob=cob, assigns=assigns, linhas=feat.lines,
        grupos_cont=grupos_cont, grupos_comp=grupos_comp,
        rec_mc=rec_mc, prc_mc=prc_mc, f1_mc=f1_mc, per_poly_mc=per_poly_mc,
        res=res, gm=gm,
    )

# ─── imprime ──────────────────────────────────────────────────────────────────

print('=' * 72)
print('Fração da máscara verde por faixa (640×640)')
print('=' * 72)
print(f"\n  {'Faixa':9s}  {'eliza':>10}  {'bitbots':>10}")
print('  ' + '-' * 35)
for i, (y0, y1) in enumerate(BANDAS):
    print(f"  {y0:3d}–{y1:3d}   "
          f"{results['eliza']['fracs'][i]:10.4f}  "
          f"{results['bitbots']['fracs'][i]:10.4f}")
print()
for img_nome in ['eliza', 'bitbots']:
    r = results[img_nome]
    print(f"  {img_nome}: y_gauss_min={r['yg_min']:.2f}  mc={r['mc_pct']:.1f}%  "
          f"campo={r['campo']}")

print()
print('=' * 72)
print('Pipeline completo')
print('=' * 72)
for img_nome in ['eliza', 'bitbots']:
    r = results[img_nome]
    n_polys = len(gabarito[img_nome])
    print(f"\n  [{img_nome}]")
    print(f"    esqueleto             : {r['skel_px']} px de grade")
    print(f"    nós (pre-simp)        : {r['n_nos_pre']}")
    print(f"    nós (post-simp)       : {r['n_nos_post']}")
    print(f"    cruzamentos L/T/X     : {r['n_L']}/{r['n_T']}/{r['n_X']}"
          f"  total={r['n_L']+r['n_T']+r['n_X']}")
    print(f"    linhas extraídas      : {r['n_linhas']}")
    print(f"    cobertura gabarito (d≤{D_MATCH}px orig):")
    for pi, c in enumerate(r['cob']):
        print(f"      poly_{pi}: {c:.1%}")
    print(f"    B_mod∩convexa: Rec={r['rec_mc']:.1%}  Prec={r['prc_mc']:.1%}  "
          f"F1={r['f1_mc']:.3f}")
    pp = '  '.join(f'p{i}={v:.0%}' for i, v in enumerate(r['per_poly_mc']))
    print(f"      rec_por_poly: {pp}")
    print(f"    linhas por grupo (d≤{D_MATCH}px orig):")
    poly_names = ['principal (poly_0)', 'ramo 1 (poly_1)', 'ramo 2 (poly_2)']
    for gi in range(n_polys):
        n = r['grupos_cont'].get(gi, 0)
        comp_cel = r['grupos_comp'].get(gi, 0.0)
        comp_px  = comp_cel * SCALE
        lbl = poly_names[gi] if gi < len(poly_names) else f'poly_{gi}'
        print(f"      {lbl:26s}: {n} linhas  {comp_cel:.1f} cel  {comp_px:.0f} px orig")
    n_sc = r['grupos_cont'].get(-1, 0)
    c_sc = r['grupos_comp'].get(-1, 0.0)
    print(f"      {'sem correspondência':26s}: {n_sc} linhas  {c_sc:.1f} cel  "
          f"{c_sc*SCALE:.0f} px orig")

# ─── sobreposições ────────────────────────────────────────────────────────────

print()
print('=' * 72)
print('Sobreposições')
print('=' * 72)

COR_L = (0, 165, 255)
COR_T = (50, 220, 50)
COR_X = (0, 220, 255)
_CORES_LINHA = [
    (0, 200, 255), (0, 255, 100), (255, 100, 0),
    (200, 0, 255), (0, 160, 255), (255, 200, 0),
    (100, 255, 200), (255, 0, 160), (150, 150, 255),
    (0, 255, 200), (255, 150, 0), (180, 0, 255),
]

for img_nome, img in [('eliza', img_e), ('bitbots', img_b)]:
    r     = results[img_nome]
    res   = r['res']
    gm    = r['gm']
    feat  = type('F', (), {'crossings': [], 'lines': r['linhas']})()
    # reextract crossings from fresh pipeline (already done above)
    feat_full = extrair_features(res)
    H, W  = img.shape[:2]

    mc_orig = cv2.resize(res['mascara_convexa'], (W, H), cv2.INTER_NEAREST)

    # Sobreposição 1: máscara verde + contorno convexa
    verde_layer = np.zeros_like(img)
    verde_layer[gm > 0] = (0, 200, 0)
    ov1 = cv2.addWeighted(img.copy(), 0.7, verde_layer, 0.3, 0)
    cnts, _ = cv2.findContours(mc_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov1, cnts, -1, (0, 0, 255), 2)
    cv2.putText(ov1, f'H {H_MIN}-{H_MAX}  verde+convexa',
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    out1 = os.path.join(OUT, f'{img_nome}_h4575_verde_convexa.png')
    cv2.imwrite(out1, ov1)
    print(f'  Salvo: debug_out/{img_nome}_h4575_verde_convexa.png')

    # Sobreposição 2: features (esqueleto + cruzamentos + linhas)
    ov2 = img.copy()
    skel_up = cv2.resize(res['skeleton_img'].astype(np.uint8),
                         (W, H), interpolation=cv2.INTER_NEAREST)
    ov2[skel_up > 0] = (210, 210, 210)
    cv2.drawContours(ov2, cnts, -1, (0, 0, 220), 1)

    for i, lf in enumerate(feat_full.lines):
        cor = _CORES_LINHA[i % len(_CORES_LINHA)]
        p1 = (int(round(lf.x1_orig)), int(round(lf.y1_orig)))
        p2 = (int(round(lf.x2_orig)), int(round(lf.y2_orig)))
        cv2.line(ov2, p1, p2, cor, 2)
        cv2.circle(ov2, p1, 4, cor, -1)
        cv2.circle(ov2, p2, 4, cor, -1)

    for cr in feat_full.crossings:
        cor = {'L': COR_L, 'T': COR_T, 'X': COR_X}.get(cr.tipo, (128,128,128))
        cx, cy = int(cr.x * SCALE), int(cr.y * SCALE)
        cv2.circle(ov2, (cx, cy), 7, cor, 2)
        cv2.putText(ov2, cr.tipo, (cx+4, cy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cor, 1, cv2.LINE_AA)

    cv2.putText(ov2, f'H {H_MIN}-{H_MAX}  features',
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    out2 = os.path.join(OUT, f'{img_nome}_h4575_features.png')
    cv2.imwrite(out2, ov2)
    print(f'  Salvo: debug_out/{img_nome}_h4575_features.png')

print()
