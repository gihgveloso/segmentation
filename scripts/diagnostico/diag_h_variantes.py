"""Diagnóstico: variantes de H_max e H_min da máscara verde.

Seções:
  1. Tabela de métricas por variante (H_max 70-90 e H_min 40-50), eliza e bitbots
  2. Pipeline completo (esqueleto, nós, cruzamentos, linhas, cobertura gabarito,
     recall/precisão máscara B_moderado∩convexa) para variantes mais promissoras
  3. Overlays das variantes selecionadas
  4. Distribuição de H (percentis 1,5,50,95,99) para carpete e fundo

Critério de seleção (Seção 2):
  Menor fração verde na faixa 0-160 do eliza que NÃO reduz mais de 3pp as faixas
  320-640 em NENHUMA das duas imagens (vs. base H 35-100).
  Até 3 variantes reportadas (melhor, 2ª, 3ª colocadas).

Uso: python3 scripts/diagnostico/diag_h_variantes.py
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

with contextlib.redirect_stdout(io.StringIO()):
    import def_white_threshold as _bm
_bm.lower = W_LOWER.copy()
_bm.up    = W_UPPER.copy()

with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment

from features import extrair_features
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao


# ─── helpers máscara ──────────────────────────────────────────────────────────

def build_lut(h_min: int, h_max: int,
              s_min: int = 40, v_min: int = 40, v_max: int = 180) -> np.ndarray:
    lut = np.zeros((180, 256, 256), dtype=np.uint8)
    for h in range(h_min, min(h_max + 1, 180)):
        lut[h, s_min:256, v_min:v_max + 1] = 255
    return lut


def apply_green(hsv: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]


BANDAS = [(0, 160), (160, 320), (320, 480), (480, 640)]


def fracao_por_faixa(gm: np.ndarray) -> list[float]:
    H, W = gm.shape[:2]
    total_col = W * 160
    return [float(int((gm[y0:y1, :] > 0).sum()) / total_col) for y0, y1 in BANDAS]


def run_pipeline(img: np.ndarray, lut: np.ndarray) -> dict:
    ps = Pixel_Segment(LUT_CSV)
    ps.lut_verde = lut
    return ps.processar(img)


# ─── helpers avaliação ────────────────────────────────────────────────────────

def sample_poly(poly: list, step: float = 1.0) -> np.ndarray:
    pts: list = []
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


def dist_to_nonzero(binary: np.ndarray) -> np.ndarray:
    if not (binary > 0).any():
        return np.full(binary.shape, np.inf, np.float32)
    return cv2.distanceTransform((binary == 0).astype(np.uint8),
                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def rasterize_polys(polys: list, H: int, W: int) -> np.ndarray:
    cl = np.zeros((H, W), np.uint8)
    for poly in polys:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in poly]
        for i in range(len(pts) - 1):
            cv2.line(cl, pts[i], pts[i+1], 255, 1)
    return cl


def recall_cl(polys, dist_to_mask, tol, H, W) -> tuple[float, list[float]]:
    total = hit = 0
    per: list[float] = []
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


def precision_d(mask, dist_to_poly, d) -> float:
    mb = mask > 0
    n = int(mb.sum())
    if n == 0:
        return 0.0
    return float(int((dist_to_poly[mb] <= d).sum()) / n)


def rasterize_lines(lines, H, W) -> np.ndarray:
    """Rasteriza LineFeatures (coords originais) numa máscara HxW."""
    img = np.zeros((H, W), np.uint8)
    for lf in lines:
        p1 = (int(round(lf.x1_orig)), int(round(lf.y1_orig)))
        p2 = (int(round(lf.x2_orig)), int(round(lf.y2_orig)))
        cv2.line(img, p1, p2, 255, 1)
    return img


def cobertura_gabarito(polys, lines, H, W, d_match=25) -> list[float]:
    """Fração de cada polilinha do gabarito coberta (a ≤ d_match px) pelas lines extraídas."""
    if not lines:
        return [0.0] * len(polys)
    lines_mask = rasterize_lines(lines, H, W)
    dist_lines  = dist_to_nonzero(lines_mask)
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
    """Para cada LineFeature, atribui índice da polilinha mais próxima (ou -1)."""
    if not polys:
        return [-1] * len(lines)
    # Distância de cada pixel ao centroide de cada polilinha
    dist_mats = []
    for poly in polys:
        cl = rasterize_polys([poly], H, W)
        dist_mats.append(dist_to_nonzero(cl))
    assigns = []
    for lf in lines:
        # Amostra a linha extraída
        p1 = [lf.x1_orig, lf.y1_orig]
        p2 = [lf.x2_orig, lf.y2_orig]
        pts = sample_poly([p1, p2], step=2.0)
        xs = np.clip(pts[:, 0].round().astype(int), 0, W-1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, H-1)
        min_dists = [float(dm[ys, xs].mean()) for dm in dist_mats]
        best_idx = int(np.argmin(min_dists))
        assigns.append(best_idx if min_dists[best_idx] <= d_match else -1)
    return assigns


# ─── carrega imagens e gabarito ───────────────────────────────────────────────

print('Carregando imagens e gabarito...', flush=True)
img_e = cv2.imread(ELIZA_PATH)
img_b = cv2.imread(BITBOTS_PATH)
hsv_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2HSV)
hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)

with open(GABARITO_PATH, encoding='utf-8') as f:
    gabarito = json.load(f)
polys_e = gabarito['eliza']    # 3 polilinhas
polys_b = gabarito['bitbots']  # 3 polilinhas

cl_e = rasterize_polys(polys_e, 640, 640)
cl_b = rasterize_polys(polys_b, 640, 640)
dtp_e = dist_to_nonzero(cl_e)  # dist de cada px à centerline eliza
dtp_b = dist_to_nonzero(cl_b)  # dist de cada px à centerline bitbots


# ── Variantes ─────────────────────────────────────────────────────────────────

VARIANTS: list[tuple[str, int, int]] = [
    ('base  H 35-100', 35, 100),
    ('Hmax  H 35-70',  35,  70),
    ('Hmax  H 35-75',  35,  75),
    ('Hmax  H 35-80',  35,  80),
    ('Hmax  H 35-85',  35,  85),
    ('Hmax  H 35-90',  35,  90),
    ('Hmin  H 40-100', 40, 100),
    ('Hmin  H 45-100', 45, 100),
    ('Hmin  H 50-100', 50, 100),
]


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1: Tabela de métricas por variante
# ══════════════════════════════════════════════════════════════════════════════

print('Seção 1: executando pipeline para todas as variantes...\n', flush=True)

# Armazena resultados de todas as variantes para reuso na seção 2
res_s1: dict[str, dict] = {}

for nome, h_min, h_max in VARIANTS:
    print(f'  {nome}...', flush=True, end=' ')
    lut = build_lut(h_min, h_max)
    row: dict[str, dict] = {}
    for img_nome, img, hsv in [('eliza', img_e, hsv_e), ('bitbots', img_b, hsv_b)]:
        gm  = apply_green(hsv, lut)
        res = run_pipeline(img, lut)
        row[img_nome] = {
            'fracs':  fracao_por_faixa(gm),
            'yg_min': float(res['y_gauss'].min()),
            'mc_pct': float((res['mascara_convexa'] > 0).mean()) * 100,
            'campo':  res['campo_detectado'],
            'res':    res,
            'gm':     gm,
            'lut':    lut,
        }
    res_s1[nome] = row
    print('ok', flush=True)

# Base de referência para critério de seleção
fracs_base_e = res_s1['base  H 35-100']['eliza']['fracs']
fracs_base_b = res_s1['base  H 35-100']['bitbots']['fracs']

print()
print('=' * 90)
print('SEÇÃO 1  Métricas por variante de H')
print('=' * 90)

for img_nome, fracs_base in [('eliza', fracs_base_e), ('bitbots', fracs_base_b)]:
    print(f'\n  [{img_nome}]')
    hdr = (f"  {'Variante':17s}  "
           f"{'0-160':>8}  {'160-320':>8}  {'320-480':>8}  {'480-640':>8}  "
           f"{'yg_min':>7}  {'mc_%':>5}  campo")
    print(hdr)
    print('  ' + '-' * 82)
    for nome, h_min, h_max in VARIANTS:
        r  = res_s1[nome][img_nome]
        fr = r['fracs']
        # Sinaliza se viola a restrição de 3pp nas faixas 320-640
        flags = ''
        for bi in [2, 3]:
            if fr[bi] < fracs_base[bi] - 0.03:
                flags += f' ↓f{bi+1}'
        print(f"  {nome:17s}  "
              f"{fr[0]:8.4f}  {fr[1]:8.4f}  {fr[2]:8.4f}  {fr[3]:8.4f}  "
              f"{r['yg_min']:7.2f}  {r['mc_pct']:5.1f}%  "
              f"{'S' if r['campo'] else 'N':1s}{flags}")

print()
print('  Restrição: faixas 320-640 ≥ base − 3pp em ambas as imagens. Violações: ↓f3/↓f4.')


# ══════════════════════════════════════════════════════════════════════════════
# Critério de seleção (automático)
# ══════════════════════════════════════════════════════════════════════════════

def passa_criterio(nome) -> bool:
    for img_nome, fracs_base in [('eliza', fracs_base_e), ('bitbots', fracs_base_b)]:
        fr = res_s1[nome][img_nome]['fracs']
        for bi in [2, 3]:  # bandas 320-480 e 480-640
            if fr[bi] < fracs_base[bi] - 0.03:
                return False
    return True

# Ordena por fração 0-160 do eliza (menor = melhor), excluindo a base
candidatos = [
    (nome, res_s1[nome]['eliza']['fracs'][0])
    for nome, _, _ in VARIANTS
    if nome != 'base  H 35-100' and passa_criterio(nome)
]
candidatos.sort(key=lambda x: x[1])

TOP3 = [nome for nome, _ in candidatos[:3]]

print()
print(f'  Candidatos que passam o critério (ordenados por fração 0-160 eliza):')
for nome, frac in candidatos:
    marcador = '→' if nome in TOP3 else ' '
    print(f'  {marcador} {nome}: eliza 0-160 = {frac:.4f}')
print(f'  Selecionados para Seção 2: {TOP3}')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2: Pipeline completo para variantes selecionadas
# ══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 90)
print('SEÇÃO 2  Pipeline completo (esqueleto, nós, cruzamentos, linhas, gabarito)')
print('=' * 90)

# Parâmetros de avaliação (B_moderado = branco V≥165 S≤60)
B_LOWER = np.array([  0,  0, 165], np.uint8)
B_UPPER = np.array([179, 60, 255], np.uint8)
TOL_E, D_E = 3, 8    # recall tol e precisão d para eliza
TOL_B, D_B = 3, 15   # recall tol e precisão d para bitbots
D_MATCH = 25         # px orig: tolerância para cobertura e classificação de linhas

for sel_nome in TOP3:
    h_min_sel = next(h for n, h, _ in VARIANTS if n == sel_nome)
    h_max_sel = next(h for n, _, h in VARIANTS if n == sel_nome)
    print(f'\n  ┌── {sel_nome} ─────────────────────────────────────────────')

    for img_nome, img, hsv, polys, dtp, tol, d_prec in [
        ('eliza',   img_e, hsv_e, polys_e, dtp_e, TOL_E, D_E),
        ('bitbots', img_b, hsv_b, polys_b, dtp_b, TOL_B, D_B),
    ]:
        res  = res_s1[sel_nome][img_nome]['res']
        H, W = img.shape[:2]

        # Esqueleto
        skel_px = int(res['skeleton_img'].astype(bool).sum())

        # Grafo (pre-simplificação) — para contar nós
        skel   = res['skeleton_img'].astype(bool)
        relevo = res['mascara_relevo'].astype(np.float32)
        branco = res['white_mask_resized'].astype(np.uint8)
        g_pre  = construir_grafo(skel, relevo, ConfigGrafo(), branco=branco)
        n_nos_pre = len(g_pre.nodes)
        # Post-simplificação
        g_post = construir_grafo(skel, relevo, ConfigGrafo(), branco=branco)
        simplificar(g_post, skel, relevo, ConfigSimplificacao())
        n_nos_post = len(g_post.nodes)

        # Features
        feat = extrair_features(res)
        n_L = sum(1 for c in feat.crossings if c.tipo == 'L')
        n_T = sum(1 for c in feat.crossings if c.tipo == 'T')
        n_X = sum(1 for c in feat.crossings if c.tipo == 'X')
        n_linhas = len(feat.lines)

        # Cobertura do gabarito pelas linhas extraídas
        cob = cobertura_gabarito(polys, feat.lines, H, W, d_match=D_MATCH)

        # Classificação das linhas extraídas por grupo do gabarito
        assigns = classificar_linhas(feat.lines, polys, H, W, d_match=D_MATCH)
        grupos = {i: 0 for i in range(len(polys))}
        grupos[-1] = 0
        for a in assigns:
            grupos[a] = grupos.get(a, 0) + 1

        # Recall / precisão — B_moderado ∩ mascara_convexa
        b_mask  = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), B_LOWER, B_UPPER)
        mc_orig = cv2.resize(res['mascara_convexa'], (W, H), cv2.INTER_NEAREST)
        mask_mc = cv2.bitwise_and(b_mask, mc_orig)
        dtm_mc  = dist_to_nonzero(mask_mc)
        rec_mc, per_poly_mc = recall_cl(polys, dtm_mc, tol, H, W)
        prc_mc  = precision_d(mask_mc, dtp, d_prec)
        f1_mc   = 2*rec_mc*prc_mc / (rec_mc+prc_mc) if rec_mc+prc_mc > 0 else 0.0

        print(f'  │  [{img_nome}]')
        print(f'  │    esqueleto: {skel_px} px')
        print(f'  │    nós: pre-simp={n_nos_pre}  post-simp={n_nos_post}')
        print(f'  │    cruzamentos: L={n_L}  T={n_T}  X={n_X}  total={n_L+n_T+n_X}')
        print(f'  │    linhas extraídas: {n_linhas}')
        print(f'  │    cobertura gabarito (d≤{D_MATCH}px):')
        for pi, cov in enumerate(cob):
            print(f'  │      poly_{pi}: {cov:.1%}')
        print(f'  │    B_moderado∩convexa: Rec={rec_mc:.1%}  Prec={prc_mc:.1%}  F1={f1_mc:.3f}')
        pp_str = '  '.join(f'p{i}={v:.0%}' for i, v in enumerate(per_poly_mc))
        print(f'  │      rec_por_poly: {pp_str}')
        print(f'  │    linhas por grupo do gabarito (d≤{D_MATCH}px):')
        for gi in range(len(polys)):
            print(f'  │      poly_{gi} (principal se 0, ramo se >0): {grupos.get(gi, 0)} linhas')
        print(f'  │      sem correspondência: {grupos.get(-1, 0)} linhas')

    print(f'  └{"─"*60}')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3: Overlays das variantes selecionadas
# ══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 90)
print('SEÇÃO 3  Overlays')
print('=' * 90)

for sel_nome in TOP3:
    tag = sel_nome.strip().replace(' ', '_').replace('-', '')
    for img_nome, img in [('eliza', img_e), ('bitbots', img_b)]:
        d  = res_s1[sel_nome][img_nome]
        gm = d['gm']
        res = d['res']

        mc_orig = cv2.resize(res['mascara_convexa'], (640, 640), cv2.INTER_NEAREST)
        verde_layer = np.zeros_like(img)
        verde_layer[gm > 0] = (0, 200, 0)
        ov = cv2.addWeighted(img.copy(), 0.7, verde_layer, 0.3, 0)
        contornos, _ = cv2.findContours(mc_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, contornos, -1, (0, 0, 255), 2)
        cv2.putText(ov, sel_nome, (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        fname = f'{img_nome}_verde_{tag}.png'
        cv2.imwrite(os.path.join(OUT, fname), ov)
        print(f'  Salvo: debug_out/{fname}')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4: Distribuição de H — carpete e fundo (pixels da imagem inteira)
# ══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 90)
print('SEÇÃO 4  Distribuição de H — carpete (linhas 400-640) e fundo (linhas 0-160)')
print('         Fonte: pixels na máscara verde base (H 35-100, S≥40, V 40-180)')
print('=' * 90)

lut_base = build_lut(35, 100)
PCTS = [1, 5, 50, 95, 99]

print(f'\n  {"Imagem":8s}  {"Classe":10s}  {"N_px":>8}  '
      + '  '.join(f'p{p:02d}' for p in PCTS))
print('  ' + '-' * 62)

for img_nome, img, hsv in [('eliza', img_e, hsv_e), ('bitbots', img_b, hsv_b)]:
    gm_base = apply_green(hsv, lut_base)
    H_ch    = hsv[:, :, 0]  # canal H

    # Carpete: linhas 400-640, verde
    mask_carp = gm_base[400:, :]   > 0
    h_carp    = H_ch[400:, :][mask_carp]

    # Fundo: linhas 0-160, verde (falsos positivos hipotéticos)
    mask_fund = gm_base[:160, :]   > 0
    h_fund    = H_ch[:160, :][mask_fund]

    for classe, h_vals in [('carpete', h_carp), ('fundo', h_fund)]:
        n = len(h_vals)
        if n == 0:
            vals_str = '   —  ' * len(PCTS)
        else:
            vals_str = '  '.join(f'{int(np.percentile(h_vals, p)):4d}' for p in PCTS)
        print(f'  {img_nome:8s}  {classe:10s}  {n:8d}  {vals_str}')

print()
