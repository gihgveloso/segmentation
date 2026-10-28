"""Diagnóstico de fragmentação das linhas extraídas nas imagens reais — v2.

Mudanças em relação à v1:
  1. Todos os 4 critérios de merge avaliados por par (não só o primeiro que falha).
  2. Ramos de junção (ângulo > 20° em relação à direção principal da polilinha)
     separados da tabela de pares e contados à parte.
  3. Análise de densidade de bifurcações por célula de esqueleto:
     faixa de borda (< 10 células) vs interior.

Uso: python3 scripts/diagnostico/diag_fragmentacao.py
"""
import sys
import os
import json
import math
import io
import contextlib

import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment

from features.segmentacao_real import PixelSegmentReal
from features.pipeline import construir_grafo, ConfigGrafo
from features.lines import (
    extrair_linhas, ConfigLinhas,
    _ajustar_reta, _angulo_diff,
    _dist_ortogonal_retas, _pontos_mais_proximos, _frac_branco,
)

CALIB = os.path.join(ROOT, 'calibracao', 'real_provisoria_n2.json')
LUT   = os.path.join(ROOT, 'green_pixels.csv')
GAB   = os.path.join(ROOT, 'tests', 'gabarito_linhas.json')
OUT   = os.path.join(ROOT, 'debug_out')
os.makedirs(OUT, exist_ok=True)

IMGS_REAL = {
    'bitbots': os.path.join(ROOT, 'images',
               'bitbots_reality_spl_only_131-16_02_2018__'
               '11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(ROOT, 'images',
               'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
}
IMGS_WEBOTS = {
    'image_1': os.path.join(ROOT, 'images', 'image_1.png'),
}

FRAG_COLORS = [
    (255,  60,   0),
    (0,   200, 255),
    (200,   0, 255),
    (0,   255,  80),
    (255, 180,   0),
    (0,   100, 255),
    (255,   0, 120),
    (80,  255, 180),
]
GAP_COLOR     = (0, 0, 220)
BRANCH_COLOR  = (0, 140, 255)

TOL_GRADE      = 3.0
FRAC_MATCH     = 0.50
FATOR          = 8
ANGULO_RAMO_GR = 20.0   # fragmentos com ângulo > este em relação à poly são "ramos de junção"

# ── Geometria ──────────────────────────────────────────────────────────────────

def _sample_lf(lf, step=0.5):
    n = max(2, int(math.ceil(lf.distancia / step)) + 1)
    return np.column_stack([np.linspace(lf.x1, lf.x2, n),
                            np.linspace(lf.y1, lf.y2, n)]).astype(np.float32)


def _rasterize_poly_grade(poly_px, H_g, W_g):
    mask = np.zeros((H_g, W_g), dtype=np.uint8)
    pts = [(int(round(p[0] / FATOR)), int(round(p[1] / FATOR))) for p in poly_px]
    for i in range(len(pts) - 1):
        cv2.line(mask, pts[i], pts[i + 1], 255, 1)
    return mask


def _dist_to_mask(mask):
    if not (mask > 0).any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((mask == 0).astype(np.uint8),
                                 cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _lf_frac_on_poly(lf, d2poly, H_g, W_g):
    pts = _sample_lf(lf)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W_g - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H_g - 1)
    return float((d2poly[ys, xs] <= TOL_GRADE).mean())


def _poly_coverage(poly_px, frags, H_g, W_g):
    if not frags:
        return 0.0
    fmask = np.zeros((H_g, W_g), dtype=np.uint8)
    for lf in frags:
        for p in _sample_lf(lf):
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < W_g and 0 <= yi < H_g:
                fmask[yi, xi] = 255
    d2f = _dist_to_mask(fmask)
    pts_g = []
    for i in range(len(poly_px) - 1):
        x1g = poly_px[i][0] / FATOR;     y1g = poly_px[i][1] / FATOR
        x2g = poly_px[i+1][0] / FATOR;   y2g = poly_px[i+1][1] / FATOR
        n = max(2, int(math.ceil(math.hypot(x2g - x1g, y2g - y1g) / 0.5)))
        for j in range(n):
            t = j / n
            pts_g.append([x1g + t * (x2g - x1g), y1g + t * (y2g - y1g)])
    if poly_px:
        pts_g.append([poly_px[-1][0] / FATOR, poly_px[-1][1] / FATOR])
    if not pts_g:
        return 0.0
    pts = np.array(pts_g, dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W_g - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H_g - 1)
    return float((d2f[ys, xs] <= TOL_GRADE).mean())


def _poly_len_cel(poly_px):
    return sum(
        math.hypot((poly_px[i + 1][0] - poly_px[i][0]) / FATOR,
                   (poly_px[i + 1][1] - poly_px[i][1]) / FATOR)
        for i in range(len(poly_px) - 1)
    )


def _poly_dir_angle(poly_px):
    """Ângulo da direção principal da polilinha (primeiro→último ponto)."""
    dx = (poly_px[-1][0] - poly_px[0][0]) / FATOR
    dy = (poly_px[-1][1] - poly_px[0][1]) / FATOR
    return math.atan2(dy, dx)


def _sort_frags(frags, poly_px):
    if len(frags) <= 1:
        return list(frags)
    p0x, p0y = poly_px[0][0] / FATOR, poly_px[0][1] / FATOR
    p1x, p1y = poly_px[-1][0] / FATOR, poly_px[-1][1] / FATOR
    dx, dy = p1x - p0x, p1y - p0y
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return list(frags)
    dx /= L; dy /= L
    return sorted(frags, key=lambda lf: ((lf.x1 + lf.x2) / 2 - p0x) * dx
                                       + ((lf.y1 + lf.y2) / 2 - p0y) * dy)


# ── Análise de merge — todos os critérios ─────────────────────────────────────

def _analise_merge_full(lf_i, lf_j, branco, cfg):
    """Avalia TODOS os 4 critérios de merge, independentemente da ordem de falha."""
    pts_i = np.array([[lf_i.x1, lf_i.y1], [lf_i.x2, lf_i.y2]], dtype=np.float32)
    pts_j = np.array([[lf_j.x1, lf_j.y1], [lf_j.x2, lf_j.y2]], dtype=np.float32)

    r_i = _ajustar_reta(pts_i)
    r_j = _ajustar_reta(pts_j)

    ang_diff_rad  = _angulo_diff(math.atan2(r_i[1], r_i[0]),
                                 math.atan2(r_j[1], r_j[0]))
    ang_diff_grau = math.degrees(ang_diff_rad)
    dist_orto     = _dist_ortogonal_retas(pts_i, r_i, pts_j, r_j)
    pi, pj, gap   = _pontos_mais_proximos(pts_i, pts_j)
    frac_b        = (_frac_branco(float(pi[0]), float(pi[1]),
                                  float(pj[0]), float(pj[1]), branco)
                     if branco is not None else None)

    ok_ang  = ang_diff_rad    <= math.radians(cfg.limiar_angulo_fusao_grau)
    ok_orto = dist_orto       <= cfg.limiar_dist_ortogonal_fusao
    ok_vao  = gap             <= cfg.lacuna_max_celulas
    ok_sup  = (frac_b         >= cfg.suporte_vao_min) if frac_b is not None else True

    bloqueios = []
    if not ok_ang:  bloqueios.append('ANG')
    if not ok_orto: bloqueios.append('ORTO')
    if not ok_vao:  bloqueios.append('VÃO')
    if not ok_sup:  bloqueios.append('SUP')

    return {
        'gap_cel':         gap,
        'ang_diff_grau':   ang_diff_grau,
        'dist_orto_cel':   dist_orto,
        'frac_branco_vao': frac_b,
        'ok_ang':   ok_ang,  'ok_orto': ok_orto,
        'ok_vao':   ok_vao,  'ok_sup':  ok_sup,
        'bloqueios': bloqueios,
        'funde':    not bloqueios,
        'pi_g': (float(pi[0]), float(pi[1])),
        'pj_g': (float(pj[0]), float(pj[1])),
    }


def _fmt_crit(label, val, lim, unit, ok):
    tick = '✓' if ok else '✗'
    op   = '≤' if label in ('ANG', 'ORTO', 'VÃO') else '≥'
    return f'{label}={val:.1f}{unit}({tick}{op}{lim}{unit})'


# ── Nós nos vãos ───────────────────────────────────────────────────────────────

def _nos_perto_ponto(g, pt, raio=2.5):
    return [(nid, g.degree(nid), node.x, node.y)
            for nid, node in g.nodes.items()
            if math.hypot(node.x - pt[0], node.y - pt[1]) <= raio]


def _classificar_vao_compacto(g, m):
    """Retorna string descrevendo os nós nas extremidades do vão."""
    raio = 2.5
    nos_i = _nos_perto_ponto(g, m['pi_g'], raio)
    nos_j = _nos_perto_ponto(g, m['pj_g'], raio)
    todos = {n[0]: n for n in nos_i + nos_j}

    bifs  = [n for n in todos.values() if n[1] >= 3]
    morts = [n for n in todos.values() if n[1] == 1]

    partes = []
    if bifs:
        b = min(bifs, key=lambda n: math.hypot(n[2] - (m['pi_g'][0] + m['pj_g'][0]) / 2,
                                               n[3] - (m['pi_g'][1] + m['pj_g'][1]) / 2))
        partes.append(f'BIFURCAÇÃO nó{b[0]} grau{b[1]}@({b[2]},{b[3]})')
    if morts:
        d = min(morts, key=lambda n: math.hypot(n[2] - (m['pi_g'][0] + m['pj_g'][0]) / 2,
                                                n[3] - (m['pi_g'][1] + m['pj_g'][1]) / 2))
        partes.append(f'DEAD-END nó{d[0]}@({d[2]},{d[3]})')
    if not partes:
        partes.append('LACUNA NO ESQUELETO')
    return '  '.join(partes)


# ── Ramos de junção ───────────────────────────────────────────────────────────

def _nearest_bif(g, lf):
    """Nó de grau≥3 mais próximo do centroide de lf."""
    cx, cy = (lf.x1 + lf.x2) / 2, (lf.y1 + lf.y2) / 2
    best_d, best = float('inf'), None
    for nid, node in g.nodes.items():
        if g.degree(nid) >= 3:
            d = math.hypot(node.x - cx, node.y - cy)
            if d < best_d:
                best_d = d
                best = (nid, g.degree(nid), node.x, node.y, d)
    return best


# ── Densidade de bifurcações por célula de esqueleto ──────────────────────────

def _densidade_borda(g, skeleton, W_g, H_g, raio_borda=10):
    xs = np.arange(W_g, dtype=np.int32)
    ys = np.arange(H_g, dtype=np.int32)
    dist_x = np.minimum(xs, W_g - 1 - xs)
    dist_y = np.minimum(ys, H_g - 1 - ys)
    dist_map = np.minimum(dist_x[np.newaxis, :], dist_y[:, np.newaxis])

    borda_mask = dist_map < raio_borda
    sk = skeleton > 0

    n_sk_b = int((sk & borda_mask).sum())
    n_sk_i = int((sk & ~borda_mask).sum())

    n_bif_b = n_bif_i = 0
    for nid, node in g.nodes.items():
        if g.degree(nid) >= 3:
            if dist_map[node.y, node.x] < raio_borda:
                n_bif_b += 1
            else:
                n_bif_i += 1

    def d(n, s): return f'{n/s:.3f}' if s > 0 else 'n/a'

    return {
        'borda':     {'bif': n_bif_b, 'sk': n_sk_b, 'dens': d(n_bif_b, n_sk_b)},
        'interior':  {'bif': n_bif_i, 'sk': n_sk_i, 'dens': d(n_bif_i, n_sk_i)},
        'dist_map':  dist_map,
    }


# ── Overlay ────────────────────────────────────────────────────────────────────

def _save_overlay(img, sk_full, frags_per_poly, gaps_per_poly, branches_per_poly, nome):
    H, W = img.shape[:2]
    ov = (img.astype(np.float32) * 0.40).astype(np.uint8)

    sk_col = np.zeros_like(ov)
    sk_col[sk_full > 0] = (100, 120, 0)
    ov = cv2.addWeighted(ov, 1.0, sk_col, 0.5, 0)

    for pname, frags in frags_per_poly.items():
        for fi, lf in enumerate(frags):
            color = FRAG_COLORS[fi % len(FRAG_COLORS)]
            p1 = (int(round(lf.x1_orig)), int(round(lf.y1_orig)))
            p2 = (int(round(lf.x2_orig)), int(round(lf.y2_orig)))
            cv2.line(ov, p1, p2, color, 3, cv2.LINE_AA)
            cv2.circle(ov, p1, 4, color, -1)
            cv2.circle(ov, p2, 4, color, -1)
            cx = (p1[0] + p2[0]) // 2
            cy = (p1[1] + p2[1]) // 2 - 8
            cv2.putText(ov, f'{pname}f{fi + 1}',
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    for pname, branches in branches_per_poly.items():
        for lf in branches:
            p1 = (int(round(lf.x1_orig)), int(round(lf.y1_orig)))
            p2 = (int(round(lf.x2_orig)), int(round(lf.y2_orig)))
            cv2.line(ov, p1, p2, BRANCH_COLOR, 1, cv2.LINE_AA)

    for pname, gaps in gaps_per_poly.items():
        for m in gaps:
            pi_g, pj_g = m['pi_g'], m['pj_g']
            p1 = (int(round(pi_g[0] * FATOR)), int(round(pi_g[1] * FATOR)))
            p2 = (int(round(pj_g[0] * FATOR)), int(round(pj_g[1] * FATOR)))
            cv2.line(ov, p1, p2, GAP_COLOR, 1, cv2.LINE_AA)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.drawMarker(ov, (mx, my), GAP_COLOR, cv2.MARKER_CROSS, 10, 1)
            cv2.putText(ov, f'{m["gap_cel"]:.1f}c',
                        (mx + 4, my - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, GAP_COLOR, 1)

    out = os.path.join(OUT, f'{nome}_fragmentacao.png')
    cv2.imwrite(out, ov)
    return out


# ── Loop principal ─────────────────────────────────────────────────────────────

def run_image(nome, img, ps, gabarito, polys_filtro=None):
    H, W = img.shape[:2]
    H_g, W_g = H // FATOR, W // FATOR

    print(f"\n{'='*66}")
    print(f"=== {nome}  ({W}×{H}px  grade {W_g}×{H_g}) ===")
    print(f"{'='*66}")

    res      = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['white_mask_resized'].astype(np.uint8)
    verde    = res['green_mask_resized'].astype(np.uint8)

    g    = construir_grafo(skeleton, relevo, ConfigGrafo(), branco=branco)
    cfg  = ConfigLinhas()
    lines = extrair_linhas(g, fator_escala=FATOR, cfg=cfg, branco=branco, verde=verde)

    print(f"  linhas extraídas: {len(lines)}  "
          f"nós no grafo: {len(g.nodes)}")

    polys_gt = gabarito.get(nome, [])
    if polys_filtro is not None:
        polys_gt_sel = [polys_gt[i] for i in polys_filtro if i < len(polys_gt)]
    else:
        polys_gt_sel = polys_gt

    frags_per_poly:    dict = {}
    gaps_per_poly:     dict = {}
    branches_per_poly: dict = {}

    lim_ang_ramo = math.radians(ANGULO_RAMO_GR)

    for pi_idx, poly in enumerate(polys_gt_sel):
        orig_idx = polys_filtro[pi_idx] if polys_filtro else pi_idx
        pname    = f'p{orig_idx + 1}'

        mask_poly = _rasterize_poly_grade(poly, H_g, W_g)
        d2poly    = _dist_to_mask(mask_poly)
        poly_dir  = _poly_dir_angle(poly)
        poly_len  = _poly_len_cel(poly)

        matching = [(lf, _lf_frac_on_poly(lf, d2poly, H_g, W_g)) for lf in lines]
        matching = [(lf, f) for lf, f in matching if f >= FRAC_MATCH]

        # Separar ramos (ângulo > 20° em relação à direção da polilinha)
        main_frags, branches = [], []
        ang_vs_poly = {}
        for lf, f in matching:
            a = math.degrees(_angulo_diff(lf.angulo, poly_dir))
            ang_vs_poly[id(lf)] = a
            if a > ANGULO_RAMO_GR:
                branches.append(lf)
            else:
                main_frags.append(lf)

        main_sorted = _sort_frags(main_frags, poly)
        frac_map    = {id(lf): f for lf, f in matching}
        cob         = _poly_coverage(poly, main_sorted, H_g, W_g)

        print(f"\n  ── {pname}  len={poly_len:.1f}cel ({poly_len*FATOR:.0f}px)  "
              f"dir={math.degrees(poly_dir):.1f}°  "
              f"fragmentos_principais={len(main_sorted)}  "
              f"ramos={len(branches)}  cobertura={100*cob:.0f}%")

        # Fragmentos principais
        if main_sorted:
            print(f"  Fragmentos ao longo da linha principal (ângulo ≤ {ANGULO_RAMO_GR}°):")
            for fi, lf in enumerate(main_sorted):
                fr = frac_map.get(id(lf), 0.0)
                av = ang_vs_poly[id(lf)]
                print(f"    f{fi+1}: len={lf.distancia:.1f}cel({lf.distancia*FATOR:.0f}px) "
                      f"ang_vs_poly={av:.1f}°  conf={lf.confianca:.2f}  "
                      f"frac={100*fr:.0f}%  "
                      f"orig=[{lf.x1_orig:.0f},{lf.y1_orig:.0f}]→"
                      f"[{lf.x2_orig:.0f},{lf.y2_orig:.0f}]")
        else:
            print(f"  (nenhum fragmento principal)")

        # Ramos de junção
        if branches:
            print(f"  Ramos de junção (ângulo > {ANGULO_RAMO_GR}°):")
            for lf in branches:
                av  = ang_vs_poly[id(lf)]
                bif = _nearest_bif(g, lf)
                bif_str = (f'bif nó{bif[0]} grau{bif[1]}@({bif[2]},{bif[3]}) '
                           f'dist={bif[4]:.1f}cel') if bif else 'sem bif próxima'
                print(f"    ramo: len={lf.distancia:.1f}cel({lf.distancia*FATOR:.0f}px) "
                      f"ang_vs_poly={av:.1f}°  {bif_str}")

        frags_per_poly[pname]    = main_sorted
        branches_per_poly[pname] = branches

        # Tabela de pares — TODOS os critérios
        gaps = []
        if len(main_sorted) >= 2:
            print(f"\n  Tabela de pares (defaults: "
                  f"ang≤{cfg.limiar_angulo_fusao_grau}°  "
                  f"orto≤{cfg.limiar_dist_ortogonal_fusao}cel  "
                  f"vão≤{cfg.lacuna_max_celulas}cel  "
                  f"sup≥{cfg.suporte_vao_min}):")
            for fi in range(len(main_sorted) - 1):
                lf_a, lf_b = main_sorted[fi], main_sorted[fi + 1]
                m = _analise_merge_full(lf_a, lf_b, branco, cfg)
                gaps.append(m)

                ang_s  = _fmt_crit('ANG',  m['ang_diff_grau'],  cfg.limiar_angulo_fusao_grau, '°', m['ok_ang'])
                orto_s = _fmt_crit('ORTO', m['dist_orto_cel'],  cfg.limiar_dist_ortogonal_fusao, 'cel', m['ok_orto'])
                vao_s  = _fmt_crit('VÃO',  m['gap_cel'],        cfg.lacuna_max_celulas, 'cel', m['ok_vao'])
                fb     = m['frac_branco_vao']
                sup_s  = (_fmt_crit('SUP', fb, cfg.suporte_vao_min, '', m['ok_sup'])
                          if fb is not None else 'SUP=n/a')

                bloq_str = ', '.join(m['bloqueios']) if m['bloqueios'] else 'NENHUM (funde)'
                print(f"    f{fi+1}↔f{fi+2}: {ang_s}  {orto_s}  {vao_s}  {sup_s}"
                      f"  → {bloq_str}")
                vao_class = _classificar_vao_compacto(g, m)
                print(f"           {vao_class}")
        elif len(main_sorted) == 1:
            print(f"  (fragmento único — sem par a analisar)")
        else:
            print(f"  (nenhum fragmento principal)")

        gaps_per_poly[pname] = gaps

    # ── Bifurcações específicas do início do bitbots p1 (item 3) ──────────────
    if nome == 'bitbots' and polys_filtro is None:
        p1_bifs_esperadas = [(9,40), (8,42), (10,43), (6,42), (22,34), (19,36)]
        print(f"\n  [Distância à borda das bifurcações iniciais de p1]")
        for (bx, by) in p1_bifs_esperadas:
            d_borda = min(bx, W_g - 1 - bx, by, H_g - 1 - by)
            zona = 'BORDA' if d_borda < 10 else 'interior'
            print(f"    ({bx:2d},{by:2d}): dist_borda={d_borda}cel  [{zona}]")

    # ── Análise de densidade de bifurcações por célula de esqueleto ───────────
    dens = _densidade_borda(g, skeleton, W_g, H_g, raio_borda=10)
    b = dens['borda'];   i = dens['interior']
    print(f"\n  [Densidade de nós grau≥3 por célula de esqueleto — borda < 10cel vs interior]")
    print(f"    borda:    bif={b['bif']:3d}  sk_cells={b['sk']:4d}  dens={b['dens']} bif/cel")
    print(f"    interior: bif={i['bif']:3d}  sk_cells={i['sk']:4d}  dens={i['dens']} bif/cel")

    # Overlay
    sk_full = cv2.resize(res['skeleton_img'], (W, H), cv2.INTER_NEAREST)
    out = _save_overlay(img, sk_full, frags_per_poly, gaps_per_poly, branches_per_poly, nome)
    print(f"\n  overlay → {out}")


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main():
    gab = json.load(open(GAB, encoding='utf-8'))

    ps_real = PixelSegmentReal.from_json(CALIB, LUT)
    for nome, img_path in IMGS_REAL.items():
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERRO] imagem não encontrada: {img_path}")
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        run_image(nome, img, ps_real, gab)

    with contextlib.redirect_stdout(io.StringIO()):
        ps_webots = Pixel_Segment(LUT)

    for nome, img_path in IMGS_WEBOTS.items():
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERRO] imagem não encontrada: {img_path}")
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        run_image(nome, img, ps_webots, gab, polys_filtro=[0])


if __name__ == '__main__':
    main()
