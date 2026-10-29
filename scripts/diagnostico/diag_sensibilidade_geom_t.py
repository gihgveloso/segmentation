#!/usr/bin/env python3
"""Diagnóstico: sensibilidade de _verifica_geometria_t (angulo_stem e distancia_arco).

Tarefas:
1. Direção dos braços a diferentes distancia_arco_geom_px para os 2 nós T reais.
2. Varredura: angulo_stem_t_max_grau x distancia_arco_geom_px.
3. Contagem de T rejeitados por geometria por imagem.
4. Análise dos 2 T FN do Webots em image_2.
"""
import math
import os
import sys
import json
from copy import deepcopy
from itertools import product

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import Pixel_Segment
from features.segmentacao_real import PixelSegmentReal
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import (
    simplificar, ConfigSimplificacao,
    remover_diagonais_espurias, remover_nos_curvatura_baixa,
    remover_espurios, fundir_juncoes, fusao_juncoes_cluster, fusao_t_par,
)
from features.crossings import (
    ConfigCruzamento, detectar_cruzamentos, classificar_candidatos,
    _verifica_geometria_t, _caminhar_ate_arco,
    _verificacao1_multi_raio, _verificacao2_caminho,
)
from features.graph import StructureGraph

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT = os.path.join(os.path.dirname(__file__), '../..')

WEBOTS_IMAGES = [
    ('image_1', os.path.join(ROOT, 'images/image_1.png')),
    ('image_2', os.path.join(ROOT, 'images/image_2.png')),
    ('image',   os.path.join(ROOT, 'images/image.png')),
]
REAL_IMAGES = [
    ('bitbots', os.path.join(ROOT,
                             'images/bitbots_reality_spl_only_131-16_02_2018__'
                             '11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg')),
    ('eliza',   os.path.join(ROOT,
                             'images/eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg')),
]
CALIB = os.path.join(ROOT, 'calibracao/real_provisoria_n2.json')

with open(os.path.join(ROOT, 'tests/gabarito_cruzamentos.json')) as _f:
    GAB_WEBOTS: dict = json.load(_f)
with open(os.path.join(ROOT, 'tests/gabarito_cruzamentos_reais.json')) as _f:
    GAB_REAIS: dict = json.load(_f)

TIPOS = ('L', 'T', 'X')

# ─── Pipeline helpers ─────────────────────────────────────────────────────────

def _processar_imagem(img_path: str, ps):
    """Retorna dict{skeleton, relevo, branco, verde} ou None."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    res = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['white_mask_resized'].astype(np.uint8)
    verde    = res['green_mask_resized'].astype(np.uint8)
    if skeleton.sum() == 0:
        return None
    return dict(skeleton=skeleton, relevo=relevo, branco=branco, verde=verde)


def _construir_e_simplificar(data: dict):
    """Constrói e simplifica o grafo; retorna (g, branco, verde)."""
    g = construir_grafo(data['skeleton'], data['relevo'], ConfigGrafo(),
                        branco=data['branco'])
    simplificar(g, data['skeleton'], data['relevo'], ConfigSimplificacao())
    return g, data['branco'], data['verde']


# ─── Casamento ───────────────────────────────────────────────────────────────

def _casar_loc(gt, det, tol=6.0):
    if not gt and not det: return 0, 0, 0, []
    if not gt: return 0, len(det), 0, []
    if not det: return 0, 0, len(gt), []
    pares = []
    for ig, gp in enumerate(gt):
        for id_, dp in enumerate(det):
            d = math.hypot(gp['x'] - dp.x, gp['y'] - dp.y)
            if d <= tol:
                pares.append((d, ig, id_))
    pares.sort()
    ug, ud = set(), set()
    pt = []
    for d, ig, id_ in pares:
        if ig in ug or id_ in ud: continue
        ug.add(ig); ud.add(id_)
        pt.append((gt[ig]['tipo'], det[id_].tipo))
    tp = len(pt)
    fp = len(det) - len(ud)
    fn = len(gt) - len(ug)
    return tp, fp, fn, pt


def _casar_tipo(gt, det, tipo, tol=6.0):
    gt_t  = [p for p in gt if p.get('tipo') == tipo]
    det_t = [d for d in det if d.tipo == tipo]
    if not gt_t and not det_t: return 0, 0, 0
    if not gt_t: return 0, len(det_t), 0
    if not det_t: return 0, 0, len(gt_t)
    pares = []
    for ig, gp in enumerate(gt_t):
        for id_, dp in enumerate(det_t):
            d = math.hypot(gp['x'] - dp.x, gp['y'] - dp.y)
            if d <= tol:
                pares.append((d, ig, id_))
    pares.sort()
    ug, ud = set(), set()
    tp = 0
    for d, ig, id_ in pares:
        if ig in ug or id_ in ud: continue
        ug.add(ig); ud.add(id_); tp += 1
    fp = len(det_t) - len(ud)
    fn = len(gt_t)  - len(ug)
    return tp, fp, fn


def _nearest_node(g: StructureGraph, gx, gy):
    best_nid, best_d = None, float('inf')
    for nid, nd in g.nodes.items():
        d = math.hypot(nd.x - gx, nd.y - gy)
        if d < best_d:
            best_d = d; best_nid = nid
    if best_nid is None:
        return None, None, None, float('inf')
    nd = g.nodes[best_nid]
    return best_nid, nd.x, nd.y, best_d


# ─── Carregar imagens ─────────────────────────────────────────────────────────

print("Iniciando carregamento de imagens...", flush=True)
print("  Pixel_Segment Webots...", flush=True)
ps_webots = Pixel_Segment(os.path.join(ROOT, 'green_pixels.csv'))
print("  PixelSegmentReal...", flush=True)
ps_real   = PixelSegmentReal.from_json(CALIB, os.path.join(ROOT, 'green_pixels.csv'))

# Dados brutos (pré-grafo) — necessários para tarefa 4 (simplificação passo a passo)
raw_webots: dict[str, dict] = {}
raw_reais:  dict[str, dict] = {}
for nome, path in WEBOTS_IMAGES:
    if not os.path.exists(path):
        print(f"  {nome}: não encontrado"); continue
    d = _processar_imagem(path, ps_webots)
    if d: raw_webots[nome] = d; print(f"  {nome}: ok")

for nome, path in REAL_IMAGES:
    if not os.path.exists(path):
        print(f"  {nome}: não encontrado"); continue
    d = _processar_imagem(path, ps_real)
    if d: raw_reais[nome] = d; print(f"  {nome}: ok")

# Grafos simplificados prontos para uso
grafos_webots: dict[str, tuple] = {}
grafos_reais:  dict[str, tuple] = {}
for nome, data in raw_webots.items():
    grafos_webots[nome] = _construir_e_simplificar(data)
for nome, data in raw_reais.items():
    grafos_reais[nome]  = _construir_e_simplificar(data)

print()

# ─── SEÇÃO 1: Direção dos braços em escala de arco ───────────────────────────

ALVOS_REAIS = [
    ('bitbots', 35, 55, 'L'),
    ('bitbots',  8, 41, 'T'),
    ('eliza',   30, 54, 'L'),
    ('eliza',   48, 43, 'T'),
]
D_ARCOS = [0, 5, 7, 10]
STEM_MAX_VARREDURA = [130, 135, 140, 150, 170]

print("=" * 70)
print("SEÇÃO 1: DIREÇÃO DOS BRAÇOS A DIFERENTES distancia_arco_geom_px")
print("=" * 70)
print()

for nome_img, gx, gy, tipo_gt in ALVOS_REAIS:
    if nome_img not in grafos_reais:
        print(f"[AVISO] {nome_img} não disponível, pulando")
        continue
    g, branco, verde = grafos_reais[nome_img]
    nid, nx, ny, dist = _nearest_node(g, gx, gy)
    if nid is None:
        print(f"  Nenhum nó encontrado para {nome_img} ({gx},{gy})"); continue
    deg = g.degree(nid)
    print(f"─── {nome_img} GT({gx},{gy}) tipo={tipo_gt} → nó ({nx},{ny}) grau={deg} dist={dist:.1f} cél ───")

    vizinhos = g.neighbors(nid)
    n = g.nodes[nid]

    for d_arco in D_ARCOS:
        print(f"  d_arco={d_arco}:")
        vecs = []
        far_pts = []
        for i, v in enumerate(vizinhos):
            if d_arco > 0:
                far = _caminhar_ate_arco(g, nid, v, float(d_arco))
            else:
                far = v
            nf = g.nodes[far]
            far_deg = g.degree(far)
            dy, dx = nf.y - n.y, nf.x - n.x
            norm = math.hypot(dy, dx) or 1.0
            vecs.append((dy / norm, dx / norm))
            far_pts.append((nf.x, nf.y, far_deg))
            geo_dist = math.hypot(nf.x - nx, nf.y - ny)
            print(f"    braço{i} → ({nf.x},{nf.y}) grau={far_deg} dist_geo={geo_dist:.1f} cél")

        if deg == 3:
            pairs = [(0,1,2),(0,2,1),(1,2,0)]
            barra_rad = math.radians(140.0)
            stem_max_rad = math.radians(130.0)
            stem_min_rad = math.radians(50.0)
            passou = False
            for i, j, k in pairs:
                dot_ij = vecs[i][0]*vecs[j][0] + vecs[i][1]*vecs[j][1]
                ang_ij = math.degrees(math.acos(max(-1., min(1., dot_ij))))
                barra_ok = ang_ij >= 140.0
                dot_ki = vecs[k][0]*vecs[i][0] + vecs[k][1]*vecs[i][1]
                dot_kj = vecs[k][0]*vecs[j][0] + vecs[k][1]*vecs[j][1]
                ang_ki = math.degrees(math.acos(max(-1., min(1., dot_ki))))
                ang_kj = math.degrees(math.acos(max(-1., min(1., dot_kj))))
                barra_str = "BARRA OK" if barra_ok else f"insuf({ang_ij:.1f}°<140)"
                stem_ki_ok = 50.0 <= ang_ki <= 130.0
                stem_kj_ok = 50.0 <= ang_kj <= 130.0
                ki_str = f"{ang_ki:.1f}° {'OK' if stem_ki_ok else 'FALHA'}"
                kj_str = f"{ang_kj:.1f}° {'OK' if stem_kj_ok else 'FALHA'}"
                print(f"    Par {i}↔{j}: {ang_ij:.1f}° [{barra_str}]"
                      f"  stem(braço{k})↔{i}: {ki_str}  stem(braço{k})↔{j}: {kj_str}")
                if barra_ok and stem_ki_ok and stem_kj_ok:
                    passou = True
            # Verificar com config temporário (d_arco variável)
            cfg_tmp = ConfigCruzamento()
            cfg_tmp.distancia_arco_geom_px = d_arco
            cfg_tmp.angulo_stem_t_max_grau = 130.0
            resultado = _verifica_geometria_t(g, nid, cfg_tmp)
            assert resultado == passou, f"divergência: calc={passou}, func={resultado}"
            print(f"    → _verifica_geometria_t(stem_max=130°) = {resultado}")
        elif deg == 2:
            i, j = 0, 1
            dot = vecs[i][0]*vecs[j][0] + vecs[i][1]*vecs[j][1]
            ang = math.degrees(math.acos(max(-1., min(1., dot))))
            print(f"    Ângulo grau-2: {ang:.1f}°")
    print()


# ─── SEÇÃO 2: Varredura stem_max × d_arco ────────────────────────────────────

print("=" * 70)
print("SEÇÃO 2: VARREDURA angulo_stem_t_max_grau × distancia_arco_geom_px")
print("=" * 70)
print()

# Juntar GT em dict por (nome, tipo)
def _gt_list(gabarito, nome):
    return gabarito.get(nome, [])

ResultadoLinha = dict  # {stem_max, arco, W_*, R_*, fp_t_reais}

linhas: list[ResultadoLinha] = []

for stem_max, d_arco in product(STEM_MAX_VARREDURA, D_ARCOS):
    cfg = ConfigCruzamento()
    cfg.angulo_stem_t_max_grau = float(stem_max)
    cfg.distancia_arco_geom_px = d_arco

    # Acumuladores Webots
    w_det = w_tp_loc = w_tp_tipo = w_fp = w_fn = 0
    w_t_det = w_t_tp = 0

    for nome, (g, branco, verde) in grafos_webots.items():
        gt = _gt_list(GAB_WEBOTS, nome)
        if not gt: continue
        aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)
        tp_loc, fp, fn, pt = _casar_loc(gt, aceitos, 6.0)
        tp_tipo = sum(1 for tg, td in pt if tg == td)
        w_det     += len(aceitos)
        w_tp_loc  += tp_loc
        w_tp_tipo += tp_tipo
        w_fp      += fp
        w_fn      += fn
        tp_t, fp_t, fn_t = _casar_tipo(gt, aceitos, 'T', 6.0)
        w_t_det += len([d for d in aceitos if d.tipo == 'T'])
        w_t_tp  += tp_t

    # Acumuladores Reais
    r_det = r_tp_loc = r_tp_tipo = r_fp = r_fn = 0
    r_t_det = r_t_tp = 0
    fp_t_reais: list[str] = []

    for nome, (g, branco, verde) in grafos_reais.items():
        gt = _gt_list(GAB_REAIS, nome)
        if not gt: continue
        aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)
        tp_loc, fp, fn, pt = _casar_loc(gt, aceitos, 6.0)
        tp_tipo = sum(1 for tg, td in pt if tg == td)
        r_det     += len(aceitos)
        r_tp_loc  += tp_loc
        r_tp_tipo += tp_tipo
        r_fp      += fp
        r_fn      += fn
        tp_t, fp_t, fn_t = _casar_tipo(gt, aceitos, 'T', 6.0)
        r_t_det += len([d for d in aceitos if d.tipo == 'T'])
        r_t_tp  += tp_t
        # Registrar T FP (detectados como T mas sem GT T próximo)
        gt_t = [p for p in gt if p.get('tipo') == 'T']
        det_t = [d for d in aceitos if d.tipo == 'T']
        usados_det: set[int] = set()
        pares = []
        for ig, gp in enumerate(gt_t):
            for id_, dp in enumerate(det_t):
                d = math.hypot(gp['x'] - dp.x, gp['y'] - dp.y)
                if d <= 6.0: pares.append((d, ig, id_))
        pares.sort()
        ug, ud = set(), set()
        for d, ig, id_ in pares:
            if ig in ug or id_ in ud: continue
            ug.add(ig); ud.add(id_); usados_det.add(id_)
        for id_, dp in enumerate(det_t):
            if id_ not in usados_det:
                fp_t_reais.append(f"{nome}({dp.x},{dp.y})conf={dp.confianca:.2f}")

    linhas.append(dict(
        stem_max=stem_max, d_arco=d_arco,
        w_det=w_det, w_tp_loc=w_tp_loc, w_tp_tipo=w_tp_tipo,
        w_fp=w_fp, w_fn=w_fn, w_t_det=w_t_det, w_t_tp=w_t_tp,
        r_det=r_det, r_tp_loc=r_tp_loc, r_tp_tipo=r_tp_tipo,
        r_fp=r_fp, r_fn=r_fn, r_t_det=r_t_det, r_t_tp=r_t_tp,
        fp_t_reais=fp_t_reais,
    ))

# Imprimir tabela
HDR = (f"{'stem_max':>8} {'arco':>5}"
       f" | {'W_Det':>5} {'W_TP_l':>6} {'W_TP_t':>6} {'W_FP':>5} {'W_FN':>5} {'W_Tdet':>6} {'W_Ttt':>6}"
       f" | {'R_Det':>5} {'R_TP_l':>6} {'R_TP_t':>6} {'R_FP':>5} {'R_FN':>5} {'R_Tdet':>6} {'R_Ttt':>6}"
       f" | FP_T_Reais")
print(HDR)
print("-" * len(HDR))
for r in linhas:
    base = (f"{r['stem_max']:>8} {r['d_arco']:>5}"
            f" | {r['w_det']:>5} {r['w_tp_loc']:>6} {r['w_tp_tipo']:>6} {r['w_fp']:>5} {r['w_fn']:>5}"
            f" {r['w_t_det']:>6} {r['w_t_tp']:>6}"
            f" | {r['r_det']:>5} {r['r_tp_loc']:>6} {r['r_tp_tipo']:>6} {r['r_fp']:>5} {r['r_fn']:>5}"
            f" {r['r_det']:>6} {r['r_t_tp']:>6}")
    fp_str = "; ".join(r['fp_t_reais']) if r['fp_t_reais'] else "—"
    print(f"{base} | {fp_str}")
print()

print("Legenda: W=Webots GT=12, R=Reais GT=4; tol=6 cél; stem_max=angulo_stem_t_max_grau; arco=distancia_arco_geom_px")
print("TP_l=TP loc, TP_t=TP tipo, Tdet=T detectados, Ttt=T TP tipo, FP_T_Reais=T FP nas imagens reais")
print()


# ─── SEÇÃO 3: Contagem de rejeições por geometria ────────────────────────────

print("=" * 70)
print("SEÇÃO 3: CONTAGEM DE REJEIÇÕES POR _verifica_geometria_t (config default)")
print("=" * 70)
print()

cfg_default = ConfigCruzamento()

# Todos os GT agrupados por imagem
all_gt = {**{k: v for k, v in GAB_WEBOTS.items()},
          **{k: v for k, v in GAB_REAIS.items()}}

def _analise_rejeicoes(nome, g: StructureGraph, gt_img: list[dict]):
    """Conta nós grau-3 rejeitados por geometria e proximidade a GT."""
    total_grau3 = 0
    rejeitados_geom: list[tuple[int, int]] = []
    for nid, nd in g.nodes.items():
        if g.degree(nid) == 3:
            total_grau3 += 1
            if not _verifica_geometria_t(g, nid, cfg_default):
                rejeitados_geom.append((nd.x, nd.y))

    # Contar rejeitados próximos a qualquer GT (tol=6)
    prox_qualquer_gt = 0
    prox_gt_t = 0
    for (rx, ry) in rejeitados_geom:
        for gp in gt_img:
            if math.hypot(gp['x'] - rx, gp['y'] - ry) <= 6.0:
                prox_qualquer_gt += 1
                break
        for gp in gt_img:
            if gp.get('tipo') == 'T' and math.hypot(gp['x'] - rx, gp['y'] - ry) <= 6.0:
                prox_gt_t += 1
                break
    return total_grau3, len(rejeitados_geom), prox_qualquer_gt, prox_gt_t, rejeitados_geom


print(f"{'Imagem':<12} {'Grau3':>6} {'RejGeom':>8} {'PercRej':>8} {'ProxGT(any)':>12} {'ProxGT(T)':>10}")
print("-" * 60)

for nome, (g, branco, verde) in list(grafos_webots.items()) + list(grafos_reais.items()):
    gt_img = all_gt.get(nome, [])
    g3, rej, prox_any, prox_t, _ = _analise_rejeicoes(nome, g, gt_img)
    pct = 100 * rej / g3 if g3 > 0 else 0
    print(f"{nome:<12} {g3:>6} {rej:>8} {pct:>7.0f}% {prox_any:>12} {prox_t:>10}")

# Listar posições dos rejeitados que estão próximos de GT-T
print()
print("Rejeitados grau=3 a ≤6 cél de um GT-T (por imagem):")
for nome, (g, branco, verde) in list(grafos_webots.items()) + list(grafos_reais.items()):
    gt_img = all_gt.get(nome, [])
    gt_t   = [p for p in gt_img if p.get('tipo') == 'T']
    if not gt_t: continue
    _, _, _, _, rejeitados = _analise_rejeicoes(nome, g, gt_img)
    proximos = []
    for (rx, ry) in rejeitados:
        for gp in gt_t:
            d = math.hypot(gp['x'] - rx, gp['y'] - ry)
            if d <= 6.0:
                proximos.append((rx, ry, gp['x'], gp['y'], d))
                break
    if proximos:
        for rx, ry, gx, gy, d in proximos:
            print(f"  {nome}: rej({rx},{ry}) ↔ GT-T({gx},{gy}) dist={d:.1f} cél")
    else:
        print(f"  {nome}: nenhum rejeitado próximo de GT-T")
print()


# ─── SEÇÃO 4: Análise dos T FN do Webots em image_2 ──────────────────────────

print("=" * 70)
print("SEÇÃO 4: T FN DO WEBOTS — image_2: T(153,53) e T(9,45)")
print("=" * 70)
print()

if 'image_2' not in raw_webots:
    print("[AVISO] image_2 não disponível")
else:
    data2 = raw_webots['image_2']
    skeleton2 = data2['skeleton']
    relevo2   = data2['relevo']
    branco2   = data2['branco']
    verde2    = data2['verde']

    GT_T_WEBOTS_2 = [
        (153, 53, 'T(153,53)'),
        (  9, 45, 'T(9,45)'),
    ]

    cfg_d = ConfigCruzamento()
    H2, W2 = branco2.shape[:2]
    r_max = max(cfg_d.raios_circulo)

    for gx, gy, label in GT_T_WEBOTS_2:
        print(f"─── {label} ───")

        # Reconstruir grafo passo a passo
        g2 = construir_grafo(skeleton2, relevo2, ConfigGrafo(), branco=branco2)
        nid, nx, ny, dist = _nearest_node(g2, gx, gy)
        if nid is None:
            print("  Nenhum nó encontrado antes da simplificação"); continue
        print(f"  ANTES simplificação: nó ({nx},{ny}) grau={g2.degree(nid)} dist={dist:.1f} cél nid={nid}")

        # Etapa 1: diagonais
        remover_diagonais_espurias(g2, ConfigSimplificacao())
        if nid in g2.nodes:
            nd = g2.nodes[nid]; d2g = g2.degree(nid)
            print(f"  Após rem_diag: nó ({nd.x},{nd.y}) grau={d2g}")
        else:
            nid2, nx2, ny2, d2 = _nearest_node(g2, gx, gy)
            print(f"  Após rem_diag: nó REMOVIDO! Mais próximo: ({nx2},{ny2}) grau={g2.degree(nid2)} dist={d2:.1f}")
            nid = nid2

        # Etapa 2: curvatura
        remover_nos_curvatura_baixa(g2, ConfigSimplificacao())
        if nid in g2.nodes:
            nd = g2.nodes[nid]; d2g = g2.degree(nid)
            print(f"  Após rem_curv: nó ({nd.x},{nd.y}) grau={d2g}")
        else:
            nid2, nx2, ny2, d2 = _nearest_node(g2, gx, gy)
            print(f"  Após rem_curv: nó REMOVIDO! Mais próximo: ({nx2},{ny2}) grau={g2.degree(nid2)} dist={d2:.1f}")
            nid = nid2

        # Etapa 3: espurios
        remover_espurios(g2, ConfigSimplificacao())
        if nid in g2.nodes:
            nd = g2.nodes[nid]; d2g = g2.degree(nid)
            print(f"  Após rem_espu: nó ({nd.x},{nd.y}) grau={d2g}")
        else:
            nid2, nx2, ny2, d2 = _nearest_node(g2, gx, gy)
            print(f"  Após rem_espu: nó REMOVIDO! Mais próximo: ({nx2},{ny2}) grau={g2.degree(nid2)} dist={d2:.1f}")
            nid = nid2

        # Etapa 4: fundir juncoes
        fundir_juncoes(g2, skeleton2, relevo2, ConfigSimplificacao())
        if nid in g2.nodes:
            nd = g2.nodes[nid]; d2g = g2.degree(nid)
            print(f"  Após fund_junc: nó ({nd.x},{nd.y}) grau={d2g}")
        else:
            nid2, nx2, ny2, d2 = _nearest_node(g2, gx, gy)
            print(f"  Após fund_junc: nó REMOVIDO! Mais próximo: ({nx2},{ny2}) grau={g2.degree(nid2)} dist={d2:.1f}")
            nid = nid2

        # Etapa 5: cluster
        fusao_juncoes_cluster(g2, skeleton2, ConfigSimplificacao())
        if nid in g2.nodes:
            nd = g2.nodes[nid]; d2g = g2.degree(nid)
            print(f"  Após clust_junc: nó ({nd.x},{nd.y}) grau={d2g}")
        else:
            nid2, nx2, ny2, d2 = _nearest_node(g2, gx, gy)
            print(f"  Após clust_junc: nó REMOVIDO! Mais próximo: ({nx2},{ny2}) grau={g2.degree(nid2)} dist={d2:.1f}")
            nid = nid2

        # Etapa 6: fusao t_par
        fusao_t_par(g2, skeleton2, ConfigSimplificacao())
        if nid in g2.nodes:
            nd = g2.nodes[nid]; d2g = g2.degree(nid)
            print(f"  Após fusao_t_par: nó ({nd.x},{nd.y}) grau={d2g}")
        else:
            nid2, nx2, ny2, d2 = _nearest_node(g2, gx, gy)
            print(f"  Após fusao_t_par: nó REMOVIDO! Mais próximo: ({nx2},{ny2}) grau={g2.degree(nid2)} dist={d2:.1f}")
            nid = nid2

        if nid not in g2.nodes:
            print("  → Nó não sobreviveu à simplificação completa. FN por simplificação.")
            print()
            continue

        nd_final = g2.nodes[nid]
        grau_final = g2.degree(nid)
        print(f"  APÓS simplificação: nó ({nd_final.x},{nd_final.y}) grau={grau_final}")

        # Verificar D29 (borda)
        cx, cy = nd_final.x, nd_final.y
        borda_falha = (cx - r_max < 0 or cx + r_max >= W2
                       or cy - r_max < 0 or cy + r_max >= H2)
        print(f"  D29 borda: cx={cx} cy={cy} r_max={r_max} H={H2} W={W2}")
        print(f"    x-r={cx-r_max} x+r={cx+r_max} y-r={cy-r_max} y+r={cy+r_max}")
        print(f"    → {'REJEITADO por borda (D29)' if borda_falha else 'Não rejeitado por borda'}")

        if borda_falha:
            print()
            continue

        # Verificar geometria (grau=3)
        if grau_final == 3:
            geom_ok = _verifica_geometria_t(g2, nid, cfg_d)
            print(f"  _verifica_geometria_t (stem_max=130°) = {geom_ok}")
            # Mostrar ângulos
            vizinhos = g2.neighbors(nid)
            n = g2.nodes[nid]
            vecs = []
            for v in vizinhos:
                nv = g2.nodes[v]
                dy, dx = nv.y - n.y, nv.x - n.x
                norm = math.hypot(dy, dx) or 1.0
                vecs.append((dy/norm, dx/norm))
                geo_d = math.hypot(nv.x - nd_final.x, nv.y - nd_final.y)
                print(f"    braço→({nv.x},{nv.y}) dist={geo_d:.1f} cél grau_viz={g2.degree(v)}")
            for i, j, k in [(0,1,2),(0,2,1),(1,2,0)]:
                dot = vecs[i][0]*vecs[j][0]+vecs[i][1]*vecs[j][1]
                ang = math.degrees(math.acos(max(-1.,min(1.,dot))))
                dot_ki = vecs[k][0]*vecs[i][0]+vecs[k][1]*vecs[i][1]
                dot_kj = vecs[k][0]*vecs[j][0]+vecs[k][1]*vecs[j][1]
                ang_ki = math.degrees(math.acos(max(-1.,min(1.,dot_ki))))
                ang_kj = math.degrees(math.acos(max(-1.,min(1.,dot_kj))))
                b_ok = ang >= 140.0
                print(f"    Par {i}↔{j}: {ang:.1f}° {'BARRA OK' if b_ok else f'insuf({ang:.1f}<140)'}"
                      f"  stem{k}↔{i}: {ang_ki:.1f}°{'OK' if 50<=ang_ki<=130 else ' FALHA'}"
                      f"  stem{k}↔{j}: {ang_kj:.1f}°{'OK' if 50<=ang_kj<=130 else ' FALHA'}")
            if not geom_ok:
                print("  → FN por _verifica_geometria_t")
                print()
                continue

        elif grau_final == 2:
            ang = math.degrees(0)
            vizinhos = g2.neighbors(nid)
            n = g2.nodes[nid]
            na = g2.nodes[vizinhos[0]]; nb = g2.nodes[vizinhos[1]]
            vay, vax = na.y-n.y, na.x-n.x; vby, vbx = nb.y-n.y, nb.x-n.x
            na_n = math.hypot(vay,vax) or 1.; nb_n = math.hypot(vby,vbx) or 1.
            dot = (vay*vby+vax*vbx)/(na_n*nb_n)
            ang = math.degrees(math.acos(max(-1.,min(1.,dot))))
            print(f"  Grau=2; ângulo={ang:.1f}° → {'candidato L' if 50<=ang<=130 else 'não é candidato L'}")
            print("  → FN: nó virou grau=2 após simplificação (fusão reduziu grau)")
            print()
            continue
        elif grau_final != 3:
            print(f"  → FN: grau final={grau_final}, não é candidato T")
            print()
            continue

        # V1 e V2
        s1, nb, nv, raio = _verificacao1_multi_raio(
            'T', cx, cy, cfg_d.raios_circulo, branco2, verde2,
            cfg_d.tolerancia_segmentos, usar_consenso=True,
            min_raios_consenso=cfg_d.min_raios_consenso,
            n_amostras=cfg_d.n_amostras_circulo,
        )
        s2 = _verificacao2_caminho(nid, g2, branco2, cfg_d.max_passos_v2)
        conf = (s1 + s2) / 2.0
        print(f"  V1={s1:.2f} (n_branco={nb} n_verde={nv} raio={raio})  V2={s2:.2f}  conf={conf:.3f}")
        print(f"  → {'ACEITO' if conf >= cfg_d.limiar_confianca else 'REJEITADO (conf abaixo do limiar)'}")
        print()

print("=" * 70)
print("FIM DO DIAGNÓSTICO")
print("=" * 70)
