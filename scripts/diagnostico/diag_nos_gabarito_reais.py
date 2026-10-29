"""Diagnóstico de nós e candidatos para os 4 pontos do gabarito real.

Rastreia cada ponto (bitbots L/T, eliza L/T) pelo pipeline completo:
  1) Nó mais próximo antes e depois de cada etapa de simplificação
  2) Classificação, verificação de geometria, V1 por raio, V2, borda (D29)
  3) Verificação de fronteira: círculo V1 sai da grade?
  4) Overlay sobre imagem original → debug_out/<nome>_diag_gabarito.png

Uso: python3 scripts/diagnostico/diag_nos_gabarito_reais.py
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
sys.path.insert(0, ROOT)

from features.segmentacao_real import PixelSegmentReal
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import (
    ConfigSimplificacao,
    remover_diagonais_espurias,
    remover_nos_curvatura_baixa,
    remover_espurios,
    fundir_juncoes,
    fusao_juncoes_cluster,
    fusao_t_par,
)
from features.crossings import (
    ConfigCruzamento,
    detectar_cruzamentos,
    _verificacao1_circulo,
    _verificacao1_multi_raio,
    _verificacao2_caminho,
    _verifica_geometria_t,
    _angulo_entre_vizinhos,
    _angulo_arco_l,
    _caminhar_ate_arco,
)

CALIB = os.path.join(ROOT, 'calibracao', 'real_provisoria_n2.json')
LUT   = os.path.join(ROOT, 'green_pixels.csv')

IMG_PATHS = {
    'bitbots': os.path.join(
        ROOT, 'images',
        'bitbots_reality_spl_only_131-16_02_2018__'
        '11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg',
    ),
    'eliza': os.path.join(
        ROOT, 'images', 'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg',
    ),
}

GABARITO = {
    'bitbots': [
        {'tipo': 'L', 'x': 35, 'y': 55},
        {'tipo': 'T', 'x': 8,  'y': 41},
    ],
    'eliza': [
        {'tipo': 'L', 'x': 30, 'y': 54},
        {'tipo': 'T', 'x': 48, 'y': 43},
    ],
}

# Etapas de simplificação na mesma ordem que simplificar()
_ETAPAS = [
    ('rem_diag',    lambda g, sk, rv, c: remover_diagonais_espurias(g, c)),
    ('rem_curv',    lambda g, sk, rv, c: remover_nos_curvatura_baixa(g, c)),
    ('rem_espu',    lambda g, sk, rv, c: remover_espurios(g, c)),
    ('fund_junc',   lambda g, sk, rv, c: fundir_juncoes(g, sk, rv, c)),
    ('clust_junc',  lambda g, sk, rv, c: fusao_juncoes_cluster(g, sk, c) if c.usar_fusao_juncoes else 0),
    ('fusao_t_par', lambda g, sk, rv, c: fusao_t_par(g, sk, c) if c.usar_fusao_t_par else 0),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _snap(g) -> dict[int, tuple[int, int, int]]:
    """Retorna {nid: (x, y, grau)} para o estado atual do grafo."""
    return {nid: (nd.x, nd.y, g.degree(nid)) for nid, nd in g.nodes.items()}


def _nearest(snap: dict, gx: int, gy: int):
    """(nid, x, y, grau, dist_células) para o nó mais próximo."""
    best_d = float('inf')
    best = None
    for nid, (x, y, deg) in snap.items():
        d = math.hypot(x - gx, y - gy)
        if d < best_d:
            best_d = d
            best = (nid, x, y, deg, d)
    return best


def _nearest_entry(entries, gx, gy):
    """(dist, entry) para o entry de debug_info mais próximo."""
    best_d = float('inf')
    best = None
    for e in entries:
        d = math.hypot(e['x'] - gx, e['y'] - gy)
        if d < best_d:
            best_d = d
            best = (d, e)
    return best


def _v1_per_raio(tipo, cx, cy, raios, branco, verde, tol, n_amostras):
    n_esp = {'L': 2, 'T': 3, 'X': 4}.get(tipo, 0)
    rows = []
    for r in raios:
        s, nb, nv = _verificacao1_circulo(tipo, cx, cy, r, branco, verde, tol, n_amostras)
        rows.append((r, nb, nv,
                     abs(nb - n_esp) <= tol,
                     nv > 0 and abs(nv - n_esp) <= tol,
                     s))
    return rows


def _geom_t_detail(g, nid, cfg: ConfigCruzamento):
    """Detalhes dos ângulos entre pares de braços de um nó grau-3."""
    vizinhos = g.neighbors(nid)
    n = g.nodes[nid]
    d_arco = cfg.distancia_arco_geom_px
    tips = []
    for v in vizinhos:
        if d_arco > 0:
            far = _caminhar_ate_arco(g, nid, v, float(d_arco))
            tips.append(g.nodes[far])
        else:
            tips.append(g.nodes[v])

    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            ni, nj = tips[i], tips[j]
            dy_i, dx_i = ni.y - n.y, ni.x - n.x
            dy_j, dx_j = nj.y - n.y, nj.x - n.x
            nm_i = math.hypot(dy_i, dx_i) or 1.0
            nm_j = math.hypot(dy_j, dx_j) or 1.0
            dot = (dy_i / nm_i) * (dy_j / nm_j) + (dx_i / nm_i) * (dx_j / nm_j)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            pairs.append((i, j,
                          (vizinhos[i], g.nodes[vizinhos[i]].x, g.nodes[vizinhos[i]].y),
                          (vizinhos[j], g.nodes[vizinhos[j]].x, g.nodes[vizinhos[j]].y),
                          ang,
                          ang >= cfg.angulo_barra_t_grau))
    return pairs


# ─── diagnóstico principal ────────────────────────────────────────────────────

def diagnosticar(nome: str) -> None:
    BAR = '=' * 70
    print(f'\n{BAR}')
    print(f'DIAGNÓSTICO: {nome}')
    print(f'{BAR}')

    img_path = IMG_PATHS.get(nome, '')
    if not os.path.exists(img_path):
        print(f'  ERRO: imagem não encontrada: {img_path}')
        return

    img = cv2.imread(img_path)
    H_orig, W_orig = img.shape[:2]

    ps  = PixelSegmentReal.from_json(CALIB, LUT)
    res = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['white_mask_resized'].astype(np.uint8)
    verde    = res['green_mask_resized'].astype(np.uint8)
    H, W = branco.shape

    cfg_grafo = ConfigGrafo()
    cfg_simp  = ConfigSimplificacao()
    cfg_cruz  = ConfigCruzamento()
    r_max     = max(cfg_cruz.raios_circulo)

    print(f'  Grade: {W}×{H}  r_max={r_max}  raios={cfg_cruz.raios_circulo}')
    print(f'  min_raios_consenso={cfg_cruz.min_raios_consenso}  '
          f'tol_segs={cfg_cruz.tolerancia_segmentos}  '
          f'limiar_conf={cfg_cruz.limiar_confianca}')
    print(f'  angulo_barra_T≥{cfg_cruz.angulo_barra_t_grau}°  '
          f'verificar_borda={cfg_cruz.verificar_borda}')

    # ── build + simplificação passo-a-passo ──────────────────────────────────
    g = construir_grafo(skeleton, relevo, cfg_grafo, branco=branco)
    snaps: list[tuple[str, dict]] = [('pre', _snap(g))]
    for nome_etapa, fn in _ETAPAS:
        fn(g, skeleton, relevo, cfg_simp)
        snaps.append((nome_etapa, _snap(g)))

    snap_pre  = snaps[0][1]
    snap_post = snaps[-1][1]
    print(f'  Nós: {len(snap_pre)} antes → {len(snap_post)} após simplificação')

    debug_info: list[dict] = []
    aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg_cruz, _debug_info=debug_info)
    print(f'  Aceitos={len(aceitos)}  candidatos_total={len(debug_info)}')

    # ── por ponto de gabarito ────────────────────────────────────────────────
    for pt in GABARITO[nome]:
        gx, gy, gtipo = pt['x'], pt['y'], pt['tipo']
        SEP = '─' * 60
        print(f'\n{SEP}')
        print(f'  Gabarito: {gtipo} ({gx},{gy})')
        print(f'{SEP}')

        nr_pre  = _nearest(snap_pre,  gx, gy)
        nr_post = _nearest(snap_post, gx, gy)

        # ── [1] Nós antes e depois ─────────────────────────────────────────
        print('\n  [1] NÓ MAIS PRÓXIMO')
        if nr_pre:
            nid_a, x_a, y_a, deg_a, d_a = nr_pre
            print(f'      ANTES: ({x_a},{y_a}) grau={deg_a}  dist={d_a:.2f} cél  nid={nid_a}')
        if nr_post:
            nid_p, x_p, y_p, deg_p, d_p = nr_post
            print(f'      APÓS:  ({x_p},{y_p}) grau={deg_p}  dist={d_p:.2f} cél  nid={nid_p}')

        # Rastrear em qual etapa nid_a desapareceu
        if nr_pre:
            nid_a = nr_pre[0]
            removed_between = None
            last_deg_known = nr_pre[3]
            for idx in range(1, len(snaps)):
                en_name = snaps[idx][0]
                snap_e  = snaps[idx][1]
                if nid_a not in snap_e:
                    removed_between = (snaps[idx - 1][0], en_name)
                    break
                last_deg_known = snap_e[nid_a][2]
            if removed_between:
                print(f'      → nid {nid_a} removido entre '
                      f"'{removed_between[0]}' → '{removed_between[1]}'"
                      f'  (grau antes da remoção: {last_deg_known})')
            else:
                print(f'      → nid {nid_a} sobrevive a toda simplificação '
                      f'(grau final: {last_deg_known})')

        if not nr_post:
            print('      (grafo vazio após simplificação?)')
            continue

        nid_p, x_p, y_p, deg_p, d_p = nr_post

        # ── [2] Classificação do nó pós-simplificação ─────────────────────
        print(f'\n  [2] CLASSIFICAÇÃO DO NÓ PÓS-SIMPL ({x_p},{y_p}) grau={deg_p}')

        bx0 = x_p - r_max
        bx1 = x_p + r_max
        by0 = y_p - r_max
        by1 = y_p + r_max
        fora = bx0 < 0 or bx1 >= W or by0 < 0 or by1 >= H
        print(f'      Borda D29: x∈[{bx0},{bx1}] (grade 0..{W-1})  '
              f'y∈[{by0},{by1}] (grade 0..{H-1})  → {"FORA — rejeitado em D29" if fora else "dentro"}')

        if deg_p == 3:
            geom_ok = _verifica_geometria_t(g, nid_p, cfg_cruz)
            print(f'      Grau 3 → candidato T; _verifica_geometria_t = {geom_ok}')
            pairs = _geom_t_detail(g, nid_p, cfg_cruz)
            for i, j, (v_i, xi, yi), (v_j, xj, yj), ang, barra in pairs:
                tag = 'BARRA OK' if barra else f'insuf (<{cfg_cruz.angulo_barra_t_grau}°)'
                print(f'        braço{i}→({xi},{yi}) ↔ braço{j}→({xj},{yj}): '
                      f'ângulo={ang:.1f}°  [{tag}]')
            if not geom_ok:
                print(f'        → nenhum par ≥ {cfg_cruz.angulo_barra_t_grau}°; '
                      f'T descartado na classificação')
        elif deg_p == 2:
            d_arco = cfg_cruz.distancia_arco_l_px
            ang = math.degrees(
                _angulo_arco_l(g, nid_p, float(d_arco))
                if d_arco > 0
                else _angulo_entre_vizinhos(g, nid_p)
            )
            in_faixa = cfg_cruz.angulo_L_min_grau <= ang <= cfg_cruz.angulo_L_max_grau
            print(f'      Grau 2 → candidato L; ângulo={ang:.1f}°  '
                  f'faixa=[{cfg_cruz.angulo_L_min_grau}°,{cfg_cruz.angulo_L_max_grau}°]  '
                  f'→ {"OK" if in_faixa else "FORA DA FAIXA"}')
        elif deg_p == 1:
            print('      Grau 1 (folha) → não classificado')
        elif deg_p == 0:
            print('      Grau 0 (isolado) → não classificado')
        else:
            print(f'      Grau {deg_p} → não classificado (esperado 2/3/4)')

        # ── [3] V1 por raio ────────────────────────────────────────────────
        print(f'\n  [3] V1 por raio — tipo={gtipo} aplicado em ({x_p},{y_p})')
        n_esp = {'L': 2, 'T': 3, 'X': 4}.get(gtipo, 0)
        rows = _v1_per_raio(gtipo, x_p, y_p,
                            cfg_cruz.raios_circulo, branco, verde,
                            cfg_cruz.tolerancia_segmentos,
                            cfg_cruz.n_amostras_circulo)
        print(f'      esperado: n_b=n_v={n_esp}  tol=±{cfg_cruz.tolerancia_segmentos}')
        print(f'      {"Raio":>5}  {"N_b":>5}  {"N_v":>5}  {"B_ok":>5}  {"V_ok":>5}  score')
        b_ok_cnt = v_ok_cnt = 0
        for r, nb, nv, b_ok, v_ok, s in rows:
            print(f'      {r:>5}  {nb:>5}  {nv:>5}  {str(b_ok):>5}  {str(v_ok):>5}  {s:.2f}')
            b_ok_cnt += b_ok
            v_ok_cnt += v_ok
        min_ok = min(cfg_cruz.min_raios_consenso, len(cfg_cruz.raios_circulo))
        s1, n_b_best, n_v_best, r_best = _verificacao1_multi_raio(
            gtipo, x_p, y_p, cfg_cruz.raios_circulo, branco, verde,
            cfg_cruz.tolerancia_segmentos,
            usar_consenso=cfg_cruz.usar_consenso_v1,
            min_raios_consenso=cfg_cruz.min_raios_consenso,
            n_amostras=cfg_cruz.n_amostras_circulo,
        )
        print(f'      consenso: b_ok={b_ok_cnt}/{min_ok}={b_ok_cnt >= min_ok}  '
              f'v_ok={v_ok_cnt}/{min_ok}={v_ok_cnt >= min_ok}')
        print(f'      V1={s1:.2f}  (raio_melhor={r_best}  n_b={n_b_best}  n_v={n_v_best})')

        s2 = _verificacao2_caminho(nid_p, g, branco, cfg_cruz.max_passos_v2)
        viz_pos = [(g.nodes[v].x, g.nodes[v].y) for v in g.neighbors(nid_p)]
        print(f'      V2={s2:.2f}  max_passos={cfg_cruz.max_passos_v2}  vizinhos={viz_pos}')
        conf = (s1 + s2) / 2.0
        print(f'      conf=(V1+V2)/2=({s1:.2f}+{s2:.2f})/2={conf:.3f}  '
              f'limiar={cfg_cruz.limiar_confianca}  '
              f'→ {"ACEITA" if conf >= cfg_cruz.limiar_confianca else "REJEITADA"}')

        # ── [4] Verificação de borda (D29) — explícita ────────────────────
        print(f'\n  [4] BORDA D29 para nó ({x_p},{y_p}) r_max={r_max}')
        checks = [
            (f'x-r = {x_p}-{r_max} = {bx0}', bx0 < 0,   'FORA (< 0)'),
            (f'x+r = {x_p}+{r_max} = {bx1}', bx1 >= W,  f'FORA (≥ {W})'),
            (f'y-r = {y_p}-{r_max} = {by0}', by0 < 0,   'FORA (< 0)'),
            (f'y+r = {y_p}+{r_max} = {by1}', by1 >= H,  f'FORA (≥ {H})'),
        ]
        for label, is_fora, reason in checks:
            print(f'      {label}  {"← " + reason if is_fora else "ok"}')
        print(f'      → {"REJEITA em D29" if fora else "NÃO rejeita em D29"}')

        # ── [5] Candidato mais próximo no debug_info ────────────────────────
        res_e = _nearest_entry(debug_info, gx, gy)
        print(f'\n  [5] CANDIDATO MAIS PRÓXIMO (debug_info)')
        if res_e:
            d_e, e = res_e
            print(f'      tipo={e["tipo"]} pos=({e["x"]},{e["y"]}) '
                  f'dist_gabarito={d_e:.2f} cél')
            print(f'      V1={e["v1"]:.3f}  V2={e["v2"]:.3f}  conf={e["confianca"]:.3f}')
            print(f'      n_branco={e["n_branco"]}  n_verde={e["n_verde"]}  '
                  f'raio_best={e["raio_usado"]}')
            print(f'      aceito={e["aceito"]}  '
                  f'motivo="{e["motivo"]}"  '
                  f'limiar={cfg_cruz.limiar_confianca}')
        else:
            print('      (debug_info vazio)')

    # ── overlay ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, 'debug_out'), exist_ok=True)
    SX = W_orig / W
    SY = H_orig / H

    canvas = img.copy()
    sk_full = cv2.resize(skeleton.astype(np.uint8) * 180, (W_orig, H_orig),
                         cv2.INTER_NEAREST)
    canvas[sk_full > 0] = (90, 70, 20)   # esqueleto em marrom escuro

    # aceitos (vermelho)
    for a in aceitos:
        cx, cy = int(round(a.x * SX)), int(round(a.y * SY))
        cv2.circle(canvas, (cx, cy), 12, (0, 0, 220), 2)
        cv2.putText(canvas, a.tipo, (cx + 14, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

    # rejeitados mais próximos de cada ponto gabarito (amarelo)
    rej_entries = [e for e in debug_info if not e['aceito']]
    drawn: set[tuple] = set()
    for pt in GABARITO[nome]:
        res_r = _nearest_entry(rej_entries, pt['x'], pt['y'])
        if res_r:
            _, e = res_r
            key = (e['x'], e['y'])
            if key not in drawn:
                cx, cy = int(round(e['x'] * SX)), int(round(e['y'] * SY))
                cv2.circle(canvas, (cx, cy), 12, (0, 220, 220), 2)
                cv2.putText(canvas, f'{e["tipo"]}:{e["motivo"][:16]}',
                            (cx + 14, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220), 1, cv2.LINE_AA)
                drawn.add(key)

    # gabarito (verde)
    for pt in GABARITO[nome]:
        cx, cy = int(round(pt['x'] * SX)), int(round(pt['y'] * SY))
        cv2.circle(canvas, (cx, cy), 18, (0, 210, 0), 2)
        cv2.putText(canvas, pt['tipo'], (cx + 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 0), 1, cv2.LINE_AA)

    out = os.path.join(ROOT, 'debug_out', f'{nome}_diag_gabarito.png')
    cv2.imwrite(out, canvas)
    print(f'\n  Overlay salvo: {out}')


if __name__ == '__main__':
    for nome in ('bitbots', 'eliza'):
        diagnosticar(nome)
    print('\n' + '=' * 70 + '\nFim do diagnóstico.\n')
