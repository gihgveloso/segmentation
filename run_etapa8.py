"""
Debug Etapa 8: extração de linhas nas imagens reais.
Salva debug_out/linhas_<imagem>.png com:
  - esqueleto em branco
  - linhas extraídas coloridas
  - linha y_topo_verde em ciano
  - nós do grafo (pré-simplificação) em cinza claro
"""
from __future__ import annotations
import math, os, sys
import cv2
import numpy as np
from scipy.signal import medfilt

sys.path.insert(0, os.path.dirname(__file__))
from main import Pixel_Segment
from features.pipeline import construir_grafo, ConfigGrafo
from features.lines import extrair_linhas, ConfigLinhas
from features.types import LineFeature

LUT_CSV   = 'green_pixels.csv'
IMAGENS   = [('image_1', 'images/image_1.png'),
             ('image_2', 'images/image_2.png'),
             ('image',   'images/image.png')]
DEBUG_OUT = 'debug_out'
os.makedirs(DEBUG_OUT, exist_ok=True)

CORRIDA_MIN = 3
MEDIANA_K   = 5
FATOR       = 8

CORES = [   # BGR para linhas
    (0, 200, 255), (0, 255, 100), (255, 100, 0),
    (200, 0, 255), (0, 160, 255), (255, 200, 0),
    (100, 255, 200),(255, 0, 160),(150, 150, 255),
]

ps  = Pixel_Segment(LUT_CSV)
cfg = ConfigLinhas()


def _topo_verde(verde_grid: np.ndarray) -> np.ndarray:
    H, W = verde_grid.shape
    y_topo = np.full(W, np.nan, dtype=np.float32)
    for c in range(W):
        col = verde_grid[:, c]
        run = 0
        for r in range(H):
            if col[r] > 0:
                run += 1
                if run >= CORRIDA_MIN:
                    y_topo[c] = float(r - CORRIDA_MIN + 1)
                    break
            else:
                run = 0
    valido = np.isfinite(y_topo)
    if valido.sum() >= 2:
        x = np.arange(W)
        y_topo[~valido] = np.interp(x[~valido], x[valido], y_topo[valido])
        y_topo = medfilt(y_topo, kernel_size=MEDIANA_K).astype(np.float32)
    return y_topo


def _draw_line_on_orig(img, x1, y1, x2, y2, color, thickness=2):
    pt1 = (int(round(x1 * FATOR)), int(round(y1 * FATOR)))
    pt2 = (int(round(x2 * FATOR)), int(round(y2 * FATOR)))
    cv2.line(img, pt1, pt2, color, thickness)


print("=" * 60)
for basename, img_path in IMAGENS:
    frame = cv2.imread(img_path)
    res   = ps.processar(frame)

    verde   = res['green_mask_resized'].astype(np.uint8)
    skel    = res['skeleton_img'].astype(bool)
    relevo  = res['mascara_relevo'].astype(np.float32)
    branco  = res['white_mask_resized'].astype(np.uint8)
    H_g, W_g = verde.shape
    H_o, W_o = frame.shape[:2]

    # Grafo pré-simplificação (Etapa 5 — antes de simplificar), com filtro D37
    g = construir_grafo(skel, relevo, ConfigGrafo(), branco=branco)

    # Linhas — passa branco (D34/D36) e verde (d_topo_celulas)
    linhas = extrair_linhas(g, fator_escala=FATOR, cfg=cfg, branco=branco, verde=verde)

    # topo verde
    y_tv = _topo_verde(verde)

    # ── Tabela de LineFeatures (2.1) ──────────────────────────────────────────
    print(f"\n{basename}  [{W_g}×{H_g} grade]  →  {len(linhas)} linhas")
    if linhas:
        comprimentos = sorted(lf.distancia for lf in linhas)
        print(f"  comprimento  mín={comprimentos[0]:.1f}  "
              f"mediana={comprimentos[len(comprimentos)//2]:.1f}  "
              f"máx={comprimentos[-1]:.1f}  células")

    acima_verde = 0
    print(f"  {'#':>3}  {'x1':>5}  {'y1':>5}  {'x2':>5}  {'y2':>5}  "
          f"{'dist':>6}  {'ang°':>6}  {'nos':>4}  {'conf':>5}  {'d_topo':>7}  {'flag':>5}")
    for i, lf in enumerate(linhas):
        d_topo = lf.d_topo_celulas
        d_str = f"{d_topo:>7.1f}" if d_topo is not None and math.isfinite(d_topo) else f"{'nan':>7}"
        flag = '***' if (d_topo is not None and d_topo <= -2.0) else ''
        if flag:
            acima_verde += 1
        ang_deg = math.degrees(lf.angulo) % 180
        print(f"  {i+1:>3}  {lf.x1:>5.1f}  {lf.y1:>5.1f}  "
              f"{lf.x2:>5.1f}  {lf.y2:>5.1f}  "
              f"{lf.distancia:>6.1f}  {ang_deg:>6.1f}  "
              f"{lf.n_nos:>4}  {lf.confianca:>5.2f}  "
              f"{d_str}  {flag:>5}")

    print(f"  (*** = d_topo ≤ -2, acima do verde: {acima_verde} de {len(linhas)})")
    # ── Tabela de componentes descartados (2.2) ───────────────────────────────
    from features.lines import _componentes_grau2
    nodes_g = g.nodes
    comps_all = _componentes_grau2(g, branco=branco,
                                   suporte_aresta_min=cfg.suporte_aresta_min,
                                   dist_aresta_min=cfg.dist_aresta_min)
    desc = [(i, c) for i, c in enumerate(comps_all) if len(c) < cfg.min_nos_componente]
    if desc:
        print(f"\n  Descartados (<{cfg.min_nos_componente} nós): {len(desc)}")
        print(f"  {'id':>4}  {'nos':>4}  {'cx':>5}  {'cy':>5}  {'fundo?':>7}")
        for idx, comp in desc:
            xs = [nodes_g[n].x for n in comp if n in nodes_g]
            ys = [nodes_g[n].y for n in comp if n in nodes_g]
            if not xs:
                continue
            cx_c = sum(xs) / len(xs)
            cy_c = sum(ys) / len(ys)
            col_c = min(int(round(cx_c)), W_g - 1)
            yt_c = float(y_tv[col_c]) if col_c < W_g else float('nan')
            in_fundo = math.isfinite(yt_c) and abs(cy_c - yt_c) <= 7
            print(f"  {idx:>4}  {len(comp):>4}  {cx_c:>5.1f}  {cy_c:>5.1f}  "
                  f"{'sim' if in_fundo else 'não':>7}")

    # Debug image (resolução original)
    debug = frame.copy()
    H_o, W_o = debug.shape[:2]

    # Esqueleto em branco
    skel_orig = cv2.resize(skel.astype(np.uint8) * 255, (W_o, H_o),
                           interpolation=cv2.INTER_NEAREST)
    debug[skel_orig > 0] = (220, 220, 220)

    # Nós do grafo (pré-simp) em cinza escuro
    for node in g.nodes.values():
        px = int(node.x * FATOR)
        py = int(node.y * FATOR)
        cv2.circle(debug, (px, py), 2, (100, 100, 100), -1)

    # Linhas coloridas
    for i, lf in enumerate(linhas):
        cor = CORES[i % len(CORES)]
        _draw_line_on_orig(debug, lf.x1, lf.y1, lf.x2, lf.y2, cor, 2)
        # marcar extremos
        cv2.circle(debug, (int(lf.x1*FATOR), int(lf.y1*FATOR)), 4, cor, -1)
        cv2.circle(debug, (int(lf.x2*FATOR), int(lf.y2*FATOR)), 4, cor, -1)

    # y_topo_verde em ciano
    for c in range(W_g):
        yt = y_tv[c]
        if not math.isfinite(yt):
            continue
        x0 = int(c * W_o / W_g)
        x1 = int((c + 1) * W_o / W_g)
        y_px = int(round(yt * H_o / H_g))
        y_px = max(0, min(H_o - 1, y_px))
        debug[max(0, y_px-1):y_px+2, x0:x1] = (255, 255, 0)  # ciano

    out = os.path.join(DEBUG_OUT, f'linhas_{basename}.png')
    cv2.imwrite(out, debug)
    print(f"  → {out}")

print("\nFim Etapa 8.")
