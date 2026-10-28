"""Features completas (nós, cruzamentos, linhas) nas imagens reais com calibração provisória.

Pipeline: construir_grafo → extrair_linhas → simplificar → detectar_cruzamentos.
As linhas usam o grafo ANTES da simplificação (conforme IV-B).

Saída:
  - Relatório: nós, cruzamentos (L/T/X + confiança), linhas (n, comp, conf, d_topo)
  - debug_out/{nome}_features.png : original + esqueleto + cruzamentos + linhas
  - Coincidência das linhas com gabarito (≥50% pontos a ≤3px de polilinha anotada)

Uso: python3 scripts/diagnostico/diag_features_real.py
"""
import sys
import os
import json
import math

import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from features.segmentacao_real import PixelSegmentReal
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao
from features.crossings import detectar_cruzamentos, ConfigCruzamento
from features.lines import extrair_linhas, ConfigLinhas

CALIB = os.path.join(ROOT, 'calibracao', 'real_provisoria_n2.json')
LUT   = os.path.join(ROOT, 'green_pixels.csv')
GAB   = os.path.join(ROOT, 'tests', 'gabarito_linhas.json')
OUT   = os.path.join(ROOT, 'debug_out')
os.makedirs(OUT, exist_ok=True)

IMGS = {
    'bitbots': os.path.join(ROOT, 'images',
               'bitbots_reality_spl_only_131-16_02_2018__'
               '11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(ROOT, 'images',
               'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
}

LINE_COLORS = [
    (255, 60,  0),   # azul escuro
    (0, 200, 255),   # amarelo
    (200,  0, 255),  # magenta
    (0,  255,  80),  # verde
    (255, 180,  0),  # ciano claro
    (0,  100, 255),  # laranja
    (255,   0, 120), # azul-rosa
    (80,  255, 180), # verde-água
]
CROSSING_COLORS = {'L': (0, 0, 220), 'T': (0, 200, 0), 'X': (220, 0, 0)}


# ── geometria ────────────────────────────────────────────────────────────────

def sample_polyline_pts(poly, step=1.0):
    pts = []
    for i in range(len(poly) - 1):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[i+1][0]), float(poly[i+1][1])
        n = max(1, int(np.ceil(np.hypot(x2-x1, y2-y1) / step)))
        for j in range(n):
            t = j / n
            pts.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
    if poly:
        pts.append([float(poly[-1][0]), float(poly[-1][1])])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)


def dist_to_nonzero(binary):
    if not (binary > 0).any():
        return np.full(binary.shape, np.inf, dtype=np.float32)
    src = (binary == 0).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def rasterize_centerlines(polys, H, W):
    cl = np.zeros((H, W), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 2: continue
        pts = [(int(round(p[0])), int(round(p[1]))) for p in poly]
        for i in range(len(pts)-1):
            cv2.line(cl, pts[i], pts[i+1], 255, 1)
    return cl


def poly_length(poly):
    total = 0.0
    for i in range(len(poly)-1):
        total += math.hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1])
    return total


# ── overlay ──────────────────────────────────────────────────────────────────

def save_features_overlay(img, sk_full, crossings, lines, nome):
    H, W = img.shape[:2]
    ov = (img.astype(np.float32) * 0.45).astype(np.uint8)

    # Esqueleto: ciano translúcido
    sk_col = np.zeros_like(ov)
    sk_col[sk_full > 0] = (180, 200, 0)
    ov = cv2.addWeighted(ov, 1.0, sk_col, 0.5, 0)

    # Linhas: cada uma em cor diferente
    for i, lf in enumerate(lines):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        p1 = (int(round(lf.x1_orig)), int(round(lf.y1_orig)))
        p2 = (int(round(lf.x2_orig)), int(round(lf.y2_orig)))
        cv2.line(ov, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(ov, p1, 3, color, -1)
        cv2.circle(ov, p2, 3, color, -1)

    # Cruzamentos: círculo + letra
    for cf in crossings:
        cx = int(cf.x * 8 + 4)
        cy = int(cf.y * 8 + 4)
        color = CROSSING_COLORS.get(cf.tipo, (255, 255, 255))
        cv2.circle(ov, (cx, cy), 8, color, 2, cv2.LINE_AA)
        cv2.putText(ov, cf.tipo, (cx - 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    legenda = (f"CIANO=esq  L=verm T=verde X=azul  linhas={len(lines)}  cruzam={len(crossings)}")
    cv2.putText(ov, legenda, (4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    out = os.path.join(OUT, f'{nome}_features.png')
    cv2.imwrite(out, ov)
    return out


# ── loop principal ────────────────────────────────────────────────────────────

def run_image(nome, img_path, ps, gabarito):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERRO] imagem não encontrada: {img_path}")
        return
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    H, W = img.shape[:2]

    print(f"\n{'='*60}")
    print(f"=== {nome}  ({W}×{H}px) ===")
    print(f"{'='*60}")

    res = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['white_mask_resized'].astype(np.uint8)
    verde    = res['green_mask_resized'].astype(np.uint8)

    # ── Features ─────────────────────────────────────────────────────────
    g = construir_grafo(skeleton, relevo, ConfigGrafo(), branco=branco)
    n_nos_antes = len(g.nodes)

    lines = extrair_linhas(g, fator_escala=8,
                           cfg=ConfigLinhas(), branco=branco, verde=verde)

    simplificar(g, skeleton, relevo, ConfigSimplificacao())
    n_nos_depois = len(g.nodes)

    crossings, _ = detectar_cruzamentos(g, branco, verde, ConfigCruzamento())

    # ── Nós ──────────────────────────────────────────────────────────────
    print(f"\n[Nós]")
    print(f"  antes  da simplificação: {n_nos_antes}")
    print(f"  depois da simplificação: {n_nos_depois}  (redução {n_nos_antes-n_nos_depois})")

    # ── Cruzamentos ───────────────────────────────────────────────────────
    by_type = {'L': [], 'T': [], 'X': []}
    for cf in crossings:
        by_type.get(cf.tipo, by_type.setdefault(cf.tipo, [])).append(cf)

    print(f"\n[Cruzamentos]  (limiar confiança=0.5)")
    for t in ('L', 'T', 'X'):
        lst = by_type.get(t, [])
        if lst:
            mean_conf = sum(c.confianca for c in lst) / len(lst)
            print(f"  {t}: {len(lst):3d}  conf média={mean_conf:.2f}")
        else:
            print(f"  {t}:   0")
    print(f"  Total: {len(crossings)}")

    # ── Linhas ───────────────────────────────────────────────────────────
    print(f"\n[Linhas]  total={len(lines)}")
    for i, lf in enumerate(lines):
        d_topo = f'{lf.d_topo_celulas:.1f}cel' if lf.d_topo_celulas is not None else 'n/a'
        print(f"  #{i+1:02d}  dist={lf.distancia:5.1f}cel  conf={lf.confianca:.2f}  "
              f"d_topo={d_topo:8s}  "
              f"orig=[{lf.x1_orig:.0f},{lf.y1_orig:.0f}]→[{lf.x2_orig:.0f},{lf.y2_orig:.0f}]")

    # ── Coincidência com gabarito ─────────────────────────────────────────
    polys_gt = gabarito.get(nome, [])
    if polys_gt and lines:
        cl_mask = rasterize_centerlines(polys_gt, H, W)
        d2cl    = dist_to_nonzero(cl_mask)

        print(f"\n[Coincidência com gabarito]  tol=3px; match se ≥50% dos pts ≤3px")
        n_match = 0
        for i, lf in enumerate(lines):
            poly_lf = [[lf.x1_orig, lf.y1_orig], [lf.x2_orig, lf.y2_orig]]
            pts = sample_polyline_pts(poly_lf)
            if len(pts) == 0:
                frac = 0.0
            else:
                xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W-1)
                ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H-1)
                frac = float((d2cl[ys, xs] <= 3.0).mean())
            match = frac >= 0.50
            if match:
                n_match += 1
            flag = f"MATCH ({100*frac:.0f}%)" if match else f"FALSA ({100*frac:.0f}%)"
            print(f"  #{i+1:02d}: {flag}")
        print(f"  → {n_match}/{len(lines)} linhas coincidem com gabarito  "
              f"({len(lines)-n_match} possíveis falsas)")
    elif not polys_gt:
        print(f"\n[Coincidência] sem gabarito para {nome}")
    else:
        print(f"\n[Coincidência] nenhuma linha detectada")

    # ── Overlay ──────────────────────────────────────────────────────────
    sk_full = cv2.resize(res['skeleton_img'], (W, H), cv2.INTER_NEAREST)
    out = save_features_overlay(img, sk_full, crossings, lines, nome)
    print(f"\n  overlay → {out}")


def main():
    ps  = PixelSegmentReal.from_json(CALIB, LUT)
    gab = json.load(open(GAB, encoding='utf-8'))
    for nome, img_path in IMGS.items():
        run_image(nome, img_path, ps, gab)


if __name__ == '__main__':
    main()
