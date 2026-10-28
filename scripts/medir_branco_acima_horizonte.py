"""Mede pixels brancos acima do horizonte anotado: dentro vs fora da convexa.

Requer que o horizonte tenha sido anotado com scripts/anotar_horizonte.py.

Uso:
    python3 scripts/medir_branco_acima_horizonte.py <nome_imagem>
                [--calib <json>] [--gabarito-horizonte <json>]

Exemplos (rodar após anotar):
    python3 scripts/medir_branco_acima_horizonte.py bitbots
    python3 scripts/medir_branco_acima_horizonte.py eliza

Saída:
    - Branco B: total, acima/abaixo do horizonte, dentro/fora da convexa
    - Fração do esqueleto acima do horizonte
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)

from features.segmentacao_real import PixelSegmentReal

CALIB_DEFAULT = os.path.join(ROOT, 'calibracao', 'real_provisoria_n2.json')
GAB_DEFAULT   = os.path.join(ROOT, 'tests', 'gabarito_horizonte.json')
LUT           = os.path.join(ROOT, 'green_pixels.csv')

IMG_PATHS = {
    'image_1': 'images/image_1.png',
    'image_2': 'images/image_2.png',
    'image':   'images/image.png',
    'bitbots': ('images/bitbots_reality_spl_only_131-16_02_2018__'
                '11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   'images/eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg',
}


def rasterize_horizon(poly: list, W: int) -> np.ndarray:
    """Interpola linearmente o horizonte para cada coluna 0..W-1.

    Retorna array float32 de shape (W,); valores fora do range de x
    da polilinha são extrapolados com o valor da extremidade.
    """
    if len(poly) < 2:
        raise ValueError("horizonte precisa de ≥2 pontos")
    poly_s = sorted(poly, key=lambda p: p[0])
    xs = [float(p[0]) for p in poly_s]
    ys = [float(p[1]) for p in poly_s]
    all_x = np.arange(W, dtype=np.float64)
    y_h = np.interp(all_x, xs, ys, left=ys[0], right=ys[-1])
    return y_h.astype(np.float32)


def above_mask(H: int, W: int, y_horizon: np.ndarray) -> np.ndarray:
    """Máscara booleana (H, W): True onde pixel está ACIMA do horizonte (y < horizon(x))."""
    rows = np.arange(H, dtype=np.float32)[:, None]    # (H, 1)
    return rows < y_horizon[None, :]                   # (H, W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('nome_imagem',
                        help='Nome da imagem (ex.: bitbots, eliza)')
    parser.add_argument('--calib', default=CALIB_DEFAULT,
                        help=f'Calibração JSON (padrão: {CALIB_DEFAULT})')
    parser.add_argument('--gabarito-horizonte', default=GAB_DEFAULT,
                        dest='gabarito_horizonte',
                        help=f'Gabarito de horizonte JSON (padrão: {GAB_DEFAULT})')
    args = parser.parse_args()

    nome = args.nome_imagem

    # ── Gabarito de horizonte ──────────────────────────────────────────────
    if not os.path.exists(args.gabarito_horizonte):
        sys.exit(f"Gabarito de horizonte não encontrado: {args.gabarito_horizonte}\n"
                 "Anote com: python3 scripts/anotar_horizonte.py")
    with open(args.gabarito_horizonte, 'r', encoding='utf-8') as f:
        gab = json.load(f)
    if nome not in gab:
        sys.exit(f"'{nome}' não encontrado em {args.gabarito_horizonte}. "
                 f"Chaves disponíveis: {list(gab.keys())}")
    horizon_poly = gab[nome]
    if len(horizon_poly) < 2:
        sys.exit(f"Horizonte de '{nome}' tem < 2 pontos.")

    # ── Imagem ────────────────────────────────────────────────────────────
    img_rel = IMG_PATHS.get(nome)
    img_path = os.path.join(ROOT, img_rel) if img_rel else None
    if not img_path or not os.path.exists(img_path):
        sys.exit(f"Imagem não encontrada: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"cv2.imread falhou: {img_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    H, W = img.shape[:2]

    # ── Pipeline ──────────────────────────────────────────────────────────
    ps  = PixelSegmentReal.from_json(args.calib, LUT)
    res = ps.processar(img)

    white_b = res['white_mask']                             # (H, W) uint8, full res
    mc_g    = res['mascara_convexa']                        # grade
    sk_g    = res['skeleton_img']                           # grade
    mc_full = cv2.resize(mc_g.astype(np.uint8), (W, H), cv2.INTER_NEAREST)
    sk_full = cv2.resize(sk_g, (W, H), cv2.INTER_NEAREST)

    # ── Horizonte ─────────────────────────────────────────────────────────
    y_h    = rasterize_horizon(horizon_poly, W)
    above  = above_mask(H, W, y_h)    # (H, W) bool
    below  = ~above

    b_bin  = white_b > 0
    mc_bin = mc_full > 0
    sk_bin = sk_full > 0

    # ── Métricas ──────────────────────────────────────────────────────────
    n_b_total = int(b_bin.sum())
    n_b_ac_in  = int((b_bin & above &  mc_bin).sum())
    n_b_ac_out = int((b_bin & above & ~mc_bin).sum())
    n_b_ab_in  = int((b_bin & below &  mc_bin).sum())
    n_b_ab_out = int((b_bin & below & ~mc_bin).sum())
    n_b_above  = n_b_ac_in + n_b_ac_out
    n_b_below  = n_b_ab_in + n_b_ab_out

    n_sk_total = int(sk_bin.sum())
    n_sk_above = int((sk_bin & above).sum())
    n_sk_below = n_sk_total - n_sk_above

    def pct(n, d):
        return f'{100*n/d:.1f}%' if d > 0 else 'n/a'

    print(f"\nMedição de branco acima do horizonte — {nome}  ({W}×{H}px)")
    print(f"  Horizonte: {len(horizon_poly)} pontos, "
          f"y mín={int(y_h.min())}px  y máx={int(y_h.max())}px")
    print(f"  Calibração: {args.calib}")
    print()
    print(f"[Branco B]  total={n_b_total} px")
    print(f"  Acima do horizonte:  {n_b_above:6d} px  ({pct(n_b_above, n_b_total)} do total)")
    print(f"    dentro convexa:    {n_b_ac_in:6d} px  ({pct(n_b_ac_in,  n_b_above)} do acima)")
    print(f"    fora convexa:      {n_b_ac_out:6d} px  ({pct(n_b_ac_out, n_b_above)} do acima)")
    print(f"  Abaixo do horizonte: {n_b_below:6d} px  ({pct(n_b_below, n_b_total)} do total)")
    print(f"    dentro convexa:    {n_b_ab_in:6d} px  ({pct(n_b_ab_in,  n_b_below)} do abaixo)")
    print(f"    fora convexa:      {n_b_ab_out:6d} px  ({pct(n_b_ab_out, n_b_below)} do abaixo)")
    print()
    print(f"[Esqueleto]  total={n_sk_total} px (resolução original)")
    print(f"  Acima do horizonte:  {n_sk_above:6d} px  ({pct(n_sk_above, n_sk_total)})")
    print(f"  Abaixo do horizonte: {n_sk_below:6d} px  ({pct(n_sk_below, n_sk_total)})")


if __name__ == '__main__':
    main()
