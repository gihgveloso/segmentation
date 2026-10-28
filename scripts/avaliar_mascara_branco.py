"""Avalia máscaras de branco contra gabarito de polilinhas (precisão/recall/F1).

DEFINIÇÕES:
    centerline_recall = fração dos pontos amostrados (1px/passo) das polilinhas
                        que ficam a ≤ --tol px de algum pixel da máscara.
    precision         = fração dos pixels da máscara que ficam a ≤ --d px
                        de alguma polilinha do gabarito.
    F1                = 2 × precision × recall / (precision + recall)

AVISO: precision é válida somente se TODAS as linhas visíveis foram anotadas.

Uso:
    python3 scripts/avaliar_mascara_branco.py <nome> [--d <px>] [--tol <px>]
                                                      [--com-convexa] [--fwhm <px>]
Exemplos:
    python3 scripts/avaliar_mascara_branco.py bitbots --d 15 --tol 3 --com-convexa
    python3 scripts/avaliar_mascara_branco.py eliza   --d 8  --tol 3 --com-convexa

Parâmetros:
    --d <px>    : distância máx (px originais) entre pixel da máscara e polilinha
                  para contar como TP na precisão.
                  sugestão: d = round(FWHM_med/2) + --tol
                  bitbots → d=15  (FWHM_med≈24px → 12+3)
                  eliza   → d=8   (FWHM_med≈10px →  5+3)
    --tol <px>  : tolerância de anotação (padrão 3px): distância máxima entre
                  ponto amostrado da centerline e pixel da máscara para recall.
    --fwhm <px> : FWHM mediano das linhas (px originais). Se fornecido, exibe o
                  teto teórico do recall da definição antiga:
                  teto_antigo = FWHM_med / (2×d+1)  (com d = raio de dilatação usado)
    --com-convexa : avaliar também máscara ∩ mascara_convexa (executa Pixel_Segment B).
    --gabarito  : caminho alternativo do gabarito JSON.

Critérios avaliados:
    A, B, C         : inRange HSV com faixas descritas na tabela
    V≥213 fixo      : V ≥ 213 (limiar Webots de referência)
    V≥T  kN.N Vc=VV : V ≥ k × V_campo  (V_campo = média V nos px verdes LUT-B)
    tophat sPPxPP   : top-hat branco no canal V, elemento estruturante elipse
                      PP×PP px, threshold = 30

Para anotar polilinhas: python3 scripts/anotar_linhas.py
"""
from __future__ import annotations
import sys, os, io, contextlib, json, argparse
from collections import OrderedDict
import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)

GABARITO_PATH = 'tests/gabarito_linhas.json'

IMG_PATHS = {
    'image_1': 'images/image_1.png',
    'image_2': 'images/image_2.png',
    'image':   'images/image.png',
    'bitbots': ('images/bitbots_reality_spl_only_131-16_02_2018__'
                '11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   'images/eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg',
}

# ── geometria e distâncias ────────────────────────────────────────────────────

def sample_polyline_pts(poly: list, step: float = 1.0) -> np.ndarray:
    """Amostra polilinha a cada `step` px; retorna array (N,2) de (x,y) float32."""
    pts: list[list[float]] = []
    for i in range(len(poly) - 1):
        x1, y1 = float(poly[i][0]),   float(poly[i][1])
        x2, y2 = float(poly[i+1][0]), float(poly[i+1][1])
        seg_len = np.hypot(x2 - x1, y2 - y1)
        n = max(1, int(np.ceil(seg_len / step)))
        for j in range(n):
            t = j / n
            pts.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
    if poly:
        pts.append([float(poly[-1][0]), float(poly[-1][1])])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)


def rasterize_centerlines(polys: list, H: int, W: int) -> np.ndarray:
    """Rasteriza centerlines (espessura=1) como máscara uint8."""
    cl = np.zeros((H, W), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 2:
            continue
        pts = [(int(round(p[0])), int(round(p[1]))) for p in poly]
        for i in range(len(pts) - 1):
            cv2.line(cl, pts[i], pts[i+1], 255, 1)
    return cl


def dist_to_nonzero(binary: np.ndarray) -> np.ndarray:
    """Distância L2 de cada pixel ao pixel não-zero mais próximo em `binary`.

    Retorna float32; pixels não-zero têm distância 0;
    se `binary` for todo-zero, retorna array de inf.
    """
    if not (binary > 0).any():
        return np.full(binary.shape, np.inf, dtype=np.float32)
    src = (binary == 0).astype(np.uint8)   # 0 nos alvos, 1 em todo o resto
    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

# ── métricas ──────────────────────────────────────────────────────────────────

def recall_centerline(polys: list, dist_to_mask: np.ndarray,
                      tol: float, H: int, W: int) -> tuple[float, list[float]]:
    """Recall centerline: fração dos pts amostrados (1px) a ≤ tol px da máscara.

    Retorna (recall_total, [recall_por_polilinha]).
    """
    total_pts = 0
    total_hit = 0
    per_poly: list[float] = []
    for poly in polys:
        pts = sample_polyline_pts(poly)
        if len(pts) == 0:
            per_poly.append(0.0)
            continue
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        dists = dist_to_mask[ys, xs]
        hit = int((dists <= tol).sum())
        n   = len(pts)
        per_poly.append(hit / n)
        total_pts += n
        total_hit += hit
    rec = total_hit / total_pts if total_pts > 0 else 0.0
    return rec, per_poly


def precision_dist(mask: np.ndarray, dist_to_poly: np.ndarray,
                   d: float) -> tuple[float, int]:
    """Precisão: fração dos pixels da máscara a ≤ d px de alguma polilinha."""
    mask_bin = mask > 0
    n_mask = int(mask_bin.sum())
    if n_mask == 0:
        return 0.0, 0
    near = int((dist_to_poly[mask_bin] <= d).sum())
    return near / n_mask, n_mask

# ── construção das máscaras ───────────────────────────────────────────────────

def compute_V_campo(img: np.ndarray) -> float:
    """Média V nos pixels verdes (LUT B: H=35-100, S=40-255, V=40-180)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gm = cv2.inRange(hsv, np.array([35,40,40],np.uint8), np.array([100,255,180],np.uint8))
    V  = hsv[:,:,2]
    return float(V[gm > 0].mean()) if (gm > 0).any() else 128.0


def build_all_masks(img: np.ndarray,
                    V_campo: float) -> 'OrderedDict[str, np.ndarray]':
    """Retorna OrderedDict {nome_verbose: máscara uint8} com todos os critérios."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V   = hsv[:,:,2]
    m: OrderedDict[str, np.ndarray] = OrderedDict()

    m['A  (V≥150 S≤80 H=0-179)']  = cv2.inRange(
        hsv, np.array([0,0,150],np.uint8), np.array([179,80,255],np.uint8))
    m['B  (V≥165 S≤60 H=0-179)']  = cv2.inRange(
        hsv, np.array([0,0,165],np.uint8), np.array([179,60,255],np.uint8))
    m['C  (V≥155 S≤70 H=0-179)']  = cv2.inRange(
        hsv, np.array([0,0,155],np.uint8), np.array([179,70,255],np.uint8))
    m[f'V≥213 (fixo, ref. Webots)'] = (V >= 213).astype(np.uint8) * 255

    for k in [1.5, 2.0, 2.5, 3.0]:
        thresh = int(k * V_campo)
        if thresh > 254:
            continue
        m[f'V≥{thresh} (k={k:.1f} Vc={V_campo:.0f})'] = (
            (V >= thresh).astype(np.uint8) * 255)

    for s in [20, 40, 60]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        tophat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, kernel)
        m[f'tophat s{s}x{s}px  th>30']  = (tophat > 30).astype(np.uint8) * 255

    return m


def get_mascara_convexa(img: np.ndarray) -> np.ndarray:
    """Executa Pixel_Segment (calibração B) e retorna mascara_convexa em resolução original."""
    H, W = img.shape[:2]
    with contextlib.redirect_stdout(io.StringIO()):
        import def_white_threshold as _bm
    _bm.lower = np.array([0,   0, 165], dtype=np.int32)
    _bm.up    = np.array([179, 60, 255], dtype=np.int32)
    with contextlib.redirect_stdout(io.StringIO()):
        from main import Pixel_Segment
    lut_b = np.zeros((180, 256, 256), dtype=np.uint8)
    for _h in range(35, 101):
        lut_b[_h, 40:256, 40:181] = 255
    ps = Pixel_Segment(os.path.join(ROOT, 'green_pixels.csv'))
    ps.lut_verde = lut_b.copy()
    res = ps.processar(img)
    mc = res['mascara_convexa']
    return cv2.resize(mc.astype(np.uint8), (W, H), cv2.INTER_NEAREST)

# ── visualização ──────────────────────────────────────────────────────────────

def save_overlay(img: np.ndarray, mask: np.ndarray,
                 polys: list, dist_to_poly: np.ndarray, dist_to_mask: np.ndarray,
                 d: float, tol: float, out_path: str, label: str) -> None:
    """Sobreposição: CIANO=TP VERMELHO=FP VERDE=rec_ok LARANJA=rec_miss."""
    H, W = img.shape[:2]
    ov = (img.astype(np.float32) * 0.50).astype(np.uint8)

    mask_bin = mask > 0
    tp_mask  = mask_bin & (dist_to_poly <= d)
    fp_mask  = mask_bin & (dist_to_poly >  d)

    ov[tp_mask] = (200, 220, 0)    # ciano  (BGR)
    ov[fp_mask] = (0,   0,   220)  # vermelho

    # Polilinhas coloridas por recall (amostradas a 0.5px)
    all_hit = 0
    all_pts = 0
    for poly in polys:
        pts = sample_polyline_pts(poly, step=0.5)
        if len(pts) == 0:
            continue
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W-1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H-1)
        dists = dist_to_mask[ys, xs]
        hit_mask = dists <= tol
        all_hit += int(hit_mask.sum())
        all_pts += len(pts)
        for xi, yi, hit in zip(xs, ys, hit_mask):
            color = (0, 240, 60) if hit else (0, 130, 255)  # verde / laranja
            ov[max(0,yi-1):yi+2, max(0,xi-1):xi+2] = color

    n_tp = int(tp_mask.sum())
    n_fp = int(fp_mask.sum())
    n_mask = n_tp + n_fp
    rec_display  = all_hit / all_pts if all_pts > 0 else 0.0
    prec_display = n_tp / n_mask     if n_mask  > 0 else 0.0
    f1_display   = (2*prec_display*rec_display / (prec_display+rec_display)
                    if prec_display+rec_display > 0 else 0.0)
    n_fn = all_pts - all_hit

    for i, txt in enumerate([
        label[:60],
        f"CIANO=TP({n_tp}px)  VERM=FP({n_fp}px)",
        f"VERDE=rec_ok  LARANJA=rec_miss ({n_fn} pts centerline)",
        f"Rec={rec_display:.1%}  Prec={prec_display:.1%}  F1={f1_display:.3f}",
    ]):
        cv2.putText(ov, txt, (4, 17 + i*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, ov)

# ── saída ─────────────────────────────────────────────────────────────────────

def _find_key(d: dict, substr: str) -> str | None:
    for k in d:
        if substr in k:
            return k
    return None


def _print_table(results: dict, W_col: int, with_mc: bool) -> None:
    if with_mc:
        hdr = (f"  {'Critério':<{W_col}}"
               f"  {'Rec':>6} {'Prec':>6} {'F1':>5}"
               f"  │  {'Rec_mc':>6} {'Pr_mc':>6} {'F1_mc':>5}")
    else:
        hdr = f"  {'Critério':<{W_col}}  {'Rec':>6} {'Prec':>6} {'F1':>5}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for name, r in results.items():
        row = (f"  {name:<{W_col}}"
               f"  {r['rec']:>5.1%} {r['prec']:>5.1%} {r['f1']:>5.3f}")
        if with_mc and r['rec2'] is not None:
            row += f"  │  {r['rec2']:>5.1%} {r['prec2']:>5.1%} {r['f12']:>5.3f}"
        print(row)
        pp = '  '.join(f"p{i+1}={v:.0%}" for i, v in enumerate(r['per_poly']))
        print(f"  {'':>{W_col}}    rec_poly: {pp}")

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('nome_imagem')
    parser.add_argument('--d', type=float, default=5.0,
                        help='Distância máx (px) do pixel da máscara à polilinha para TP (padrão 5)')
    parser.add_argument('--tol', type=float, default=3.0,
                        help='Tolerância de anotação para recall centerline, em px (padrão 3)')
    parser.add_argument('--fwhm', type=float, default=None,
                        help='FWHM mediano (px) das linhas para calcular teto do recall antigo')
    parser.add_argument('--com-convexa', action='store_true',
                        help='Avaliar também máscara ∩ mascara_convexa (executa Pixel_Segment B)')
    parser.add_argument('--gabarito', default=None,
                        help=f'Caminho alternativo do gabarito JSON (padrão: {GABARITO_PATH})')
    args = parser.parse_args()

    nome          = args.nome_imagem
    d             = float(args.d)
    tol           = float(args.tol)
    gabarito_path = args.gabarito or GABARITO_PATH

    if not os.path.exists(gabarito_path):
        sys.exit(f"Gabarito não encontrado: {gabarito_path}\n"
                 "Anote primeiro com: python3 scripts/anotar_linhas.py")
    with open(gabarito_path, 'r', encoding='utf-8') as f:
        gabarito = json.load(f)
    if nome not in gabarito:
        sys.exit(f"'{nome}' não encontrado. Chaves: {list(gabarito.keys())}")
    polys = gabarito[nome]
    if not polys:
        sys.exit(f"Gabarito de '{nome}' está vazio.")

    img  = load_img(nome)
    H, W = img.shape[:2]

    # Distância de cada pixel às centerlines (fixa para toda a imagem)
    cl_mask      = rasterize_centerlines(polys, H, W)
    dist_to_poly = dist_to_nonzero(cl_mask)

    V_campo = compute_V_campo(img)
    masks   = build_all_masks(img, V_campo)

    mc_full = None
    if args.com_convexa:
        mc_full = get_mascara_convexa(img)

    # ── cabeçalho ──────────────────────────────────────────────────────────
    print(f"Avaliação de máscara branca — {nome}  ({W}x{H}px)")
    print(f"  Gabarito : {len(polys)} polilinha(s)")
    print(f"  V_campo  : {V_campo:.1f}  (média V nos px verdes, LUT-B)")
    print(f"  Recall   : centerline, passo=1px, tolerância tol={tol:.0f}px")
    print(f"  Precisão : fração da máscara a ≤ d={d:.0f}px da polilinha")
    if args.fwhm is not None:
        band = 2 * d + 1
        teto = min(1.0, args.fwhm / band)
        print(f"  FWHM_med : {args.fwhm:.1f}px  →  teto recall_antigo (faixa={band:.0f}px) "
              f"≈ {teto:.1%}")
        print(f"    (recall_antigo = TP_faixa / GT_faixa ≤ FWHM / largura_faixa)")
    print()
    print("  *** AVISO: precisão válida somente se TODAS as linhas visíveis foram anotadas ***")
    print()

    # ── loop de avaliação ──────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for name, mask in masks.items():
        dist_to_mask = dist_to_nonzero(mask)
        rec,  per_poly = recall_centerline(polys, dist_to_mask, tol, H, W)
        prec, n_mask   = precision_dist(mask, dist_to_poly, d)
        f1 = 2*rec*prec / (rec+prec) if rec+prec > 0 else 0.0

        rec2 = prec2 = f12 = None
        if mc_full is not None:
            mask_mc = cv2.bitwise_and(mask, mc_full)
            dist_mc = dist_to_nonzero(mask_mc)
            rec2,  _ = recall_centerline(polys, dist_mc, tol, H, W)
            prec2, _ = precision_dist(mask_mc, dist_to_poly, d)
            f12 = 2*rec2*prec2 / (rec2+prec2) if rec2+prec2 > 0 else 0.0

        results[name] = dict(rec=rec, prec=prec, f1=f1, per_poly=per_poly,
                             rec2=rec2, prec2=prec2, f12=f12, mask=mask)

    W_col = max(len(k) for k in results) + 1
    _print_table(results, W_col, mc_full is not None)

    # ── overlays (só com --com-convexa) ────────────────────────────────────
    if mc_full is not None:
        os.makedirs('debug_out', exist_ok=True)

        best_key = max(
            (k for k in results if results[k]['f12'] is not None),
            key=lambda k: results[k]['f12'] or 0.0,
        )

        overlay_targets = [
            (best_key,                   f'{nome}_best_f1_mc'),
            (_find_key(results, 's40x'), f'{nome}_tophat_s40_mc'),
            (_find_key(results, 'V≥165'),f'{nome}_B_moderado_mc'),
        ]

        print()
        for crit_key, file_tag in overlay_targets:
            if crit_key is None or crit_key not in results:
                continue
            r = results[crit_key]
            mask_mc = cv2.bitwise_and(r['mask'], mc_full)
            dist_mc = dist_to_nonzero(mask_mc)
            out = f'debug_out/{file_tag}.png'
            save_overlay(img, mask_mc, polys, dist_to_poly, dist_mc,
                         d, tol, out, f"{nome} | {crit_key}")
            print(f"  overlay → {out}")

        print(f"\n  melhor F1_mc: '{best_key}'  F1={results[best_key]['f12']:.3f}")


def load_img(nome: str) -> np.ndarray:
    path = IMG_PATHS.get(nome)
    if path is None or not os.path.exists(path):
        sys.exit(f"Imagem não encontrada: {path}")
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cv2.imread falhou: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


if __name__ == '__main__':
    main()
