"""Pipeline completo nas imagens reais com calibração provisória (bitbots, eliza).

Saída por imagem:
  - Estatísticas por etapa (white, green, campo, convexa, relevo, esqueleto, segmentos)
  - Cobertura do gabarito de linhas pelo esqueleto (tol=3px originais)
  - debug_out/{nome}_pipeline_esqueleto.png   (esqueleto sobre original)
  - debug_out/{nome}_pipeline_gabarito_sk.png (gabarito colorido sobre esqueleto)
  - debug_out/{nome}_convexa_borda.png        (item 4: convexa + y_gauss + horizonte)

Uso: python3 scripts/diagnostico/diag_pipeline_real.py
"""
import sys
import os
import json

import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from features.segmentacao_real import PixelSegmentReal

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

# ── métricas e geometria ─────────────────────────────────────────────────────

def dist_to_nonzero(binary: np.ndarray) -> np.ndarray:
    if not (binary > 0).any():
        return np.full(binary.shape, np.inf, dtype=np.float32)
    src = (binary == 0).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def sample_polyline_pts(poly: list, step: float = 1.0) -> np.ndarray:
    pts: list = []
    for i in range(len(poly) - 1):
        x1, y1 = float(poly[i][0]),   float(poly[i][1])
        x2, y2 = float(poly[i+1][0]), float(poly[i+1][1])
        n = max(1, int(np.ceil(np.hypot(x2 - x1, y2 - y1) / step)))
        for j in range(n):
            t = j / n
            pts.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
    if poly:
        pts.append([float(poly[-1][0]), float(poly[-1][1])])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)


def cobertura_esqueleto(polys: list, sk_full: np.ndarray, H: int, W: int,
                        tol: float = 3.0):
    d2sk = dist_to_nonzero(sk_full)
    total_pts = total_hit = 0
    per_poly: list[float] = []
    for poly in polys:
        pts = sample_polyline_pts(poly)
        if len(pts) == 0:
            per_poly.append(0.0)
            continue
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        hit = int((d2sk[ys, xs] <= tol).sum())
        n   = len(pts)
        per_poly.append(hit / n)
        total_pts += n
        total_hit += hit
    return (total_hit / total_pts if total_pts > 0 else 0.0), per_poly, d2sk

# ── overlays ─────────────────────────────────────────────────────────────────

def save_esqueleto_overlay(img: np.ndarray, sk_full: np.ndarray, nome: str) -> str:
    ov = (img.astype(np.float32) * 0.5).astype(np.uint8)
    ov[sk_full > 0] = (200, 220, 0)   # ciano
    out = os.path.join(OUT, f'{nome}_pipeline_esqueleto.png')
    cv2.imwrite(out, ov)
    return out


def save_gabarito_sk_overlay(sk_full: np.ndarray, polys: list, d2sk: np.ndarray,
                              H: int, W: int, nome: str, tol: float = 3.0) -> str:
    ov = cv2.cvtColor(sk_full, cv2.COLOR_GRAY2BGR)
    for poly in polys:
        pts = sample_polyline_pts(poly, step=0.5)
        if len(pts) == 0:
            continue
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        for xi, yi in zip(xs, ys):
            color = (0, 200, 60) if d2sk[yi, xi] <= tol else (0, 80, 220)
            ov[max(0, yi-1):yi+2, max(0, xi-1):xi+2] = color
    out = os.path.join(OUT, f'{nome}_pipeline_gabarito_sk.png')
    cv2.imwrite(out, ov)
    return out


def save_convexa_borda(img: np.ndarray, res: dict, polys: list, nome: str):
    H, W = img.shape[:2]
    gW   = W // 8

    ov = img.copy()

    # mascara_convexa: overlay âmbar translúcido
    mc      = res['mascara_convexa']
    mc_full = cv2.resize(mc.astype(np.uint8), (W, H), cv2.INTER_NEAREST)
    col     = np.zeros_like(ov)
    col[mc_full > 0] = (0, 180, 255)   # âmbar em BGR
    ov = cv2.addWeighted(ov, 0.70, col, 0.30, 0)

    # y_gauss: linha vermelha
    y_gauss  = res['y_gauss']
    pts_gauss = np.array(
        [(int(x * 8 + 4), int(y_gauss[x] * 8 + 4)) for x in range(gW)],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(ov, [pts_gauss], False, (0, 0, 220), 2)

    # horizonte_anotado: linha verde (mín y das polilinhas do gabarito)
    all_ys  = [p[1] for poly in polys for p in poly]
    y_horiz = int(min(all_ys)) if all_ys else -1
    if y_horiz >= 0:
        cv2.line(ov, (0, y_horiz), (W - 1, y_horiz), (0, 210, 0), 2)

    cv2.putText(ov,
        'AMBAR=convexa  VERM=y_gauss  VERDE=horizonte_anotado',
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if y_horiz >= 0:
        cv2.putText(ov, f'horizonte_anotado: y={y_horiz}px (aprox. — mín y das linhas)',
            (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 240, 200), 1, cv2.LINE_AA)

    out = os.path.join(OUT, f'{nome}_convexa_borda.png')
    cv2.imwrite(out, ov)
    return out, y_horiz

# ── loop principal ────────────────────────────────────────────────────────────

def run_image(nome: str, img_path: str, ps: PixelSegmentReal, gabarito: dict):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERRO] imagem não encontrada: {img_path}")
        return
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    H, W = img.shape[:2]
    gH, gW = H // 8, W // 8

    print(f"\n{'='*62}")
    print(f"=== {nome}  ({W}×{H}px, grade {gW}×{gH}, fator=8) ===")
    print(f"{'='*62}")

    res = ps.processar(img)

    wm = res['white_mask']
    gm = res['green_mask']
    mc = res['mascara_convexa']
    mr = res['mascara_relevo']
    sk = res['skeleton_img']

    n_w  = int((wm > 0).sum())
    n_g  = int((gm > 0).sum())
    n_mc = int((mc > 0).sum())
    n_sk = int((sk > 0).sum())
    mr_nz   = mr[mr > 0]
    mr_max  = float(mr_nz.max())  if mr_nz.size > 0 else 0.0
    mr_mean = float(mr_nz.mean()) if mr_nz.size > 0 else 0.0
    n_segs  = len(res.get('segmentos_linhas', []))
    polys_r = res.get('polylines_linhas', [])
    total_pts_r = sum(len(p['pontos']) for p in polys_r)

    print(f"\n[Etapas do pipeline]")
    print(f"  white_mask (B):       {n_w:6d} px    ({100*n_w/(H*W):.1f}% da imagem)")
    print(f"  green_mask (LUT-B):   {n_g:6d} px    ({100*n_g/(H*W):.1f}% da imagem)")
    print(f"  campo_detectado:      {res['campo_detectado']}")
    print(f"  mascara_convexa:      {n_mc:6d} células  ({100*n_mc/(gH*gW):.1f}% da grade)")
    print(f"  mascara_relevo:       max={mr_max:.1f}  média(>0)={mr_mean:.1f}")
    print(f"  esqueleto:            {n_sk:6d} px (grade)")
    print(f"  segmentos conectados: {n_segs} segs, {total_pts_r} pts em {len(polys_r)} polylines")

    # ── Cobertura do gabarito ───────────────────────────────────────────────
    polys_gt = gabarito.get(nome, [])
    print(f"\n[Cobertura do gabarito pelo esqueleto]  tol=3px (pixels originais)")
    sk_full = cv2.resize(sk, (W, H), cv2.INTER_NEAREST)

    if polys_gt:
        rec_total, per_poly, d2sk = cobertura_esqueleto(polys_gt, sk_full, H, W, tol=3.0)

        for pi, (poly, rec) in enumerate(zip(polys_gt, per_poly)):
            p0 = poly[0]; p_end = poly[-1]
            n_pts = len(sample_polyline_pts(poly))
            n_hit = int(round(rec * n_pts))
            label = f"[{p0[0]},{p0[1]}]→[{p_end[0]},{p_end[1]}]"
            flag  = "  ← PERDIDA" if rec < 0.50 else ""
            print(f"  p{pi+1} {label}: ~{n_pts} pts, hit={n_hit}  ({100*rec:.0f}%){flag}")

        print(f"  Total: rec={100*rec_total:.0f}%")

        perdidas = [f"p{i+1}" for i, r in enumerate(per_poly) if r < 0.50]
        if perdidas:
            print(f"\n  Linhas com cobertura < 50%: {', '.join(perdidas)}")
            print(f"  Causa esperada: linhas largas (FWHM≈24px bitbots/≈10px eliza) →")
            print(f"  GaussBlur(5,5) cria plateau → c_xy < 3 não ativa no centro da linha.")
            print(f"  (Parâmetros não ajustados nesta fase — apenas diagnóstico)")
    else:
        _, _, d2sk = cobertura_esqueleto([], sk_full, H, W)
        per_poly = []
        print(f"  (sem anotações para {nome})")

    # ── Imagens de debug ────────────────────────────────────────────────────
    out1 = save_esqueleto_overlay(img, sk_full, nome)
    out2 = save_gabarito_sk_overlay(sk_full, polys_gt, d2sk, H, W, nome)
    out3, y_horiz = save_convexa_borda(img, res, polys_gt, nome)

    # ── Item 4: borda e horizonte ───────────────────────────────────────────
    y_gauss      = res['y_gauss']
    y_gauss_min  = float(np.min(y_gauss))
    y_gauss_max  = float(np.max(y_gauss))

    # Branco-B dentro da convexa
    hsv_img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    branco_b = cv2.inRange(hsv_img,
                           np.array([0, 0, 165], np.uint8),
                           np.array([179, 60, 255], np.uint8))
    b_in_mc  = (branco_b > 0) & (mc_full := cv2.resize(mc.astype(np.uint8), (W, H),
                                                         cv2.INTER_NEAREST)) > 0
    n_b_mc   = int(b_in_mc.sum())

    print(f"\n[Borda do campo e horizonte real  (item 4)]")
    print(f"  y_gauss (grade): min={y_gauss_min:.1f}  max={y_gauss_max:.1f}")
    print(f"  y_gauss (px):    min≈{y_gauss_min*8:.0f}  max≈{y_gauss_max*8:.0f}")

    if y_horiz >= 0:
        print(f"  horizonte_anotado: y={y_horiz}px  (mínimo y das linhas do gabarito — NÃO é o horizonte real)")
        print(f"  Branco (B) dentro da convexa: {n_b_mc} px")
        print(f"  Fração acima de y={y_horiz}px: NÃO MEDIDO — falta anotação do horizonte real.")
        print(f"  Use: python3 scripts/anotar_horizonte.py {nome}")
        print(f"        python3 scripts/medir_branco_acima_horizonte.py {nome}")
    else:
        print(f"  Branco (B) dentro da convexa: {n_b_mc} px")
        print(f"  Fração acima do horizonte: NÃO MEDIDO — falta anotação do horizonte real.")

    print(f"\n[Imagens salvas]")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")


def main() -> None:
    ps  = PixelSegmentReal.from_json(CALIB, LUT)
    gab = json.load(open(GAB, encoding='utf-8'))

    for nome, img_path in IMGS.items():
        run_image(nome, img_path, ps, gab)


if __name__ == '__main__':
    main()
