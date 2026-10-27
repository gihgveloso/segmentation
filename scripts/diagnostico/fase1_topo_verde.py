"""
FASE 1 — Medição do topo verde e impacto nos FP/GT.
NÃO modifica nenhum arquivo existente.
"""
from __future__ import annotations
import json, math, os, sys
import cv2
import numpy as np
from scipy.signal import medfilt

sys.path.insert(0, os.path.dirname(__file__))
from main import Pixel_Segment
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao
from features.crossings import detectar_cruzamentos, ConfigCruzamento

GABARITO_PATH = 'tests/gabarito_cruzamentos.json'
DEBUG_OUT     = 'debug_out'
os.makedirs(DEBUG_OUT, exist_ok=True)

IMAGES = [
    ('image_1', 'images/image_1.png'),
    ('image_2', 'images/image_2.png'),
    ('image',   'images/image.png'),
]
CORRIDA_MIN = 3      # células verdes consecutivas para definir o topo
MARGEM      = 2      # células acima do topo verde
TOL_CASAMENTO = 6.0

ps  = Pixel_Segment('green_pixels.csv')
cfg = ConfigCruzamento()

with open(GABARITO_PATH) as f:
    gabarito: dict = json.load(f)


def _topo_verde_por_coluna(verde_grid: np.ndarray, corrida_min: int) -> np.ndarray:
    """Para cada coluna, primeira linha (de cima para baixo) em que há ≥ corrida_min
    células verdes consecutivas. Retorna array shape=(W,); NaN se não encontrou."""
    H, W = verde_grid.shape
    y_topo = np.full(W, np.nan, dtype=np.float32)
    for c in range(W):
        col = verde_grid[:, c]
        run = 0
        for r in range(H):
            if col[r] > 0:
                run += 1
                if run >= corrida_min:
                    y_topo[c] = float(r - corrida_min + 1)
                    break
            else:
                run = 0
    return y_topo


def _suavizar_topo(y_topo: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Mediana de 5 colunas; NaN → interpolado antes, depois restaurado."""
    valido = np.isfinite(y_topo)
    if valido.sum() < 2:
        return y_topo.copy()
    x = np.arange(len(y_topo))
    y_interp = y_topo.copy()
    y_interp[~valido] = np.interp(x[~valido], x[valido], y_topo[valido])
    y_suav = medfilt(y_interp, kernel_size=kernel).astype(np.float32)
    y_suav[~valido] = np.nan   # restaura NaN onde não havia verde
    return y_suav


def _casar_localizacao(gt_img, aceitos, tol):
    """Casamento guloso por posição (qualquer tipo). Retorna (tp_set_gt, tp_set_det)."""
    pares = []
    for ig, gp in enumerate(gt_img):
        for id_, dp in enumerate(aceitos):
            d = math.hypot(gp['x'] - dp.x, gp['y'] - dp.y)
            if d <= tol:
                pares.append((d, ig, id_))
    pares.sort()
    usados_gt, usados_det = set(), set()
    for d, ig, id_ in pares:
        if ig in usados_gt or id_ in usados_det:
            continue
        usados_gt.add(ig)
        usados_det.add(id_)
    return usados_gt, usados_det


# ─────────────────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────────────────
total_fp_acima   = 0
total_fp         = 0
total_gt_acima   = 0
total_gt         = 0
linhas_fp_tabela = []
linhas_gt_tabela = []

for basename, img_path in IMAGES:
    frame = cv2.imread(img_path)
    assert frame is not None, img_path

    res    = ps.processar(frame)
    verde  = res['green_mask_resized'].astype(np.uint8)  # grade
    flm    = res['field_line_mask']                      # grade uint8
    skel   = res['skeleton']                             # grade bool
    branco = res['white_mask_resized'].astype(np.uint8)

    H, W = verde.shape

    # 1a) topo verde por coluna (corrida ≥ 3)
    y_topo_raw  = _topo_verde_por_coluna(verde, CORRIDA_MIN)
    y_topo_suav = _suavizar_topo(y_topo_raw, kernel=5)

    # Para colunas sem verde usa H (linha de fundo = sem restrição)
    y_limite = np.where(np.isfinite(y_topo_suav),
                        y_topo_suav - MARGEM,
                        float(H)).astype(np.float32)

    # 1b) pixels field_line_mask acima do limite
    px_flm_acima  = 0
    px_skel_acima = 0
    for c in range(W):
        ylim = y_limite[c]
        if ylim > 0:
            yt = int(math.ceil(max(ylim, 0)))
            if yt > 0:
                px_flm_acima  += int(np.count_nonzero(flm[:yt, c]))
                px_skel_acima += int(np.count_nonzero(skel[:yt, c]))

    print(f"\n{'='*60}")
    print(f"  {basename}  [grade {W}×{H}]")
    print(f"  {'='*58}")
    print(f"  field_line_mask total:          {int(np.count_nonzero(flm))}")
    print(f"  field_line_mask acima do limite:{px_flm_acima}")
    print(f"  skeleton total:                 {int(np.count_nonzero(skel))}")
    print(f"  skeleton acima do limite:       {px_skel_acima}")

    # Detecções
    relevo = res['mascara_relevo'].astype(np.float32)
    g = construir_grafo(res['skeleton_img'].astype(bool), relevo, ConfigGrafo())
    simplificar(g, res['skeleton_img'].astype(bool), relevo, ConfigSimplificacao())
    aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)

    gt_img = gabarito.get(basename, [])
    usados_gt, usados_det = _casar_localizacao(gt_img, aceitos, TOL_CASAMENTO)

    # Falsos positivos
    fp_lista = [dp for id_, dp in enumerate(aceitos) if id_ not in usados_det]
    total_fp += len(fp_lista)

    # 1c) tabela FP
    print(f"\n  FP ({len(fp_lista)} pontos):")
    print(f"  {'tipo':>4}  {'x':>5}  {'y':>5}  {'y_topo[c]':>10}  {'acima?':>6}")
    print(f"  {'----':>4}  {'---':>5}  {'---':>5}  {'---------':>10}  {'------':>6}")
    for dp in fp_lista:
        col = min(int(round(dp.x)), W - 1)
        yt  = y_topo_suav[col]
        ylim_c = float(yt - MARGEM) if np.isfinite(yt) else float(H)
        acima = dp.y < ylim_c
        acima_str = "SIM" if acima else "não"
        yt_str = f"{yt:.1f}" if np.isfinite(yt) else "N/A"
        print(f"  {dp.tipo:>4}  {dp.x:>5}  {dp.y:>5}  {yt_str:>10}  {acima_str:>6}")
        linhas_fp_tabela.append((basename, dp.tipo, dp.x, dp.y,
                                  yt_str, ylim_c, acima))
        if acima:
            total_fp_acima += 1

    # GT
    print(f"\n  GT ({len(gt_img)} pontos):")
    print(f"  {'tipo':>4}  {'x':>5}  {'y':>5}  {'y_topo[c]':>10}  {'acima?':>6}")
    print(f"  {'----':>4}  {'---':>5}  {'---':>5}  {'---------':>10}  {'------':>6}")
    total_gt += len(gt_img)
    for gp in gt_img:
        col = min(int(round(gp['x'])), W - 1)
        yt  = y_topo_suav[col]
        ylim_c = float(yt - MARGEM) if np.isfinite(yt) else float(H)
        acima = gp['y'] < ylim_c
        acima_str = "SIM" if acima else "não"
        yt_str = f"{yt:.1f}" if np.isfinite(yt) else "N/A"
        print(f"  {gp['tipo']:>4}  {gp['x']:>5}  {gp['y']:>5}  {yt_str:>10}  {acima_str:>6}")
        linhas_gt_tabela.append((basename, gp['tipo'], gp['x'], gp['y'],
                                  yt_str, ylim_c, acima))
        if acima:
            total_gt_acima += 1

    # 1d) debug image
    frame_orig = cv2.imread(img_path)
    H_orig, W_orig = frame_orig.shape[:2]
    debug = frame_orig.copy()

    # branco acima do limite → vermelho
    for c in range(W):
        ylim_c = y_limite[c]
        if ylim_c <= 0:
            continue
        yt = int(math.ceil(max(0.0, ylim_c)))
        if yt <= 0:
            continue
        # coluna no espaço original
        x0 = int(c * W_orig / W)
        x1 = int((c + 1) * W_orig / W)
        y1 = int(yt * H_orig / H)
        # pixels field_line_mask acima
        px_acima = np.count_nonzero(flm[:yt, c])
        if px_acima > 0:
            debug[:y1, x0:x1] = (0, 0, 255)

    # linha y_topo_verde em amarelo
    for c in range(W):
        yt_s = y_topo_suav[c]
        if not np.isfinite(yt_s):
            continue
        x0 = int(c * W_orig / W)
        x1 = int((c + 1) * W_orig / W)
        y_px = int(round(yt_s * H_orig / H))
        y_px = max(0, min(H_orig - 1, y_px))
        debug[max(0, y_px - 1):y_px + 2, x0:x1] = (0, 255, 255)

    # linha y_limite (topo_verde - 2) em laranja
    for c in range(W):
        ylim_c = y_limite[c]
        if ylim_c >= H or ylim_c <= 0:
            continue
        x0 = int(c * W_orig / W)
        x1 = int((c + 1) * W_orig / W)
        y_px = int(round(ylim_c * H_orig / H))
        y_px = max(0, min(H_orig - 1, y_px))
        debug[max(0, y_px - 1):y_px + 2, x0:x1] = (0, 165, 255)

    out_path = os.path.join(DEBUG_OUT, f'topo_verde_{basename}.png')
    cv2.imwrite(out_path, debug)
    print(f"\n  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Critério
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESUMO DO CRITÉRIO")
print(f"{'='*60}")
print(f"  FP totais:          {total_fp}")
print(f"  FP acima do limite: {total_fp_acima}  ({100*total_fp_acima/max(total_fp,1):.0f}%)")
print(f"  GT totais:          {total_gt}")
print(f"  GT acima do limite: {total_gt_acima}")
print()
crit_fp = total_fp_acima >= total_fp / 2
crit_gt = total_gt_acima <= 1
print(f"  Critério FP (≥ metade acima): {'PASSA ✓' if crit_fp else 'FALHA ✗'}"
      f"  ({total_fp_acima}/{total_fp})")
print(f"  Critério GT (≤ 1 acima):      {'PASSA ✓' if crit_gt else 'FALHA ✗'}"
      f"  ({total_gt_acima}/{total_gt})")
print()
if crit_fp and crit_gt:
    print("  → CRITÉRIO APROVADO — prosseguir para Fase 2.")
else:
    print("  → CRITÉRIO NÃO APROVADO — parar e apresentar tabelas.")
