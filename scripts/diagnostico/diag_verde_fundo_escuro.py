"""Diagnóstico: fundo escuro do eliza entra na máscara verde (H 35-100, S>=40, V 40-180)?

Hipótese: pixels de fundo escuro no topo da imagem eliza satisfazem a calibração
provisória e isso estende a mascara_convexa para o topo.

Uso: python3 scripts/diagnostico/diag_verde_fundo_escuro.py
Saída: texto no stdout + debug_out/eliza_verde_mask_overlay.png
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import cv2

BASE  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT   = os.path.join(BASE, 'debug_out')
os.makedirs(OUT, exist_ok=True)

ELIZA   = os.path.join(BASE, 'images', 'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg')
BITBOTS = os.path.join(BASE, 'images',
          'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg')
LUT_CSV = os.path.join(BASE, 'green_pixels.csv')

# Calibração provisória base
H_MIN, H_MAX = 35, 100
S_BASE       = 40
V_MIN, V_MAX = 40, 180

# Branco calibrado (real_provisoria_n2.json)
W_LOWER = np.array([  0,  0, 165], dtype=np.int32)
W_UPPER = np.array([179, 60, 255], dtype=np.int32)

# Patch global de def_white_threshold (antes de importar main)
with contextlib.redirect_stdout(io.StringIO()):
    import def_white_threshold as _bm
_bm.lower = W_LOWER.copy()
_bm.up    = W_UPPER.copy()

with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment


# ─── helpers ──────────────────────────────────────────────────────────────────

def build_lut(h_min=H_MIN, h_max=H_MAX, s_min=S_BASE, v_min=V_MIN, v_max=V_MAX):
    lut = np.zeros((180, 256, 256), dtype=np.uint8)
    for h in range(h_min, h_max + 1):
        lut[h, s_min:256, v_min:v_max + 1] = 255
    return lut


def apply_green(hsv, lut):
    return lut[hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]


def apply_white(hsv):
    return cv2.inRange(hsv, W_LOWER, W_UPPER)


def median3x3(hsv, x, y):
    """Mediana 3x3 no espaço HSV; coords: x=coluna, y=linha."""
    H, W = hsv.shape[:2]
    r0, r1 = max(0, y - 1), min(H, y + 2)
    c0, c1 = max(0, x - 1), min(W, x + 2)
    patch = hsv[r0:r1, c0:c1]
    return (int(np.median(patch[:, :, 0])),
            int(np.median(patch[:, :, 1])),
            int(np.median(patch[:, :, 2])))


def run_pipeline(img, lut):
    ps = Pixel_Segment(LUT_CSV)
    ps.lut_verde = lut
    return ps.processar(img)


BANDAS = [(0, 160), (160, 320), (320, 480), (480, 640)]


def fracao_por_faixa(gm):
    """Fração de pixels verdes por faixa de linhas (imagem original)."""
    fracs = []
    for y0, y1 in BANDAS:
        faixa = gm[y0:y1, :]
        fracs.append(int((faixa > 0).sum()) / faixa.size)
    return fracs


# ─── carrega imagens ───────────────────────────────────────────────────────────

img_e = cv2.imread(ELIZA)
img_b = cv2.imread(BITBOTS)
hsv_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2HSV)
hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)

lut_base = build_lut()
gm_e = apply_green(hsv_e, lut_base)
wm_e = apply_white(hsv_e)
gm_b = apply_green(hsv_b, lut_base)
wm_b = apply_white(hsv_b)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1: Amostragem de pixels (mediana 3x3)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SEÇÃO 1  Amostragem HSV (mediana 3x3) + pertencimento às máscaras")
print("=" * 72)
print("  calibração verde: H 35-100, S>=40, V 40-180")
print("  calibração branco: V>=165, S<=60, H 0-179")

coords_eliza = [
    ("fundo (200,120)",    200, 120),
    ("fundo (500,80)",     500,  80),
    ("fundo (580,150)",    580, 150),
    ("fundo (100,60)",     100,  60),
    ("poste (320,150)",    320, 150),
    ("carpete (300,550)",  300, 550),
    ("carpete (150,450)",  150, 450),
    ("carpete (500,500)",  500, 500),
]

coords_bitbots = [
    ("fundo (200,60)",     200,  60),
    ("fundo (300,40)",     300,  40),
    ("robô (360,100)",     360, 100),
    ("trave (165,80)",     165,  80),
    ("carpete (300,550)",  300, 550),
    ("carpete (120,480)",  120, 480),
]


def imprimir_amostras(nome, hsv, gm, wm, coords):
    print(f"\n  [{nome}]")
    print(f"  {'Label':22s}  (x,y)       H    S    V  | verde | branco")
    print("  " + "-" * 64)
    for label, x, y in coords:
        hv, sv, vv = median3x3(hsv, x, y)
        g = "SIM" if gm[y, x] > 0 else "nao"
        w = "SIM" if wm[y, x] > 0 else "nao"
        print(f"  {label:22s} ({x:3d},{y:3d})  {hv:3d}  {sv:3d}  {vv:3d}  |  {g:3s}  |  {w:3s}")


imprimir_amostras("eliza",   hsv_e, gm_e, wm_e, coords_eliza)
imprimir_amostras("bitbots", hsv_b, gm_b, wm_b, coords_bitbots)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2: Fração da máscara verde por faixa de linhas
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SEÇÃO 2  Fração da máscara verde por faixa de linhas (640×640, full-res)")
print("=" * 72)


def imprimir_fracao(nome, gm):
    fracs = fracao_por_faixa(gm)
    print(f"\n  [{nome}]")
    print(f"  {'Faixa':9s}  pixels_verdes  total_pixels  fração")
    print("  " + "-" * 50)
    for (y0, y1), f in zip(BANDAS, fracs):
        h_band = y1 - y0
        total = 640 * h_band
        n = int(f * total)
        print(f"  {y0:3d}–{y1:3d}    {n:10d}    {total:8d}    {f:.4f}  ({f*100:.2f}%)")


imprimir_fracao("eliza",   gm_e)
imprimir_fracao("bitbots", gm_b)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3: Overlay eliza → debug_out/eliza_verde_mask_overlay.png
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SEÇÃO 3  Overlay: máscara verde (translúcida) + contorno mascara_convexa")
print("=" * 72)

res_e = run_pipeline(img_e, lut_base)
mascara_convexa_e = res_e['mascara_convexa']   # 80×80
y_gauss_e         = res_e['y_gauss']           # len 80, valores 0..79

mc_orig = cv2.resize(mascara_convexa_e, (640, 640), interpolation=cv2.INTER_NEAREST)

verde_layer = np.zeros_like(img_e)
verde_layer[gm_e > 0] = (0, 200, 0)
ov = cv2.addWeighted(img_e.copy(), 0.7, verde_layer, 0.3, 0)

contornos, _ = cv2.findContours(mc_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ov, contornos, -1, (0, 0, 255), 2)

out_path = os.path.join(OUT, 'eliza_verde_mask_overlay.png')
cv2.imwrite(out_path, ov)

mc_frac_e = float((mascara_convexa_e > 0).mean())
yg_min_e  = float(y_gauss_e.min())

print(f"\n  Salvo: debug_out/eliza_verde_mask_overlay.png")
print(f"  y_gauss min (eliza) : {yg_min_e:.2f}  (grade 0=topo … 79=base)")
print(f"  mascara_convexa eliza: {mc_frac_e * 100:.1f}% da grade (base)")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4: Teste de sensibilidade — APENAS MEDIÇÃO, SEM ESCOLHA DE VARIANTE
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SEÇÃO 4  Sensibilidade (eliza)  — APENAS MEDIÇÃO, SEM ESCOLHA")
print("         variando S  : H 35-100 fixo, V 40-180 fixo")
print("         variando V_min: H 35-100 fixo, S>=40 fixo, V_max=180 fixo")
print("=" * 72)

variantes_s = [
    ("base   S>=40",  40, V_MIN, V_MAX),
    ("       S>=60",  60, V_MIN, V_MAX),
    ("       S>=80",  80, V_MIN, V_MAX),
]
variantes_v = [
    ("base V>=40",    S_BASE, 40, V_MAX),
    ("     V>=60",    S_BASE, 60, V_MAX),
    ("     V>=80",    S_BASE, 80, V_MAX),
]

header = (f"  {'Variante':20s}  "
          f"{'0-160':>8}  {'160-320':>8}  {'320-480':>8}  {'480-640':>8}  "
          f"{'yg_min':>8}  {'mc_%':>6}")
sep = "  " + "-" * 80


def avaliar_variante(nome, s_min, v_min, v_max):
    lut_v  = build_lut(s_min=s_min, v_min=v_min, v_max=v_max)
    gm_v   = apply_green(hsv_e, lut_v)
    fracs  = fracao_por_faixa(gm_v)
    res_v  = run_pipeline(img_e, lut_v)
    yg_min = float(res_v['y_gauss'].min())
    mc_pct = float((res_v['mascara_convexa'] > 0).mean()) * 100
    return fracs, yg_min, mc_pct


print(f"\n  ── variando limiar mínimo de S (H e V iguais) ──")
print(header)
print(sep)
for nome, s_min, v_min, v_max in variantes_s:
    fracs, yg, mc = avaliar_variante(nome, s_min, v_min, v_max)
    print(f"  {nome:20s}  "
          f"{fracs[0]:8.4f}  {fracs[1]:8.4f}  {fracs[2]:8.4f}  {fracs[3]:8.4f}  "
          f"{yg:8.2f}  {mc:6.1f}%")

print(f"\n  ── variando limiar mínimo de V (H e S iguais) ──")
print(header)
print(sep)
for nome, s_min, v_min, v_max in variantes_v:
    fracs, yg, mc = avaliar_variante(nome, s_min, v_min, v_max)
    print(f"  {nome:20s}  "
          f"{fracs[0]:8.4f}  {fracs[1]:8.4f}  {fracs[2]:8.4f}  {fracs[3]:8.4f}  "
          f"{yg:8.2f}  {mc:6.1f}%")

print()
