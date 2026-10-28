"""(4) Verde no topo do eliza: onde estão os pixels verdes (calibração B) na metade
superior da grade, e o que são na imagem original.

Uso: python3 scripts/diagnostico/diag_verde_topo_eliza.py
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import cv2

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
OUT  = os.path.join(BASE, 'debug_out')
os.makedirs(OUT, exist_ok=True)

with contextlib.redirect_stdout(io.StringIO()):
    import def_white_threshold as _bm
_bm.lower = np.array([0,  0, 165], dtype=np.int32)
_bm.up    = np.array([179,60, 255], dtype=np.int32)
with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment

lut_b = np.zeros((180,256,256), dtype=np.uint8)
for _h in range(35,101):
    lut_b[_h, 40:256, 40:181] = 255

LUT_CSV = os.path.join(BASE, 'green_pixels.csv')
IMG_PATH = os.path.join(BASE,'images','eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg')

img = cv2.imread(IMG_PATH)
if img.shape[2]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
H, W = img.shape[:2]
gH, gW = H//8, W//8

ps = Pixel_Segment(LUT_CSV)
ps.lut_verde = lut_b.copy()
res = ps.processar(img)

green_res = res['green_mask_resized']
green_bin = (green_res > 127).astype(np.uint8)

top_half = gH // 2   # = 40 células

green_top = green_bin[:top_half, :]
n_green_top = int(green_top.sum())
print(f"eliza ({W}x{H}px  grade {gW}x{gH})")
print(f"Células verdes na metade superior da grade (y=0..{top_half-1}): {n_green_top}")
print()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
H_ch = hsv[:,:,0]; S_ch = hsv[:,:,1]; V_ch = hsv[:,:,2]

print(f"  {'r_grade':>7} {'c_grade':>7} {'y_orig':>7} {'x_orig':>7}  "
      f"{'H_med':>6} {'S_med':>6} {'V_med':>6}  {'y_frac':>6}")
print("  " + "-" * 60)

rows, cols = np.where(green_top > 0)
for r, c in sorted(zip(rows, cols)):
    y0, y1 = r*8, (r+1)*8
    x0, x1 = c*8, (c+1)*8
    h_m = float(H_ch[y0:y1, x0:x1].mean())
    s_m = float(S_ch[y0:y1, x0:x1].mean())
    v_m = float(V_ch[y0:y1, x0:x1].mean())
    y_frac = (r + 0.5) / gH
    print(f"  {r:>7} {c:>7} {r*8:>7} {c*8:>7}  "
          f"{h_m:>6.1f} {s_m:>6.1f} {v_m:>6.1f}  {y_frac:>5.2f}")

# ── Visualização sobre imagem original ───────────────────────────────────────
ov = img.copy()

# Células verdes na metade superior: VERDE brilhante
for r, c in zip(rows, cols):
    y0, y1 = r*8, min((r+1)*8, H)
    x0, x1 = c*8, min((c+1)*8, W)
    ov[y0:y1, x0:x1, 1] = np.minimum(255, ov[y0:y1, x0:x1, 1].astype(int) + 120)
    ov[y0:y1, x0:x1, 0] = ov[y0:y1, x0:x1, 0] // 2
    ov[y0:y1, x0:x1, 2] = ov[y0:y1, x0:x1, 2] // 2

# Células verdes na metade inferior: roxo translúcido (referência)
rows_bot, cols_bot = np.where(green_bin[top_half:, :] > 0)
rows_bot = rows_bot + top_half
for r, c in zip(rows_bot, cols_bot):
    y0, y1 = r*8, min((r+1)*8, H)
    x0, x1 = c*8, min((c+1)*8, W)
    ov[y0:y1, x0:x1, 2] = np.minimum(255, ov[y0:y1, x0:x1, 2].astype(int) + 80)

# Linha de corte (y = H//2 em pixels originais)
cv2.line(ov, (0, H//2), (W, H//2), (0, 0, 255), 2)

# Grade horizontal a cada 40px (5 células)
for r in range(0, gH+1, 5):
    cv2.line(ov, (0, r*8), (W, r*8), (80, 80, 80), 1)

cv2.putText(ov,
    f"VERDE_BRILHANTE=verde_topo ({n_green_top} celulas)  ROXO=verde_inferior  VERMELHO=y={H//2}",
    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
cv2.putText(ov, f"eliza {W}x{H}px  grade {gW}x{gH}  top={top_half} linhas",
    (4, H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,200), 1)

out = os.path.join(OUT, 'eliza_verde_topo.png')
cv2.imwrite(out, ov)
print(f"\nImagem salva: debug_out/eliza_verde_topo.png")
