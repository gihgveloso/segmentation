"""(5) Bitbots: 214 células brancas FORA da mascara_convexa — o que são?

Salva debug_out/bitbots_branco_fora_mc.png com esses pixels em cor distinta.

Uso: python3 scripts/diagnostico/diag_branco_fora_mc_bitbots.py
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
IMG_PATH = os.path.join(BASE, 'images',
    'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg')

img = cv2.imread(IMG_PATH)
if img.shape[2]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
H, W = img.shape[:2]
gH, gW = H//8, W//8

ps = Pixel_Segment(LUT_CSV)
ps.lut_verde = lut_b.copy()
res = ps.processar(img)

white_res = res['white_mask_resized']
mc        = res['mascara_convexa']

white_bin   = (white_res > 127).astype(np.uint8)
mc_bin      = (mc > 0).astype(np.uint8)
fora_mask   = (white_bin & ~mc_bin).astype(np.uint8)
dentro_mask = (white_bin & mc_bin).astype(np.uint8)

n_fora   = int(fora_mask.sum())
n_dentro = int(dentro_mask.sum())
print(f"bitbots ({W}x{H}px  grade {gW}x{gH})")
print(f"Branco fora   da mascara_convexa: {n_fora} células")
print(f"Branco dentro da mascara_convexa: {n_dentro} células")
print()

print("Distribuição das células fora por faixa vertical:")
for r0, r1 in [(0, gH//4), (gH//4, gH//2), (gH//2, 3*gH//4), (3*gH//4, gH)]:
    n = int(fora_mask[r0:r1,:].sum())
    print(f"  grade y={r0:3d}–{r1:3d}  (orig y={r0*8}–{r1*8}):  {n} células")

print()
print("Células fora (r_grade, c_grade, y_orig, x_orig, H_med, S_med, V_med):")
rows, cols = np.where(fora_mask > 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
H_ch = hsv[:,:,0]; S_ch = hsv[:,:,1]; V_ch = hsv[:,:,2]
for r, c in sorted(zip(rows, cols)):
    y0, y1 = r*8, min((r+1)*8, H)
    x0, x1 = c*8, min((c+1)*8, W)
    h_m = float(H_ch[y0:y1, x0:x1].mean())
    s_m = float(S_ch[y0:y1, x0:x1].mean())
    v_m = float(V_ch[y0:y1, x0:x1].mean())
    print(f"  r={r:3d} c={c:3d}  y={r*8:4d} x={c*8:4d}  "
          f"H={h_m:.0f} S={s_m:.0f} V={v_m:.0f}")

# ── Visualização ─────────────────────────────────────────────────────────────
ov = img.copy()

# mascara_convexa — azul translúcido
mc_full = cv2.resize(mc.astype(np.uint8), (W,H), cv2.INTER_NEAREST)
mc_col = np.zeros_like(ov); mc_col[mc_full>0] = (120,0,0)
ov = cv2.addWeighted(ov, 0.8, mc_col, 0.2, 0)

# Branco dentro da mc — ciano
wd_full = cv2.resize((dentro_mask*255).astype(np.uint8), (W,H), cv2.INTER_NEAREST)
ov[wd_full>0] = (0,220,220)

# Branco FORA da mc — AMARELO (BGR: 0,220,255)
wf_full = cv2.resize((fora_mask*255).astype(np.uint8), (W,H), cv2.INTER_NEAREST)
ov[wf_full>0] = (0,220,255)

cv2.putText(ov,
    f"AMARELO=branco_fora_mc ({n_fora}cel)  CIANO=branco_dentro ({n_dentro}cel)  AZUL=campo",
    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

out = os.path.join(OUT, 'bitbots_branco_fora_mc.png')
cv2.imwrite(out, ov)
print(f"\nImagem salva: debug_out/bitbots_branco_fora_mc.png")
