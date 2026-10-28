"""(4) mascara_convexa — eliza e bitbots com calibração B.

Mostra: máscara verde, mascara_convexa, branco×mascara_convexa.
Conta pixels brancos dentro vs fora da mascara_convexa.
Identifica se a região saturada da metade superior entra no campo.

Uso: python3 scripts/diagnostico/diag_mascara_convexa.py
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

IMG_PATHS = {
    'bitbots': os.path.join(BASE,'images',
        'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(BASE,'images','eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
}

print("=" * 70)
print("(4) MASCARA_CONVEXA — calibração B")
print("=" * 70)

for nome in ['bitbots', 'eliza']:
    img = cv2.imread(IMG_PATHS[nome])
    if img.shape[2]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
    H, W = img.shape[:2]
    gH, gW = H//8, W//8

    # Rodar pipeline
    ps = Pixel_Segment(LUT_CSV)
    ps.lut_verde = lut_b.copy()
    res = ps.processar(img)

    white_res = res['white_mask_resized']        # grade
    green_res = res['green_mask_resized']        # grade
    mc        = res['mascara_convexa']           # grade
    wm_orig   = res.get('white_mask', None)      # resolução original (branco B)
    flm       = res['field_line_mask']           # grade

    # Branco × mascara_convexa (em grade)
    white_dentro = cv2.bitwise_and((white_res>127).astype(np.uint8)*255, mc)
    white_fora_g = (white_res>127).astype(np.uint8)*255
    white_fora_g[mc>0] = 0

    n_wres_tot = int((white_res > 127).sum())
    n_dentro   = int((white_dentro > 0).sum())
    n_fora     = n_wres_tot - n_dentro
    n_mc       = int((mc > 0).sum())
    n_green    = int((green_res > 127).sum())

    print(f"\n{nome} ({W}x{H}px → grade {gW}x{gH}):")
    print(f"  green_mask_resized: {n_green} px ({n_green/gH/gW*100:.1f}% da grade)")
    print(f"  mascara_convexa:    {n_mc} px ({n_mc/gH/gW*100:.1f}% da grade)")
    print(f"  white_mask_resized: {n_wres_tot} px total")
    print(f"    dentro da mascara_convexa: {n_dentro} px  "
          f"({n_dentro/max(n_wres_tot,1)*100:.1f}% do branco)")
    print(f"    fora   da mascara_convexa: {n_fora} px  "
          f"({n_fora/max(n_wres_tot,1)*100:.1f}% do branco)")

    # Região saturada (V >= 230 na imagem original, topo = y < H//2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V   = hsv[:,:,2]
    sat_mask_orig = ((V >= 230) & (np.arange(H)[:,None] < H//2)).astype(np.uint8)
    n_sat_orig = int(sat_mask_orig.sum())

    # Reescalar para grade
    sat_grade = cv2.resize((sat_mask_orig*255), (gW,gH), cv2.INTER_AREA)
    sat_grade = (sat_grade > 30).astype(np.uint8)*255   # >30 = ≥1 px sat nessa célula

    n_sat_grade = int((sat_grade > 0).sum())
    n_sat_dentro = int(((sat_grade > 0) & (mc > 0)).sum())

    print(f"\n  Região saturada (V≥230, y<{H//2}) na imagem original:")
    print(f"    n_sat_orig = {n_sat_orig} px ({n_sat_orig/(H//2*W)*100:.1f}% da metade sup.)")
    print(f"    n_sat_grade = {n_sat_grade} células de grade")
    print(f"    n_sat_dentro_mc = {n_sat_dentro} células "
          f"({n_sat_dentro/max(n_sat_grade,1)*100:.1f}% da sat_grade)")

    if n_sat_dentro > 0:
        print(f"\n  → A região saturada ENTRA na mascara_convexa ({n_sat_dentro} células).")
        # Verificar causa: onde a mascara_convexa chega no topo?
        mc_top_row = 0
        for r in range(gH):
            if (mc[r,:] > 0).any():
                mc_top_row = r
                break
        print(f"  → mascara_convexa começa na linha de grade r={mc_top_row} "
              f"(original y≈{mc_top_row*8}px)")

        # Causa: máscara verde ou cálculo do contorno?
        # A mascara_convexa é construída a partir de y_vals = borda inferior do campo
        # (percorre de baixo pra cima em cada coluna). Se green_mask tem pixels no topo,
        # uniq_img também terá, borda pode subir.
        green_top_half = int((green_res[:gH//2,:] > 127).sum())
        print(f"\n  Diagnóstico de causa:")
        print(f"    green_res na metade superior da grade ({gH//2} linhas): "
              f"{green_top_half} px")

        # Verificar y_gauss da borda
        y_gauss = res.get('y_gauss', None)
        if y_gauss is not None:
            y_min = float(np.min(y_gauss))
            y_max = float(np.max(y_gauss))
            y_mean = float(np.mean(y_gauss))
            print(f"    y_gauss (borda do campo, em px de grade): "
                  f"min={y_min:.1f} max={y_max:.1f} mean={y_mean:.1f}")
            print(f"    → borda mínima está em y_grade={y_min:.0f} "
                  f"= y_orig≈{y_min*8:.0f}px")
            # Colunas onde borda está muito alta (y < gH//4 = top 25%)
            n_cols_alto = int((y_gauss < gH//4).sum())
            print(f"    Colunas onde borda está no top 25% da grade: {n_cols_alto}/{gW}")

            if green_top_half > 0:
                print(f"\n  CAUSA PROVÁVEL: máscara verde tem {green_top_half} px na metade")
                print(f"  superior da grade → uniq_img tem sinal no topo → borda sobe")
                print(f"  → mascara_convexa engloba região do topo.")
                print(f"  Verificar: quais pixels verdes estão no topo?")
                # Contar verde na grade por faixa
                for band in [(0, gH//4), (gH//4, gH//2), (gH//2, gH)]:
                    r0, r1 = band
                    n_g = int((green_res[r0:r1,:] > 127).sum())
                    print(f"    grade y={r0}-{r1}: {n_g} px verdes")
            else:
                print(f"\n  CAUSA: não é a máscara verde (sem pixels verdes no topo).")
                print(f"  A mascara_convexa usa fillPoly com o contorno da borda —")
                print(f"  mesmo sem verde no topo, a forma convexa pode incluí-lo.")
    else:
        print(f"\n  → A região saturada NÃO entra na mascara_convexa.")

    # ── Salvar figuras ───────────────────────────────────────────────────────
    DISP = 320
    def to_d(arr, size=DISP):
        c = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.resize(c, (size,size), interpolation=cv2.INTER_NEAREST)
    def orig_to_d(arr_orig, size=DISP):
        return cv2.resize(arr_orig, (size,size), interpolation=cv2.INTER_AREA)

    # Figura 1: grade (4 panels)
    green_vis  = (green_res > 127).astype(np.uint8) * 255
    white_vis  = (white_res > 127).astype(np.uint8) * 255

    panels_g = [to_d(green_vis), to_d(mc), to_d(white_vis), to_d(white_dentro)]
    labels_g = ['green_mask', 'mascara_convexa', 'white_mask', 'white*mc']
    hdr = np.zeros((18, DISP*4, 3), np.uint8)
    for i, lbl in enumerate(labels_g):
        cv2.putText(hdr, lbl, (i*DISP+2,13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    title = np.zeros((20, DISP*4, 3), np.uint8)
    cv2.putText(title, f"{nome} calibB  grade {gW}x{gH}  "
                f"wbr_dentro={n_dentro}px  wbr_fora={n_fora}px",
                (2,14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)
    row_g = np.concatenate(panels_g, axis=1)
    fig_g = np.concatenate([title, hdr, row_g], axis=0)
    cv2.imwrite(os.path.join(OUT, f"{nome}_mc_grade.png"), fig_g)

    # Figura 2: imagem original com sobreposições
    ov = img.copy()
    # mascara_convexa (azul translucido)
    mc_full = cv2.resize(mc, (W,H), cv2.INTER_NEAREST)
    mc_col = np.zeros_like(ov)
    mc_col[mc_full>0] = (120,0,0)
    ov = cv2.addWeighted(ov, 0.8, mc_col, 0.2, 0)
    # green_mask (verde)
    gm_full = cv2.resize(green_vis, (W,H), cv2.INTER_NEAREST)
    ov[gm_full>0, 1] = np.minimum(255, ov[gm_full>0, 1].astype(int)+60)
    # white*mc (branco-amarelo)
    wd_full = cv2.resize(white_dentro, (W,H), cv2.INTER_NEAREST)
    ov[wd_full>0] = (0,220,220)
    # linha de corte topo (H//2)
    cv2.line(ov, (0,H//2),(W,H//2),(0,0,255),1)
    cv2.putText(ov, f"AZUL=campo VERDE=grass CIANO=branco*campo RED=y={H//2}",
                (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(ov, f"branco dentro={n_dentro} fora={n_fora}  "
                f"sat_grade={n_sat_grade} sat_dentro={n_sat_dentro}",
                (5,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.imwrite(os.path.join(OUT, f"{nome}_mc_original.png"), ov)

    print(f"\n  Imagens salvas:")
    print(f"    debug_out/{nome}_mc_grade.png")
    print(f"    debug_out/{nome}_mc_original.png")

print()
NEW_FILES = [
    'scripts/diagnostico/diag_b2_perp.py',
    'scripts/diagnostico/diag_c2_etapas.py',
    'scripts/anotar_linhas.py',
    'scripts/avaliar_mascara_branco.py',
    'scripts/diagnostico/diag_mascara_convexa.py',
]
print("=" * 70)
print("ARQUIVOS NOVOS:")
for f in NEW_FILES: print(f"  {f}")
