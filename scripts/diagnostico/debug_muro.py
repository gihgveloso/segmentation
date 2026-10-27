"""
Diagnóstico do muro branco (WEBOTS) e esqueleto vazio em bitbots/eliza.
Itens 2 e 4 do pedido do usuário — NÃO modificar main.py.
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from main import Pixel_Segment

LUT_CSV   = os.path.join(os.path.dirname(__file__), 'green_pixels.csv')
IMAGENS   = os.path.join(os.path.dirname(__file__), 'images')
DEBUG_OUT = os.path.join(os.path.dirname(__file__), 'debug_out')
os.makedirs(DEBUG_OUT, exist_ok=True)

seg = Pixel_Segment(LUT_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# ITEM 2 — blocos grossos nas três imagens de referência
# ─────────────────────────────────────────────────────────────────────────────
GROSSAS = ['image_1.png', 'image_2.png', 'image.png']
LIMIAR_DT = 3.0   # células da grade

print("=" * 65)
print("ITEM 2 — blocos grossos (transform. distância ≥ 3)")
print("=" * 65)

for nome in GROSSAS:
    path = os.path.join(IMAGENS, nome)
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERRO] não encontrou {path}")
        continue

    res = seg.processar(frame)

    field_line_mask = res['field_line_mask']         # grade, uint8 0/1..255
    mascara_convexa = res['mascara_convexa']          # grade, uint8 0/255
    contour_points  = res['contour_points']

    # distance transform nos pixels de linha dentro do campo
    mascara_bin = (field_line_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mascara_bin, cv2.DIST_L2, 3)

    # pixels "grossos" = distância >= LIMIAR_DT
    grossos = (dist >= LIMIAR_DT).astype(np.uint8)

    # componentes conectados dos pixels grossos
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        grossos, connectivity=8
    )

    total_area = int(np.sum(grossos))
    print(f"\n  {nome}  [grade {mascara_bin.shape[1]}×{mascara_bin.shape[0]}]")
    print(f"    pixels brancos no campo: {int(np.sum(mascara_bin))}")
    print(f"    pixels grossos (dt≥{LIMIAR_DT:.0f}):  {total_area}")
    print(f"    componentes conectados: {n_cc - 1}")

    for i in range(1, n_cc):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        print(f"      CC{i}: bbox=({x},{y},{w},{h})  área={area}  "
              f"centro=({cx:.1f},{cy:.1f})")

    # ── debug image ──────────────────────────────────────────────────────────
    # Ampliar grade → resolução original para visualização
    H_orig, W_orig = frame.shape[:2]
    H_grid, W_grid = mascara_bin.shape[:2]

    # Fundo = imagem original
    debug = frame.copy()

    # Limite do campo em verde sobre a imagem original
    if len(contour_points) >= 3:
        pts_orig = (contour_points.astype(np.float32)
                    * np.array([W_orig / W_grid, H_orig / H_grid])
                    ).astype(np.int32)
        cv2.polylines(debug, [pts_orig], isClosed=True,
                      color=(0, 255, 0), thickness=2)

    # Blocos grossos em vermelho
    grossos_orig = cv2.resize(
        (grossos * 255).astype(np.uint8),
        (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
    )
    debug[grossos_orig > 0] = (0, 0, 255)

    # Borda do campo (mascara_convexa) em semitransparente azul para contexto
    mascara_orig = cv2.resize(mascara_convexa, (W_orig, H_orig),
                              interpolation=cv2.INTER_NEAREST)
    overlay = debug.copy()
    overlay[mascara_orig == 0] = (40, 40, 40)
    debug = cv2.addWeighted(debug, 0.75, overlay, 0.25, 0)

    stem = os.path.splitext(nome)[0]
    out_path = os.path.join(DEBUG_OUT, f'muro_{stem}.png')
    cv2.imwrite(out_path, debug)
    print(f"    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 4 — diagnóstico do esqueleto vazio em bitbots e eliza
# ─────────────────────────────────────────────────────────────────────────────
DIAGNOSTICAR = [
    'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg',
    'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg',
]

print("\n" + "=" * 65)
print("ITEM 4 — diagnóstico esqueleto vazio (bitbots / eliza)")
print("=" * 65)

for nome in DIAGNOSTICAR:
    path = os.path.join(IMAGENS, nome)
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERRO] não encontrou {path}")
        continue

    print(f"\n  {nome[:50]}…")
    print(f"    imagem original: {frame.shape[1]}×{frame.shape[0]}")

    # Etapa 1: _preparar_frame
    seg._preparar_frame(frame)
    print(f"    grade: {seg.grid_width}×{seg.grid_height}")

    # Etapa 2: _masks_and_resize
    res_masks = seg._masks_and_resize()
    wm  = res_masks['white_mask']             # resolução original
    wmr = res_masks['white_mask_resized']     # grade, float32
    gm  = res_masks['green_mask']
    gmr = res_masks['green_mask_resized']

    px_white_orig = int(np.count_nonzero(wm))
    px_white_grid = int(np.count_nonzero(wmr))
    px_green_orig = int(np.count_nonzero(gm))
    px_green_grid = int(np.count_nonzero(gmr))
    print(f"    white_mask (original): {px_white_orig} px")
    print(f"    white_mask (grade):    {px_white_grid} px")
    print(f"    green_mask (original): {px_green_orig} px")
    print(f"    green_mask (grade):    {px_green_grid} px")

    # Etapa 3: _binarization
    res_bin = seg._binarization(wmr, gmr)
    uniq_img       = res_bin['uniq_img']
    borda_bin      = res_bin['borda_binaria']
    mascara_convexa= res_bin['mascara_convexa']
    campo_det      = res_bin['campo_detectado']
    trow_ef        = res_bin['trow_efetivo']

    px_uniq_positivos = int(np.count_nonzero(uniq_img))
    px_borda_bin      = int(np.count_nonzero(borda_bin))
    area_campo        = int(np.count_nonzero(mascara_convexa))
    print(f"    uniq_img > 0:          {px_uniq_positivos} px")
    print(f"    borda_binaria > 0:     {px_borda_bin} px")
    print(f"    campo_detectado:       {campo_det}")
    print(f"    trow_efetivo:          {trow_ef:.1f}")
    print(f"    mascara_convexa > 0:   {area_campo} px de grade")

    # Etapa 4: _skeletonization_and_connect
    res_skel = seg._skeletonization_and_connect(wm, wmr, mascara_convexa)
    flm      = res_skel['field_line_mask']
    relevo   = res_skel['mascara_relevo']
    skeleton = res_skel['skeleton']

    px_field_line = int(np.count_nonzero(flm))
    px_relevo     = int(np.count_nonzero(relevo))
    px_skeleton   = int(np.count_nonzero(skeleton))
    print(f"    field_line_mask > 0:   {px_field_line} px de grade")
    print(f"    mascara_relevo > 0:    {px_relevo} px de grade")
    print(f"    skeleton True:         {px_skeleton} px de grade")

    # Histograma de y_gauss para ver onde está o topo detectado
    y_gauss = res_bin['y_gauss']
    print(f"    y_gauss min/max:       {y_gauss.min():.1f} / {y_gauss.max():.1f}")
    print(f"    y_gauss médio:         {y_gauss.mean():.1f}  (0=topo, {seg.grid_height-1}=base)")

    # Distribuição de c_xy no relevo (para entender por que o esqueleto é vazio)
    if px_relevo > 0:
        import numpy as np
        padded = np.pad(relevo, ((1,1),(1,1)), mode='constant', constant_values=0)
        vizinhos = np.stack([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, 2:],  padded[1:-1, :-2],
            padded[:-2, 2:],   padded[2:, 2:],
            padded[:-2, :-2],  padded[2:, :-2]
        ])
        c_xy = np.sum(vizinhos >= relevo[np.newaxis], axis=0, dtype=np.uint8)
        nao_fundo = relevo > 0
        cumes = nao_fundo & (c_xy < 3)
        print(f"    pixels relevo > 0:     {int(np.sum(nao_fundo))}")
        print(f"    cumes (c_xy < 3):      {int(np.sum(cumes))}")
        print(f"    distribuição c_xy (nos pixels relevo):")
        for v in range(9):
            n = int(np.sum((c_xy == v) & nao_fundo))
            if n > 0:
                print(f"      c_xy={v}: {n}")

print("\nFim do diagnóstico.")
