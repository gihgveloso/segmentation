"""(c) v2 — Pipeline de produção: etapas, kernels, pixels por estágio.

Uso: python3 scripts/diagnostico/diag_c2_etapas.py
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import cv2

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
OUT  = os.path.join(BASE, 'debug_out')
os.makedirs(OUT, exist_ok=True)

# ── Monkeypatch (sem tocar main.py) ─────────────────────────────────────────
with contextlib.redirect_stdout(io.StringIO()):
    import def_white_threshold as _bm
_bm.lower = np.array([0,  0, 165], dtype=np.int32)
_bm.up    = np.array([179,60, 255], dtype=np.int32)

with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment

lut_b = np.zeros((180,256,256), dtype=np.uint8)
for _h in range(35,101):
    lut_b[_h, 40:256, 40:181] = 255

class PSFator(Pixel_Segment):
    _fator = 8
    def _preparar_frame(self, frame):
        super()._preparar_frame(frame)
        self.grid_height = self.height // self._fator
        self.grid_width  = self.width  // self._fator

LUT_CSV = os.path.join(BASE, 'green_pixels.csv')

IMG_PATHS = {
    'bitbots': os.path.join(BASE,'images',
        'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(BASE,'images','eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
}

# ── Constantes do main.py ────────────────────────────────────────────────────
print("=" * 70)
print("[main.py] CONSTANTES EM PIXELS DE GRADE E KERNELS MORFOLÓGICOS")
print("=" * 70)
TRECHO = """\
Linha 128–131  _binarization():
    kernel_8x4 = np.ones((4, 8), dtype=np.float32)
    → janela 4 linhas × 8 colunas em pixels de GRADE
    → equivalente original: f=8 → 32×64px, f=4 → 16×32px, f=2 → 8×16px

Linha 133–138  filter2D com anchor=(4,0):
    cv2.filter2D(uniq_img, -1, kernel_8x4, anchor=(4, 0))
    → NÃO escala: a janela é sempre 4×8 px de grade, independente do fator

Linha 243      medfilt border smoothing:
    y_median = medfilt(y_vals, kernel_size=3)
    → 3 px de grade; NÃO escala

Linha 247      gaussian_filter1d border smoothing:
    y_gauss = gaussian_filter1d(y_median, sigma=1.0)
    → sigma=1.0 px de grade; NÃO escala

Linha 360–363  _skeletonization_and_connect():
    mascara_relevo = cv2.GaussianBlur(field_line_mask, (5,5), 0)
    → 5×5 px de grade → f=8: blur sobre 40×40 px originais
                      → f=4: blur sobre 20×20 px originais
                      → f=2: blur sobre 10×10 px originais
    → NÃO escala com o fator; a mesma constante (5×5) é usada sempre

Linha 391      ridge detector:
    e_cume_montanha = c_xy < 3
    → threshold de contagem de vizinhos (não é tamanho em px);
      indireto: em grade fina, linhas mais largas → mais vizinhos em plateau
      → mais pixels excluídos → menos esqueleto (ver tabela abaixo)

QUAL NÃO ESCALA: kernel_8x4, medfilt(3), gaussian(sigma=1), GaussianBlur(5,5).
Todos são tamanhos fixos em pixels de GRADE, portanto representam tamanhos
diferentes em pixels originais conforme o fator.
"""
print(TRECHO)

# ── Contagem de pixels por etapa ─────────────────────────────────────────────
print("=" * 70)
print("PIXELS POR ETAPA  (calibração B,  white_res = máscara B na grade)")
print()
print(f"{'Img':<8} {'f':>2} {'gHxgW':>9} {'white_res':>10} {'mc':>8} "
      f"{'flm':>8} {'relevo_nz':>10} {'skeleton':>9} {'sk/relevo':>10}")
print("-" * 80)

STAGE_DATA = {}   # para análise posterior

for nome in ['bitbots','eliza']:
    img = cv2.imread(IMG_PATHS[nome])
    if img.shape[2]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)

    STAGE_DATA[nome] = {}
    for fator in [8, 4, 2]:
        PSFator._fator = fator
        ps = PSFator(LUT_CSV); ps.lut_verde = lut_b.copy()
        ps._preparar_frame(img)
        rm = ps._masks_and_resize()
        rb = ps._binarization(rm['white_mask_resized'], rm['green_mask_resized'])

        wres = rm['white_mask_resized']
        mc   = rb['mascara_convexa']
        flm  = cv2.bitwise_and(wres.astype(np.uint8), mc)
        rlv  = cv2.GaussianBlur(flm, (5,5), 0)
        rs   = ps._skeletonization_and_connect(rm['white_mask'], wres, mc)

        n_wres = int((wres > 127).sum())
        n_mc   = int((mc > 0).sum())
        n_flm  = int((flm > 0).sum())
        n_rlv  = int((rlv > 0).sum())
        n_sk   = int((rs['skeleton_img'] > 0).sum())
        ratio  = n_sk / max(n_rlv, 1)

        STAGE_DATA[nome][fator] = dict(wres=n_wres, mc=n_mc, flm=n_flm,
                                       rlv=n_rlv, sk=n_sk)

        print(f"{nome:<8} {fator:>2} {ps.grid_height:>4}x{ps.grid_width:<4} "
              f"{n_wres:>10} {n_mc:>8} {n_flm:>8} {n_rlv:>10} {n_sk:>9} "
              f"{ratio:>9.1%}")

        # ── Figura side-by-side (white_resized | relevo | skeleton) ──
        DISP = 240   # tamanho de exibição para cada panel

        def to_display(arr8, size=DISP):
            """Uint8 2D → 3-canal RGB redimensionado."""
            c = cv2.cvtColor(arr8.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            return cv2.resize(c, (size,size), interpolation=cv2.INTER_NEAREST)

        # white_mask_resized thresholded
        wres_bin = ((wres > 127)*255).astype(np.uint8)
        # relevo normalizado
        rlv_norm = cv2.normalize(rlv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # skeleton
        sk_img = rs['skeleton_img']
        # mascara_convexa
        mc_vis = mc.copy()

        row = np.concatenate([
            to_display(wres_bin),
            to_display(mc_vis),
            to_display(flm),
            to_display(rlv_norm),
            to_display(sk_img),
        ], axis=1)

        # Cabeçalho
        labels = ['white_res', 'mascara_conv', 'field_line_mask', 'relevo', 'skeleton']
        hdr = np.zeros((20, DISP*5, 3), np.uint8)
        for i, lbl in enumerate(labels):
            cv2.putText(hdr, lbl, (i*DISP+2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        title_bar = np.zeros((20, DISP*5, 3), np.uint8)
        cv2.putText(title_bar, f"{nome}  fator={fator}  {ps.grid_height}x{ps.grid_width}grid  "
                    f"wres={n_wres} flm={n_flm} rlv={n_rlv} sk={n_sk}",
                    (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        panel = np.concatenate([title_bar, hdr, row], axis=0)
        out = os.path.join(OUT, f"{nome}_c3_etapas_f{fator}.png")
        cv2.imwrite(out, panel)
        print(f"         → {out}")

    print()

# ── Análise: onde se perdem os pixels no bitbots f=4→f=2 ───────────────────
print("=" * 70)
print("ANÁLISE: sk cai de 260 (f=4) para 131 (f=2) no bitbots")
print("=" * 70)

d4 = STAGE_DATA['bitbots'][4]
d2 = STAGE_DATA['bitbots'][2]

print(f"""
  f=4: wres={d4['wres']}  flm={d4['flm']}  relevo_nz={d4['rlv']}  skeleton={d4['sk']}
  f=2: wres={d2['wres']}  flm={d2['flm']}  relevo_nz={d2['rlv']}  skeleton={d2['sk']}

  Razão sk/relevo_nz:
    f=4: {d4['sk']}/{d4['rlv']} = {d4['sk']/d4['rlv']:.1%}
    f=2: {d2['sk']}/{d2['rlv']} = {d2['sk']/d2['rlv']:.1%}

  → A PERDA OCORRE EM: relevo → skeleton (ridge detector).
    relevo_nz aumentou de {d4['rlv']} para {d2['rlv']} (+{(d2['rlv']/d4['rlv']-1)*100:.0f}%),
    mas sk CAIU de {d4['sk']} para {d2['sk']} (−{(1-d2['sk']/d4['sk'])*100:.0f}%).

  Causa (mecanismo do ridge detector, main.py linha 384–391):
    skeleton = pixels onde c_xy < 3
    (c_xy = nº de vizinhos 8-conectados com valor >= pixel atual)

    A condição c_xy < 3 exige que o pixel seja maior que pelo menos 6 dos
    8 vizinhos. Isso funciona bem quando as linhas têm 1-2 px de grade.

    A linha real tem ~7 px originais.
    Em grade f=8: 7/8 ≈ 1 px  → linha fina → ridge detector funciona.
    Em grade f=4: 7/4 ≈ 2 px  → linha fina/média → ainda funciona razoavelmente.
    Em grade f=2: 7/2 ≈ 4 px  → linha larga de 4 px na grade.

    Após GaussianBlur(5,5) em grade f=2, uma linha de 4 px vira um plateau
    largo onde muitos pixels têm vizinhos de valor similar (c_xy ≥ 3).
    Apenas os pixels exatamente no topo do plateau passam (c_xy < 3),
    resultando em muito menos pixels de esqueleto por pixel de relevo.

    Evidência direta: sk/relevo = 5.8% em f=4 vs 1.2% em f=2
    (medido na tabela acima).

    Para eliza f=4→f=2: sk AUMENTA (618→737), pois as linhas em eliza têm
    V mais alto que o campo (maior contraste), então o plateau pós-blur é
    mais pronunciado e o ridge detector ainda encontra cristas bem definidas.

  Parâmetro responsável: GaussianBlur((5,5), 0) em main.py linha 360.
    Em f=2, o blur (5 px de grade = 10 px originais) é proporcional à largura
    da linha (~7 px originais), criando o plateau.
    Em f=8, o blur (5 px de grade = 40 px originais) é muito maior que a
    linha (7 px), mas a linha no grid é só 1 px → o plateau não ocorre.
""")
