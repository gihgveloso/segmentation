"""diag_fase1_v2.py — Diagnóstico Fase 1, versão 2.

Cobre (a) geometria, (b) perfis de linha, [trecho] main.py,
(c) pipeline produção, (e) métricas honestas, (d) frações.
Uso: python3 scripts/diagnostico/diag_fase1_v2.py
"""
import io, contextlib, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import cv2

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
OUT  = os.path.join(BASE, 'debug_out')
os.makedirs(OUT, exist_ok=True)

IMG_PATHS = {
    'bitbots': os.path.join(BASE, 'images',
        'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    'eliza':   os.path.join(BASE, 'images', 'eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
    'image_1': os.path.join(BASE, 'images', 'image_1.png'),
}

CALIB_B_WHITE = (np.array([0, 0, 165], dtype=np.int32),
                 np.array([179, 60, 255], dtype=np.int32))

NEW_FILES = []

def save(path, img):
    cv2.imwrite(path, img)
    rel = os.path.relpath(path, BASE)
    NEW_FILES.append(rel)

def load(name):
    p = IMG_PATHS[name]
    img = cv2.imread(p)
    if img is None:
        sys.exit(f"ERRO: não encontrou {p}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def white_mask_b(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, CALIB_B_WHITE[0], CALIB_B_WHITE[1])

def draw_profile_chart(values, label, pw=300, ph=150):
    """Desenha gráfico V×posição usando só OpenCV."""
    canvas = np.ones((ph, pw, 3), np.uint8) * 240
    n = len(values)
    if n < 2:
        return canvas
    xs = np.linspace(10, pw-10, n).astype(int)
    ys = (ph - 10 - np.array(values, float) * (ph-20) / 255).astype(int)
    ys = np.clip(ys, 5, ph-5)
    pts = np.stack([xs, ys], axis=1).reshape(-1,1,2).astype(np.int32)
    cv2.polylines(canvas, [pts], False, (20, 20, 200), 1)
    peak = int(np.argmax(values))
    cv2.circle(canvas, (xs[peak], ys[peak]), 3, (0,0,255), -1)
    cv2.putText(canvas, label, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    cv2.putText(canvas, f"V_max={values[peak]:.0f} px={peak}",
                (5, ph-4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80,0,0), 1)
    cv2.line(canvas, (10,ph-10), (pw-10,ph-10), (0,0,0), 1)
    cv2.line(canvas, (10,5), (10,ph-10), (0,0,0), 1)
    return canvas

def fwhm(profile, background_pct=10):
    """FWHM: distância entre pontos onde V = (V_pico + V_fundo)/2.
    V_fundo = média dos 10% menores valores.
    """
    arr = np.array(profile, float)
    n = len(arr)
    bg = np.mean(np.sort(arr)[:max(1, n//10)])
    pk = arr.max()
    half = (pk + bg) / 2.0
    above = arr >= half
    transitions = np.diff(above.astype(int))
    rises  = np.where(transitions == 1)[0]
    falls  = np.where(transitions == -1)[0]
    if len(rises) == 0 or len(falls) == 0:
        return int(above.sum()), bg, pk, half
    w = falls[-1] - rises[0]
    return int(w), float(bg), float(pk), float(half)


# ─────────────────────────────────────────────────────────────
# (a) GEOMETRIA DAS 2 IMAGENS REAIS
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("(a) GEOMETRIA")
print("=" * 70)

for nome in ['bitbots', 'eliza']:
    img  = load(nome)
    H, W = img.shape[:2]
    ch   = img.shape[2] if img.ndim == 3 else 1
    asp  = W / H

    print(f"\n{nome}:")
    print(f"  Dimensão:      {W} × {H}  canais={ch}  aspecto={asp:.4f}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_s = gray.sum(axis=1).astype(float)   # soma por linha (0..H-1)
    col_s = gray.sum(axis=0).astype(float)   # soma por coluna (0..W-1)

    # Bordas pretas: linhas/colunas com soma < 0.5% do máximo
    thr_row = row_s.max() * 0.005
    thr_col = col_s.max() * 0.005
    black_rows = np.where(row_s < thr_row)[0]
    black_cols = np.where(col_s < thr_col)[0]

    # Bordas consecutivas no topo/base/esq/dir
    def consec_border(arr, arr_len):
        top = 0
        for i in range(arr_len):
            if arr[i] < thr_row:
                top = i + 1
            else:
                break
        bot = 0
        for i in range(arr_len-1, -1, -1):
            if arr[i] < thr_row:
                bot = arr_len - i
            else:
                break
        return top, bot

    top_b, bot_b = consec_border(row_s, H)
    thr_col2 = col_s.max() * 0.005
    left_b = 0
    for i in range(W):
        if col_s[i] < thr_col2: left_b = i+1
        else: break
    right_b = 0
    for i in range(W-1, -1, -1):
        if col_s[i] < thr_col2: right_b = W-i
        else: break

    print(f"  Bordas pretas: top={top_b}px bot={bot_b}px left={left_b}px right={right_b}px")
    print(f"  row_sum min={row_s.min():.0f} max={row_s.max():.0f} "
          f"(5 menores linhas: {sorted(row_s)[:5]})")
    print(f"  col_sum min={col_s.min():.0f} max={col_s.max():.0f} "
          f"(5 menores colunas: {sorted(col_s)[:5]})")
    print(f"  Linhas totais com soma < thr: {len(black_rows)}  "
          f"Colunas: {len(black_cols)}")

    # Distorção: Hough sobre máscara branca
    wmask = white_mask_b(img)
    edges = cv2.Canny(wmask, 50, 150)
    segs  = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                             minLineLength=30, maxLineGap=10)
    if segs is not None:
        # OpenCV 5: shape (N,4); OpenCV 4: shape (N,1,4)
        segs2 = segs.reshape(-1, 4)
        lens = [np.hypot(s[2]-s[0], s[3]-s[1]) for s in segs2]
        print(f"  Hough (branco B): {len(segs2)} segmentos  "
              f"comprimento max={max(lens):.0f}px  médio={np.mean(lens):.1f}px")
        top3 = sorted(segs2, key=lambda s: np.hypot(s[2]-s[0], s[3]-s[1]),
                      reverse=True)[:3]
        for i, s in enumerate(top3):
            x1,y1,x2,y2 = s
            ang = np.degrees(np.arctan2(y2-y1, x2-x1))
            print(f"    seg{i+1}: ({x1},{y1})→({x2},{y2}) "
                  f"len={np.hypot(x2-x1,y2-y1):.0f}px ang={ang:.1f}°")
    else:
        print("  Hough: nenhum segmento detectado (branco B)")

    # Anotar e salvar
    ann = img.copy()
    # Row sum: barra lateral direita (20px de largura, escala)
    bar_w = 30
    bar_x = W - bar_w - 2
    rs_norm = (row_s / row_s.max() * bar_w).astype(int)
    for r in range(H):
        cv2.line(ann, (bar_x, r), (bar_x + rs_norm[r], r), (0, 200, 200), 1)
    # Col sum: barra inferior
    bar_h2 = 20
    bar_y = H - bar_h2 - 2
    cs_norm = (col_s / col_s.max() * bar_h2).astype(int)
    for c in range(W):
        cv2.line(ann, (c, bar_y + bar_h2 - cs_norm[c]), (c, bar_y + bar_h2),
                 (200, 200, 0), 1)
    # Marcar bordas pretas
    for r in black_rows:
        cv2.line(ann, (0,r), (W//4, r), (0,0,255), 1)
    for c in black_cols:
        cv2.line(ann, (c,0), (c, H//4), (0,0,255), 1)
    # Hough lines
    if segs is not None:
        segs2_ann = segs.reshape(-1, 4)
        for s in segs2_ann[:20]:
            x1,y1,x2,y2 = s
            cv2.line(ann, (x1,y1),(x2,y2), (0,255,0), 1)
    cv2.putText(ann, f"{W}x{H} asp={asp:.3f}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(ann, "cyan=row_sum  yellow=col_sum  green=Hough  red=black_px",
                (5, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    out_a = os.path.join(OUT, f"{nome}_a2_bordas.png")
    save(out_a, ann)
    print(f"  → {out_a}")


# ─────────────────────────────────────────────────────────────
# (b) LARGURA DAS LINHAS — 3+ PERFIS PERPENDICULARES
# ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("(b) LARGURA DE LINHAS BRANCAS — PERFIS PERPENDICULARES (px originais)")
print("=" * 70)

PERFIL_Y = {   # linhas horizontais conhecidas ou a procurar
    'bitbots': None,    # procurar automaticamente
    'eliza':   None,
    'image_1': None,
}

def find_horiz_line_rows(img, n=4):
    """Encontra n fileiras com padrão de linha horizontal (pico local alto)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2].astype(float)
    H_img, W_img = V.shape
    wm = cv2.inRange(hsv, CALIB_B_WHITE[0], CALIB_B_WHITE[1])
    row_w = wm.sum(axis=1).astype(float)  # pixels brancos por linha
    # suavização
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(row_w, size=5)
    # encontrar picos locais
    peaks = []
    for r in range(5, H_img - 5):
        if smooth[r] == smooth[r-5:r+6].max() and smooth[r] > smooth.mean() * 2:
            peaks.append((smooth[r], r))
    peaks.sort(reverse=True)
    # escolher n picos separados por pelo menos 30px
    chosen = []
    for _, r in peaks:
        if all(abs(r - c) > 30 for c in chosen):
            chosen.append(r)
        if len(chosen) == n:
            break
    return chosen

def find_vert_line_cols(img, n=2):
    """Encontra n colunas com padrão de linha vertical."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wm = cv2.inRange(hsv, CALIB_B_WHITE[0], CALIB_B_WHITE[1])
    col_w = wm.sum(axis=0).astype(float)
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(col_w, size=5)
    peaks = []
    H_img, W_img = img.shape[:2]
    for c in range(5, W_img - 5):
        if smooth[c] == smooth[max(0,c-5):c+6].max() and smooth[c] > smooth.mean() * 2:
            peaks.append((smooth[c], c))
    peaks.sort(reverse=True)
    chosen = []
    for _, c in peaks:
        if all(abs(c - ch) > 30 for ch in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return chosen

for nome in ['bitbots', 'eliza', 'image_1']:
    img = load(nome)
    H_img, W_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V_img = hsv[:,:,2]

    print(f"\n{nome} ({W_img}x{H_img} px originais):")

    horiz_rows = find_horiz_line_rows(img, n=4)
    vert_cols  = find_vert_line_cols(img, n=2)

    ann = img.copy()
    charts = []
    profile_data = []

    # Perfis perpendiculares a linhas HORIZONTAIS (corte vertical)
    for i, y_c in enumerate(horiz_rows[:3]):
        # Escolher x onde o pico de V é maior nessa linha
        row_V = V_img[y_c, :].astype(float)
        x_best = int(np.argmax(row_V))
        x_best = np.clip(x_best, 30, W_img-31)

        # Perfil vertical de 61px centrado em y_c, na coluna x_best
        y0, y1 = max(0, y_c-30), min(H_img, y_c+31)
        profile = V_img[y0:y1, x_best].tolist()
        w, bg, pk, half = fwhm(profile)
        center_rel = y_c - y0  # posição do centro dentro do perfil

        print(f"  Perfil H{i+1}: y_linha={y_c} x_corte={x_best} "
              f"V_pico={pk:.0f} V_fundo={bg:.0f} V_meio={half:.0f} "
              f"largura_FWHM={w}px")
        profile_data.append(f"H{i+1}_y{y_c}")

        # Anotar linha vertical de corte na imagem
        cv2.line(ann, (x_best, y0), (x_best, y1), (0, 255, 255), 2)
        cv2.circle(ann, (x_best, y_c), 4, (0, 0, 255), -1)
        cv2.putText(ann, f"H{i+1} w={w}px", (x_best+3, y_c-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        chart = draw_profile_chart(profile, f"H{i+1} y={y_c} x={x_best}")
        charts.append(chart)

    # Perfis perpendiculares a linhas VERTICAIS (corte horizontal)
    for j, x_c in enumerate(vert_cols[:1]):   # 1 vertical para completar 3+
        # Escolher y onde pico de V é maior nessa coluna
        col_V = V_img[:, x_c].astype(float)
        y_best = int(np.argmax(col_V))
        y_best = np.clip(y_best, 30, H_img-31)

        x0, x1 = max(0, x_c-30), min(W_img, x_c+31)
        profile = V_img[y_best, x0:x1].tolist()
        w, bg, pk, half = fwhm(profile)

        print(f"  Perfil V{j+1}: x_linha={x_c} y_corte={y_best} "
              f"V_pico={pk:.0f} V_fundo={bg:.0f} V_meio={half:.0f} "
              f"largura_FWHM={w}px")
        profile_data.append(f"V{j+1}_x{x_c}")

        cv2.line(ann, (x0, y_best), (x1, y_best), (255, 200, 0), 2)
        cv2.circle(ann, (x_c, y_best), 4, (255, 0, 0), -1)
        cv2.putText(ann, f"V{j+1} w={w}px", (x_c+3, y_best-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)

        chart = draw_profile_chart(profile, f"V{j+1} x={x_c} y={y_best}")
        charts.append(chart)

    # Compor: imagem original + gráficos empilhados
    chart_h = 150
    chart_w = 300
    charts_resized = [cv2.resize(c, (chart_w, chart_h)) for c in charts]
    total_chart_h = chart_h * len(charts_resized)
    chart_stack = np.concatenate(charts_resized, axis=0) if charts_resized else \
                  np.ones((100, chart_w, 3), np.uint8)*200

    # Redimensionar anotação para a mesma altura
    ann_h = total_chart_h if total_chart_h > 0 else H_img
    ann_scaled = cv2.resize(ann, (int(W_img * ann_h / H_img), ann_h))
    if chart_stack.shape[0] != ann_scaled.shape[0]:
        diff = ann_scaled.shape[0] - chart_stack.shape[0]
        if diff > 0:
            chart_stack = np.pad(chart_stack, ((0,diff),(0,0),(0,0)),
                                 mode='constant', constant_values=240)
        else:
            ann_scaled = ann_scaled[:chart_stack.shape[0]]
    composite = np.concatenate([ann_scaled, chart_stack], axis=1)
    out_b = os.path.join(OUT, f"{nome}_b2_perfis.png")
    save(out_b, composite)
    print(f"  → {out_b}")


# ─────────────────────────────────────────────────────────────
# [trecho] MAIN.PY — _masks_and_resize, fator 8, monkeypatch
# ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("[trecho] MAIN.PY — linhas relevantes")
print("=" * 70)
TRECHO = """\
main.py:4   import def_white_threshold as branco
main.py:33  def _preparar_frame(self, frame):
main.py:44      self.grid_height = self.height // 8   ← FATOR 8 FIXO
main.py:45      self.grid_width  = self.width  // 8   ← FATOR 8 FIXO

main.py:58  def _masks_and_resize(self):
main.py:60      lower = branco.lower    ← lido do módulo, em tempo de execução
main.py:61      up    = branco.up       ← idem
main.py:63      white_mask = cv2.inRange(self.img_hsv, lower, up)
main.py:66      green_mask = self.lut_verde[self.h_img, self.s_img, self.v_img]
main.py:68      white_mask_resized = cv2.resize(white_mask, (self.grid_width,
                   self.grid_height), interpolation=cv2.INTER_AREA).astype(float32)
main.py:69      green_mask_resized = cv2.resize(green_mask, ...)

Como monkeypatch sem tocar main.py:

  1. Calibração branco (branco.lower/up):
     import def_white_threshold as branco_mod   # antes de importar main
     branco_mod.lower = np.array([0, 0, 165])   # set B
     branco_mod.up    = np.array([179, 60, 255])
     → main.py usa o mesmo objeto de módulo (sys.modules compartilhado)
       e lê branco.lower/up em tempo de chamada (não no import)

  2. Calibração verde (lut_verde):
     ps.lut_verde = lut_nova  # antes de processar; é atributo de instância

  3. Fator de subamostragem:
     class PSFator(Pixel_Segment):
         _fator = 8
         def _preparar_frame(self, frame):
             super()._preparar_frame(frame)
             self.grid_height = self.height // self._fator
             self.grid_width  = self.width  // self._fator
     PSFator._fator = 4   # troca sem tocar main.py
     → _preparar_frame é o único lugar onde grid_height/width são setados;
       sobrescrever via subclasse é suficiente.

Limitação: Pixel_Segment.processar chama _preparar_frame internamente,
então o patch precisa estar na subclasse (não pode fazer ps.grid_height = X
antes de processar; seria sobrescrito).
"""
print(TRECHO)


# ─────────────────────────────────────────────────────────────
# (c) PIPELINE DE PRODUÇÃO — fator 8/4/2 × calibração B
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("(c) PIPELINE PRODUÇÃO — fator 8/4/2, calibração B, bitbots e eliza")
print("=" * 70)

# Patch branco ANTES de importar main
with contextlib.redirect_stdout(io.StringIO()):
    import def_white_threshold as branco_mod

branco_mod.lower = np.array([0, 0, 165], dtype=np.int32)
branco_mod.up    = np.array([179, 60, 255], dtype=np.int32)

with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment

# LUT verde calibração B: H=35-100, S=40-255, V=40-180
lut_b = np.zeros((180, 256, 256), dtype=np.uint8)
for _h in range(35, 101):
    lut_b[_h, 40:256, 40:181] = 255

# Subclasse para fator configurável
class PSFator(Pixel_Segment):
    _fator = 8
    def _preparar_frame(self, frame):
        super()._preparar_frame(frame)
        self.grid_height = self.height // self._fator
        self.grid_width  = self.width  // self._fator

LUT_CSV = os.path.join(BASE, 'green_pixels.csv')

# Importar features
from features.pipeline   import construir_grafo, ConfigGrafo
from features.simplify   import simplificar, ConfigSimplificacao
from features.crossings  import detectar_cruzamentos, ConfigCruzamento

print(f"{'Img':<8} {'Fat':>3} {'gHxgW':>10} {'campo':>6} "
      f"{'sk_px':>7} {'nós':>6} {'cruzam.':>8}")
print("-" * 60)

for nome in ['bitbots', 'eliza']:
    img = load(nome)
    H_img, W_img = img.shape[:2]

    for fator in [8, 4, 2]:
        PSFator._fator = fator
        ps = PSFator(LUT_CSV)
        ps.lut_verde = lut_b.copy()
        # Patch permanece em branco_mod (não resetado entre iterações)

        try:
            res = ps.processar(img)
        except Exception as e:
            print(f"{nome:<8} {fator:>3} ERRO: {e}")
            continue

        campo = res.get('campo_detectado', False)
        sk    = res.get('skeleton_img', None)
        sk_px = int((sk > 0).sum()) if sk is not None else 0
        gH, gW = sk.shape if sk is not None else (0, 0)

        n_nos  = 0
        n_cruz = 0
        if sk_px > 0:
            try:
                relevo = res['mascara_relevo'].astype(np.float32)
                branco_r = res['white_mask_resized'].astype(np.uint8)
                verde_r  = res['green_mask_resized'].astype(np.uint8)
                sk_bool  = sk.astype(bool)
                g = construir_grafo(sk_bool, relevo, ConfigGrafo(), branco=branco_r)
                n_nos = len(g.nodes)
                simplificar(g, sk_bool, relevo, ConfigSimplificacao())
                aceitos, _ = detectar_cruzamentos(g, branco_r, verde_r,
                                                  ConfigCruzamento())
                n_cruz = len(aceitos)
            except Exception as e:
                n_nos  = -1
                n_cruz = -1

        print(f"{nome:<8} {fator:>3} {gH:>4}x{gW:<5} "
              f"{'SIM' if campo else 'NAO':>6} "
              f"{sk_px:>7} {n_nos:>6} {n_cruz:>8}")

        # Overlay do esqueleto na imagem ORIGINAL
        if sk is not None and sk_px > 0:
            sk_full = cv2.resize(sk, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
            ov = img.copy()
            ov[sk_full > 0] = (0, 255, 0)       # verde brilhante
            # Também mostrar mascara_convexa em translucido
            mc = res.get('mascara_convexa', None)
            if mc is not None:
                mc_full = cv2.resize(mc, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
                mc_col = np.zeros_like(ov)
                mc_col[mc_full > 0] = (100, 0, 0)
                ov = cv2.addWeighted(ov, 0.85, mc_col, 0.15, 0)
            cv2.putText(ov, f"fator={fator} sk={sk_px}px nos={n_nos} cruz={n_cruz}",
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            out_c = os.path.join(OUT, f"{nome}_c2_sk_f{fator}.png")
            save(out_c, ov)
            print(f"         → {out_c}")
    print()


# ─────────────────────────────────────────────────────────────
# (e) MÉTRICAS HONESTAS: dentro e fora da faixa de linha
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("(e) MÉTRICAS HONESTAS — pixels dentro/fora da faixa ±15px")
print()
print("Definições:")
print("  faixa_linha = ±15 linhas ao redor de y_linha (linha conhecida)")
print("  dentro_captdos = pixels capturados DENTRO da faixa")
print("  fora_captdos   = pixels capturados FORA da faixa")
print("  razão = fora_captdos / max(dentro_captdos, 1)")
print("  (razão>1 significa que captura mais fundo que linha)")
print("=" * 70)

LINHA_Y = {'bitbots': 464, 'eliza': 450, 'image_1': 392}

CRITERIOS = [
    ('V>=213_fixo',    lambda hsv: (hsv[:,:,2] >= 213)),
    ('V>=1.5xcamp',    None),  # calculado por imagem
    ('V>=2.0xcamp',    None),
    ('V>=2.5xcamp',    None),
    ('V>=3.0xcamp',    None),
    ('tophat>=20',     None),
    ('tophat>=40',     None),
    ('tophat>=60',     None),
]

# LUT verde para mediana do campo (usar B green range para consistência)
def verde_mask_b(hsv):
    lo = np.array([35, 40, 40], dtype=np.uint8)
    hi = np.array([100, 255, 180], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)

def tophat_v(img_bgr, kernel_size=25):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(V, cv2.MORPH_TOPHAT, k)

for nome in ['bitbots', 'eliza', 'image_1']:
    img = load(nome)
    H_img, W_img = img.shape[:2]
    y_c = LINHA_Y[nome]
    yw0, yw1 = max(0, y_c-15), min(H_img, y_c+16)
    n_dentro_total = (yw1 - yw0) * W_img
    n_fora_total   = H_img * W_img - n_dentro_total

    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V    = hsv[:,:,2].astype(float)
    gm   = verde_mask_b(hsv)
    V_campo = float(np.median(V[gm > 0])) if gm.any() else 0.0
    th   = tophat_v(img)

    print(f"\n{nome}:  y_linha={y_c}  V_campo(verde B)={V_campo:.1f}")
    print(f"  faixa: linhas {yw0}-{yw1}  n_dentro_total={n_dentro_total}  "
          f"n_fora_total={n_fora_total}")
    print()
    hdr = f"  {'Critério':<22} {'dent_capt':>10} {'fora_capt':>10} "
    hdr += f"{'razão':>7}  {'dent%':>6}  {'fora%':>6}"
    print(hdr)
    print("  " + "-"*65)

    # Acumular fora-da-linha masks para a imagem final
    masks_fora = {}

    for label, fn in CRITERIOS:
        if label == 'V>=213_fixo':
            mask = (V >= 213).astype(np.uint8)
        elif label.startswith('V>=') and 'xcamp' in label:
            k_val = float(label.split('>=')[1].replace('xcamp',''))
            thr_v = V_campo * k_val
            mask = (V >= thr_v).astype(np.uint8)
        elif label.startswith('tophat'):
            thr_th = int(label.split('>=')[1])
            mask = (th >= thr_th).astype(np.uint8)
        else:
            continue

        dent = int(mask[yw0:yw1, :].sum())
        fora = int(mask.sum()) - dent
        razao = fora / max(dent, 1)
        dent_pct = dent / n_dentro_total * 100
        fora_pct = fora / n_fora_total * 100

        print(f"  {label:<22} {dent:>10} {fora:>10} {razao:>7.1f}  "
              f"{dent_pct:>5.1f}%  {fora_pct:>5.1f}%")

        masks_fora[label] = mask

    # Salvar máscara "fora da linha" — para identificar o que é capturado além das linhas
    # Usar V>=1.5xcamp como representante (captura mais coisas)
    crit_rep = 'V>=1.5xcamp'
    if crit_rep in masks_fora:
        mask_fora_img = masks_fora[crit_rep].copy()
        mask_fora_img[yw0:yw1, :] = 0   # zera a faixa da linha
        ov_e = img.copy()
        ov_e[mask_fora_img > 0] = (0, 0, 255)   # vermelho = capturado fora da linha
        # Marcar a faixa da linha em verde translúcido
        ov_e[yw0:yw1, :] = cv2.addWeighted(
            ov_e[yw0:yw1,:], 0.7,
            np.full_like(ov_e[yw0:yw1,:], (0,200,0)), 0.3, 0)
        cv2.line(ov_e, (0,y_c),(W_img,y_c),(0,255,0),1)
        cv2.putText(ov_e, f"RED=V>={V_campo*1.5:.0f}(1.5x) fora linha  GREEN=faixa linha",
                    (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out_e = os.path.join(OUT, f"{nome}_e2_fora_linha.png")
        save(out_e, ov_e)
        print(f"\n  → fora-linha anotado: {out_e}")

    # Identificar regiões brilhantes fora da linha
    if 'V>=213_fixo' in masks_fora:
        mask_f = masks_fora['V>=213_fixo'].copy()
        mask_f[yw0:yw1, :] = 0
        n_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_f)
        big = [(i, stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP]) for i in range(1, n_cc)
               if stats[i, cv2.CC_STAT_AREA] > 10]
        big.sort(key=lambda x: -x[1])
        if big:
            print(f"  Regiões brilhantes V>=213 FORA da linha (top-5 por área):")
            for idx, (i, area, xl, yl) in enumerate(big[:5]):
                print(f"    CC{idx+1}: área={area}px  pos=({xl},{yl})  "
                      f"  ← julgar visualmente em imagem")
        else:
            print("  Nenhuma região brilhante V>=213 fora da linha (n_cc=0)")


# ─────────────────────────────────────────────────────────────
# (d) FRAÇÕES DA IMAGEM CAPTURADAS
# ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("(d) FRAÇÕES DA IMAGEM CAPTURADAS (branco e verde)")
print("=" * 70)

CALIBS_D = {
    'original': {
        'white': None,   # usará branco_mod (original de def_white_threshold)
        'green': 'lut',  # usará LUT csv
    },
    'A_liberal':      {'wlo': (0,0,150), 'whi': (179,80,255), 'glo': (30,25,25), 'ghi': (110,255,200)},
    'B_moderado':     {'wlo': (0,0,165), 'whi': (179,60,255), 'glo': (35,40,40), 'ghi': (100,255,180)},
    'C_verde_aperto': {'wlo': (0,0,155), 'whi': (179,70,255), 'glo': (45,70,50), 'ghi': (95,255,190)},
}

# Recarregar original
with contextlib.redirect_stdout(io.StringIO()):
    import importlib, def_white_threshold as _dw
    importlib.reload(_dw)
import def_white_threshold as branco_orig

print(f"  Calibração original: lower={branco_orig.lower}  up={branco_orig.up}")
print()

hdr = f"  {'Imagem':<10} {'Calib':<18} {'W_px':>8} {'W%':>7} {'G_px':>9} {'G%':>7}"
print(hdr)
print("  " + "-" * 62)

frac_data = {}
for nome in ['bitbots', 'eliza', 'image_1']:
    img = load(nome)
    H_img, W_img = img.shape[:2]
    N_total = H_img * W_img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frac_data[nome] = {}

    for cal_nome, cal in CALIBS_D.items():
        if cal_nome == 'original':
            # Usar branco original + LUT csv
            with contextlib.redirect_stdout(io.StringIO()):
                ps_orig = Pixel_Segment(LUT_CSV)
            lo_orig = np.array(branco_orig.lower, dtype=np.int32)
            up_orig = np.array(branco_orig.up, dtype=np.int32)
            wm = cv2.inRange(hsv, lo_orig, up_orig.astype(np.int32))
            h_, s_, v_ = cv2.split(hsv)
            gm = ps_orig.lut_verde[h_, s_, v_]
        else:
            lo_w = np.array(cal['wlo'], dtype=np.uint8)
            hi_w = np.array(cal['whi'], dtype=np.uint8)
            lo_g = np.array(cal['glo'], dtype=np.uint8)
            hi_g = np.array(cal['ghi'], dtype=np.uint8)
            wm = cv2.inRange(hsv, lo_w, hi_w)
            gm = cv2.inRange(hsv, lo_g, hi_g)

        w_px = int((wm > 0).sum())
        g_px = int((gm > 0).sum())
        w_pct = w_px / N_total * 100
        g_pct = g_px / N_total * 100
        frac_data[nome][cal_nome] = (w_px, w_pct, g_px, g_pct)

        print(f"  {nome:<10} {cal_nome:<18} {w_px:>8} {w_pct:>6.1f}% "
              f"{g_px:>9} {g_pct:>6.1f}%")

    print()

# Destaque: branco real vs Webots
print("  Destaque — branco % em calibração B:")
for nome in ['bitbots', 'eliza', 'image_1']:
    w_px, w_pct, _, _ = frac_data[nome].get('B_moderado', (0,0,0,0))
    print(f"    {nome:<10}: {w_pct:.1f}%  ({w_px} px de {IMG_PATHS[nome].split('/')[-1][:30]})")

print()
print("  Destaque — branco % ORIGINAL (def_white_threshold.py):")
for nome in ['bitbots', 'eliza', 'image_1']:
    w_px, w_pct, _, _ = frac_data[nome].get('original', (0,0,0,0))
    print(f"    {nome:<10}: {w_pct:.3f}%  ({w_px} px)")

# Anotação: o que o branco B captura ALÉM das linhas
print()
print("  Anotando regiões brancas B por image (imagens salvas):")
for nome in ['bitbots', 'eliza', 'image_1']:
    img = load(nome)
    H_img, W_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo_w = np.array(CALIBS_D['B_moderado']['wlo'], dtype=np.uint8)
    hi_w = np.array(CALIBS_D['B_moderado']['whi'], dtype=np.uint8)
    wm = cv2.inRange(hsv, lo_w, hi_w)

    # Componentes conectados do branco
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(wm)
    big_ccs = [(i, stats[i, cv2.CC_STAT_AREA],
                stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
               for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] > 50]
    big_ccs.sort(key=lambda x: -x[1])

    print(f"\n  {nome}: top-8 componentes brancos (calibração B):")
    for idx, (i, area, xl, yl, wcc, hcc) in enumerate(big_ccs[:8]):
        print(f"    CC{idx+1}: área={area:6d}px  bbox=({xl},{yl})+{wcc}x{hcc}")

    ov_d = img.copy()
    ov_d[wm > 0] = (0, 150, 255)   # laranja = branco capturado
    # Numerar os top-8 no overlay
    for idx, (i, area, xl, yl, wcc, hcc) in enumerate(big_ccs[:8]):
        cx = xl + wcc//2
        cy = yl + hcc//2
        cv2.rectangle(ov_d, (xl,yl),(xl+wcc,yl+hcc),(0,0,255),1)
        cv2.putText(ov_d, str(idx+1), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(ov_d, "ORANGE=branco B  numeros=CC por tamanho",
                (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    out_d = os.path.join(OUT, f"{nome}_d2_branco_anotado.png")
    save(out_d, ov_d)
    print(f"  → {out_d}")


# ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("ARQUIVOS NOVOS CRIADOS:")
for f in NEW_FILES:
    print(f"  {f}")
