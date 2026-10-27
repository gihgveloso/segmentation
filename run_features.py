"""Visualizador de debug integrado — Etapas III-C + IV.

Uso:
    python run_features.py [--salvar] [imagem ...]

Teclas (modo interativo):
    a / d   — imagem anterior / próxima
    q       — sair
    n       — alternar: nós do grafo (cor por grau)
    g       — alternar: arestas do grafo (pré-simp)
    c       — alternar: cruzamentos L/T/X coloridos + confiança
    l       — alternar: linhas (IV-B) coloridas
    t       — alternar: linha y_topo_verde em ciano
    h       — mostrar esta ajuda no terminal

Opções:
    --salvar   Grava PNGs em debug_out/ sem abrir janela.
               Ativado automaticamente quando não há DISPLAY.
    imagem     Caminho(s) de imagem; padrão: images/image_1.png
               images/image_2.png images/image.png

Saída salva em debug_out/features_<basename>.png.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Pixel_Segment
from features import extrair_features, FeaturesResult, ConfigFeatures

# ─── Constantes de visualização ──────────────────────────────────────────────

SCALE = 8   # fator grade→imagem original

COR_L = (0, 165, 255)     # laranja
COR_T = (50, 220, 50)     # verde
COR_X = (0, 220, 255)     # ciano
COR_REJ = (110, 110, 110) # cinza

_CORES_LINHA = [
    (0, 200, 255), (0, 255, 100), (255, 100, 0),
    (200, 0, 255), (0, 160, 255), (255, 200, 0),
    (100, 255, 200), (255, 0, 160), (150, 150, 255),
    (0, 255, 200), (255, 150, 0), (180, 0, 255),
]

_COR_NOS = {
    0: (180, 180,   0),
    1: (255, 200,   0),
    2: (200, 200, 200),
    3: (  0, 255,   0),
    4: (  0, 200, 255),
}

_IMAGENS_PADRAO = [
    'images/image_1.png',
    'images/image_2.png',
    'images/image.png',
]

# ─── Verificação de display ───────────────────────────────────────────────────

def _tem_display() -> bool:
    """Retorna True se consegue abrir uma janela OpenCV (nome ASCII)."""
    if not os.environ.get('DISPLAY', ''):
        return False
    _win = '_runfeat_test'
    try:
        _img = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.namedWindow(_win, cv2.WINDOW_NORMAL)
        cv2.imshow(_win, _img)
        cv2.waitKey(1)
        cv2.destroyWindow(_win)
        return True
    except Exception:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return False


# ─── Pipeline por imagem ──────────────────────────────────────────────────────

class _DadosImagem:
    __slots__ = ('basename', 'img_orig', 'result', 'verde', 'skel', 'branco',
                 'relevo', 'H_g', 'W_g', 'y_topo')

    def __init__(self, basename: str, img_orig, result: FeaturesResult,
                 verde, skel, branco, relevo):
        self.basename = basename
        self.img_orig = img_orig
        self.result   = result
        self.verde    = verde
        self.skel     = skel
        self.branco   = branco
        self.relevo   = relevo
        H, W = verde.shape
        self.H_g, self.W_g = H, W
        self.y_topo = _calcular_topo(verde)


def _calcular_topo(verde: np.ndarray, corrida_min: int = 3) -> np.ndarray:
    H, W = verde.shape
    y_topo = np.full(W, float('nan'), dtype=np.float32)
    for c in range(W):
        run = 0
        for r in range(H):
            if verde[r, c] > 0:
                run += 1
                if run >= corrida_min:
                    y_topo[c] = float(r - corrida_min + 1)
                    break
            else:
                run = 0
    valido = np.isfinite(y_topo)
    if valido.sum() >= 2:
        x = np.arange(W)
        y_topo[~valido] = np.interp(x[~valido], x[valido], y_topo[valido])
    return y_topo


def _carregar(img_path: str, ps: Pixel_Segment) -> _DadosImagem | None:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f'  [aviso] {img_path}: não encontrado.')
        return None
    res = ps.processar(img)
    skel   = res['skeleton_img'].astype(bool)
    verde  = res['green_mask_resized'].astype(np.uint8)
    branco = res['white_mask_resized'].astype(np.uint8)
    relevo = res['mascara_relevo'].astype(np.float32)
    if skel.sum() == 0:
        print(f'  [aviso] {basename}: sem esqueleto (imagem sem campo detectado).')
    result = extrair_features(res)
    print(f'  {basename}: {len(result.crossings)} cruzamentos, {len(result.lines)} linhas.')
    return _DadosImagem(basename, img, result, verde, skel, branco, relevo)


# ─── Desenho de camadas ───────────────────────────────────────────────────────

def _desenhar_grafo_nos(
    canvas: np.ndarray,
    dados: _DadosImagem,
    mostrar_nos: bool,
    mostrar_arestas: bool,
) -> None:
    """Reconstrói o grafo pré-simp e desenha nós e/ou arestas sobre canvas."""
    if not mostrar_nos and not mostrar_arestas:
        return
    from features.pipeline import construir_grafo, ConfigGrafo
    g = construir_grafo(dados.skel, dados.relevo, ConfigGrafo(), branco=dados.branco)

    if mostrar_arestas:
        for a, b in g.edges:
            if a not in g.nodes or b not in g.nodes:
                continue
            na, nb = g.nodes[a], g.nodes[b]
            pt1 = (int(na.x * SCALE), int(na.y * SCALE))
            pt2 = (int(nb.x * SCALE), int(nb.y * SCALE))
            cv2.line(canvas, pt1, pt2, (160, 160, 160), 1)

    if mostrar_nos:
        for nid, nd in g.nodes.items():
            d = g.degree(nid)
            cor = _COR_NOS.get(d, (0, 0, 255))
            r = 3 if d >= 3 else 2
            cx = int(nd.x * SCALE)
            cy = int(nd.y * SCALE)
            cv2.circle(canvas, (cx, cy), r, cor, -1)


def _desenhar_cruzamentos(canvas: np.ndarray, dados: _DadosImagem) -> None:
    for cr in dados.result.crossings:
        cor = {'L': COR_L, 'T': COR_T, 'X': COR_X}.get(cr.tipo, COR_REJ)
        cx = int(cr.x * SCALE)
        cy = int(cr.y * SCALE)
        cv2.circle(canvas, (cx, cy), 7, cor, 2)
        cv2.putText(canvas, f'{cr.tipo} {cr.confianca:.1f}',
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cor, 1, cv2.LINE_AA)


def _desenhar_linhas(canvas: np.ndarray, dados: _DadosImagem) -> None:
    for i, lf in enumerate(dados.result.lines):
        cor = _CORES_LINHA[i % len(_CORES_LINHA)]
        pt1 = (int(lf.x1 * SCALE), int(lf.y1 * SCALE))
        pt2 = (int(lf.x2 * SCALE), int(lf.y2 * SCALE))
        cv2.line(canvas, pt1, pt2, cor, 2)
        cv2.circle(canvas, pt1, 4, cor, -1)
        cv2.circle(canvas, pt2, 4, cor, -1)


def _desenhar_topo(canvas: np.ndarray, dados: _DadosImagem) -> None:
    H_o, W_o = canvas.shape[:2]
    y_topo = dados.y_topo
    W_g = dados.W_g
    for c in range(W_g):
        yt = y_topo[c]
        if not math.isfinite(yt):
            continue
        x0 = int(c * W_o / W_g)
        x1 = int((c + 1) * W_o / W_g)
        yp = int(round(yt * H_o / dados.H_g))
        yp = max(0, min(H_o - 1, yp))
        canvas[max(0, yp - 1):yp + 2, x0:x1] = (255, 255, 0)  # ciano


# ─── Renderizar frame com camadas ativas ─────────────────────────────────────

def _renderizar(dados: _DadosImagem, camadas: dict) -> np.ndarray:
    canvas = dados.img_orig.copy()
    # Esqueleto em cinza claro
    H_o, W_o = canvas.shape[:2]
    skel_up = cv2.resize(dados.skel.astype(np.uint8) * 200,
                         (W_o, H_o), interpolation=cv2.INTER_NEAREST)
    canvas[skel_up > 0] = (210, 210, 210)
    if camadas['g'] or camadas['n']:
        _desenhar_grafo_nos(canvas, dados, camadas['n'], camadas['g'])
    if camadas['l']:
        _desenhar_linhas(canvas, dados)
    if camadas['c']:
        _desenhar_cruzamentos(canvas, dados)
    if camadas['t']:
        _desenhar_topo(canvas, dados)
    return canvas


# ─── Salvar modo offline ──────────────────────────────────────────────────────

def _salvar(dados: _DadosImagem) -> None:
    os.makedirs('debug_out', exist_ok=True)
    camadas = {'n': True, 'g': True, 'c': True, 'l': True, 't': True}
    canvas = _renderizar(dados, camadas)
    saida = f'debug_out/features_{dados.basename}.png'
    cv2.imwrite(saida, canvas)
    print(f'  → {saida}')


# ─── Loop interativo ──────────────────────────────────────────────────────────

_AJUDA = """
Teclas:
  a/d  — imagem anterior/próxima
  n    — nós do grafo (cor por grau: amarelo=term, cinza=deg2, verde=deg3, ciano=deg4)
  g    — arestas do grafo (pré-simp)
  c    — cruzamentos L/T/X (laranja/verde/ciano) + confiança
  l    — linhas IV-B
  t    — y_topo_verde em ciano
  h    — esta ajuda
  q    — sair
"""


def _loop_interativo(lista: list[_DadosImagem]) -> None:
    WIN = 'run_features'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    idx = 0
    camadas = {'n': False, 'g': False, 'c': True, 'l': True, 't': True}
    redraw = True

    while True:
        dados = lista[idx]
        if redraw:
            frame = _renderizar(dados, camadas)
            info = (f'{dados.basename}  [{dados.W_g}x{dados.H_g}]  '
                    f'{idx+1}/{len(lista)}  '
                    f'c={len(dados.result.crossings)} l={len(dados.result.lines)}  '
                    f'camadas: {"n" if camadas["n"] else "."}{"g" if camadas["g"] else "."}' 
                    f'{"c" if camadas["c"] else "."}{"l" if camadas["l"] else "."}' 
                    f'{"t" if camadas["t"] else "."}')
            cv2.setWindowTitle(WIN, info)
            cv2.imshow(WIN, frame)
            redraw = False

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            idx = (idx + 1) % len(lista)
            redraw = True
        elif key == ord('a'):
            idx = (idx - 1) % len(lista)
            redraw = True
        elif key in (ord('n'), ord('g'), ord('c'), ord('l'), ord('t')):
            k = chr(key)
            camadas[k] = not camadas[k]
            redraw = True
        elif key == ord('h'):
            print(_AJUDA)

    cv2.destroyAllWindows()


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualizador de features (III-C + IV).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('imagens', nargs='*', help='Imagens a processar.')
    parser.add_argument('--salvar', action='store_true',
                        help='Salva PNGs em debug_out/ sem abrir janela.')
    args = parser.parse_args()

    paths = args.imagens if args.imagens else _IMAGENS_PADRAO
    os.makedirs('debug_out', exist_ok=True)

    ps = Pixel_Segment('green_pixels.csv')
    print('Carregando imagens...')
    lista = [d for p in paths for d in [_carregar(p, ps)] if d is not None]

    if not lista:
        print('Nenhuma imagem carregada.')
        sys.exit(1)

    modo_salvar = args.salvar or not _tem_display()
    if modo_salvar:
        print('Modo --salvar (sem display):')
        for dados in lista:
            _salvar(dados)
        print('Concluído.')
    else:
        print(_AJUDA)
        _loop_interativo(lista)


if __name__ == '__main__':
    main()
