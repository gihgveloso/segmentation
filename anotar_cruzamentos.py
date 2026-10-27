"""Ferramenta de anotação de gabarito para cruzamentos L/T/X.

Uso:
    python anotar_cruzamentos.py

Modos:
  OFFLINE  — sempre gera debug_out/anotar_<basename>.png para cada imagem.
             O arquivo mostra a grade subamostrada ampliada 8× com os cruzamentos
             detectados. Use-o para anotar manualmente o JSON abaixo.

  INTERATIVO — se um display OpenCV estiver disponível: clique na janela para
               marcar cruzamentos e pressione as teclas indicadas na janela.

Formato do JSON gerado (tests/gabarito_cruzamentos.json):
    {
      "image_1": [{"tipo": "T", "x": 50, "y": 81}, ...],
      "image_2": [...],
      "image":  [...]
    }
    x, y são coordenadas na grade SUBAMOSTRADA (não pixels da imagem original).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import cv2
import numpy as np

from main import Pixel_Segment
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao
from features.crossings import detectar_cruzamentos, ConfigCruzamento

# ─── Configuração global ──────────────────────────────────────────────────────

SCALE = 8          # fator de ampliação da grade subamostrada
GABARITO_PATH = 'tests/gabarito_cruzamentos.json'
IMAGES = [
    ('image_1', 'images/image_1.png'),
    ('image_2', 'images/image_2.png'),
    ('image',   'images/image.png'),
]

# Cores por tipo (BGR)
COR_TIPO = {
    'L': (0, 165, 255),    # laranja
    'T': (50, 220, 50),    # verde
    'X': (0, 220, 255),    # ciano
}
COR_REJEITADO = (120, 120, 120)   # cinza
COR_PENDENTE  = (0, 220, 220)     # amarelo-esverdeado


# ─── Detecção de display ─────────────────────────────────────────────────────

def _tem_display() -> bool:
    """Verifica se display funciona: abre janela ASCII, testa imshow+setMouseCallback."""
    display = os.environ.get('DISPLAY', '')
    if not display:
        return False
    _win = '_test_display'
    try:
        _img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.namedWindow(_win, cv2.WINDOW_NORMAL)
        cv2.imshow(_win, _img)
        cv2.waitKey(1)
        cv2.setMouseCallback(_win, lambda *a: None)
        cv2.destroyWindow(_win)
        return True
    except Exception:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return False


# ─── Pipeline completo ───────────────────────────────────────────────────────

def _rodar_pipeline(img_path: str, ps: Pixel_Segment):
    """Executa o pipeline e retorna (aceitos, rejeitados, branco, verde, skeleton)."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None, None

    res = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['white_mask_resized'].astype(np.uint8)
    verde    = res['green_mask_resized'].astype(np.uint8)

    if skeleton.sum() == 0:
        return [], [], branco, verde, skeleton

    cfg_grafo = ConfigGrafo()
    cfg_simp  = ConfigSimplificacao()
    cfg_cruz  = ConfigCruzamento()

    g = construir_grafo(skeleton, relevo, cfg_grafo)
    simplificar(g, skeleton, relevo, cfg_simp)
    aceitos, rejeitados = detectar_cruzamentos(g, branco, verde, cfg_cruz)
    return aceitos, rejeitados, branco, verde, skeleton


# ─── Construção do canvas anotado ────────────────────────────────────────────

_INSTRUCOES = (
    "click=marcar | l=L  t=T  x=X | z=desfaz | h=overlay(off) | "
    "n=proxima  s=salvar  ESC=sair"
)
_INSTRUCOES_OFFLINE = (
    "MODO OFFLINE: edite tests/gabarito_cruzamentos.json | "
    "formato: {\"tipo\":\"T\",\"x\":50,\"y\":81}  (coordenadas subamostradas)"
)


def _construir_canvas(
    branco: np.ndarray,
    verde: np.ndarray,
    aceitos,
    rejeitados,
    anotacoes: list[dict],
    pendente: Optional[tuple[int, int]] = None,
    interativo: bool = False,
    mostrar_overlay: bool = True,
) -> np.ndarray:
    """Constroi o canvas 8x com fundo, grid, cruzamentos e anotacoes.

    mostrar_overlay=False omite aceitos/rejeitados (modo anotacao sem vies).
    """
    h, w = branco.shape
    ch, cw = h * SCALE, w * SCALE

    # Fundo
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            y0, y1 = y * SCALE, (y + 1) * SCALE
            x0, x1 = x * SCALE, (x + 1) * SCALE
            if branco[y, x] > 0:
                canvas[y0:y1, x0:x1] = (255, 255, 255)
            elif verde[y, x] > 0:
                canvas[y0:y1, x0:x1] = (0, 80, 0)
            else:
                canvas[y0:y1, x0:x1] = (15, 15, 15)

    # Linhas de grade a cada 10 px subamostrados
    for gx in range(0, w, 10):
        px = gx * SCALE
        cv2.line(canvas, (px, 0), (px, ch - 1), (60, 60, 60), 1)
    for gy in range(0, h, 10):
        py = gy * SCALE
        cv2.line(canvas, (0, py), (cw - 1, py), (60, 60, 60), 1)

    # Rótulos de coordenadas nas interseções da grade
    for gx in range(0, w, 10):
        for gy in range(0, h, 10):
            px = gx * SCALE + 2
            py = gy * SCALE + 9
            cv2.putText(
                canvas, f'{gx},{gy}', (px, py),
                cv2.FONT_HERSHEY_PLAIN, 0.55, (80, 80, 80), 1, cv2.LINE_AA,
            )

    # Deteccoes (aceitos/rejeitados) — so se mostrar_overlay=True
    if mostrar_overlay:
        if rejeitados:
            for r in rejeitados:
                cx = r.x * SCALE + SCALE // 2
                cy = r.y * SCALE + SCALE // 2
                cv2.circle(canvas, (cx, cy), 4, COR_REJEITADO, 1)
        if aceitos:
            for a in aceitos:
                cx = a.x * SCALE + SCALE // 2
                cy = a.y * SCALE + SCALE // 2
                cor = COR_TIPO.get(a.tipo, (255, 255, 255))
                cv2.circle(canvas, (cx, cy), 7, cor, 2)
                cv2.putText(
                    canvas, a.tipo, (cx + 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA,
                )

    # Anotações do gabarito atual
    for ann in anotacoes:
        cx = ann['x'] * SCALE + SCALE // 2
        cy = ann['y'] * SCALE + SCALE // 2
        cor = COR_TIPO.get(ann['tipo'], (255, 255, 0))
        cv2.circle(canvas, (cx, cy), 10, cor, 2)
        cv2.drawMarker(canvas, (cx, cy), cor, cv2.MARKER_CROSS, 10, 1)
        cv2.putText(
            canvas, ann['tipo'], (cx + 11, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA,
        )

    # Posição pendente (clique sem tipo ainda)
    if pendente is not None:
        pcx = pendente[0] * SCALE + SCALE // 2
        pcy = pendente[1] * SCALE + SCALE // 2
        cv2.drawMarker(canvas, (pcx, pcy), (0, 220, 220), cv2.MARKER_CROSS, 14, 2)

    # Texto de instrucoes no canto superior esquerdo (overlay state embutido)
    if interativo:
        overlay_label = 'on' if mostrar_overlay else 'off'
        texto = _INSTRUCOES.replace('overlay(off)', f'overlay({overlay_label})')
    else:
        texto = _INSTRUCOES_OFFLINE
    # Fundo semi-opaco
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (cw, 16), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    cv2.putText(
        canvas, texto, (4, 12),
        cv2.FONT_HERSHEY_PLAIN, 0.75, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return canvas


# ─── Modo OFFLINE ─────────────────────────────────────────────────────────────

def gerar_offline(
    ps: Pixel_Segment,
    gabarito: dict[str, list[dict]],
) -> None:
    """Gera debug_out/anotar_<basename>.png para cada imagem."""
    os.makedirs('debug_out', exist_ok=True)
    for basename, img_path in IMAGES:
        if not os.path.exists(img_path):
            print(f"  [offline] {img_path} não encontrado, pulando.")
            continue

        print(f"  [offline] Processando {basename}…")
        aceitos, rejeitados, branco, verde, skeleton = _rodar_pipeline(img_path, ps)
        if branco is None:
            print(f"  [offline] Não foi possível carregar {img_path}, pulando.")
            continue

        anotacoes = gabarito.get(basename, [])
        canvas = _construir_canvas(
            branco, verde,
            aceitos or [], rejeitados or [],
            anotacoes,
            pendente=None,
            interativo=False,
            mostrar_overlay=True,
        )
        out = f'debug_out/anotar_{basename}.png'
        cv2.imwrite(out, canvas)
        print(f"  [offline] Salvo: {out}")


# ─── Modo INTERATIVO ──────────────────────────────────────────────────────────

class _Estado:
    """Estado mutable compartilhado com o callback do mouse."""
    def __init__(self):
        self.pendente: Optional[tuple[int, int]] = None  # (sub_x, sub_y)
        self.anotacoes: list[dict] = []
        self.precisa_redesenhar: bool = True
        self.mostrar_overlay: bool = False  # H key: liga/desliga deteccoes


def _callback_mouse(event, x, y, flags, param):
    estado: _Estado = param
    if event == cv2.EVENT_LBUTTONDOWN:
        sub_x = x // SCALE
        sub_y = y // SCALE
        estado.pendente = (sub_x, sub_y)
        estado.precisa_redesenhar = True


def _modo_interativo(
    ps: Pixel_Segment,
    gabarito: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Loop interativo por imagem. Retorna gabarito atualizado."""
    resultado = dict(gabarito)  # cópia

    n_imgs = len(IMAGES)
    for idx, (basename, img_path) in enumerate(IMAGES):
        if not os.path.exists(img_path):
            print(f"  [interativo] {img_path} não encontrado, pulando.")
            continue

        print(f"\n  [interativo] {basename} ({idx+1}/{n_imgs})")
        print("    Clique para marcar, depois: l=L  t=T  x=X  z=desfaz  n=próxima  s=salvar  ESC=sair")

        aceitos, rejeitados, branco, verde, skeleton = _rodar_pipeline(img_path, ps)
        if branco is None:
            print(f"  [interativo] Não foi possível carregar {img_path}, pulando.")
            continue

        estado = _Estado()
        estado.anotacoes = list(resultado.get(basename, []))

        # Nome 100% ASCII — evita bug OpenCV 5 com nomes nao-ASCII
        win = f'Anotar - {basename}'

        # Dimensionar janela para caber em ~1280x720 sem ultrapassar o canvas
        h_canvas = branco.shape[0] * SCALE
        w_canvas = branco.shape[1] * SCALE
        scale_fit = min(1280 / w_canvas, 720 / h_canvas, 1.0)
        win_w = max(int(w_canvas * scale_fit), 400)
        win_h = max(int(h_canvas * scale_fit), 300)

        # Criar janela e mostrar primeiro frame ANTES de setMouseCallback
        canvas_inicial = _construir_canvas(
            branco, verde,
            aceitos or [], rejeitados or [],
            estado.anotacoes,
            pendente=None,
            interativo=True,
            mostrar_overlay=estado.mostrar_overlay,
        )
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, win_w, win_h)
        cv2.imshow(win, canvas_inicial)
        cv2.waitKey(1)
        cv2.setMouseCallback(win, _callback_mouse, estado)
        # frame inicial ja desenhado
        estado.precisa_redesenhar = False

        salvar_e_continuar = False
        proxima = False

        while True:
            if estado.precisa_redesenhar:
                canvas = _construir_canvas(
                    branco, verde,
                    aceitos or [], rejeitados or [],
                    estado.anotacoes,
                    pendente=estado.pendente,
                    interativo=True,
                    mostrar_overlay=estado.mostrar_overlay,
                )
                cv2.imshow(win, canvas)
                estado.precisa_redesenhar = False

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("  ESC: saindo sem salvar.")
                cv2.destroyAllWindows()
                return resultado

            elif key in (ord('l'), ord('L')):
                if estado.pendente:
                    estado.anotacoes.append({'tipo': 'L', 'x': estado.pendente[0], 'y': estado.pendente[1]})
                    print(f"    + L ({estado.pendente[0]}, {estado.pendente[1]})")
                    estado.pendente = None
                    estado.precisa_redesenhar = True

            elif key in (ord('t'), ord('T')):
                if estado.pendente:
                    estado.anotacoes.append({'tipo': 'T', 'x': estado.pendente[0], 'y': estado.pendente[1]})
                    print(f"    + T ({estado.pendente[0]}, {estado.pendente[1]})")
                    estado.pendente = None
                    estado.precisa_redesenhar = True

            elif key in (ord('x'), ord('X')):
                if estado.pendente:
                    estado.anotacoes.append({'tipo': 'X', 'x': estado.pendente[0], 'y': estado.pendente[1]})
                    print(f"    + X ({estado.pendente[0]}, {estado.pendente[1]})")
                    estado.pendente = None
                    estado.precisa_redesenhar = True

            elif key in (ord('h'), ord('H')):
                estado.mostrar_overlay = not estado.mostrar_overlay
                estado.precisa_redesenhar = True
                print(f"    overlay {'ligado' if estado.mostrar_overlay else 'desligado'}")

            elif key in (ord('z'), ord('Z')):
                if estado.anotacoes:
                    removido = estado.anotacoes.pop()
                    print(f"    desfaz: {removido}")
                    estado.precisa_redesenhar = True
                estado.pendente = None

            elif key in (ord('s'), ord('S')):
                resultado[basename] = list(estado.anotacoes)
                _salvar_gabarito(resultado)
                print(f"    Salvo: {GABARITO_PATH}  ({len(estado.anotacoes)} anotações em {basename})")
                salvar_e_continuar = True

            elif key in (ord('n'), ord('N')):
                resultado[basename] = list(estado.anotacoes)
                print(f"    Próxima imagem. {len(estado.anotacoes)} anotações em {basename}.")
                proxima = True
                break

        cv2.destroyWindow(win)

        if not proxima and not salvar_e_continuar:
            break

    # Última imagem: salvar automaticamente
    _salvar_gabarito(resultado)
    print(f"\n  Gabarito final salvo: {GABARITO_PATH}")
    return resultado


# ─── I/O do gabarito ──────────────────────────────────────────────────────────

def _carregar_gabarito() -> dict[str, list[dict]]:
    if os.path.exists(GABARITO_PATH):
        try:
            with open(GABARITO_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  Gabarito existente carregado: {GABARITO_PATH}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Aviso: não foi possível ler {GABARITO_PATH}: {e}")
    return {}


def _salvar_gabarito(gabarito: dict[str, list[dict]]) -> None:
    os.makedirs(os.path.dirname(GABARITO_PATH) or '.', exist_ok=True)
    with open(GABARITO_PATH, 'w', encoding='utf-8') as f:
        json.dump(gabarito, f, ensure_ascii=False, indent=2)


# ─── Ponto de entrada ─────────────────────────────────────────────────────────

def main() -> None:
    print(__doc__)
    print("=" * 70)

    ps = Pixel_Segment('green_pixels.csv')
    gabarito = _carregar_gabarito()

    interativo = _tem_display()

    if interativo:
        print("  Display detectado: modo INTERATIVO ativado.")
        print("  As imagens offline também serão geradas.")
    else:
        print("  Sem display detectado: modo OFFLINE.")

    # Sempre gera as imagens offline
    gerar_offline(ps, gabarito)

    if interativo:
        gabarito = _modo_interativo(ps, gabarito)

    print("\nConcluído.")
    if not interativo:
        print(f"  Edite {GABARITO_PATH} manualmente se necessário,")
        print("  ou execute em um ambiente com display para o modo interativo.")


if __name__ == '__main__':
    main()
