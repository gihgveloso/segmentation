"""Anotação do horizonte do campo: onde o campo (verde) termina e começa o fundo.

Uma polilinha por imagem, salva em tests/gabarito_horizonte.json.

Uso:
    python3 scripts/anotar_horizonte.py [imagens...] [--test]

Exemplos:
    python3 scripts/anotar_horizonte.py bitbots
    python3 scripts/anotar_horizonte.py eliza
    python3 scripts/anotar_horizonte.py bitbots eliza
    python3 scripts/anotar_horizonte.py              # todas as imagens
    python3 scripts/anotar_horizonte.py --test       # modo teste (arquivo temporário)

Teclas no modo interativo:
    clique esquerdo : adiciona ponto ao horizonte
    f / F           : finaliza o horizonte atual e passa para a próxima imagem
    z / Z           : desfaz último ponto
    s / S           : salva (sem avançar)
    n / N           : próxima imagem (salva automaticamente)
    ESC             : sai sem salvar a imagem atual

Formato de saída (tests/gabarito_horizonte.json):
    {
      "bitbots": [[x1,y1], [x2,y2], ...],   <- uma polilinha por imagem
      "eliza":   [[x1,y1], ...]
    }
    x, y em PIXELS DA IMAGEM ORIGINAL (não da grade).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np

GABARITO_PATH = 'tests/gabarito_horizonte.json'
IMAGES = [
    ('bitbots',
     'images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    ('eliza',   'images/eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
    ('image_1', 'images/image_1.png'),
    ('image_2', 'images/image_2.png'),
    ('image',   'images/image.png'),
]

COR_HORIZONTE  = (0, 220, 255)   # ciano
COR_PENDENTE   = (0, 130, 255)   # laranja (em andamento)
COR_PONTO      = (255, 255, 255)


# ─── detecção de display ──────────────────────────────────────────────────────

def _tem_display() -> bool:
    display = os.environ.get('DISPLAY', '')
    if not display:
        return False
    win = '_test_display_horizonte'
    try:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        cv2.waitKey(1)
        cv2.setMouseCallback(win, lambda *a: None)
        cv2.destroyWindow(win)
        return True
    except Exception:
        try: cv2.destroyAllWindows()
        except Exception: pass
        return False


# ─── canvas ───────────────────────────────────────────────────────────────────

def _construir_canvas(
    img_orig: np.ndarray,
    horizonte: list[list[int]],
    pendente: list[list[int]],
    scale: float,
    interativo: bool,
) -> np.ndarray:
    H, W = img_orig.shape[:2]
    dW = int(W * scale)
    dH = int(H * scale)
    canvas = cv2.resize(img_orig, (dW, dH), interpolation=cv2.INTER_AREA)

    def td(x, y):
        return (int(round(x * scale)), int(round(y * scale)))

    # Horizonte salvo: ciano
    if len(horizonte) >= 2:
        pts = np.array([[int(p[0]*scale), int(p[1]*scale)] for p in horizonte],
                       dtype=np.int32)
        cv2.polylines(canvas, [pts], False, COR_HORIZONTE, 2)
    for p in horizonte:
        cv2.circle(canvas, td(p[0], p[1]), 4, COR_HORIZONTE, -1)

    # Em andamento: laranja
    if len(pendente) >= 2:
        pts = np.array([[int(p[0]*scale), int(p[1]*scale)] for p in pendente],
                       dtype=np.int32)
        cv2.polylines(canvas, [pts], False, COR_PENDENTE, 2)
    for p in pendente:
        cv2.circle(canvas, td(p[0], p[1]), 4, COR_PENDENTE, -1)

    # Barra de instrução
    if interativo:
        instr = "click=ponto | f=finaliza | z=desfaz | s=salva | n=proxima | ESC=sair"
    else:
        instr = f"MODO OFFLINE: edite {GABARITO_PATH}"
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (dW, 18), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, instr, (4, 13),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (200, 200, 200), 1)
    n_salvo  = len(horizonte)
    n_pend   = len(pendente)
    cv2.putText(canvas, f"salvo={n_salvo}pts  pendente={n_pend}pts",
                (4, dH - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (180, 255, 180), 1)
    return canvas


# ─── I/O do gabarito ──────────────────────────────────────────────────────────

def _carregar(path: str = GABARITO_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Aviso: não foi possível ler {path}: {e}")
    return {}


def _salvar(gabarito: dict, path: str = GABARITO_PATH) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(gabarito, f, ensure_ascii=False, indent=2)
    print(f"  Salvo: {path}")


# ─── modo OFFLINE ─────────────────────────────────────────────────────────────

def _gerar_offline(gabarito: dict, imagens_filtro: list[str]) -> None:
    os.makedirs('debug_out', exist_ok=True)
    for basename, img_path in IMAGES:
        if imagens_filtro and basename not in imagens_filtro:
            continue
        if not os.path.exists(img_path):
            print(f"  [offline] {img_path} não encontrado, pulando.")
            continue
        img = cv2.imread(img_path)
        if img is None: continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        horiz = gabarito.get(basename, [])
        canvas = _construir_canvas(img, horiz, [], 1.0, interativo=False)
        out = f'debug_out/anotar_horizonte_{basename}.png'
        cv2.imwrite(out, canvas)
        print(f"  [offline] {basename}: {len(horiz)} pts → {out}")


# ─── modo INTERATIVO ──────────────────────────────────────────────────────────

class _Estado:
    def __init__(self):
        self.pendente: list[list[int]] = []
        self.precisa_redesenhar: bool = True


def _callback_mouse(event, x, y, flags, param):
    estado: _Estado = param[0]
    scale: float    = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        estado.pendente.append([ox, oy])
        estado.precisa_redesenhar = True


def _modo_interativo(gabarito: dict, imagens_filtro: list[str]) -> dict:
    resultado = dict(gabarito)

    for basename, img_path in IMAGES:
        if imagens_filtro and basename not in imagens_filtro:
            continue
        if not os.path.exists(img_path):
            print(f"  [interativo] {img_path} não encontrado, pulando.")
            continue
        img = cv2.imread(img_path)
        if img is None: continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        H, W = img.shape[:2]

        scale = min(1280/W, 720/H, 1.0)
        dW, dH = max(int(W*scale), 400), max(int(H*scale), 300)

        estado = _Estado()
        horizonte_salvo: list[list[int]] = list(resultado.get(basename) or [])
        estado.pendente = list(horizonte_salvo)   # inicia com o horizonte já salvo

        win = f'Horizonte - {basename}'
        print(f"\n  [interativo] {basename}  ({W}x{H}px)")
        print("    click=ponto  f=finaliza  z=desfaz  s=salva  n=proxima  ESC=sair")
        if horizonte_salvo:
            print(f"    Horizonte existente: {len(horizonte_salvo)} pts — continue clicando ou use z para apagar")

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, dW, dH)
        cv2.setMouseCallback(win, _callback_mouse, [estado, scale])
        cv2.imshow(win, _construir_canvas(img, [], estado.pendente, scale, True))
        cv2.waitKey(1)
        estado.precisa_redesenhar = False

        while True:
            if estado.precisa_redesenhar:
                canvas = _construir_canvas(img, [], estado.pendente, scale, True)
                cv2.imshow(win, canvas)
                estado.precisa_redesenhar = False

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("  ESC: saindo sem salvar.")
                cv2.destroyAllWindows()
                return resultado

            elif key in (ord('f'), ord('F')):
                if len(estado.pendente) >= 2:
                    horizonte_salvo = list(estado.pendente)
                    resultado[basename] = horizonte_salvo
                    print(f"    Horizonte finalizado: {len(horizonte_salvo)} pts")
                    break
                else:
                    print("    Horizonte precisa de ≥2 pontos")

            elif key in (ord('z'), ord('Z')):
                if estado.pendente:
                    pt = estado.pendente.pop()
                    print(f"    desfaz ponto {pt}")
                    estado.precisa_redesenhar = True

            elif key in (ord('s'), ord('S')):
                if len(estado.pendente) >= 2:
                    resultado[basename] = list(estado.pendente)
                    _salvar(resultado)
                    print(f"    Salvo: {len(estado.pendente)} pts em {basename}")

            elif key in (ord('n'), ord('N')):
                if len(estado.pendente) >= 2:
                    resultado[basename] = list(estado.pendente)
                    print(f"    Próxima. {len(estado.pendente)} pts em {basename}.")
                else:
                    print("    Pulando sem salvar (< 2 pts).")
                break

        cv2.destroyWindow(win)

    _salvar(resultado)
    return resultado


# ─── modo TEST ────────────────────────────────────────────────────────────────

def _modo_test() -> None:
    """Cria horizonte sintético para bitbots em arquivo temporário; não toca gabarito real."""
    print("=== MODO TEST ===")
    horizonte_synth = [[0, 200], [160, 190], [320, 185], [480, 195], [640, 200]]
    gabarito_test = {'bitbots': horizonte_synth}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False, encoding='utf-8') as f:
        tmp_path = f.name
        json.dump(gabarito_test, f, ensure_ascii=False, indent=2)
    print(f"  Escrito em arquivo temporário: {tmp_path}")
    print(f"  Horizonte sintético: {len(horizonte_synth)} pts — {horizonte_synth[0]} → {horizonte_synth[-1]}")

    with open(tmp_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert 'bitbots' in loaded, "ERRO: 'bitbots' não encontrado"
    h = loaded['bitbots']
    assert len(h) == len(horizonte_synth), f"ERRO: esperado {len(horizonte_synth)} pts, got {len(h)}"
    assert h[0] == horizonte_synth[0], "ERRO: primeiro ponto não bate"
    print(f"  Carga verificada: {len(h)} pts")

    os.unlink(tmp_path)
    print(f"  Arquivo temporário removido: {tmp_path}")
    print("=== TEST OK ===")


# ─── ponto de entrada ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'imagens', nargs='*',
        help="Nomes das imagens a anotar (ex.: bitbots eliza). "
             "Padrão: todas as imagens configuradas.",
    )
    parser.add_argument(
        '--test', action='store_true',
        help="Modo teste: gabarito sintético em arquivo temporário, não toca gabarito real.",
    )
    args = parser.parse_args()

    if args.test:
        _modo_test()
        return

    gabarito = _carregar()
    imagens_filtro = args.imagens  # lista vazia = todas

    interativo = _tem_display()
    print(f"  Display detectado: {'sim (modo interativo)' if interativo else 'não (modo offline)'}")
    print(f"  Gabarito de saída: {GABARITO_PATH}")
    if imagens_filtro:
        print(f"  Imagens selecionadas: {imagens_filtro}")
    print()

    _gerar_offline(gabarito, imagens_filtro)

    if interativo:
        _modo_interativo(gabarito, imagens_filtro)

    print("\nConcluído.")
    if not interativo:
        print(f"  Para anotar: execute com display ou edite {GABARITO_PATH} manualmente.")


if __name__ == '__main__':
    main()
