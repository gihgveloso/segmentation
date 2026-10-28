"""Ferramenta de anotação de gabarito para linhas do campo (polilinhas).

Uso:
    python3 scripts/anotar_linhas.py [--test]

Modos:
  OFFLINE  — sem display: gera debug_out/anotar_linhas_<basename>.png com
             as polilinhas já anotadas (se existirem no JSON).

  INTERATIVO — com display OpenCV: clique para adicionar pontos, teclas para
               controlar. Coordenadas gravadas em PIXELS DA IMAGEM ORIGINAL.

Teclas no modo interativo:
    clique esquerdo : adiciona ponto à polilinha atual
    f / F           : fecha polilinha atual e começa nova
    z / Z           : desfaz último ponto
    s / S           : salva gabarito
    n / N           : passa para a próxima imagem (salva automaticamente)
    ESC             : sai sem salvar a imagem atual

Formato de saída (tests/gabarito_linhas.json):
    {
      "image_1": [
        [[x1,y1], [x2,y2], [x3,y3]],   <- polilinha 1
        [[x4,y4], [x5,y5]]               <- polilinha 2
      ],
      "bitbots": [...]
    }
    x, y são coordenadas em PIXELS DA IMAGEM ORIGINAL (não da grade).

Argumento --test: cria gabarito sintético para image_1 e verifica carregamento.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Optional

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────

GABARITO_PATH = 'tests/gabarito_linhas.json'
IMAGES = [
    ('image_1', 'images/image_1.png'),
    ('image_2', 'images/image_2.png'),
    ('image',   'images/image.png'),
    ('bitbots',
     'images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'),
    ('eliza',   'images/eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg'),
]

COR_POLILINHA = [
    (0, 220, 255),    # ciano
    (255, 120, 0),    # laranja
    (180, 0, 255),    # roxo
    (0, 255, 100),    # verde-claro
    (200, 200, 0),    # amarelo
]
COR_PONTO    = (255, 255, 255)
COR_PENDENTE = (0, 180, 255)   # azul-laranja = ponto ainda não confirmado


# ─── Detecção de display ──────────────────────────────────────────────────────

def _tem_display() -> bool:
    display = os.environ.get('DISPLAY', '')
    if not display:
        return False
    win = '_test_display_linhas'
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


# ─── Canvas ───────────────────────────────────────────────────────────────────

def _construir_canvas(
    img_orig: np.ndarray,
    polilinhas: list[list[list[int]]],
    polilinha_atual: list[list[int]],
    display_scale: float,
    interativo: bool,
) -> np.ndarray:
    """Constrói canvas com imagem + polilinhas sobrepostas."""
    H, W = img_orig.shape[:2]
    dW = int(W * display_scale)
    dH = int(H * display_scale)
    canvas = cv2.resize(img_orig, (dW, dH), interpolation=cv2.INTER_AREA)

    def to_disp(x, y):
        return (int(round(x * display_scale)), int(round(y * display_scale)))

    # Polilinhas salvas
    for i, poly in enumerate(polilinhas):
        cor = COR_POLILINHA[i % len(COR_POLILINHA)]
        if len(poly) >= 2:
            pts = np.array([[int(p[0]*display_scale), int(p[1]*display_scale)]
                             for p in poly], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, cor, 2)
        for pt in poly:
            cv2.circle(canvas, to_disp(pt[0], pt[1]), 4, cor, -1)
        if poly:
            cv2.putText(canvas, str(i+1), to_disp(poly[0][0], poly[0][1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    # Polilinha atual (em andamento)
    cor_atual = COR_POLILINHA[len(polilinhas) % len(COR_POLILINHA)]
    if len(polilinha_atual) >= 2:
        pts = np.array([[int(p[0]*display_scale), int(p[1]*display_scale)]
                         for p in polilinha_atual], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, COR_PENDENTE, 2)
    for pt in polilinha_atual:
        cv2.circle(canvas, to_disp(pt[0], pt[1]), 4, COR_PENDENTE, -1)

    # Instruções
    if interativo:
        instrucoes = (
            "click=ponto | f=fecha_polilinha | z=desfaz | s=salvar | n=proxima | ESC=sair"
        )
    else:
        instrucoes = "MODO OFFLINE: edite tests/gabarito_linhas.json"
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (dW, 18), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, instrucoes, (4, 13),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (200,200,200), 1)

    info = (f"polilinhas={len(polilinhas)+bool(polilinha_atual)}  "
            f"atual={len(polilinha_atual)}pts")
    cv2.putText(canvas, info, (4, dH-5),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (180,255,180), 1)

    return canvas


# ─── I/O do gabarito ──────────────────────────────────────────────────────────

def _carregar() -> dict:
    if os.path.exists(GABARITO_PATH):
        try:
            with open(GABARITO_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Aviso: não foi possível ler {GABARITO_PATH}: {e}")
    return {}


def _salvar(gabarito: dict) -> None:
    os.makedirs(os.path.dirname(GABARITO_PATH) or '.', exist_ok=True)
    with open(GABARITO_PATH, 'w', encoding='utf-8') as f:
        json.dump(gabarito, f, ensure_ascii=False, indent=2)
    print(f"  Salvo: {GABARITO_PATH}")


# ─── Modo OFFLINE ─────────────────────────────────────────────────────────────

def _gerar_offline(gabarito: dict) -> None:
    os.makedirs('debug_out', exist_ok=True)
    for basename, img_path in IMAGES:
        if not os.path.exists(img_path):
            print(f"  [offline] {img_path} não encontrado, pulando.")
            continue
        img = cv2.imread(img_path)
        if img is None: continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        polys = gabarito.get(basename, [])
        canvas = _construir_canvas(img, polys, [], 1.0, interativo=False)
        out = f'debug_out/anotar_linhas_{basename}.png'
        cv2.imwrite(out, canvas)
        n = sum(len(p) for p in polys)
        print(f"  [offline] {basename}: {len(polys)} polilinhas, {n} pts → {out}")


# ─── Modo INTERATIVO ──────────────────────────────────────────────────────────

class _Estado:
    def __init__(self):
        self.pendente: Optional[list[int]] = None   # último clique (não usado)
        self.polilinha_atual: list[list[int]] = []
        self.precisa_redesenhar: bool = True
        # coordenada do cursor (para pré-visualização, não implementado)


def _callback_mouse(event, x, y, flags, param):
    estado: _Estado = param[0]
    scale: float    = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(round(x / scale))
        orig_y = int(round(y / scale))
        estado.polilinha_atual.append([orig_x, orig_y])
        estado.precisa_redesenhar = True


def _modo_interativo(gabarito: dict) -> dict:
    resultado = dict(gabarito)

    for basename, img_path in IMAGES:
        if not os.path.exists(img_path):
            print(f"  [interativo] {img_path} não encontrado, pulando.")
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        H, W = img.shape[:2]

        scale = min(1280/W, 720/H, 1.0)
        dW = max(int(W*scale), 400)
        dH = max(int(H*scale), 300)

        polilinhas: list[list[list[int]]] = list(resultado.get(basename, []))
        estado = _Estado()
        estado.polilinha_atual = []

        win = f'Anotar linhas - {basename}'
        print(f"\n  [interativo] {basename}  ({W}x{H}px, display {dW}x{dH})")
        print("    click=ponto  f=fecha  z=desfaz  s=salvar  n=proxima  ESC=sair")

        canvas_0 = _construir_canvas(img, polilinhas, [], scale, True)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, dW, dH)
        cv2.imshow(win, canvas_0)
        cv2.waitKey(1)
        cv2.setMouseCallback(win, _callback_mouse, [estado, scale])
        estado.precisa_redesenhar = False

        while True:
            if estado.precisa_redesenhar:
                canvas = _construir_canvas(img, polilinhas,
                                           estado.polilinha_atual, scale, True)
                cv2.imshow(win, canvas)
                estado.precisa_redesenhar = False

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("  ESC: saindo sem salvar.")
                cv2.destroyAllWindows()
                return resultado

            elif key in (ord('f'), ord('F')):
                if len(estado.polilinha_atual) >= 2:
                    polilinhas.append(list(estado.polilinha_atual))
                    print(f"    + polilinha {len(polilinhas)} "
                          f"({len(estado.polilinha_atual)} pts)")
                    estado.polilinha_atual = []
                    estado.precisa_redesenhar = True
                elif estado.polilinha_atual:
                    print("    polilinha precisa de ≥2 pontos; use z para desfazer")

            elif key in (ord('z'), ord('Z')):
                if estado.polilinha_atual:
                    pt = estado.polilinha_atual.pop()
                    print(f"    desfaz ponto {pt}")
                elif polilinhas:
                    estado.polilinha_atual = polilinhas.pop()
                    print(f"    desfaz polilinha (recuperando {len(estado.polilinha_atual)} pts)")
                estado.precisa_redesenhar = True

            elif key in (ord('s'), ord('S')):
                polys_save = list(polilinhas)
                if len(estado.polilinha_atual) >= 2:
                    polys_save.append(list(estado.polilinha_atual))
                resultado[basename] = polys_save
                _salvar(resultado)
                print(f"    Salvo: {len(polys_save)} polilinhas em {basename}")

            elif key in (ord('n'), ord('N')):
                polys_save = list(polilinhas)
                if len(estado.polilinha_atual) >= 2:
                    polys_save.append(list(estado.polilinha_atual))
                resultado[basename] = polys_save
                print(f"    Próxima. {len(polys_save)} polilinhas salvas em {basename}.")
                break

        cv2.destroyWindow(win)

    _salvar(resultado)
    return resultado


# ─── Modo TEST ────────────────────────────────────────────────────────────────

def _modo_test() -> None:
    """Cria 2 polilinhas sintéticas em arquivo temporário e verifica carga/salva.

    Não toca em tests/gabarito_linhas.json.
    """
    print("=== MODO TEST ===")
    poly1 = [[100,500],[400,498],[700,502],[1050,497],[1200,500]]
    poly2 = [[670,200],[668,400],[672,600],[670,800]]

    gabarito_test = {'image_1': [poly1, poly2]}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False, encoding='utf-8') as f:
        tmp_path = f.name
        json.dump(gabarito_test, f, ensure_ascii=False, indent=2)
    print(f"  Escrito em arquivo temporário: {tmp_path}")
    print(f"  Criadas 2 polilinhas sintéticas para image_1:")
    print(f"    poly1 ({len(poly1)} pts): {poly1[0]} → {poly1[-1]}")
    print(f"    poly2 ({len(poly2)} pts): {poly2[0]} → {poly2[-1]}")

    # Verificar carga
    with open(tmp_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert 'image_1' in loaded, "ERRO: image_1 não encontrado"
    loaded_polys = loaded['image_1']
    assert len(loaded_polys) == 2, f"ERRO: esperado 2 polilinhas, got {len(loaded_polys)}"
    assert loaded_polys[0][0] == poly1[0], "ERRO: primeiro ponto da poly1 não bate"
    print(f"  Carga verificada: {len(loaded_polys)} polilinhas, "
          f"pts={[len(p) for p in loaded_polys]}")

    # Apagar arquivo temporário
    os.unlink(tmp_path)
    print(f"  Arquivo temporário removido: {tmp_path}")

    # Gerar imagem offline para visual (em debug_out/, não em tests/)
    img_path = 'images/image_1.png'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Aviso: não conseguiu ler {img_path}")
        else:
            if img.ndim==3 and img.shape[2]==4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            canvas = _construir_canvas(img, [poly1, poly2], [], 1.0, False)
            out = 'debug_out/anotar_linhas_test_image_1.png'
            os.makedirs('debug_out', exist_ok=True)
            cv2.imwrite(out, canvas)
            print(f"  Imagem de teste salva: {out}")

    print("=== TEST OK ===")


# ─── Ponto de entrada ─────────────────────────────────────────────────────────

def main() -> None:
    print(__doc__)
    print("=" * 70)

    if '--test' in sys.argv:
        _modo_test()
        return

    gabarito = _carregar()
    interativo = _tem_display()

    if interativo:
        print("  Display detectado: modo INTERATIVO.")
    else:
        print("  Sem display: modo OFFLINE.")

    _gerar_offline(gabarito)

    if interativo:
        gabarito = _modo_interativo(gabarito)

    print("\nConcluído.")
    if not interativo:
        print(f"  Para anotar: execute com display ou edite {GABARITO_PATH} manualmente.")


if __name__ == '__main__':
    main()
