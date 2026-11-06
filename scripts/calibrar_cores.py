"""Calibração de cor para imagens reais.

Modo headless (--sem-gui ou sem DISPLAY):
    python3 scripts/calibrar_cores.py img1.jpg [img2.jpg ...] \\
        --amostras specs.json [--saida calib.json] [--force]

Modo interativo (requer DISPLAY):
    python3 scripts/calibrar_cores.py img1.jpg [img2.jpg ...] \\
        [--amostras specs.json] [--saida calib.json] [--force]

Formato de specs.json (--amostras):
    {
      "carpete": [{"imagem": "<nome>", "tipo": "patch", "x": 100, "y": 200, "raio": 10}],
      "linha":   [{"imagem": "<nome>", "tipo": "patch", "x": 100, "y": 50,  "raio": 5}]
    }
    onde <nome> é o nome base do arquivo sem extensão.

Teclas (modo interativo):
    clique esquerdo   : adiciona patch de carpete
    clique direito    : adiciona patch de linha
    l                 : alterna modo clique (carpete/linha)
    Enter / e         : estimar calibração
    y                 : confirmar proposta pendente
    n                 : rejeitar proposta pendente
    u                 : desfazer último passo
    r                 : reiniciar sessão
    s                 : salvar e sair
    q                 : sair sem salvar
    ← →               : imagem anterior / próxima
    espaço            : próxima imagem
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.calibracao_cor import (
    Aviso,
    ResultadoPreview,
    avaliar_calibracao,
)
from features.sessao_calibracao import (
    LimiteRodadasError,
    SaltoVerdeGrandeError,
    SessaoCalibracao,
    SessaoNaoProntaError,
)


# ---------------------------------------------------------------------------
# Constantes visuais
# ---------------------------------------------------------------------------

_RAIO_CARP: int = 15
_RAIO_LIN: int = 8
_COR_CARP = (255, 80, 0)      # azul
_COR_LIN = (0, 0, 220)        # vermelho
_COR_VERDE = (0, 200, 0)
_COR_BRANCO = (255, 255, 0)   # ciano em BGR
_COR_CONVEXA = (0, 255, 255)  # amarelo em BGR
_ALPHA_VERDE: float = 0.35
_ALPHA_BRANCO: float = 0.55


# ---------------------------------------------------------------------------
# Detecção de display
# ---------------------------------------------------------------------------

def _tem_display() -> bool:
    if not os.environ.get('DISPLAY', '') and not os.environ.get('WAYLAND_DISPLAY', ''):
        return False
    win = '_calib_test_display_'
    try:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        cv2.waitKey(1)
        cv2.destroyWindow(win)
        return True
    except Exception:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Overlay visual
# ---------------------------------------------------------------------------

def _alpha_blend(
    img: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple,
    alpha: float,
) -> np.ndarray:
    out = img.copy()
    m = mask > 0
    for c, v in enumerate(color_bgr):
        out[:, :, c] = np.where(
            m,
            np.clip(
                img[:, :, c].astype(float) * (1 - alpha) + v * alpha, 0, 255
            ).astype(np.uint8),
            img[:, :, c],
        )
    return out


def _put_text_shadowed(
    img: np.ndarray,
    text: str,
    pos: tuple,
    scale: float,
    color: tuple,
    thickness: int,
) -> None:
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


def _specs_do_frame(specs: list, nome_img: str) -> list:
    return [s for s in specs if s.get('imagem') == nome_img]


def gerar_overlay(
    img_bgr: np.ndarray,
    nome_img: str,
    resultado: Optional[ResultadoPreview],
    specs_carp: list,
    specs_lin: list,
    avisos: List[Aviso],
    extras: List[str] = (),
) -> np.ndarray:
    """Gera overlay visual sobre img_bgr com máscaras, contornos e métricas."""
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()

    if resultado is not None:
        out = _alpha_blend(out, resultado.green_mask, _COR_VERDE, _ALPHA_VERDE)
        out = _alpha_blend(out, resultado.white_mask, _COR_BRANCO, _ALPHA_BRANCO)
        mc_full = cv2.resize(
            resultado.mascara_convexa, (W, H), interpolation=cv2.INTER_NEAREST
        )
        ctrs, _ = cv2.findContours(mc_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, _COR_CONVEXA, 2)

    for s in _specs_do_frame(specs_carp, nome_img):
        if s.get('tipo') == 'patch':
            x, y, r = s['x'], s['y'], s.get('raio', _RAIO_CARP)
            cv2.rectangle(out, (x - r, y - r), (x + r, y + r), _COR_CARP, 1)

    for s in _specs_do_frame(specs_lin, nome_img):
        if s.get('tipo') == 'patch':
            x, y, r = s['x'], s['y'], s.get('raio', _RAIO_LIN)
            cv2.rectangle(out, (x - r, y - r), (x + r, y + r), _COR_LIN, 1)

    linhas: List[str] = []
    if resultado is not None:
        rc = resultado.recall_carpete
        rl = resultado.recall_linha
        if rc is not None and rl is not None:
            linhas.append(f"Rec_c={rc*100:.0f}%  Rec_l={rl*100:.0f}%")
        linhas.append(
            f"frV={resultado.fracao_verde*100:.1f}%  "
            f"frCv={resultado.fracao_convexa*100:.1f}%  "
            f"y_min={resultado.y_gauss_min:.1f}"
        )
    for a in avisos:
        if a.nivel in ('atencao', 'grave'):
            linhas.append(f"[{a.nivel.upper()}] {a.codigo}")
    linhas.extend(extras)

    if linhas:
        tx, ty, lh = 8, 20, 22
        box_w = max(len(l) for l in linhas) * 8 + 16
        box_h = len(linhas) * lh + 8
        cv2.rectangle(out, (tx - 4, ty - 16), (tx + box_w, ty + box_h - 10), (20, 20, 20), -1)
        for i, linha in enumerate(linhas):
            _put_text_shadowed(out, linha, (tx, ty + i * lh), 0.48, (255, 255, 255), 1)

    return out


# ---------------------------------------------------------------------------
# Carregamento de imagens
# ---------------------------------------------------------------------------

def _carregar_imagens(caminhos: List[str]) -> Dict[str, np.ndarray]:
    """Carrega imagens BGR; nome = basename sem extensão."""
    imagens: Dict[str, np.ndarray] = {}
    for caminho in caminhos:
        nome = os.path.splitext(os.path.basename(caminho))[0]
        img = cv2.imread(caminho)
        if img is None:
            raise FileNotFoundError(f'Não foi possível carregar: {caminho}')
        imagens[nome] = img
    return imagens


# ---------------------------------------------------------------------------
# Modo headless
# ---------------------------------------------------------------------------

def _modo_headless(
    imagens: Dict[str, np.ndarray],
    amostras: dict,
    saida: str,
    force: bool,
) -> None:
    """Roda calibração sem GUI: specs JSON → estimar → salvar JSON + PNG overlay."""
    sessao = SessaoCalibracao(imagens)

    for spec in amostras.get('carpete', []):
        avisos_add = sessao.adicionar_carpete(spec)
        for a in avisos_add:
            print(f'  [AVISO_CARP] {a}')

    for spec in amostras.get('linha', []):
        avisos_add = sessao.adicionar_linha(spec)
        for a in avisos_add:
            print(f'  [AVISO_LIN] {a}')

    if not sessao.pronto():
        raise SystemExit(
            f'Specs insuficientes após --amostras: '
            f'carpete={len(sessao._specs_carp)}/4, linha={len(sessao._specs_lin)}/1.'
        )

    print('Estimando calibração...')
    calib, avisos = sessao.estimar()

    if avisos:
        print(f'\nAvisos ({len(avisos)}):')
        for a in avisos:
            print(f'  [{a.nivel.upper()}] {a.codigo}: {a.mensagem}')

    print(f'\nVerde:  {calib["verde"]}')
    print(f'Branco: {calib["branco"]}')

    sessao.salvar(saida, confirmado=True, sobrescrever=force)
    print(f'\nCalibração salva: {saida}')

    stem = os.path.splitext(saida)[0]
    for nome, img_bgr in imagens.items():
        resultado = avaliar_calibracao(img_bgr, calib)
        avisos_img = [a for a in avisos if a.nivel in ('atencao', 'grave')]
        overlay = gerar_overlay(
            img_bgr, nome, resultado,
            sessao._specs_carp, sessao._specs_lin,
            avisos_img,
        )
        png_path = f'{stem}_overlay_{nome}.png'
        cv2.imwrite(png_path, overlay)
        print(f'Overlay salvo: {png_path}')


# ---------------------------------------------------------------------------
# Modo interativo
# ---------------------------------------------------------------------------

class _Estado:
    NORMAL = 'normal'
    AGUARDA_SALTO = 'aguarda_salto'
    AGUARDA_EXTRA = 'aguarda_extra'


def _modo_gui(
    imagens: Dict[str, np.ndarray],
    amostras_iniciais: Optional[dict],
    saida: str,
    force: bool,
) -> None:
    nomes = list(imagens.keys())
    idx_atual = [0]  # mutável via closure

    sessao = SessaoCalibracao(imagens)
    if amostras_iniciais is not None:
        for spec in amostras_iniciais.get('carpete', []):
            sessao.adicionar_carpete(spec)
        for spec in amostras_iniciais.get('linha', []):
            sessao.adicionar_linha(spec)

    resultado_atual: List[Optional[ResultadoPreview]] = [None]
    avisos_atuais: List[List[Aviso]] = [[]]
    modo_adicao = ['carp']
    estado = [_Estado.NORMAL]
    delta_pp = [0.0]
    extras = [[]]
    status = ['']

    win = 'Calibrar Cores  (s=salvar  q=sair  h=ajuda)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _atualizar():
        if sessao.calibracao is not None:
            nome = nomes[idx_atual[0]]
            resultado_atual[0] = avaliar_calibracao(imagens[nome], sessao.calibracao)
        else:
            resultado_atual[0] = None

    def _mostrar():
        nome = nomes[idx_atual[0]]
        ex = list(extras[0])
        ex.append(
            f'[{"CARP" if modo_adicao[0]=="carp" else "LIN "}]  '
            f'{idx_atual[0]+1}/{len(nomes)}: {nome}'
        )
        if status[0]:
            ex.append(status[0])
        if estado[0] == _Estado.AGUARDA_SALTO:
            ex.append(f'>>> SALTO +{delta_pp[0]:.1f}pp  y=confirmar  n=rejeitar')
        elif estado[0] == _Estado.AGUARDA_EXTRA:
            ex.append('>>> LIMITE RODADAS  y=confirmar_extra  n=cancelar')
        frame = gerar_overlay(
            imagens[nome], nome, resultado_atual[0],
            sessao._specs_carp, sessao._specs_lin,
            avisos_atuais[0], ex,
        )
        cv2.imshow(win, frame)

    def _tentar_estimar(confirmar_extra: bool) -> None:
        try:
            calib, av = sessao.estimar(confirmar_extra=confirmar_extra)
            avisos_atuais[0] = [a for a in av if a.nivel in ('atencao', 'grave')]
            estado[0] = _Estado.NORMAL
            extras[0] = []
            status[0] = 'estimado'
            _atualizar()
        except SessaoNaoProntaError as e:
            status[0] = str(e)
        except LimiteRodadasError:
            estado[0] = _Estado.AGUARDA_EXTRA
            extras[0] = []
            status[0] = 'limite de rodadas atingido'
        except SaltoVerdeGrandeError as exc:
            delta_pp[0] = exc.delta_pp
            estado[0] = _Estado.AGUARDA_SALTO
            extras[0] = [f'Δ(frV)=+{exc.delta_pp:.1f}pp > limiar']
            status[0] = 'salto grande — confirmar? (y/n)'

    def _salvar_e_overlays() -> bool:
        try:
            sessao.salvar(saida, confirmado=True, sobrescrever=force)
            stem = os.path.splitext(saida)[0]
            for nome, img_bgr in imagens.items():
                res = avaliar_calibracao(img_bgr, sessao.calibracao) if sessao.calibracao else None
                ov = gerar_overlay(img_bgr, nome, res, sessao._specs_carp, sessao._specs_lin, [])
                cv2.imwrite(f'{stem}_overlay_{nome}.png', ov)
                print(f'Overlay: {stem}_overlay_{nome}.png')
            print(f'Salvo: {saida}')
            return True
        except FileExistsError as exc:
            status[0] = f'Erro: {exc}. Use --force.'
            return False

    def _on_mouse(event, x, y, flags, _param):
        if estado[0] != _Estado.NORMAL:
            return
        nome = nomes[idx_atual[0]]
        if event == cv2.EVENT_LBUTTONDOWN:
            modo = modo_adicao[0]
        elif event == cv2.EVENT_RBUTTONDOWN:
            modo = 'lin'
        else:
            return
        spec = {
            'imagem': nome, 'tipo': 'patch', 'x': x, 'y': y,
            'raio': _RAIO_CARP if modo == 'carp' else _RAIO_LIN,
        }
        avisos_add = sessao.adicionar_carpete(spec) if modo == 'carp' else sessao.adicionar_linha(spec)
        status[0] = '  '.join(avisos_add) if avisos_add else ''
        _mostrar()

    cv2.setMouseCallback(win, _on_mouse)
    _mostrar()

    print(f'\nAberto: {win}')
    print('esq=carpete  dir=linha  l=alternar  Enter=estimar  '
          'u=desfazer  r=reiniciar  s=salvar  q=sair  ←→=imagem')

    while True:
        _mostrar()
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('l'):
            modo_adicao[0] = 'lin' if modo_adicao[0] == 'carp' else 'carp'
            status[0] = f'modo: {"CARPETE" if modo_adicao[0]=="carp" else "LINHA"}'

        elif key == ord('u') and estado[0] == _Estado.NORMAL:
            ok = sessao.desfazer()
            status[0] = 'desfeito' if ok else 'nada a desfazer'
            _atualizar()

        elif key == ord('r'):
            sessao.reiniciar()
            resultado_atual[0] = None
            avisos_atuais[0] = []
            estado[0] = _Estado.NORMAL
            extras[0] = []
            status[0] = 'reiniciado'

        elif key in (13, ord('e')) and estado[0] == _Estado.NORMAL:
            _tentar_estimar(False)

        elif key == ord('y'):
            if estado[0] == _Estado.AGUARDA_SALTO:
                calib, av = sessao.confirmar_salto()
                avisos_atuais[0] = [a for a in av if a.nivel in ('atencao', 'grave')]
                estado[0] = _Estado.NORMAL
                extras[0] = []
                status[0] = f'salto confirmado (+{delta_pp[0]:.1f}pp)'
                _atualizar()
            elif estado[0] == _Estado.AGUARDA_EXTRA:
                _tentar_estimar(True)

        elif key == ord('n'):
            if estado[0] == _Estado.AGUARDA_SALTO:
                sessao.rejeitar_salto()
                estado[0] = _Estado.NORMAL
                extras[0] = []
                status[0] = 'salto rejeitado'
            elif estado[0] == _Estado.AGUARDA_EXTRA:
                estado[0] = _Estado.NORMAL
                extras[0] = []
                status[0] = 'cancelado'

        elif key == ord('s'):
            if sessao.calibracao is None:
                status[0] = 'Estimar antes de salvar (Enter)'
            elif _salvar_e_overlays():
                break

        elif key in (81, 2):   # seta esquerda
            idx_atual[0] = (idx_atual[0] - 1) % len(nomes)
            _atualizar()
            status[0] = ''

        elif key in (83, 3, ord(' ')):  # seta direita / espaço
            idx_atual[0] = (idx_atual[0] + 1) % len(nomes)
            _atualizar()
            status[0] = ''

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('imagens', nargs='+', help='Caminhos para as imagens.')
    parser.add_argument(
        '--saida', default='calibracao.json',
        help='Arquivo JSON de saída (padrão: calibracao.json).',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Sobrescreve o arquivo de saída se já existir.',
    )
    parser.add_argument(
        '--amostras', metavar='SPECS.JSON',
        help='JSON com specs de amostras (obrigatório no modo headless).',
    )
    parser.add_argument(
        '--sem-gui', action='store_true',
        help='Força modo headless mesmo se DISPLAY disponível.',
    )
    args = parser.parse_args()

    imagens = _carregar_imagens(args.imagens)

    amostras: Optional[dict] = None
    if args.amostras:
        with open(args.amostras, 'r', encoding='utf-8') as f:
            amostras = json.load(f)

    headless = args.sem_gui or not _tem_display()

    if headless:
        if amostras is None:
            raise SystemExit(
                'Modo headless requer --amostras. '
                'Passe um JSON de specs ou use o modo interativo.'
            )
        _modo_headless(imagens, amostras, args.saida, args.force)
    else:
        _modo_gui(imagens, amostras, args.saida, args.force)


if __name__ == '__main__':
    main()
