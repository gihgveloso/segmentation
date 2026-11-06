"""Calibração de cor para imagens reais (campo real, iluminação natural).

Sem janela, sem cv2.imshow, sem cv2.namedWindow.

Seção de referência: fora do artigo Schulz 2009 — módulo auxiliar de calibração.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.segmentacao_real import PixelSegmentReal  # noqa: E402

# ---------------------------------------------------------------------------
# Arquivo temporário de LUT (reutilizado entre chamadas)
# ---------------------------------------------------------------------------
_lut_csv_tmp: Optional[str] = None


def _get_lut_csv_tmp() -> str:
    """Retorna (criando se necessário) o CSV mínimo temporário para LUT."""
    global _lut_csv_tmp
    if _lut_csv_tmp is None or not os.path.exists(_lut_csv_tmp):
        fd, path = tempfile.mkstemp(suffix='.csv', prefix='calibracao_lut_')
        os.close(fd)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('H,S,V\n0,0,0\n')
        _lut_csv_tmp = path
    return _lut_csv_tmp


# ---------------------------------------------------------------------------
# 1. Amostragem
# ---------------------------------------------------------------------------

def amostrar_patch(
    img_bgr: np.ndarray,
    x: int,
    y: int,
    raio: int,
) -> np.ndarray:
    """Amostra patch quadrado de raio `raio` centrado em (x, y).

    Retorna array (N,3) uint8 em espaço HSV, recortado nas bordas da imagem.
    `x` é coluna, `y` é linha.
    """
    h, w = img_bgr.shape[:2]
    x0 = max(0, x - raio)
    x1 = min(w, x + raio + 1)
    y0 = max(0, y - raio)
    y1 = min(h, y + raio + 1)
    patch = img_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).astype(np.uint8)


def amostrar_poligono(
    img_bgr: np.ndarray,
    pontos: list,
) -> np.ndarray:
    """Amostra pixels dentro do polígono definido por `pontos`.

    `pontos`: lista de (x, y) em pixels da imagem (x=coluna, y=linha).
    Retorna array (N,3) uint8 em espaço HSV.
    """
    h, w = img_bgr.shape[:2]
    pts = np.array(pontos, dtype=np.int32).reshape((-1, 1, 2))
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts], 255)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mascara > 0]
    return pixels.astype(np.uint8)


def reamostrar(
    specs: List[dict],
    imagens_por_nome: dict,
) -> np.ndarray:
    """Reamostrar pixels conforme lista de specs.

    Cada spec deve ter: 'imagem' (str), 'tipo' ('patch'|'poligono'), e:
      - tipo='patch': x, y, raio
      - tipo='poligono': pontos
    Retorna concatenação de todos os pixels amostrados, shape (N,3) uint8 HSV.
    """
    partes = []
    for spec in specs:
        img = imagens_por_nome[spec['imagem']]
        if spec['tipo'] == 'patch':
            pxs = amostrar_patch(img, spec['x'], spec['y'], spec['raio'])
        elif spec['tipo'] == 'poligono':
            pxs = amostrar_poligono(img, spec['pontos'])
        else:
            raise ValueError(f"tipo desconhecido: {spec['tipo']}")
        partes.append(pxs)
    return np.vstack(partes).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. ParametrosEstimacao
# ---------------------------------------------------------------------------

@dataclass
class ParametrosEstimacao:
    """PROVISÓRIO — derivado de n=2 imagens (bitbots, eliza)."""
    p_baixo: float = 5       # percentil inferior — provisório, Rodada 6 (D47)
    p_alto: float = 95       # percentil superior — provisório, Rodada 6 (D47)
    margem_h: int = 3        # margem em unidades de H
    margem_s: int = 15       # margem em unidades de S — provisório, Rodada 6 (D47)
    margem_v: int = 15       # margem em unidades de V — provisório, Rodada 6 (D47)
    p_linha_v_baixo: float = 1   # percentil V para v_min branco — provisório, Rodada 6 (D47)
    p_linha_s_alto: float = 99   # percentil S para s_max branco — provisório, Rodada 6 (D47)
    margem_branco_s: int = 10    # provisório, Rodada 6 (D47)
    margem_branco_v: int = 10    # provisório, Rodada 6 (D47)
    min_pixels: int = 30


# ---------------------------------------------------------------------------
# 3. estimar_calibracao
# ---------------------------------------------------------------------------

def estimar_calibracao(
    carpete: np.ndarray,
    linha: np.ndarray,
    params: Optional[ParametrosEstimacao] = None,
) -> dict:
    """Estima faixas HSV de calibração a partir de amostras de pixels.

    Parameters
    ----------
    carpete : (N,3) uint8 HSV — pixels do carpete verde
    linha   : (M,3) uint8 HSV — pixels da linha branca
    params  : ParametrosEstimacao (usa defaults se None)

    Returns
    -------
    dict com chaves 'branco' e 'verde', cada um com 6 ints Python nativos.

    Raises
    ------
    ValueError se len(carpete) < min_pixels ou len(linha) < min_pixels,
               ou se faixa invertida (h_min > h_max ou v_min > v_max).
    """
    if params is None:
        params = ParametrosEstimacao()

    if len(carpete) < params.min_pixels:
        raise ValueError(
            f"Carpete tem {len(carpete)} pixels, mínimo é {params.min_pixels}"
        )
    if len(linha) < params.min_pixels:
        raise ValueError(
            f"Linha tem {len(linha)} pixels, mínimo é {params.min_pixels}"
        )

    def ci(val: float, lo: int, hi: int) -> int:
        return int(np.clip(int(round(val)), lo, hi))

    H = carpete[:, 0].astype(float)
    S = carpete[:, 1].astype(float)
    V = carpete[:, 2].astype(float)
    S_l = linha[:, 1].astype(float)
    V_l = linha[:, 2].astype(float)

    h_min = ci(float(np.percentile(H, params.p_baixo)) - params.margem_h, 0, 179)
    h_max = ci(float(np.percentile(H, params.p_alto))  + params.margem_h, 0, 179)
    s_min = ci(float(np.percentile(S, params.p_baixo)) - params.margem_s, 0, 255)
    s_max = 255
    v_min = ci(float(np.percentile(V, params.p_baixo)) - params.margem_v, 0, 255)
    v_max = ci(float(np.percentile(V, params.p_alto))  + params.margem_v, 0, 255)

    if h_min > h_max:
        raise ValueError(f"Verde: faixa H invertida ({h_min} > {h_max})")
    if v_min > v_max:
        raise ValueError(f"Verde: faixa V invertida ({v_min} > {v_max})")

    b_s_max = ci(float(np.percentile(S_l, params.p_linha_s_alto)) + params.margem_branco_s, 0, 255)
    b_v_min = ci(float(np.percentile(V_l, params.p_linha_v_baixo)) - params.margem_branco_v, 0, 255)

    verde = {
        'h_min': h_min,
        'h_max': h_max,
        's_min': s_min,
        's_max': s_max,
        'v_min': v_min,
        'v_max': v_max,
    }
    branco = {
        'h_min': 0,
        'h_max': 179,
        's_min': 0,
        's_max': b_s_max,
        'v_min': b_v_min,
        'v_max': 255,
    }

    return {'verde': verde, 'branco': branco}


# ---------------------------------------------------------------------------
# 4. criar_segmentador
# ---------------------------------------------------------------------------

def criar_segmentador(
    calibracao: dict,
    lut_csv: Optional[str] = None,
) -> PixelSegmentReal:
    """Cria PixelSegmentReal a partir de calibração.

    Se lut_csv is None, cria CSV mínimo em tempfile e reutiliza entre chamadas.
    NUNCA modifica o CSV do Webots (green_pixels.csv).
    """
    if lut_csv is None:
        lut_csv = _get_lut_csv_tmp()
    return PixelSegmentReal(calibracao, lut_csv)


# ---------------------------------------------------------------------------
# 5. ResultadoPreview e avaliar_calibracao
# ---------------------------------------------------------------------------

@dataclass
class ResultadoPreview:
    """Resultado de avaliação de uma calibração em um frame."""
    white_mask: np.ndarray           # (H,W) uint8 da resolução original
    green_mask: np.ndarray           # (H,W) uint8 da resolução original
    mascara_convexa: np.ndarray      # (grid_H, grid_W) uint8
    campo_detectado: bool
    y_gauss_min: float
    fracao_branco: float             # pixels white / total pixels originais
    fracao_verde: float              # pixels green / total pixels originais
    fracao_convexa: float            # cells convexa / total cells da grade
    overlap_pixels: int              # pixels onde white & green ambos != 0
    overlap_fracao: float            # overlap_pixels / total pixels originais (fração, não %)
    recall_carpete: Optional[float]  # fração de amostras carpete dentro da caixa verde
    recall_linha: Optional[float]    # fração de amostras linha dentro da caixa branca
    fracao_negativos_no_verde: Optional[float]   # fração de negativos dentro da caixa verde
    separacao_v: Optional[float]                 # v_min_branco - p99(V_carpete); informacional
    fracao_carpete_na_caixa_branca: Optional[float]   # fração de carpete dentro da caixa branca
    fracao_negativos_na_caixa_branca: Optional[float] # fração de negativos dentro da caixa branca


def _recall_em_caixa(pixels_hsv: np.ndarray, caixa: dict) -> float:
    """Fração de pixels dentro da caixa HSV (inclusive)."""
    H = pixels_hsv[:, 0].astype(int)
    S = pixels_hsv[:, 1].astype(int)
    V = pixels_hsv[:, 2].astype(int)
    dentro = (
        (H >= caixa['h_min']) & (H <= caixa['h_max']) &
        (S >= caixa['s_min']) & (S <= caixa['s_max']) &
        (V >= caixa['v_min']) & (V <= caixa['v_max'])
    )
    return float(np.mean(dentro))


def avaliar_calibracao(
    frame_bgr: np.ndarray,
    calibracao: dict,
    segmentador: Optional[PixelSegmentReal] = None,
    carpete: Optional[np.ndarray] = None,
    linha: Optional[np.ndarray] = None,
    negativos: Optional[np.ndarray] = None,
) -> ResultadoPreview:
    """Avalia calibração em um frame BGR.

    Parameters
    ----------
    frame_bgr   : imagem BGR
    calibracao  : dict com 'branco' e 'verde'
    segmentador : PixelSegmentReal (criado temporariamente se None)
    carpete     : (N,3) uint8 HSV — amostras de carpete (opcional)
    linha       : (M,3) uint8 HSV — amostras de linha (opcional)
    negativos   : (K,3) uint8 HSV — amostras negativas (opcional)

    Returns
    -------
    ResultadoPreview com todas as métricas.
    """
    if segmentador is None:
        segmentador = criar_segmentador(calibracao)

    res = segmentador.processar(frame_bgr)

    white_mask = res['white_mask']
    green_mask = res['green_mask']
    mascara_convexa = res['mascara_convexa']
    campo_detectado = bool(res.get('campo_detectado', False))

    h, w = frame_bgr.shape[:2]
    total_pixels = h * w

    y_gauss = res.get('y_gauss', None)
    if y_gauss is not None and len(y_gauss) > 0:
        y_gauss_min = float(np.min(y_gauss))
    else:
        y_gauss_min = 0.0

    fracao_branco = float(np.count_nonzero(white_mask)) / total_pixels
    fracao_verde = float(np.count_nonzero(green_mask)) / total_pixels

    total_cells = mascara_convexa.size
    fracao_convexa = float(np.count_nonzero(mascara_convexa)) / total_cells if total_cells > 0 else 0.0

    overlap_pixels = int(np.count_nonzero((white_mask > 0) & (green_mask > 0)))

    recall_carpete = None
    separacao_v = None
    fracao_carpete_na_caixa_branca = None
    if carpete is not None and len(carpete) > 0:
        recall_carpete = _recall_em_caixa(carpete, calibracao['verde'])
        separacao_v = calibracao['branco']['v_min'] - int(np.percentile(carpete[:, 2], 99))
        fracao_carpete_na_caixa_branca = _recall_em_caixa(carpete, calibracao['branco'])

    recall_linha = None
    if linha is not None and len(linha) > 0:
        recall_linha = _recall_em_caixa(linha, calibracao['branco'])

    fracao_negativos_no_verde = None
    fracao_negativos_na_caixa_branca = None
    if negativos is not None and len(negativos) > 0:
        fracao_negativos_no_verde = _recall_em_caixa(negativos, calibracao['verde'])
        fracao_negativos_na_caixa_branca = _recall_em_caixa(negativos, calibracao['branco'])

    return ResultadoPreview(
        white_mask=white_mask,
        green_mask=green_mask,
        mascara_convexa=mascara_convexa,
        campo_detectado=campo_detectado,
        y_gauss_min=y_gauss_min,
        fracao_branco=fracao_branco,
        fracao_verde=fracao_verde,
        fracao_convexa=fracao_convexa,
        overlap_pixels=overlap_pixels,
        overlap_fracao=overlap_pixels / total_pixels,
        recall_carpete=recall_carpete,
        recall_linha=recall_linha,
        fracao_negativos_no_verde=fracao_negativos_no_verde,
        separacao_v=separacao_v,
        fracao_carpete_na_caixa_branca=fracao_carpete_na_caixa_branca,
        fracao_negativos_na_caixa_branca=fracao_negativos_na_caixa_branca,
    )


# ---------------------------------------------------------------------------
# 6. Aviso e verificar_sanidade
# ---------------------------------------------------------------------------

@dataclass
class Aviso:
    nivel: str    # 'info' | 'atencao' | 'grave'
    codigo: str
    mensagem: str


# Constantes nomeadas (PROVISÓRIO, n=2 imagens: bitbots, eliza)
LIMIAR_PROVISORIO_FRACAO_VERDE_QUEBRADA = 0.75
# acima disto E convexa > 0.80 → atencao
LIMIAR_PROVISORIO_CONVEXA_QUEBRADA = 0.80
# y_gauss_min: boa 9.02 (bitbots) / 20.01 (eliza), quebrada 0.00
LIMIAR_PROVISORIO_YGAUSS_TOPO = 0.0   # <= este valor → atencao
LIMIAR_PROVISORIO_RECALL_MIN = 0.90
LIMIAR_PROVISORIO_NEGATIVOS_MAX = 0.01
# Fração de amostras de carpete (ou negativos) dentro da caixa branca — PROVISÓRIO (n=2)
LIMIAR_PROVISORIO_CARPETE_NA_BRANCA = 0.01
# overlap referência medida: 107px/409600px bitbots, 261px/409600px eliza (inventário 29/10/2026)
REFERENCIA_OVERLAP_BITBOTS_FRACAO = 0.00026
REFERENCIA_OVERLAP_ELIZA_FRACAO = 0.00064


def verificar_sanidade(resultado: ResultadoPreview) -> List[Aviso]:
    """Verifica sanidade do ResultadoPreview e retorna lista de avisos."""
    avisos: List[Aviso] = []

    # 1. campo_detectado False → grave
    if not resultado.campo_detectado:
        avisos.append(Aviso(
            nivel='grave',
            codigo='CAMPO_NAO_DETECTADO',
            mensagem='Campo não detectado: campo_detectado=False. Verificar iluminação e calibração.',
        ))

    # 2. y_gauss_min <= LIMIAR → atencao
    if resultado.y_gauss_min <= LIMIAR_PROVISORIO_YGAUSS_TOPO:
        avisos.append(Aviso(
            nivel='atencao',
            codigo='CONVEXA_AO_TOPO',
            mensagem=(
                f'y_gauss_min={resultado.y_gauss_min:.2f} <= {LIMIAR_PROVISORIO_YGAUSS_TOPO} '
                f'— máscara convexa pode estar cobrindo o topo da imagem.'
            ),
        ))

    # 3. fracao_verde alta AND fracao_convexa alta → atencao
    if (resultado.fracao_verde > LIMIAR_PROVISORIO_FRACAO_VERDE_QUEBRADA and
            resultado.fracao_convexa > LIMIAR_PROVISORIO_CONVEXA_QUEBRADA):
        avisos.append(Aviso(
            nivel='atencao',
            codigo='VERDE_E_CONVEXA_ALTAS',
            mensagem=(
                f'fracao_verde={resultado.fracao_verde:.3f} > {LIMIAR_PROVISORIO_FRACAO_VERDE_QUEBRADA} '
                f'e fracao_convexa={resultado.fracao_convexa:.3f} > {LIMIAR_PROVISORIO_CONVEXA_QUEBRADA} '
                f'— faixa HSV pode estar muito abrangente (ref: eliza H35-100: verde=78.8%, convexa=91.9%).'
            ),
        ))

    # 4. fracao_carpete_na_caixa_branca > LIMIAR → atencao
    if (resultado.fracao_carpete_na_caixa_branca is not None and
            resultado.fracao_carpete_na_caixa_branca > LIMIAR_PROVISORIO_CARPETE_NA_BRANCA):
        avisos.append(Aviso(
            nivel='atencao',
            codigo='CARPETE_NA_CAIXA_BRANCA',
            mensagem=(
                f'fracao_carpete_na_caixa_branca={resultado.fracao_carpete_na_caixa_branca:.3f} '
                f'> {LIMIAR_PROVISORIO_CARPETE_NA_BRANCA} (PROVISÓRIO, n=2) '
                f'— amostras de carpete classificadas como branco; ajustar s_max ou v_min do branco.'
            ),
        ))

    # 4b. fracao_negativos_na_caixa_branca > LIMIAR → atencao
    if (resultado.fracao_negativos_na_caixa_branca is not None and
            resultado.fracao_negativos_na_caixa_branca > LIMIAR_PROVISORIO_CARPETE_NA_BRANCA):
        avisos.append(Aviso(
            nivel='atencao',
            codigo='NEGATIVOS_NA_CAIXA_BRANCA',
            mensagem=(
                f'fracao_negativos_na_caixa_branca={resultado.fracao_negativos_na_caixa_branca:.3f} '
                f'> {LIMIAR_PROVISORIO_CARPETE_NA_BRANCA} (PROVISÓRIO, n=2) '
                f'— pixels negativos (parede/robôs) classificados como branco.'
            ),
        ))

    # 5. recall_carpete < 0.90 → atencao
    if resultado.recall_carpete is not None and resultado.recall_carpete < LIMIAR_PROVISORIO_RECALL_MIN:
        avisos.append(Aviso(
            nivel='atencao',
            codigo='RECALL_CARPETE_BAIXO',
            mensagem=(
                f'recall_carpete={resultado.recall_carpete:.3f} < {LIMIAR_PROVISORIO_RECALL_MIN} '
                f'— muitas amostras de carpete fora da caixa verde.'
            ),
        ))

    # 6. recall_linha < 0.90 → atencao
    if resultado.recall_linha is not None and resultado.recall_linha < LIMIAR_PROVISORIO_RECALL_MIN:
        avisos.append(Aviso(
            nivel='atencao',
            codigo='RECALL_LINHA_BAIXO',
            mensagem=(
                f'recall_linha={resultado.recall_linha:.3f} < {LIMIAR_PROVISORIO_RECALL_MIN} '
                f'— muitas amostras de linha fora da caixa branca.'
            ),
        ))

    # 7. fracao_negativos_no_verde > 0.01 → atencao
    if (resultado.fracao_negativos_no_verde is not None and
            resultado.fracao_negativos_no_verde > LIMIAR_PROVISORIO_NEGATIVOS_MAX):
        avisos.append(Aviso(
            nivel='atencao',
            codigo='NEGATIVOS_NO_VERDE',
            mensagem=(
                f'fracao_negativos_no_verde={resultado.fracao_negativos_no_verde:.3f} > '
                f'{LIMIAR_PROVISORIO_NEGATIVOS_MAX} '
                f'— muitos pixels negativos sendo classificados como verde.'
            ),
        ))

    # 8. overlap_fracao → info sempre
    avisos.append(Aviso(
        nivel='info',
        codigo='OVERLAP_INFO',
        mensagem=(
            f'overlap_fracao={resultado.overlap_fracao:.6f} '
            f'({resultado.overlap_fracao * 100:.4f}%) '
            f'(ref: bitbots={REFERENCIA_OVERLAP_BITBOTS_FRACAO * 100:.3f}%, '
            f'eliza={REFERENCIA_OVERLAP_ELIZA_FRACAO * 100:.3f}%)'
        ),
    ))

    return avisos


# ---------------------------------------------------------------------------
# 7. JSON
# ---------------------------------------------------------------------------

def montar_json(
    calibracao: dict,
    *,
    status: Optional[str] = None,
    data: Optional[str] = None,
    origem: Optional[dict] = None,
    imagens: Optional[list] = None,
    amostras: Optional[dict] = None,
    parametros_estimacao: Optional[dict] = None,
    metricas_sanidade: Optional[dict] = None,
    avisos: Optional[list] = None,
    observacoes_camera: str = '',
    validada_visualmente: bool = False,
    fator: int = 8,
) -> dict:
    """Monta dict JSON de calibração (versão 1).

    NÃO inclui chave 'cruzamentos'.
    """
    now = datetime.now().isoformat()
    if data is None:
        data = now

    if status is None:
        status = 'manual_assistida; NAO validada visualmente'
    if validada_visualmente:
        status = status + f'; validada visualmente em {data}'

    if origem is None:
        origem = {
            'branco': 'Estimado por estimar_calibracao (percentis+margem).',
            'verde': 'Estimado por estimar_calibracao (percentis+margem).',
            'fator': f'{fator} — herdado do Pixel_Segment (main.py).',
        }

    if imagens is None:
        imagens = []

    if amostras is None:
        amostras = {'carpete': [], 'linha': [], 'negativos': []}

    if parametros_estimacao is None:
        parametros_estimacao = {}

    if metricas_sanidade is None:
        metricas_sanidade = {}

    if avisos is None:
        avisos = []

    return {
        'versao': 1,
        'status': status,
        'data': data,
        'origem': origem,
        'imagens': imagens,
        'amostras': amostras,
        'parametros_estimacao': parametros_estimacao,
        'metricas_sanidade': metricas_sanidade,
        'avisos': avisos,
        'observacoes_camera': observacoes_camera,
        'validada_visualmente': validada_visualmente,
        'branco': calibracao['branco'],
        'verde': calibracao['verde'],
        'fator': fator,
    }


def salvar_json(path: str, dados: dict, sobrescrever: bool = False) -> None:
    """Salva dict como JSON em `path`.

    Raises FileExistsError se path existe e sobrescrever=False.
    """
    if os.path.exists(path) and not sobrescrever:
        raise FileExistsError(f'Arquivo já existe: {path}. Use sobrescrever=True.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def carregar_json(path: str) -> dict:
    """Carrega JSON de calibração de `path`."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
