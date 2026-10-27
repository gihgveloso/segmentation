"""Avaliação quantitativa dos cruzamentos detectados contra o gabarito.

Uso:
    python avaliar_cruzamentos.py [--tolerancia N] [--sem-geometria] [--sem-consenso]

Opcoes:
    --tolerancia N     Tolerancia em pixels subamostrados para casamento GT<->det
                       (padrao: 6.0).
    --sem-geometria    Desativa verificacao de geometria T/X em ConfigCruzamento
                       (verificar_geometria_t=False, verificar_geometria_x=False).
    --sem-consenso     Desativa consenso V1 (usar_consenso_v1=False).

O gabarito deve estar em tests/gabarito_cruzamentos.json.
Se nao existir, instrucoes serao impressas e o programa saira com codigo 0.

Metricas reportadas
-------------------
A  Localizacao (qualquer tipo): casa GT<->det por posicao, um-para-um guloso
   pela menor distancia, dentro de `tolerancia` px.
   Reporta: precisao e revocacao globais.

B  Classificacao: entre os pares casados em A, matriz de confusao tipo_gt x tipo_det.

C  Por tipo (original): casa GT<->det por posicao E tipo (L, T, X separados).
   Reporta: precisao/revocacao por tipo e total.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from main import Pixel_Segment
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao
from features.crossings import detectar_cruzamentos, ConfigCruzamento

GABARITO_PATH = 'tests/gabarito_cruzamentos.json'
IMAGES = [
    ('image_1', 'images/image_1.png'),
    ('image_2', 'images/image_2.png'),
    ('image',   'images/image.png'),
]
TIPOS = ('L', 'T', 'X')


# --- Pipeline ----------------------------------------------------------------

def _rodar_pipeline(
    img_path: str,
    ps: Pixel_Segment,
    cfg_cruz: ConfigCruzamento,
):
    """Retorna (aceitos, rejeitados) ou (None, None) se nao foi possivel processar."""
    img = __import__('cv2').imread(img_path)
    if img is None:
        return None, None

    res = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['white_mask_resized'].astype(np.uint8)
    verde    = res['green_mask_resized'].astype(np.uint8)

    if skeleton.sum() == 0:
        return [], []

    g = construir_grafo(skeleton, relevo, ConfigGrafo(), branco=branco)
    simplificar(g, skeleton, relevo, ConfigSimplificacao())
    aceitos, rejeitados = detectar_cruzamentos(g, branco, verde, cfg_cruz)
    return aceitos, rejeitados


# --- Casamento por tipo (metrica C) ------------------------------------------

def _casar(
    gt_pontos: list[dict],
    det_pontos,
    tolerancia: float,
) -> tuple[int, int, int]:
    """Casamento guloso por vizinho mais proximo (mesmo tipo).

    Retorna (TP, FP, FN).
    """
    if not gt_pontos and not det_pontos:
        return 0, 0, 0
    if not gt_pontos:
        return 0, len(det_pontos), 0
    if not det_pontos:
        return 0, 0, len(gt_pontos)

    pares: list[tuple[float, int, int]] = []
    for ig, g_pt in enumerate(gt_pontos):
        for id_, d_pt in enumerate(det_pontos):
            dist = math.hypot(g_pt['x'] - d_pt.x, g_pt['y'] - d_pt.y)
            if dist <= tolerancia:
                pares.append((dist, ig, id_))

    pares.sort()

    usados_gt: set[int] = set()
    usados_det: set[int] = set()
    tp = 0
    for dist, ig, id_ in pares:
        if ig in usados_gt or id_ in usados_det:
            continue
        usados_gt.add(ig)
        usados_det.add(id_)
        tp += 1

    fp = len(det_pontos) - len(usados_det)
    fn = len(gt_pontos)  - len(usados_gt)
    return tp, fp, fn


# --- Casamento por localizacao (metrica A + B) --------------------------------

def _casar_localizacao(
    gt_pontos: list[dict],
    det_pontos,
    tolerancia: float,
) -> tuple[int, int, int, list[tuple[str, str]]]:
    """Casamento guloso por posicao (ignora tipo).

    Retorna (TP, FP, FN, pares_tipo) onde pares_tipo = [(tipo_gt, tipo_det), ...].
    """
    if not gt_pontos and not det_pontos:
        return 0, 0, 0, []
    if not gt_pontos:
        return 0, len(det_pontos), 0, []
    if not det_pontos:
        return 0, 0, len(gt_pontos), []

    pares: list[tuple[float, int, int]] = []
    for ig, g_pt in enumerate(gt_pontos):
        for id_, d_pt in enumerate(det_pontos):
            dist = math.hypot(g_pt['x'] - d_pt.x, g_pt['y'] - d_pt.y)
            if dist <= tolerancia:
                pares.append((dist, ig, id_))

    pares.sort()

    usados_gt: set[int] = set()
    usados_det: set[int] = set()
    pares_tipo: list[tuple[str, str]] = []

    for dist, ig, id_ in pares:
        if ig in usados_gt or id_ in usados_det:
            continue
        usados_gt.add(ig)
        usados_det.add(id_)
        pares_tipo.append((gt_pontos[ig]['tipo'], det_pontos[id_].tipo))

    tp = len(pares_tipo)
    fp = len(det_pontos) - len(usados_det)
    fn = len(gt_pontos)  - len(usados_gt)
    return tp, fp, fn, pares_tipo


# --- Formatacao --------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    if den == 0:
        return '  nan'
    return f'{100 * num / den:5.0f}%'


def _imprimir_localizacao(
    nome: str,
    tp: int,
    fp: int,
    fn: int,
) -> None:
    prec = _pct(tp, tp + fp)
    rev  = _pct(tp, tp + fn)
    gt   = tp + fn
    det  = tp + fp
    print(f'  {nome}: GT={gt}  Det={det}  TP={tp}  FP={fp}  FN={fn}'
          f'  Precisao={prec}  Revocacao={rev}')


def _imprimir_matriz_confusao(
    nome: str,
    pares_tipo: list[tuple[str, str]],
) -> None:
    """Imprime matriz de confusao tipo_gt x tipo_det."""
    if not pares_tipo:
        print(f'  {nome}: (nenhum par casado)')
        return

    matriz: dict[str, dict[str, int]] = {t: {t2: 0 for t2 in TIPOS} for t in TIPOS}
    for tg, td in pares_tipo:
        if tg in matriz and td in matriz[tg]:
            matriz[tg][td] += 1

    print(f'  {nome} - matriz de confusao (linhas=GT, colunas=det):')
    header = f'    {"GT\\Det":<6}' + ''.join(f'  {t:>4}' for t in TIPOS)
    print(header)
    for tg in TIPOS:
        row = f'    {tg:<6}' + ''.join(f'  {matriz[tg][td]:>4}' for td in TIPOS)
        print(row)
    corretos = sum(1 for tg, td in pares_tipo if tg == td)
    print(f'    Acuracia de classificacao: {corretos}/{len(pares_tipo)}'
          f' = {100*corretos/len(pares_tipo):.0f}%')


def _imprimir_tabela_c(
    nome: str,
    resultados: dict[str, tuple[int, int, int, int, int]],
) -> None:
    """Imprime tabela de resultados por tipo (metrica C)."""
    print(f'  {nome} - por tipo:')
    linha = f'    {"Tipo":<5}  {"GT":>4}  {"Det":>4}  {"TP":>4}  {"FP":>4}  {"FN":>4}  ' \
            f'{"Precisao":>9}  {"Revocacao":>9}'
    print(linha)
    print('    ' + '-' * (len(linha) - 4))

    tot_gt = tot_det = tot_tp = tot_fp = tot_fn = 0
    for tipo in TIPOS:
        gt, det, tp, fp, fn = resultados[tipo]
        tot_gt  += gt
        tot_det += det
        tot_tp  += tp
        tot_fp  += fp
        tot_fn  += fn
        prec = _pct(tp, tp + fp)
        rev  = _pct(tp, tp + fn)
        print(f'    {tipo:<5}  {gt:>4}  {det:>4}  {tp:>4}  {fp:>4}  {fn:>4}  {prec:>9}  {rev:>9}')

    print('    ' + chr(0x2500) * (len(linha) - 4))
    prec_t = _pct(tot_tp, tot_tp + tot_fp)
    rev_t  = _pct(tot_tp, tot_tp + tot_fn)
    print(f'    {"Total":<5}  {tot_gt:>4}  {tot_det:>4}  {tot_tp:>4}  {tot_fp:>4}  {tot_fn:>4}  '
          f'{prec_t:>9}  {rev_t:>9}')


# --- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Avalia deteccao de cruzamentos contra o gabarito.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--tolerancia', type=float, default=6.0,
                        help='Tolerancia em px subamostrados (padrao: 6.0)')
    parser.add_argument('--sem-geometria', action='store_true',
                        help='Desativa verificacao de geometria T/X')
    parser.add_argument('--sem-consenso', action='store_true',
                        help='Desativa consenso V1')
    args = parser.parse_args()

    if not os.path.exists(GABARITO_PATH):
        print(f'Gabarito nao encontrado: {GABARITO_PATH}')
        print()
        print('Para criar o gabarito, execute:')
        print('    python anotar_cruzamentos.py')
        print()
        print('Em modo OFFLINE (sem display), edite manualmente:')
        print(f'    {GABARITO_PATH}')
        print('Formato:')
        print('    {')
        print('      "image_1": [{"tipo": "T", "x": 50, "y": 81}, ...],')
        print('      "image_2": [...],')
        print('      "image":  [...]')
        print('    }')
        sys.exit(0)

    with open(GABARITO_PATH, 'r', encoding='utf-8') as f:
        gabarito: dict[str, list[dict]] = json.load(f)

    if not gabarito:
        print(f'Gabarito vazio em {GABARITO_PATH}.')
        print('Execute python anotar_cruzamentos.py para criar anotacoes.')
        sys.exit(0)

    cfg_cruz = ConfigCruzamento()
    if args.sem_geometria:
        if hasattr(cfg_cruz, 'verificar_geometria_t'):
            cfg_cruz.verificar_geometria_t = False
        if hasattr(cfg_cruz, 'verificar_geometria_x'):
            cfg_cruz.verificar_geometria_x = False
    if args.sem_consenso:
        if hasattr(cfg_cruz, 'usar_consenso_v1'):
            cfg_cruz.usar_consenso_v1 = False

    ps = Pixel_Segment('green_pixels.csv')
    tolerancia = float(args.tolerancia)

    print(f'Tolerancia: {tolerancia} px subamostrados')
    if args.sem_geometria:
        print('  [sem-geometria] verificacao de geometria T/X desativada')
    if args.sem_consenso:
        print('  [sem-consenso] consenso V1 desativado')

    # Acumuladores para metrica A (localizacao)
    acum_a_tp = acum_a_fp = acum_a_fn = 0
    acum_pares_tipo: list[tuple[str, str]] = []

    # Acumuladores para metrica C (por tipo)
    acum_gt  = {t: 0 for t in TIPOS}
    acum_det = {t: 0 for t in TIPOS}
    acum_tp  = {t: 0 for t in TIPOS}
    acum_fp  = {t: 0 for t in TIPOS}
    acum_fn  = {t: 0 for t in TIPOS}

    imagens_avaliadas = 0

    for basename, img_path in IMAGES:
        gt_img = gabarito.get(basename)
        if not gt_img:
            continue

        if not os.path.exists(img_path):
            print(f'\n[aviso] {img_path} nao encontrado; pulando {basename}.')
            continue

        print(f'\nProcessando {basename}...', end=' ', flush=True)
        aceitos, rejeitados = _rodar_pipeline(img_path, ps, cfg_cruz)
        if aceitos is None:
            print('falha ao carregar imagem, pulando.')
            continue
        print(f'{len(aceitos)} deteccoes aceitas.')

        # --- Metrica A: localizacao (qualquer tipo) ---------------------------
        a_tp, a_fp, a_fn, pares_tipo = _casar_localizacao(gt_img, aceitos, tolerancia)
        acum_a_tp += a_tp
        acum_a_fp += a_fp
        acum_a_fn += a_fn
        acum_pares_tipo.extend(pares_tipo)

        print(f'\n=== {basename} ===')

        print('\n[A] Localizacao (qualquer tipo):')
        _imprimir_localizacao(basename, a_tp, a_fp, a_fn)

        # --- Metrica B: classificacao dos pares de A -------------------------
        print('\n[B] Classificacao dos pares casados:')
        _imprimir_matriz_confusao(basename, pares_tipo)

        # --- Metrica C: por tipo (original) ----------------------------------
        resultados_c: dict[str, tuple[int, int, int, int, int]] = {}
        for tipo in TIPOS:
            gt_tipo  = [p for p in gt_img if p.get('tipo') == tipo]
            det_tipo = [d for d in aceitos if d.tipo == tipo]
            tp, fp, fn = _casar(gt_tipo, det_tipo, tolerancia)
            resultados_c[tipo] = (len(gt_tipo), len(det_tipo), tp, fp, fn)

            acum_gt[tipo]  += len(gt_tipo)
            acum_det[tipo] += len(det_tipo)
            acum_tp[tipo]  += tp
            acum_fp[tipo]  += fp
            acum_fn[tipo]  += fn

        print('\n[C] Por tipo (matching GT=det):')
        _imprimir_tabela_c(basename, resultados_c)

        imagens_avaliadas += 1

    if imagens_avaliadas == 0:
        print('\nNenhuma imagem com gabarito foi encontrada/processada.')
        print('Execute python anotar_cruzamentos.py para criar anotacoes.')
        sys.exit(0)

    if imagens_avaliadas > 1:
        print('\n' + '=' * 50)
        print('=== TOTAL (todas as imagens) ===')

        print('\n[A] Localizacao (qualquer tipo):')
        _imprimir_localizacao('TOTAL', acum_a_tp, acum_a_fp, acum_a_fn)

        print('\n[B] Classificacao dos pares casados:')
        _imprimir_matriz_confusao('TOTAL', acum_pares_tipo)

        resultados_c_glob: dict[str, tuple[int, int, int, int, int]] = {}
        for tipo in TIPOS:
            resultados_c_glob[tipo] = (
                acum_gt[tipo], acum_det[tipo],
                acum_tp[tipo], acum_fp[tipo], acum_fn[tipo],
            )
        print('\n[C] Por tipo (matching GT=det):')
        _imprimir_tabela_c('TOTAL', resultados_c_glob)


if __name__ == '__main__':
    main()
