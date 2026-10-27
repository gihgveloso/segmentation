"""Teste de regressão: métricas do avaliador no gabarito das 3 imagens reais.

Congela os valores medidos em 2026-10-27 com o pipeline completo padrão:
  construir_grafo → simplificar → detectar_cruzamentos
Tolerância de localização: 6.0 px na grade subamostrada.

Comando para reproduzir manualmente:
  python3 avaliar_cruzamentos.py
"""
import math
import json
import os
import sys
import pytest
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

GABARITO = os.path.join(os.path.dirname(__file__), 'gabarito_cruzamentos.json')
_BASE = os.path.join(os.path.dirname(__file__), '..')
IMAGES = [
    ('image_1', os.path.join(_BASE, 'images', 'image_1.png')),
    ('image_2', os.path.join(_BASE, 'images', 'image_2.png')),
    ('image',   os.path.join(_BASE, 'images', 'image.png')),
]
TOL = 6.0

# Valores medidos em 27/10/2026, config padrão (D27+D29+D37 ativos, D26 desligado).
# Código idêntico ao commitado; discrepância vs. medição de 26/10 documentada em docs/decisoes.md.
EXPECTED = {
    'image_1': dict(gt=4, det=2,  tp=1),
    'image_2': dict(gt=6, det=16, tp=3),
    'image':   dict(gt=2, det=9,  tp=2),
    'total':   dict(gt=12, det=27, tp=6, fp=21, fn=6),
}


def _rodar_pipeline(img_path):
    from main import Pixel_Segment
    from features.pipeline import construir_grafo, ConfigGrafo
    from features.simplify import simplificar, ConfigSimplificacao
    from features.crossings import detectar_cruzamentos, ConfigCruzamento

    lut = os.path.join(_BASE, 'green_pixels.csv')
    img = cv2.imread(img_path)
    if img is None:
        pytest.skip(f"imagem não encontrada: {img_path}")
    ps = Pixel_Segment(lut)
    res = ps.processar(img)
    sk = res['skeleton_img'].astype(bool)
    rv = res['mascara_relevo'].astype('float32')
    br = res['white_mask_resized'].astype('uint8')
    vd = res['green_mask_resized'].astype('uint8')
    g = construir_grafo(sk, rv, ConfigGrafo(), branco=br)
    simplificar(g, sk, rv, ConfigSimplificacao())
    aceitos, _ = detectar_cruzamentos(g, br, vd, ConfigCruzamento())
    return aceitos


def _contar(nome, aceitos):
    gt_all = json.load(open(GABARITO))
    gt_pts = gt_all.get(nome, [])
    gt_n = len(gt_pts)
    det_n = len(aceitos)
    usado_gt = [False] * gt_n
    usado_det = [False] * det_n
    tp = 0
    pares = sorted(
        [(math.hypot(gp['x'] - dp.x, gp['y'] - dp.y), ig, id_)
         for ig, gp in enumerate(gt_pts)
         for id_, dp in enumerate(aceitos)
         if math.hypot(gp['x'] - dp.x, gp['y'] - dp.y) <= TOL]
    )
    for d, ig, id_ in pares:
        if not usado_gt[ig] and not usado_det[id_]:
            tp += 1
            usado_gt[ig] = True
            usado_det[id_] = True
    return gt_n, det_n, tp


@pytest.mark.parametrize("nome,img_path", IMAGES)
def test_por_imagem(nome, img_path):
    aceitos = _rodar_pipeline(img_path)
    gt_n, det_n, tp = _contar(nome, aceitos)
    exp = EXPECTED[nome]
    assert gt_n  == exp['gt'],  f"{nome}: GT={gt_n} esperado={exp['gt']}"
    assert det_n == exp['det'], f"{nome}: Det={det_n} esperado={exp['det']}"
    assert tp    == exp['tp'],  f"{nome}: TP={tp} esperado={exp['tp']}"


def test_total():
    totais = dict(gt=0, det=0, tp=0, fp=0, fn=0)
    for nome, img_path in IMAGES:
        aceitos = _rodar_pipeline(img_path)
        gt_n, det_n, tp = _contar(nome, aceitos)
        totais['gt']  += gt_n
        totais['det'] += det_n
        totais['tp']  += tp
        totais['fp']  += det_n - tp
        totais['fn']  += gt_n - tp
    exp = EXPECTED['total']
    assert totais['gt']  == exp['gt'],  f"GT total={totais['gt']} esperado={exp['gt']}"
    assert totais['det'] == exp['det'], f"Det total={totais['det']} esperado={exp['det']}"
    assert totais['tp']  == exp['tp'],  f"TP total={totais['tp']} esperado={exp['tp']}"
    assert totais['fp']  == exp['fp'],  f"FP total={totais['fp']} esperado={exp['fp']}"
    assert totais['fn']  == exp['fn'],  f"FN total={totais['fn']} esperado={exp['fn']}"
