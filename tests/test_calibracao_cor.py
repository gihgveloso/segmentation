"""Testes TDD para features/calibracao_cor.py.

Imagem sintética 160×160:
  linhas 0-59:   fundo/parede   H=90, S=100, V=80  (HSV)
  linhas 60-69:  linha branca   H=0,  S=10,  V=220 (HSV)
  linhas 70-159: carpete verde  H=60, S=150, V=100 (HSV)
  Ruído determinístico em H: (i*7 + j*13) % 5 - 2, clipado a [0,179]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Imagem sintética
# ---------------------------------------------------------------------------

def _make_img() -> np.ndarray:
    """Cria imagem sintética BGR 160×160."""
    hsv = np.zeros((160, 160, 3), dtype=np.uint8)
    for i in range(160):
        for j in range(160):
            ruido = (i * 7 + j * 13) % 5 - 2
            if i < 60:       # fundo/parede
                h, s, v = 90, 100, 80
            elif i < 70:     # linha branca
                h, s, v = 0, 10, 220
            else:             # carpete verde
                h, s, v = 60, 150, 100
            h_r = int(np.clip(h + ruido, 0, 179))
            hsv[i, j] = [h_r, s, v]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.fixture(scope="module")
def img_sint():
    return _make_img()


@pytest.fixture(scope="module")
def img_hsv_sint(img_sint):
    return cv2.cvtColor(img_sint, cv2.COLOR_BGR2HSV)


@pytest.fixture(scope="module")
def carpete_hsv(img_hsv_sint):
    """Pixels HSV do carpete (linhas 70-159)."""
    return img_hsv_sint[70:, :].reshape(-1, 3).copy()


@pytest.fixture(scope="module")
def linha_hsv(img_hsv_sint):
    """Pixels HSV da linha branca (linhas 60-69)."""
    return img_hsv_sint[60:70, :].reshape(-1, 3).copy()


@pytest.fixture(scope="module")
def calib_padrao(carpete_hsv, linha_hsv):
    from features.calibracao_cor import estimar_calibracao
    return estimar_calibracao(carpete_hsv, linha_hsv)


# ---------------------------------------------------------------------------
# 1. test_estimar_calibracao_valores_exatos
# ---------------------------------------------------------------------------

def test_estimar_calibracao_valores_exatos(carpete_hsv, linha_hsv):
    from features.calibracao_cor import estimar_calibracao, ParametrosEstimacao
    p = ParametrosEstimacao()
    calib = estimar_calibracao(carpete_hsv, linha_hsv, p)

    H = carpete_hsv[:, 0].astype(float)
    S = carpete_hsv[:, 1].astype(float)
    V = carpete_hsv[:, 2].astype(float)
    S_l = linha_hsv[:, 1].astype(float)
    V_l = linha_hsv[:, 2].astype(float)

    def ci(a, lo, hi): return int(np.clip(a, lo, hi))

    exp_verde = {
        'h_min': ci(np.percentile(H, p.p_baixo) - p.margem_h, 0, 179),
        'h_max': ci(np.percentile(H, p.p_alto)  + p.margem_h, 0, 179),
        's_min': ci(np.percentile(S, p.p_baixo) - p.margem_s, 0, 255),
        's_max': 255,
        'v_min': ci(np.percentile(V, p.p_baixo) - p.margem_v, 0, 255),
        'v_max': ci(np.percentile(V, p.p_alto)  + p.margem_v, 0, 255),
    }
    exp_branco = {
        'h_min': 0,
        'h_max': 179,
        's_min': 0,
        's_max': ci(np.percentile(S_l, p.p_linha_s_alto) + p.margem_branco_s, 0, 255),
        'v_min': ci(np.percentile(V_l, p.p_linha_v_baixo) - p.margem_branco_v, 0, 255),
        'v_max': 255,
    }

    assert calib['verde'] == exp_verde, f"verde esperado {exp_verde}, obtido {calib['verde']}"
    assert calib['branco'] == exp_branco, f"branco esperado {exp_branco}, obtido {calib['branco']}"

    # todos int Python nativo
    for sec in ('verde', 'branco'):
        for k, v in calib[sec].items():
            assert type(v) is int, f"{sec}.{k} deve ser int, é {type(v)}"


# ---------------------------------------------------------------------------
# 2. test_estimar_calibracao_poucos_pixels
# ---------------------------------------------------------------------------

def test_estimar_calibracao_poucos_pixels():
    from features.calibracao_cor import estimar_calibracao
    arr29 = np.ones((29, 3), dtype=np.uint8)
    arr100 = np.ones((100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        estimar_calibracao(arr29, arr100)
    with pytest.raises(ValueError):
        estimar_calibracao(arr100, arr29)


# ---------------------------------------------------------------------------
# 3. test_estimar_calibracao_int_nativo
# ---------------------------------------------------------------------------

def test_estimar_calibracao_int_nativo(carpete_hsv, linha_hsv):
    from features.calibracao_cor import estimar_calibracao
    calib = estimar_calibracao(carpete_hsv, linha_hsv)
    for sec in ('verde', 'branco'):
        for k, v in calib[sec].items():
            assert isinstance(v, int) and not isinstance(v, np.integer), \
                f"{sec}.{k}={v!r} não é int Python nativo"


# ---------------------------------------------------------------------------
# 4. test_amostrar_patch_centro
# ---------------------------------------------------------------------------

def test_amostrar_patch_centro(img_sint):
    from features.calibracao_cor import amostrar_patch
    # centro do carpete: linha 115, col 80
    pxs = amostrar_patch(img_sint, x=80, y=115, raio=1)
    assert pxs.shape == (9, 3), f"esperado 9 pixels, obtido {pxs.shape}"
    assert pxs.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. test_amostrar_patch_borda
# ---------------------------------------------------------------------------

def test_amostrar_patch_borda(img_sint):
    from features.calibracao_cor import amostrar_patch
    pxs = amostrar_patch(img_sint, x=0, y=0, raio=5)
    # Patch de raio=5 completo seria (2*5+1)^2 = 121 pixels;
    # em (0,0) é recortado para 6×6=36 (apenas quadrante positivo)
    assert pxs.shape[0] <= 36, f"esperado <=36 pixels (recortado), obtido {pxs.shape[0]}"
    assert pxs.shape[0] < 121, "patch não foi recortado: obteve pixels de fora da imagem"


# ---------------------------------------------------------------------------
# 6. test_amostrar_patch_retorna_hsv
# ---------------------------------------------------------------------------

def test_amostrar_patch_retorna_hsv(img_sint):
    from features.calibracao_cor import amostrar_patch
    # patch no centro do carpete
    pxs = amostrar_patch(img_sint, x=80, y=115, raio=3)
    H = pxs[:, 0]
    # H do carpete é ~60 com ruído ±2
    assert np.all(H >= 55) and np.all(H <= 65), f"H esperado ~60, obtido {H}"


# ---------------------------------------------------------------------------
# 7. test_amostrar_poligono_retangulo
# ---------------------------------------------------------------------------

def test_amostrar_poligono_retangulo(img_sint):
    from features.calibracao_cor import amostrar_poligono
    # retângulo 10×10 no carpete: x in [75,85), y in [110,120)
    pts = [(75, 110), (85, 110), (85, 120), (75, 120)]
    pxs = amostrar_poligono(img_sint, pts)
    assert pxs.shape[0] >= 90, f"esperado ~100 pixels, obtido {pxs.shape[0]}"
    assert pxs.dtype == np.uint8


# ---------------------------------------------------------------------------
# 8. test_reamostrar_patch
# ---------------------------------------------------------------------------

def test_reamostrar_patch(img_sint):
    from features.calibracao_cor import amostrar_patch, reamostrar
    imagens = {'sint': img_sint}
    spec = [{'imagem': 'sint', 'tipo': 'patch', 'x': 80, 'y': 115, 'raio': 3}]
    resultado = reamostrar(spec, imagens)
    direto = amostrar_patch(img_sint, 80, 115, 3)
    assert resultado.shape == direto.shape
    assert np.array_equal(resultado, direto)


# ---------------------------------------------------------------------------
# 9. test_reamostrar_poligono
# ---------------------------------------------------------------------------

def test_reamostrar_poligono(img_sint):
    from features.calibracao_cor import amostrar_poligono, reamostrar
    pts = [(75, 110), (85, 110), (85, 120), (75, 120)]
    imagens = {'sint': img_sint}
    spec = [{'imagem': 'sint', 'tipo': 'poligono', 'pontos': pts}]
    resultado = reamostrar(spec, imagens)
    direto = amostrar_poligono(img_sint, pts)
    assert resultado.shape == direto.shape
    assert np.array_equal(resultado, direto)


# ---------------------------------------------------------------------------
# 10. test_criar_segmentador_sem_csv
# ---------------------------------------------------------------------------

def test_criar_segmentador_sem_csv(calib_padrao):
    from features.calibracao_cor import criar_segmentador
    seg = criar_segmentador(calib_padrao)
    assert seg is not None


# ---------------------------------------------------------------------------
# 11. test_criar_segmentador_lut_shape
# ---------------------------------------------------------------------------

def test_criar_segmentador_lut_shape(calib_padrao):
    from features.calibracao_cor import criar_segmentador
    seg = criar_segmentador(calib_padrao)
    assert seg.lut_verde.shape == (180, 256, 256)


# ---------------------------------------------------------------------------
# 12. test_criar_segmentador_lut_nonzeros
# ---------------------------------------------------------------------------

def test_criar_segmentador_lut_nonzeros(calib_padrao):
    from features.calibracao_cor import criar_segmentador
    seg = criar_segmentador(calib_padrao)
    v = calib_padrao['verde']
    esperado = (v['h_max'] - v['h_min'] + 1) * (v['s_max'] - v['s_min'] + 1) * (v['v_max'] - v['v_min'] + 1)
    obtido = int(np.count_nonzero(seg.lut_verde))
    assert obtido == esperado, f"esperado {esperado} nonzeros, obtido {obtido}"


# ---------------------------------------------------------------------------
# 13. test_criar_segmentador_restaura_white_threshold
# ---------------------------------------------------------------------------

def test_criar_segmentador_restaura_white_threshold(calib_padrao, img_sint):
    import def_white_threshold as _m
    from features.calibracao_cor import criar_segmentador
    lower_antes = _m.lower.copy() if isinstance(_m.lower, np.ndarray) else _m.lower
    up_antes = _m.up.copy() if isinstance(_m.up, np.ndarray) else _m.up
    seg = criar_segmentador(calib_padrao)
    seg.processar(img_sint)
    lower_depois = _m.lower.copy() if isinstance(_m.lower, np.ndarray) else _m.lower
    up_depois = _m.up.copy() if isinstance(_m.up, np.ndarray) else _m.up
    assert np.array_equal(lower_antes, lower_depois), "lower não restaurado"
    assert np.array_equal(up_antes, up_depois), "up não restaurado"


# ---------------------------------------------------------------------------
# 14. test_avaliar_calibracao_chaves
# ---------------------------------------------------------------------------

def test_avaliar_calibracao_chaves(calib_padrao, img_sint):
    from features.calibracao_cor import avaliar_calibracao, ResultadoPreview
    import dataclasses
    res = avaliar_calibracao(img_sint, calib_padrao)
    campos = {f.name for f in dataclasses.fields(ResultadoPreview)}
    assert campos.issubset(res.__dataclass_fields__), "campos ausentes no ResultadoPreview"
    # verifica que tem os campos corretos
    assert hasattr(res, 'white_mask')
    assert hasattr(res, 'green_mask')
    assert hasattr(res, 'campo_detectado')
    assert hasattr(res, 'overlap_pixels')
    assert hasattr(res, 'overlap_fracao')


# ---------------------------------------------------------------------------
# 15. test_avaliar_calibracao_overlap
# ---------------------------------------------------------------------------

def test_avaliar_calibracao_overlap(calib_padrao, img_sint):
    from features.calibracao_cor import avaliar_calibracao
    res = avaliar_calibracao(img_sint, calib_padrao)
    overlap_direto = int(np.count_nonzero((res.white_mask > 0) & (res.green_mask > 0)))
    assert res.overlap_pixels == overlap_direto
    h, w = img_sint.shape[:2]
    assert abs(res.overlap_fracao - overlap_direto / (h * w)) < 1e-9


# ---------------------------------------------------------------------------
# Helpers para sanidade
# ---------------------------------------------------------------------------

def _preview_basico(**kwargs):
    """Cria ResultadoPreview com valores padrão válidos, substituindo kwargs."""
    from features.calibracao_cor import ResultadoPreview
    defaults = dict(
        white_mask=np.zeros((10, 10), dtype=np.uint8),
        green_mask=np.zeros((10, 10), dtype=np.uint8),
        mascara_convexa=np.zeros((2, 2), dtype=np.uint8),
        campo_detectado=True,
        y_gauss_min=5.0,
        fracao_branco=0.01,
        fracao_verde=0.30,
        fracao_convexa=0.50,
        overlap_pixels=0,
        overlap_fracao=0.0,
        recall_carpete=None,
        recall_linha=None,
        fracao_negativos_no_verde=None,
        separacao_v=None,
        fracao_carpete_na_caixa_branca=None,
        fracao_negativos_na_caixa_branca=None,
    )
    defaults.update(kwargs)
    return ResultadoPreview(**defaults)


# ---------------------------------------------------------------------------
# 16. test_verificar_sanidade_campo_nao_detectado
# ---------------------------------------------------------------------------

def test_verificar_sanidade_campo_nao_detectado():
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(campo_detectado=False)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'CAMPO_NAO_DETECTADO' in codigos
    grave = [a for a in avisos if a.codigo == 'CAMPO_NAO_DETECTADO']
    assert grave[0].nivel == 'grave'


# ---------------------------------------------------------------------------
# 17. test_verificar_sanidade_convexa_ao_topo
# ---------------------------------------------------------------------------

def test_verificar_sanidade_convexa_ao_topo():
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(y_gauss_min=0.0)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'CONVEXA_AO_TOPO' in codigos
    av = [a for a in avisos if a.codigo == 'CONVEXA_AO_TOPO']
    assert av[0].nivel == 'atencao'


# ---------------------------------------------------------------------------
# 18. test_verificar_sanidade_verde_convexa_altas
# ---------------------------------------------------------------------------

def test_verificar_sanidade_verde_convexa_altas():
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(fracao_verde=0.80, fracao_convexa=0.85)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'VERDE_E_CONVEXA_ALTAS' in codigos
    av = [a for a in avisos if a.codigo == 'VERDE_E_CONVEXA_ALTAS']
    assert av[0].nivel == 'atencao'


# ---------------------------------------------------------------------------
# 19. test_verificar_sanidade_separacao_v_nao_gera_aviso
# ---------------------------------------------------------------------------

def test_verificar_sanidade_separacao_v_nao_gera_aviso():
    """separacao_v negativa é informacional — não gera SEPARACAO_V_NULA_OU_NEGATIVA grave."""
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(separacao_v=-5)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'SEPARACAO_V_NULA_OU_NEGATIVA' not in codigos
    graves = [a for a in avisos if a.nivel == 'grave']
    # campo_detectado=True → nenhum aviso grave
    assert len(graves) == 0


# ---------------------------------------------------------------------------
# 20. test_verificar_sanidade_recall_baixo
# ---------------------------------------------------------------------------

def test_verificar_sanidade_recall_baixo():
    from features.calibracao_cor import verificar_sanidade
    res_c = _preview_basico(recall_carpete=0.85)
    avisos_c = verificar_sanidade(res_c)
    assert 'RECALL_CARPETE_BAIXO' in [a.codigo for a in avisos_c]
    av = [a for a in avisos_c if a.codigo == 'RECALL_CARPETE_BAIXO']
    assert av[0].nivel == 'atencao'

    res_l = _preview_basico(recall_linha=0.85)
    avisos_l = verificar_sanidade(res_l)
    assert 'RECALL_LINHA_BAIXO' in [a.codigo for a in avisos_l]
    av2 = [a for a in avisos_l if a.codigo == 'RECALL_LINHA_BAIXO']
    assert av2[0].nivel == 'atencao'


# ---------------------------------------------------------------------------
# 21. test_verificar_sanidade_negativos
# ---------------------------------------------------------------------------

def test_verificar_sanidade_negativos():
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(fracao_negativos_no_verde=0.05)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'NEGATIVOS_NO_VERDE' in codigos
    av = [a for a in avisos if a.codigo == 'NEGATIVOS_NO_VERDE']
    assert av[0].nivel == 'atencao'


# ---------------------------------------------------------------------------
# 22. test_verificar_sanidade_overlap_info
# ---------------------------------------------------------------------------

def test_verificar_sanidade_overlap_info():
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico()
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'OVERLAP_INFO' in codigos
    av = [a for a in avisos if a.codigo == 'OVERLAP_INFO']
    assert av[0].nivel == 'info'


# ---------------------------------------------------------------------------
# 23. test_json_ida_volta
# ---------------------------------------------------------------------------

def test_json_ida_volta(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json, carregar_json
    path = str(tmp_path / 'calib_test.json')
    dados = montar_json(calib_padrao)
    salvar_json(path, dados)
    recarregado = carregar_json(path)
    assert recarregado['verde'] == calib_padrao['verde']
    assert recarregado['branco'] == calib_padrao['branco']
    assert 'versao' in recarregado


# ---------------------------------------------------------------------------
# 24. test_json_sem_cruzamentos
# ---------------------------------------------------------------------------

def test_json_sem_cruzamentos(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json, carregar_json
    path = str(tmp_path / 'calib_nocruz.json')
    dados = montar_json(calib_padrao)
    salvar_json(path, dados)
    recarregado = carregar_json(path)
    assert 'cruzamentos' not in recarregado


# ---------------------------------------------------------------------------
# 25. test_json_from_json_lut_igual
# ---------------------------------------------------------------------------

def test_json_from_json_lut_igual(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json, criar_segmentador
    from features.segmentacao_real import PixelSegmentReal
    path = str(tmp_path / 'calib_lut.json')
    dados = montar_json(calib_padrao)
    salvar_json(path, dados)

    # CSV mínimo
    csv_tmp = str(tmp_path / 'lut.csv')
    with open(csv_tmp, 'w') as f:
        f.write('H,S,V\n0,0,0\n')

    seg_from_json = PixelSegmentReal.from_json(path, csv_tmp)
    seg_criador = criar_segmentador(calib_padrao)
    assert np.array_equal(seg_from_json.lut_verde, seg_criador.lut_verde)


# ---------------------------------------------------------------------------
# 26. test_json_cfg_cruzamento_defaults
# ---------------------------------------------------------------------------

def test_json_cfg_cruzamento_defaults(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json
    from features.crossings import ConfigCruzamento
    # import inline para evitar dependência circular
    import importlib, sys
    path = str(tmp_path / 'calib_cfg.json')
    dados = montar_json(calib_padrao)
    salvar_json(path, dados)

    # _cfg_cruzamento_from_calib está em avaliar_cruzamentos
    import avaliar_cruzamentos as _ac
    cfg = _ac._cfg_cruzamento_from_calib(path)
    cfg_default = ConfigCruzamento()
    assert cfg.angulo_stem_t_max_grau == cfg_default.angulo_stem_t_max_grau
    assert cfg.distancia_arco_geom_px == cfg_default.distancia_arco_geom_px


# ---------------------------------------------------------------------------
# 27. test_json_chaves_extras
# ---------------------------------------------------------------------------

def test_json_chaves_extras(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json, carregar_json
    path = str(tmp_path / 'calib_extra.json')
    dados = montar_json(calib_padrao)
    dados['chave_extra_desconhecida'] = 'valor'
    salvar_json(path, dados)
    recarregado = carregar_json(path)
    assert 'chave_extra_desconhecida' in recarregado


# ---------------------------------------------------------------------------
# 28. test_salvar_json_recusa_sobrescrever
# ---------------------------------------------------------------------------

def test_salvar_json_recusa_sobrescrever(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json
    path = str(tmp_path / 'calib_exists.json')
    dados = montar_json(calib_padrao)
    salvar_json(path, dados)
    with pytest.raises(FileExistsError):
        salvar_json(path, dados, sobrescrever=False)


# ---------------------------------------------------------------------------
# 29. test_salvar_json_sobrescreve_se_autorizado
# ---------------------------------------------------------------------------

def test_salvar_json_sobrescreve_se_autorizado(calib_padrao, tmp_path):
    from features.calibracao_cor import montar_json, salvar_json, carregar_json
    path = str(tmp_path / 'calib_overwrite.json')
    dados1 = montar_json(calib_padrao, observacoes_camera='v1')
    salvar_json(path, dados1)
    dados2 = montar_json(calib_padrao, observacoes_camera='v2')
    salvar_json(path, dados2, sobrescrever=True)
    recarregado = carregar_json(path)
    assert recarregado['observacoes_camera'] == 'v2'


# ---------------------------------------------------------------------------
# 30. test_verificar_sanidade_carpete_na_branca_atencao
# ---------------------------------------------------------------------------

def test_verificar_sanidade_carpete_na_branca_atencao():
    """fracao_carpete_na_caixa_branca > 0.01 → CARPETE_NA_CAIXA_BRANCA em atencao."""
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(fracao_carpete_na_caixa_branca=0.05)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'CARPETE_NA_CAIXA_BRANCA' in codigos
    av = [a for a in avisos if a.codigo == 'CARPETE_NA_CAIXA_BRANCA']
    assert av[0].nivel == 'atencao'


# ---------------------------------------------------------------------------
# 31. test_verificar_sanidade_carpete_na_branca_abaixo_limiar
# ---------------------------------------------------------------------------

def test_verificar_sanidade_carpete_na_branca_abaixo_limiar():
    """fracao_carpete_na_caixa_branca <= 0.01 ou None → sem CARPETE_NA_CAIXA_BRANCA."""
    from features.calibracao_cor import verificar_sanidade
    for val in (None, 0.005, 0.01):
        res = _preview_basico(fracao_carpete_na_caixa_branca=val)
        codigos = [a.codigo for a in verificar_sanidade(res)]
        assert 'CARPETE_NA_CAIXA_BRANCA' not in codigos, f"esperado sem aviso para val={val}"


# ---------------------------------------------------------------------------
# 32. test_verificar_sanidade_negativos_na_branca_atencao
# ---------------------------------------------------------------------------

def test_verificar_sanidade_negativos_na_branca_atencao():
    """fracao_negativos_na_caixa_branca > 0.01 → NEGATIVOS_NA_CAIXA_BRANCA em atencao."""
    from features.calibracao_cor import verificar_sanidade
    res = _preview_basico(fracao_negativos_na_caixa_branca=0.05)
    avisos = verificar_sanidade(res)
    codigos = [a.codigo for a in avisos]
    assert 'NEGATIVOS_NA_CAIXA_BRANCA' in codigos
    av = [a for a in avisos if a.codigo == 'NEGATIVOS_NA_CAIXA_BRANCA']
    assert av[0].nivel == 'atencao'


# ---------------------------------------------------------------------------
# 33. test_referencia_overlap_sao_fracoes
# ---------------------------------------------------------------------------

def test_referencia_overlap_sao_fracoes():
    """Pina que REFERENCIA_OVERLAP_*_FRACAO são frações (0-1), não percentagens (0-100).

    Valores medidos em inventário 2026-10-29: bitbots=107px/409600px, eliza=261px/409600px.
    """
    from features.calibracao_cor import (
        REFERENCIA_OVERLAP_BITBOTS_FRACAO,
        REFERENCIA_OVERLAP_ELIZA_FRACAO,
    )
    # Frações: devem estar bem abaixo de 0.001 (seriam >=1.0 se fossem %)
    assert 0 < REFERENCIA_OVERLAP_BITBOTS_FRACAO < 0.001, (
        f"esperado fração (< 0.001), obtido {REFERENCIA_OVERLAP_BITBOTS_FRACAO}"
    )
    assert 0 < REFERENCIA_OVERLAP_ELIZA_FRACAO < 0.001, (
        f"esperado fração (< 0.001), obtido {REFERENCIA_OVERLAP_ELIZA_FRACAO}"
    )
    # Valores exatos (inventário 29/10/2026)
    assert abs(REFERENCIA_OVERLAP_BITBOTS_FRACAO - 0.00026) < 1e-10
    assert abs(REFERENCIA_OVERLAP_ELIZA_FRACAO - 0.00064) < 1e-10


def test_defaults_g2w3_d47():
    """D47: defaults de ParametrosEstimacao correspondem ao estimador G2W3 (provisório, n=2 imagens).

    Verde G2: percentis p5/p95 com margem 15 em S e V, margem 3 em H.
    Branco W3: p_linha_v_baixo=1, p_linha_s_alto=99, margem_branco_s=10, margem_branco_v=10.
    """
    from features.calibracao_cor import ParametrosEstimacao
    p = ParametrosEstimacao()
    assert p.p_baixo == 5
    assert p.p_alto == 95
    assert p.margem_h == 3
    assert p.margem_s == 15
    assert p.margem_v == 15
    assert p.p_linha_v_baixo == 1
    assert p.p_linha_s_alto == 99
    assert p.margem_branco_s == 10
    assert p.margem_branco_v == 10
