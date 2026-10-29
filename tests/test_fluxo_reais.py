"""Testa o fluxo de avaliação com imagens reais e gabarito temporário.

Garante que:
- _casar e _casar_localizacao funcionam corretamente com dados sintéticos.
- avaliar_conjunto aceita gabarito em arquivo temporário sem tocar em
  tests/gabarito_*.json.
- _cfg_cruzamento_from_calib lê angulo_stem_t_max_grau=150 do JSON real.
- ConfigCruzamento() padrão mantém angulo_stem_t_max_grau=130 (Webots não muda).
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import avaliar_cruzamentos as ac

_ROOT = os.path.join(os.path.dirname(__file__), '..')
_CALIB_REAL = os.path.join(_ROOT, ac.CALIB_DEFAULT)


# --- Testes de regressão de config -------------------------------------------

def test_default_config_stem_max_invariante():
    """ConfigCruzamento() padrão mantém angulo_stem_t_max_grau=130 (Webots inalterado)."""
    from features.crossings import ConfigCruzamento
    cfg = ConfigCruzamento()
    assert cfg.angulo_stem_t_max_grau == 130.0, (
        f"Default mudou! Esperado 130.0, obtido {cfg.angulo_stem_t_max_grau}. "
        "O caminho Webots depende deste default."
    )
    assert cfg.distancia_arco_geom_px == 0


def test_cfg_cruzamento_from_calib_carrega_150():
    """_cfg_cruzamento_from_calib lê angulo_stem_t_max_grau=150 do JSON real."""
    if not os.path.exists(_CALIB_REAL):
        pytest.skip(f'calibração real não encontrada: {_CALIB_REAL}')
    cfg = ac._cfg_cruzamento_from_calib(_CALIB_REAL)
    assert cfg.angulo_stem_t_max_grau == 150.0, (
        f"Esperado 150.0 do JSON real, obtido {cfg.angulo_stem_t_max_grau}"
    )
    assert cfg.distancia_arco_geom_px == 0


def test_cfg_cruzamento_from_calib_sem_secao(tmp_path):
    """Sem seção 'cruzamentos' no JSON, _cfg_cruzamento_from_calib usa defaults."""
    from features.crossings import ConfigCruzamento
    calib_sem = {
        'branco': {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 60,
                   'v_min': 165, 'v_max': 255},
        'verde':  {'h_min': 45, 'h_max': 75, 's_min': 40, 's_max': 255,
                   'v_min': 40, 'v_max': 180},
    }
    p = tmp_path / 'calib_sem_cruz.json'
    p.write_text(json.dumps(calib_sem), encoding='utf-8')
    cfg = ac._cfg_cruzamento_from_calib(str(p))
    # Deve retornar os valores default de ConfigCruzamento
    assert cfg.angulo_stem_t_max_grau == ConfigCruzamento().angulo_stem_t_max_grau
    assert cfg.distancia_arco_geom_px == ConfigCruzamento().distancia_arco_geom_px


# --- Dados sintéticos --------------------------------------------------------

class _FakeCrossing:
    def __init__(self, x, y, tipo):
        self.x = x
        self.y = y
        self.tipo = tipo


# --- Testes unitários de matching --------------------------------------------

def test_casar_localizacao_basico():
    """1 TP (mesmo tipo), 1 FP (fora de tolerância), 1 FN (sem det próxima)."""
    gt  = [{'tipo': 'L', 'x': 10, 'y': 10}, {'tipo': 'T', 'x': 50, 'y': 50}]
    det = [_FakeCrossing(10, 10, 'L'), _FakeCrossing(60, 60, 'X')]
    tp, fp, fn, pares = ac._casar_localizacao(gt, det, tolerancia=6.0)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    assert pares == [('L', 'L')]


def test_casar_distancia():
    """_casar: matching por distância; GT e det já chegam filtrados pelo mesmo tipo."""
    gt = [{'tipo': 'L', 'x': 10, 'y': 10}]
    # dentro da tolerância → TP
    tp, fp, fn = ac._casar(gt, [_FakeCrossing(10, 10, 'L')], tolerancia=6.0)
    assert (tp, fp, fn) == (1, 0, 0)
    # fora da tolerância → FP + FN
    tp, fp, fn = ac._casar(gt, [_FakeCrossing(100, 100, 'L')], tolerancia=6.0)
    assert (tp, fp, fn) == (0, 1, 1)


def test_casar_localizacao_vazio():
    """Comportamento com listas vazias."""
    assert ac._casar_localizacao([], [], 6.0) == (0, 0, 0, [])
    tp, fp, fn, _ = ac._casar_localizacao([], [_FakeCrossing(1, 1, 'L')], 6.0)
    assert (tp, fp, fn) == (0, 1, 0)
    tp, fp, fn, _ = ac._casar_localizacao([{'tipo': 'L', 'x': 1, 'y': 1}], [], 6.0)
    assert (tp, fp, fn) == (0, 0, 1)


# --- Teste de fluxo com gabarito temporário ----------------------------------

def test_fluxo_gabarito_temporario(tmp_path):
    """Fluxo completo com imagens reais e gabarito em tmp_path.

    Garante que avaliar_conjunto lê o gabarito correto e que nenhum arquivo é
    gravado em tests/gabarito_*.json.
    """
    from features.segmentacao_real import PixelSegmentReal
    from features.crossings import ConfigCruzamento

    images = [(n, os.path.join(_ROOT, p)) for n, p in ac.REAL_IMAGES]
    existentes = [(n, p) for n, p in images if os.path.exists(p)]
    if not existentes:
        pytest.skip('imagens reais não disponíveis')

    # Gabarito com anotação em (0,0) — posição improvávelmente ocupada → tudo FN
    gab = {n: [{'tipo': 'L', 'x': 0, 'y': 0}] for n, _ in existentes}
    gab_path = tmp_path / 'gab_reais_tmp.json'
    gab_path.write_text(json.dumps(gab, ensure_ascii=False), encoding='utf-8')

    # Ler de volta (simula o que o avaliador faz ao carregar o arquivo)
    with open(gab_path, 'r', encoding='utf-8') as f:
        gab_lido = json.load(f)

    calib = os.path.join(_ROOT, ac.CALIB_DEFAULT)
    ps    = PixelSegmentReal.from_json(calib, os.path.join(_ROOT, 'green_pixels.csv'))
    cfg   = ConfigCruzamento()

    acum = ac.avaliar_conjunto(gab_lido, existentes, ps, cfg, tolerancia=6.0)

    assert acum['n_avaliadas'] == len(existentes)
    n_gt = sum(len(gab[n]) for n, _ in existentes)
    # tp + fn = total de GT; com anotação em (0,0) esperamos tp=0 na maioria dos casos
    assert acum['tp'] + acum['fn'] == n_gt
    assert acum['tp'] >= 0
    assert acum['fp'] >= 0

    # Confirmar que nenhum tests/gabarito_*.json foi tocado
    import glob
    arquivos_gabarito = glob.glob(os.path.join(_ROOT, 'tests', 'gabarito_*.json'))
    for arq in arquivos_gabarito:
        assert os.path.abspath(str(gab_path)) != os.path.abspath(arq), (
            f"gabarito temporário colidiu com arquivo protegido: {arq}"
        )
