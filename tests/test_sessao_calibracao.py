"""Testes TDD para features/sessao_calibracao.py.

Imagem sintética 160×160:
  linhas 0-59:   parede  H=90, S=100, V=80
  linhas 60-69:  linha   H=0,  S=10,  V=220
  linhas 70-159: carpete H=h_carpete, S=150, V=100  (default h=60)
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_BITBOTS = os.path.join(
    ROOT,
    'images',
    'bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg',
)


# ---------------------------------------------------------------------------
# Imagem sintética
# ---------------------------------------------------------------------------

def _make_img(h_carpete: int = 60) -> np.ndarray:
    hsv = np.zeros((160, 160, 3), dtype=np.uint8)
    for i in range(160):
        for j in range(160):
            ruido = (i * 7 + j * 13) % 5 - 2
            if i < 60:
                h, s, v = 90, 100, 80
            elif i < 70:
                h, s, v = 0, 10, 220
            else:
                h, s, v = h_carpete, 150, 100
            hsv[i, j] = [int(np.clip(h + ruido, 0, 179)), s, v]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _spec_carp(x: int, y: int, imagem: str = 'img1') -> dict:
    return {'imagem': imagem, 'tipo': 'patch', 'x': x, 'y': y, 'raio': 5}


def _spec_lin(x: int = 80, y: int = 65, imagem: str = 'img1') -> dict:
    return {'imagem': imagem, 'tipo': 'patch', 'x': x, 'y': y, 'raio': 5}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def imagens_uma():
    """Uma imagem sintética (carpete H=60)."""
    return {'img1': _make_img(60)}


@pytest.fixture(scope='module')
def imagens_duas():
    """Duas imagens: img1 carpete H=60, img2 carpete H=75."""
    return {'img1': _make_img(60), 'img2': _make_img(75)}


@pytest.fixture
def sessao_vazia(imagens_uma):
    from features.sessao_calibracao import SessaoCalibracao
    return SessaoCalibracao(imagens_uma)


@pytest.fixture
def sessao_com_calib(imagens_uma):
    """Sessão com 4 specs de carpete + 1 de linha + calibração inicial."""
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    for x in [20, 60, 100, 140]:
        s.adicionar_carpete(_spec_carp(x, 90))
    s.adicionar_linha(_spec_lin())
    s.estimar()
    return s


# ---------------------------------------------------------------------------
# 1. pronto()
# ---------------------------------------------------------------------------

def test_pronto_exige_quatro_carp_e_uma_lin(imagens_uma):
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    assert not s.pronto()

    for x in [20, 60, 100]:
        s.adicionar_carpete(_spec_carp(x, 90))
    assert not s.pronto()   # apenas 3 carp

    s.adicionar_carpete(_spec_carp(140, 90))
    assert not s.pronto()   # 4 carp mas sem linha

    s.adicionar_linha(_spec_lin())
    assert s.pronto()


# ---------------------------------------------------------------------------
# 2. SessaoNaoProntaError
# ---------------------------------------------------------------------------

def test_estimar_sem_pronto_levanta_erro(sessao_vazia):
    from features.sessao_calibracao import SessaoNaoProntaError
    with pytest.raises(SessaoNaoProntaError):
        sessao_vazia.estimar()


# ---------------------------------------------------------------------------
# 3. Primeira estimação
# ---------------------------------------------------------------------------

def test_primeira_estimacao_retorna_calibracao(imagens_uma):
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    for x in [20, 60, 100, 140]:
        s.adicionar_carpete(_spec_carp(x, 90))
    s.adicionar_linha(_spec_lin())

    calib, avisos = s.estimar()

    assert 'verde' in calib and 'branco' in calib
    assert s.calibracao is not None
    assert s.n_rodadas == 0        # primeira estimação não conta como corretiva
    assert isinstance(avisos, list)
    # OVERLAP_INFO sempre presente
    assert any(a.codigo == 'OVERLAP_INFO' for a in avisos)


# ---------------------------------------------------------------------------
# 4. Limite de rodadas corretivas
# ---------------------------------------------------------------------------

def test_limite_rodadas_levanta_erro(sessao_com_calib):
    from features.sessao_calibracao import LimiteRodadasError, MAX_RODADAS_CORRETIVAS
    s = sessao_com_calib

    # 1ª rodada corretiva — deve ser permitida
    s.adicionar_carpete(_spec_carp(80, 90))
    s.estimar()
    assert s.n_rodadas == 1

    # 2ª rodada corretiva sem confirmar_extra → LimiteRodadasError
    s.adicionar_carpete(_spec_carp(80, 85))
    with pytest.raises(LimiteRodadasError):
        s.estimar()


def test_limite_rodadas_com_confirmar_extra(sessao_com_calib):
    from features.sessao_calibracao import LimiteRodadasError
    s = sessao_com_calib

    s.adicionar_carpete(_spec_carp(80, 90))
    s.estimar()   # 1ª rodada
    assert s.n_rodadas == 1

    s.adicionar_carpete(_spec_carp(80, 85))
    # confirmar_extra=True → deve passar (sem LimiteRodadasError)
    calib, _ = s.estimar(confirmar_extra=True)
    assert s.n_rodadas == 2
    assert 'verde' in calib


# ---------------------------------------------------------------------------
# 5. Growth guard
# ---------------------------------------------------------------------------

def test_growth_guard_salto_grande_levanta_erro(sessao_com_calib):
    from features.sessao_calibracao import SaltoVerdeGrandeError
    s = sessao_com_calib

    # Simula frV atual artificialmente baixa para forçar Δ > LIMIAR_SALTO_VERDE_PP
    s._fracao_verde_atual = 0.001

    s.adicionar_carpete(_spec_carp(80, 90))
    with pytest.raises(SaltoVerdeGrandeError) as exc_info:
        s.estimar()

    assert exc_info.value.delta_pp > 5.0
    assert s._proposta is not None
    # calibração NÃO deve ter sido atualizada ainda
    assert s._calibracao is not None  # ainda a antiga


def test_growth_guard_confirmar_salto(sessao_com_calib):
    from features.sessao_calibracao import SaltoVerdeGrandeError
    s = sessao_com_calib

    s._fracao_verde_atual = 0.001
    s.adicionar_carpete(_spec_carp(80, 90))

    with pytest.raises(SaltoVerdeGrandeError):
        s.estimar()

    calib_antes = s._calibracao

    nova_calib, avisos = s.confirmar_salto()
    assert s._proposta is None
    assert s.calibracao is not None
    assert 'verde' in nova_calib and 'branco' in nova_calib
    assert isinstance(avisos, list)


def test_growth_guard_rejeitar_salto(sessao_com_calib):
    from features.sessao_calibracao import SaltoVerdeGrandeError
    s = sessao_com_calib

    calib_original = s._calibracao
    frv_original = s._fracao_verde_atual
    s._fracao_verde_atual = 0.001
    s.adicionar_carpete(_spec_carp(80, 90))

    with pytest.raises(SaltoVerdeGrandeError):
        s.estimar()

    s.rejeitar_salto()
    assert s._proposta is None
    # frV continua no valor artificialmente baixo (proposta descartada)
    assert s._fracao_verde_atual == 0.001


def test_confirmar_salto_sem_proposta_levanta_erro(sessao_com_calib):
    from features.sessao_calibracao import SemPropostaError
    with pytest.raises(SemPropostaError):
        sessao_com_calib.confirmar_salto()


# ---------------------------------------------------------------------------
# 6. Avisos suaves — adicionar_*
# ---------------------------------------------------------------------------

def test_aviso_a_linha_no_verde(sessao_com_calib):
    """Aviso (a): linha clicada em área verde da calibração atual."""
    # spec de linha em área de CARPETE (y=90, não y=65)
    avisos = sessao_com_calib.adicionar_linha(_spec_carp(80, 90))
    assert any('AVISO_LINHA_NO_VERDE' in a for a in avisos)


def test_aviso_a_linha_real_nao_dispara(sessao_com_calib):
    """Aviso (a) NÃO dispara quando linha está fora da caixa verde."""
    avisos = sessao_com_calib.adicionar_linha(_spec_lin(80, 65))
    assert not any('AVISO_LINHA_NO_VERDE' in a for a in avisos)


def test_aviso_b_h_desvio(imagens_duas):
    """Aviso (b): H_med do novo patch difere >LIMIAR_H_DESVIO do conjunto existente."""
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_duas)
    # 4 specs de img1 (H=60)
    for x in [20, 60, 100, 140]:
        s.adicionar_carpete(_spec_carp(x, 90, imagem='img1'))
    # spec de img2 (H=75) — difere 15 > LIMIAR_H_DESVIO=12
    avisos = s.adicionar_carpete(_spec_carp(80, 90, imagem='img2'))
    assert any('AVISO_H_DESVIO' in a for a in avisos)


def test_aviso_b_sem_desvio_nao_dispara(imagens_uma):
    """Aviso (b) NÃO dispara quando mesma imagem (H consistente)."""
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    for x in [20, 60, 100]:
        s.adicionar_carpete(_spec_carp(x, 90))
    avisos = s.adicionar_carpete(_spec_carp(140, 90))
    assert not any('AVISO_H_DESVIO' in a for a in avisos)


def test_aviso_c_teco_unico(imagens_uma):
    """Aviso (c): todas as amostras no terço inferior (y=120, teco=2)."""
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    todos_avisos = []
    for x in [20, 60, 100, 140]:
        ws = s.adicionar_carpete(_spec_carp(x, 120))
        todos_avisos.extend(ws)
    assert any('AVISO_TECO_UNICO' in a for a in todos_avisos)


def test_aviso_c_teco_diverso_nao_dispara(imagens_uma):
    """Aviso (c) NÃO dispara quando specs abrangem tecos diferentes."""
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    s.adicionar_carpete(_spec_carp(80, 80))   # teco 1 (médio)
    ws = s.adicionar_carpete(_spec_carp(80, 120))  # teco 2 (inferior)
    assert not any('AVISO_TECO_UNICO' in a for a in ws)


# ---------------------------------------------------------------------------
# 7. Avisos de verificar_sanidade via estimar()
# ---------------------------------------------------------------------------

def test_verificar_sanidade_em_estimar(imagens_uma):
    """estimar() retorna List[Aviso] com pelo menos OVERLAP_INFO."""
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    for x in [20, 60, 100, 140]:
        s.adicionar_carpete(_spec_carp(x, 90))
    s.adicionar_linha(_spec_lin())
    _, avisos = s.estimar()
    assert any(a.codigo == 'OVERLAP_INFO' for a in avisos)


# ---------------------------------------------------------------------------
# 8. desfazer()
# ---------------------------------------------------------------------------

def test_desfazer_remove_ultimo_carp(sessao_vazia):
    s = sessao_vazia
    s.adicionar_carpete(_spec_carp(20, 90))
    s.adicionar_carpete(_spec_carp(60, 90))
    assert len(s._specs_carp) == 2

    result = s.desfazer()
    assert result is True
    assert len(s._specs_carp) == 1


def test_desfazer_vazio_retorna_false(sessao_vazia):
    result = sessao_vazia.desfazer()
    assert result is False


def test_desfazer_apos_estimar_remove_calibracao(imagens_uma):
    from features.sessao_calibracao import SessaoCalibracao
    s = SessaoCalibracao(imagens_uma)
    for x in [20, 60, 100, 140]:
        s.adicionar_carpete(_spec_carp(x, 90))
    s.adicionar_linha(_spec_lin())
    s.estimar()
    assert s.calibracao is not None

    s.desfazer()
    assert s.calibracao is None


# ---------------------------------------------------------------------------
# 9. reiniciar()
# ---------------------------------------------------------------------------

def test_reiniciar_limpa_tudo(sessao_com_calib):
    s = sessao_com_calib
    s.reiniciar()
    assert not s.pronto()
    assert s.calibracao is None
    assert len(s._specs_carp) == 0
    assert len(s._specs_lin) == 0
    assert len(s._historico) == 0
    assert s.n_rodadas == 0


# ---------------------------------------------------------------------------
# 10. salvar()
# ---------------------------------------------------------------------------

def test_salvar_sem_confirmar_levanta_erro(sessao_com_calib, tmp_path):
    from features.sessao_calibracao import SalvarSemConfirmacaoError
    p = tmp_path / 'calib.json'
    with pytest.raises(SalvarSemConfirmacaoError):
        sessao_com_calib.salvar(str(p))


def test_salvar_sobrescrever_sem_force_levanta_erro(sessao_com_calib, tmp_path):
    p = tmp_path / 'calib.json'
    p.write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError):
        sessao_com_calib.salvar(str(p), confirmado=True)


def test_salvar_ok(sessao_com_calib, tmp_path):
    import json
    p = tmp_path / 'calib.json'
    sessao_com_calib.salvar(str(p), confirmado=True)
    dados = json.loads(p.read_text(encoding='utf-8'))
    assert 'verde' in dados
    assert 'branco' in dados
    assert dados.get('validada_visualmente') is False
    assert 'amostras' in dados


# ---------------------------------------------------------------------------
# 11. Headless end-to-end com imagem real
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(_BITBOTS), reason='imagem real não disponível')
def test_headless_e2e_real_image(tmp_path):
    import json
    from features.sessao_calibracao import SessaoCalibracao

    img_bgr = cv2.imread(_BITBOTS)
    assert img_bgr is not None, f'cv2.imread falhou: {_BITBOTS}'

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]

    # encontra pixels verdes e brancos
    mascara_verde = (H >= 35) & (H <= 85) & (S > 50) & (V > 50)
    mascara_branco = (S < 50) & (V > 150)
    ys_v, xs_v = np.where(mascara_verde)
    ys_b, xs_b = np.where(mascara_branco)

    assert len(ys_v) >= 4, 'Imagem não tem pixels verdes suficientes'
    assert len(ys_b) >= 1, 'Imagem não tem pixels brancos'

    idx_v = np.linspace(0, len(ys_v) - 1, 6, dtype=int)
    idx_b = np.linspace(0, len(ys_b) - 1, 2, dtype=int)

    imagens = {'bitbots': img_bgr}
    s = SessaoCalibracao(imagens)

    for i in idx_v[:4]:
        s.adicionar_carpete({
            'imagem': 'bitbots', 'tipo': 'patch',
            'x': int(xs_v[i]), 'y': int(ys_v[i]), 'raio': 10,
        })
    for i in idx_b:
        s.adicionar_linha({
            'imagem': 'bitbots', 'tipo': 'patch',
            'x': int(xs_b[i]), 'y': int(ys_b[i]), 'raio': 5,
        })

    assert s.pronto()
    calib, avisos = s.estimar()

    assert 'verde' in calib and 'branco' in calib
    assert isinstance(avisos, list)

    p = tmp_path / 'calib_bitbots.json'
    s.salvar(str(p), confirmado=True)
    dados = json.loads(p.read_text(encoding='utf-8'))
    assert dados['verde']['h_min'] >= 0 and dados['verde']['h_max'] <= 179


# ---------------------------------------------------------------------------
# 12. JSON → criar_segmentador
# ---------------------------------------------------------------------------

def test_json_criar_segmentador(sessao_com_calib, tmp_path):
    from features.calibracao_cor import carregar_json, criar_segmentador
    p = tmp_path / 'calib.json'
    sessao_com_calib.salvar(str(p), confirmado=True)

    dados = carregar_json(str(p))
    seg = criar_segmentador({'verde': dados['verde'], 'branco': dados['branco']})
    assert seg is not None
