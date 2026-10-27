"""Testes de subdivisão de arcos longos em dividir_grau_zero (Etapa 4, fix aresta longa)."""
import numpy as np
import pytest
from features.graph import Node, StructureGraph
from features.connections import dividir_grau_zero, ConfigSubdivisao, contar_componentes


def _no_isolado(g: StructureGraph, nid: int, y: int, x: int, gray: float = 5.0):
    g.add_node(Node(id=nid, x=x, y=y, gray=gray))


# ── Arco em U ────────────────────────────────────────────────────────────────

def _skel_u(rows=9, cols=21, arm_col_l=2, arm_col_r=20, base_row=8):
    """Arco em U: braço esq (col arm_col_l, rows 0..base_row),
    base (row base_row, cols arm_col_l..arm_col_r),
    braço dir (col arm_col_r, rows 0..base_row)."""
    s = np.zeros((rows + 1, cols), dtype=bool)
    s[:base_row + 1, arm_col_l] = True
    s[base_row, arm_col_l:arm_col_r + 1] = True
    s[:base_row + 1, arm_col_r] = True
    return s


def test_arco_u_vira_cadeia():
    """Arco em U de ~35 px com 1 nó de grau 0 subdivide em cadeia conectada. III-C fix."""
    skel = _skel_u()   # ~35 pixels (8+18+8 - 2 cantos = 32 pixels)
    relevo = np.ones(skel.shape, dtype=np.float32) * 5.0

    g = StructureGraph()
    _no_isolado(g, 0, y=0, x=2)   # grau 0, na ponta esquerda do U

    cfg = ConfigSubdivisao(max_segmento_px=20, passo_override=5.0)
    n = dividir_grau_zero(g, skel, relevo, cfg_sub=cfg)

    assert n == 1, "uma componente dividida"
    assert len(g.nodes) >= 3, "pelo menos 3 nós para um arco de ~32 px com passo=5"
    assert contar_componentes(g) == 1, "cadeia deve ser 1 componente conexa"

    # Nenhuma aresta deve cruzar o 'vazio' do U — a mais longa deve ser ≤ 2*passo
    for a, b in g.edges:
        na, nb = g.nodes[a], g.nodes[b]
        dist = np.hypot(na.y - nb.y, na.x - nb.x)
        assert dist <= 2 * cfg.passo_override + 1, \
            f"aresta {a}-{b} dist={dist:.1f} > 2*passo — pode cruzar o vazio"


def test_arco_u_endpoints_nos_extremos():
    """Os extremos da cadeia devem estar nos vértices do U (ponta esq e dir). III-C fix."""
    skel = _skel_u()
    relevo = np.ones(skel.shape, dtype=np.float32) * 5.0

    g = StructureGraph()
    _no_isolado(g, 0, y=4, x=2)   # nó no meio do braço esq, grau 0

    cfg = ConfigSubdivisao(max_segmento_px=20, passo_override=5.0)
    dividir_grau_zero(g, skel, relevo, cfg_sub=cfg)

    posicoes = {(n.y, n.x) for n in g.nodes.values()}
    # Um dos extremos deve estar na ponta esquerda ou direita do U
    assert (0, 2) in posicoes or (0, 20) in posicoes, \
        f"extremo do U não encontrado; posicoes={posicoes}"


# ── Trecho reto longo ─────────────────────────────────────────────────────────

def test_trecho_reto_30px_vira_cadeia():
    """Segmento reto de 30 px subdivide em cadeia com passo=5 → 7 nós, 6 arestas."""
    skel = np.zeros((5, 32), dtype=bool)
    skel[2, 0:30] = True   # row 2, cols 0..29

    relevo = np.ones((5, 32), dtype=np.float32) * 5.0

    g = StructureGraph()
    _no_isolado(g, 0, y=2, x=15)   # nó no centro, grau 0

    cfg = ConfigSubdivisao(max_segmento_px=20, passo_override=5.0)
    n = dividir_grau_zero(g, skel, relevo, cfg_sub=cfg)

    assert n == 1
    assert len(g.nodes) == 7, f"esperado 7 nós (30px/passo5 + 1), obteve {len(g.nodes)}"
    assert len(g.edges) == 6, f"esperado 6 arestas, obteve {len(g.edges)}"
    assert contar_componentes(g) == 1

    # Extremos nos cantos
    posicoes = {(n.y, n.x) for n in g.nodes.values()}
    assert (2, 0) in posicoes, "extremo esq (2,0) não encontrado"
    assert (2, 29) in posicoes, "extremo dir (2,29) não encontrado"

    # Todas as arestas devem ter comprimento ≈ passo
    for a, b in g.edges:
        na, nb = g.nodes[a], g.nodes[b]
        dist = np.hypot(na.y - nb.y, na.x - nb.x)
        assert dist <= cfg.passo_override + 1, \
            f"aresta {a}-{b} comprimento {dist:.1f} > passo+1"


def test_trecho_curto_nao_subdivide():
    """Segmento de 10 px com max_segmento_px=20 → comportamento original: 2 extremos. III-C."""
    skel = np.zeros((3, 12), dtype=bool)
    skel[1, 1:10] = True   # 9 pixels

    relevo = np.ones((3, 12), dtype=np.float32) * 5.0

    g = StructureGraph()
    _no_isolado(g, 0, y=1, x=5)   # nó no meio, grau 0

    cfg = ConfigSubdivisao(max_segmento_px=20, passo_override=5.0)
    n = dividir_grau_zero(g, skel, relevo, cfg_sub=cfg)

    assert n == 1
    assert len(g.nodes) == 2, "apenas 2 extremos para segmento curto"
    assert len(g.edges) == 1
    assert contar_componentes(g) == 1


# ── Sem cfg_sub: comportamento original mantido ──────────────────────────────

def test_sem_cfg_sub_mantém_comportamento_original():
    """Sem cfg_sub, dividir_grau_zero usa ConfigSubdivisao() default — arcos curtos ok."""
    skel = np.zeros((3, 5), dtype=bool)
    skel[1, 1:4] = True   # 3 pixels

    g = StructureGraph()
    _no_isolado(g, 0, y=1, x=2)

    n = dividir_grau_zero(g, skel)   # sem cfg_sub

    assert n == 1
    assert len(g.nodes) == 2
    assert contar_componentes(g) == 1


# ── Múltiplos arcos: long + short na mesma chamada ───────────────────────────

def test_dois_arcos_longo_e_curto():
    """Um arco longo (subdivide) e um arco curto (não subdivide) na mesma chamada."""
    skel = np.zeros((5, 35), dtype=bool)
    skel[1, 0:30] = True   # arco longo: 30 px
    skel[3, 0:5] = True    # arco curto: 5 px

    relevo = np.ones((5, 35), dtype=np.float32) * 5.0

    g = StructureGraph()
    _no_isolado(g, 0, y=1, x=15)   # no longo, grau 0
    _no_isolado(g, 1, y=3, x=2)    # no curto, grau 0

    cfg = ConfigSubdivisao(max_segmento_px=20, passo_override=5.0)
    n = dividir_grau_zero(g, skel, relevo, cfg_sub=cfg)

    assert n == 2, "dois componentes processados"
    assert contar_componentes(g) == 2, "dois componentes distintos"

    # Arco longo deve ter múltiplos nós
    nos_linha1 = [nid for nid, no in g.nodes.items() if no.y == 1]
    assert len(nos_linha1) >= 4, "arco longo deve ter >= 4 nós"

    # Arco curto deve ter apenas 2 nós
    nos_linha3 = [nid for nid, no in g.nodes.items() if no.y == 3]
    assert len(nos_linha3) == 2, "arco curto deve ter apenas 2 nós"
