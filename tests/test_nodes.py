import numpy as np
import pytest
from features.nodes import neighbor_count, place_nodes, ConfigNos


# ─── Auxiliares ──────────────────────────────────────────────────────────────

def linha_horizontal(largura: int, relevo_linha: float = 5.0, relevo_fundo: float = 0.0):
    """Retorna (relevo, skeleton) para uma linha horizontal no meio de uma grade 3×L."""
    h, w = 3, largura
    relevo = np.full((h, w), relevo_fundo, dtype=np.float32)
    relevo[1, :] = relevo_linha
    skeleton = np.zeros((h, w), dtype=bool)
    skeleton[1, :] = True
    return relevo, skeleton


# ─── neighbor_count: Opção A (relevo float) ──────────────────────────────────

def test_pico_isolado_c0():
    """Pixel sem vizinhos de relevo >= central → c=0. Seção III-C."""
    relevo = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.float32)
    skel = np.zeros((3, 3), dtype=bool)
    skel[1, 1] = True
    c = neighbor_count(relevo, skel)
    assert c[1, 1] == 0


def test_extremidade_tem_c1():
    """Extremidade de linha: o pixel menor tem 1 vizinho de relevo >= central → c=1."""
    relevo = np.array([[0, 0, 0], [0, 2, 5], [0, 0, 0]], dtype=np.float32)
    skel = np.zeros((3, 3), dtype=bool)
    skel[1, 1] = True
    skel[1, 2] = True
    c = neighbor_count(relevo, skel)
    # pixel (1,1)=2: vizinho (1,2)=5 >= 2 → c=1
    assert c[1, 1] == 1
    # pixel (1,2)=5: vizinho (1,1)=2 < 5 → c=0
    assert c[1, 2] == 0


def test_passagem_com_c2():
    """Dois vizinhos diagonais de relevo >= central → c=2."""
    relevo = np.array([[5, 0, 5], [0, 3, 0], [0, 0, 0]], dtype=np.float32)
    skel = np.zeros((3, 3), dtype=bool)
    skel[0, 0] = True
    skel[0, 2] = True
    skel[1, 1] = True
    c = neighbor_count(relevo, skel)
    # (1,1)=3: (0,0)=5 >= 3 e (0,2)=5 >= 3 → c=2
    assert c[1, 1] == 2


def test_fora_do_esqueleto_retorna_zero():
    relevo = np.array([[5, 5, 5], [5, 3, 5], [5, 5, 5]], dtype=np.float32)
    skel = np.zeros((3, 3), dtype=bool)
    c = neighbor_count(relevo, skel)
    assert c.sum() == 0


def test_linha_constante_extremidades_c1_interior_c2():
    """Linha horizontal com relevo constante: extremidades c=1, interior c=2."""
    relevo, skel = linha_horizontal(7)
    c = neighbor_count(relevo, skel)
    # Extremidades: (1,0) e (1,6)
    assert c[1, 0] == 1, f"extremidade esq c={c[1, 0]}"
    assert c[1, 6] == 1, f"extremidade dir c={c[1, 6]}"
    # Interior
    for col in range(1, 6):
        assert c[1, col] == 2, f"interior col={col} c={c[1, col]}"


# ─── neighbor_count: Opção B (binário, fallback) ─────────────────────────────

cfg_bin = ConfigNos(usar_relevo=False)


def test_binario_isolado():
    skel = np.zeros((3, 3), dtype=bool)
    skel[1, 1] = True
    c = neighbor_count(None, skel, cfg_bin)
    assert c[1, 1] == 0


def test_binario_linha_vertical():
    skel = np.zeros((5, 1), dtype=bool)
    skel[1, 0] = True
    skel[2, 0] = True
    skel[3, 0] = True
    c = neighbor_count(None, skel, cfg_bin)
    assert c[1, 0] == 1  # extremidade superior
    assert c[3, 0] == 1  # extremidade inferior
    assert c[2, 0] == 2  # interior


def test_binario_juncao_t():
    """T-junction binário: pixel central tem 3 vizinhos no esqueleto."""
    skel = np.zeros((3, 3), dtype=bool)
    skel[0, 1] = True  # cima
    skel[1, 0] = True  # esquerda
    skel[1, 1] = True  # centro
    skel[1, 2] = True  # direita
    c = neighbor_count(None, skel, cfg_bin)
    assert c[1, 1] == 3


# ─── place_nodes: Etapa 2 ─────────────────────────────────────────────────────

def test_pixel_isolado_vira_no():
    """D3: pixel isolado (c=0) sempre recebe nó."""
    relevo = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.float32)
    skel = np.zeros((3, 3), dtype=bool)
    skel[1, 1] = True
    c = neighbor_count(relevo, skel)
    g = place_nodes(skel, relevo, c)
    assert len(g.nodes) == 1
    n = list(g.nodes.values())[0]
    assert n.x == 1 and n.y == 1
    assert n.gray == pytest.approx(5.0)


def test_linha_nos_espacados_e_extremos_cobertos():
    """Linha horizontal de 7 px: spacing >= 2, ambos os extremos com nó. PLANO Etapa 2."""
    relevo, skel = linha_horizontal(7)
    c = neighbor_count(relevo, skel)
    g = place_nodes(skel, relevo, c)

    posicoes = np.array([[n.y, n.x] for n in g.nodes.values()])

    # Distância mínima entre quaisquer dois nós >= 2.0
    for i in range(len(posicoes)):
        for j in range(i + 1, len(posicoes)):
            d = np.hypot(posicoes[i, 0] - posicoes[j, 0],
                         posicoes[i, 1] - posicoes[j, 1])
            assert d >= 2.0, f"nós {i},{j} muito próximos: d={d:.2f}"

    # Ambos os extremos cobertos (col 0 e col 6 têm nós)
    cols = set(posicoes[:, 1].astype(int))
    assert 0 in cols, "extremidade esquerda sem nó"
    assert 6 in cols, "extremidade direita sem nó"


def test_linha_longa_nos_espaçados():
    """Linha de 20 px: todos os nós colocados têm spacing >= 2."""
    relevo, skel = linha_horizontal(20)
    c = neighbor_count(relevo, skel)
    g = place_nodes(skel, relevo, c)
    posicoes = np.array([[n.y, n.x] for n in g.nodes.values()])
    for i in range(len(posicoes)):
        for j in range(i + 1, len(posicoes)):
            d = np.hypot(posicoes[i, 0] - posicoes[j, 0],
                         posicoes[i, 1] - posicoes[j, 1])
            assert d >= 2.0, f"spacing violado: d={d:.2f}"


def test_pico_relevo_sempre_recebe_no():
    """Pixel de pico (c_relevo=0) recebe nó mesmo no meio de muitos pixels."""
    relevo = np.zeros((3, 7), dtype=np.float32)
    skel = np.zeros((3, 7), dtype=bool)
    # pico no centro
    relevo[1, :] = [1, 2, 3, 10, 3, 2, 1]
    skel[1, :] = True
    c = neighbor_count(relevo, skel)
    g = place_nodes(skel, relevo, c)
    cols = {n.x for n in g.nodes.values()}
    assert 3 in cols, "pico central sem nó"


def test_arco_com_cbin3_fica_conexo():
    """Arco com pixels c_bin≥3 no meio fica conexo após place_nodes+connect_nodes. D1 fix."""
    # Esqueleto: row 2 cols 0-6 + (1,3).
    # Pixel (1,3) faz (2,2),(2,3),(2,4) terem c_bin=3 — excluídos pelo critério antigo c_bin==2.
    # Com c_xy>=1 (D1 opção A), esses pixels entram no passo guloso e bridgeiam o arco.
    skel = np.zeros((4, 8), dtype=bool)
    skel[2, 0:7] = True   # caminho principal
    skel[1, 3] = True     # pixel lateral que eleva c_bin dos vizinhos de row 2

    relevo = np.zeros((4, 8), dtype=np.float32)
    relevo[2, :7] = [6, 5, 4, 3, 4, 5, 6]   # picos nas extremidades, vale no centro
    relevo[1, 3] = 3

    c = neighbor_count(relevo, skel)
    g = place_nodes(skel, relevo, c)

    from features.connections import connect_nodes, contar_componentes
    connect_nodes(g, skel)

    posicoes = {(n.y, n.x) for n in g.nodes.values()}
    assert (2, 0) in posicoes, "extremo esq deve ter nó"
    assert (2, 6) in posicoes, "extremo dir deve ter nó"
    assert contar_componentes(g) == 1, "arco inteiro deve ser 1 componente conexa"


def test_features_nao_importam_polylines():
    """D6: nenhum módulo de features/ deve importar polylines_linhas ou segmentos_linhas."""
    import ast
    import os
    features_dir = os.path.join(os.path.dirname(__file__), '..', 'features')
    for fname in os.listdir(features_dir):
        if not fname.endswith('.py'):
            continue
        path = os.path.join(features_dir, fname)
        with open(path) as fh:
            src = fh.read()
        assert 'polylines_linhas' not in src, f"{fname} referencia polylines_linhas"
        assert 'segmentos_linhas' not in src, f"{fname} referencia segmentos_linhas"
