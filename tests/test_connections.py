"""Testes de Etapa 3: conexão de nós por vizinhança 3×3 (Seção III-C)."""
import numpy as np
import pytest
from features.graph import Node, StructureGraph
from features.connections import connect_nodes


# ─── Auxiliar: grafo com nós colocados manualmente ───────────────────────────

def grafo_com_nos(*posicoes):
    """Cria StructureGraph com nós em (y, x) dado como tuplas."""
    g = StructureGraph()
    for i, (y, x) in enumerate(posicoes):
        g.add_node(Node(id=i, x=x, y=y, gray=5.0))
    return g


# ─── Linha reta ───────────────────────────────────────────────────────────────

def test_linha_cadeia_graus():
    """Linha de 7 px: nós nas colunas 1, 4, 7. Cadeia com graus 1-2-1. III-C."""
    skel = np.zeros((3, 9), dtype=bool)
    skel[1, 1:8] = True  # colunas 1..7

    # Nós colocados manualmente (endpoints + interior)
    g = grafo_com_nos((1, 1), (1, 4), (1, 7))
    connect_nodes(g, skel)

    assert g.degree(0) == 1, "endpoint esq deve ter grau 1"
    assert g.degree(1) == 2, "interior deve ter grau 2"
    assert g.degree(2) == 1, "endpoint dir deve ter grau 1"


def test_linha_sem_aresta_espuria_entre_extremos():
    """Extremos de linha reta não devem ser conectados diretamente."""
    skel = np.zeros((3, 9), dtype=bool)
    skel[1, 1:8] = True
    g = grafo_com_nos((1, 1), (1, 4), (1, 7))
    connect_nodes(g, skel)
    assert 2 not in g.neighbors(0)  # extremo esq não liga ao extremo dir


# ─── L-shape ──────────────────────────────────────────────────────────────────

def test_l_shape_cadeia():
    """L: canto com grau 2, extremos com grau 1. Nós a dist. 3 do canto."""
    # Esqueleto: col 1 linhas 0-3 + linha 3 colunas 1-4
    skel = np.zeros((5, 6), dtype=bool)
    skel[0:4, 1] = True   # braço vertical
    skel[3, 1:5] = True   # braço horizontal

    # Nós: topo (0,1), canto (3,1), extremo dir (3,4)
    g = grafo_com_nos((0, 1), (3, 1), (3, 4))
    connect_nodes(g, skel)

    assert 1 in g.neighbors(0), "topo deve conectar ao canto"
    assert 1 in g.neighbors(2), "extremo dir deve conectar ao canto"
    assert g.degree(1) == 2, "canto deve ter grau 2"
    assert g.degree(0) == 1, "topo deve ter grau 1"
    assert g.degree(2) == 1, "extremo dir deve ter grau 1"


# ─── T-junction ───────────────────────────────────────────────────────────────

def test_t_junction_grau_3():
    """T: nó de junção com grau 3, três extremos com grau 1. III-C."""
    # Grade 7×7. Junção em (3,3).
    # Esqueleto: coluna 3 linhas 0-3 (braço superior) + linha 3 colunas 0-6
    skel = np.zeros((7, 7), dtype=bool)
    skel[0:4, 3] = True   # braço vertical (cima até a junção)
    skel[3, :] = True     # barra horizontal completa

    # Nós: topo (0,3), junção (3,3), esquerda (3,0), direita (3,6)
    g = grafo_com_nos((0, 3), (3, 3), (3, 0), (3, 6))
    connect_nodes(g, skel)

    # Junção deve conectar aos três braços
    assert set(g.neighbors(1)) >= {0, 2, 3}, \
        f"junção deve conectar a 0,2,3; tem {g.neighbors(1)}"
    assert g.degree(1) >= 3

    # Extremos devem ter pelo menos grau 1
    for eid in (0, 2, 3):
        assert g.degree(eid) >= 1, f"extremo {eid} sem conexão"


def test_t_sem_aresta_entre_extremos_opostos():
    """Extremos dos dois lados da barra horizontal do T não se conectam diretamente."""
    skel = np.zeros((7, 7), dtype=bool)
    skel[0:4, 3] = True
    skel[3, :] = True
    g = grafo_com_nos((0, 3), (3, 3), (3, 0), (3, 6))
    connect_nodes(g, skel)
    # esquerda (id=2) e direita (id=3) não devem estar ligados diretamente
    assert 3 not in g.neighbors(2), "esquerda e direita não devem ligar direto"


# ─── X-junction ───────────────────────────────────────────────────────────────

def test_x_junction_grau_4():
    """X: nó de junção com grau 4, quatro extremos com grau 1. III-C."""
    # Grade 7×7. Junção em (3,3).
    skel = np.zeros((7, 7), dtype=bool)
    skel[3, :] = True     # horizontal completo
    skel[:, 3] = True     # vertical completo

    # Nós: junção (3,3) + 4 extremos
    g = grafo_com_nos((3, 3), (3, 0), (3, 6), (0, 3), (6, 3))
    connect_nodes(g, skel)

    # Junção conecta a todos os 4 braços
    assert set(g.neighbors(0)) >= {1, 2, 3, 4}, \
        f"junção deve ter vizinhos 1,2,3,4; tem {g.neighbors(0)}"
    assert g.degree(0) >= 4


# ─── Propriedades gerais ──────────────────────────────────────────────────────

def test_sem_auto_lacos():
    """Nenhum nó deve ter aresta para si mesmo."""
    skel = np.zeros((5, 5), dtype=bool)
    skel[2, :] = True
    skel[:, 2] = True
    g = grafo_com_nos((2, 0), (2, 2), (2, 4), (0, 2), (4, 2))
    connect_nodes(g, skel)
    for nid in g.nodes:
        assert nid not in g.neighbors(nid), f"auto-laço no nó {nid}"


def test_grafo_sem_esqueleto_sem_arestas():
    """Sem pixels de esqueleto nas regiões 3×3, nenhuma aresta é criada."""
    skel = np.zeros((5, 5), dtype=bool)
    g = grafo_com_nos((0, 0), (4, 4))
    connect_nodes(g, skel)
    assert len(g.edges) == 0


def test_nos_sobrepostos_conectados():
    """Dois nós cujas regiões 3×3 se sobrepõem num pixel de esqueleto são conectados."""
    skel = np.zeros((3, 3), dtype=bool)
    skel[1, 1] = True  # pixel central de esqueleto
    # Nó 0 em (1,0): sua região 3×3 cobre (1,1)
    # Nó 1 em (1,2): sua região 3×3 cobre (1,1)
    g = grafo_com_nos((1, 0), (1, 2))
    connect_nodes(g, skel)
    assert g.degree(0) == 1
    assert g.degree(1) == 1
    assert 1 in g.neighbors(0)
