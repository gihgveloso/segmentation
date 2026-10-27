import pytest
from features.graph import Node, StructureGraph


def test_add_node_e_recuperar():
    g = StructureGraph()
    n = Node(id=0, x=3, y=7, gray=4.5)
    g.add_node(n)
    assert 0 in g.nodes
    assert g.nodes[0].x == 3
    assert g.nodes[0].y == 7
    assert g.nodes[0].gray == pytest.approx(4.5)


def test_grau_sem_arestas():
    g = StructureGraph()
    g.add_node(Node(id=0, x=0, y=0, gray=0.0))
    assert g.degree(0) == 0


def test_add_aresta_atualiza_grau():
    g = StructureGraph()
    g.add_node(Node(id=0, x=0, y=0, gray=0.0))
    g.add_node(Node(id=1, x=1, y=0, gray=0.0))
    g.add_edge(0, 1)
    assert g.degree(0) == 1
    assert g.degree(1) == 1


def test_aresta_bidirecional():
    """add_edge deve criar aresta nos dois sentidos."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=0, y=0, gray=0.0))
    g.add_node(Node(id=1, x=1, y=0, gray=0.0))
    g.add_edge(0, 1)
    assert 1 in g.neighbors(0)
    assert 0 in g.neighbors(1)


def test_vizinhos_cadeia():
    g = StructureGraph()
    for i in range(3):
        g.add_node(Node(id=i, x=i, y=0, gray=0.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    assert set(g.neighbors(1)) == {0, 2}
    assert g.neighbors(0) == [1]
    assert g.neighbors(2) == [1]


def test_remover_no_atualiza_vizinhos():
    g = StructureGraph()
    for i in range(3):
        g.add_node(Node(id=i, x=i, y=0, gray=0.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.remove_node(1)
    assert 1 not in g.nodes
    assert g.degree(0) == 0
    assert g.degree(2) == 0


def test_arestas_sem_duplicatas():
    g = StructureGraph()
    for i in range(3):
        g.add_node(Node(id=i, x=i, y=0, gray=0.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    edges = g.edges
    assert len(edges) == 2
    # Não importa a ordem, mas pares devem ser únicos
    pares = {(min(a, b), max(a, b)) for a, b in edges}
    assert pares == {(0, 1), (1, 2)}


def test_add_aresta_idempotente():
    """Adicionar a mesma aresta duas vezes não deve duplicar o grau."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=0, y=0, gray=0.0))
    g.add_node(Node(id=1, x=1, y=0, gray=0.0))
    g.add_edge(0, 1)
    g.add_edge(0, 1)
    assert g.degree(0) == 1
    assert g.degree(1) == 1


def test_no_grau_zero_permitido():
    """D3: nós de grau 0 são válidos no grafo (descarte é da Etapa 7)."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=5, y=5, gray=3.0))
    assert 0 in g.nodes
    assert g.degree(0) == 0
