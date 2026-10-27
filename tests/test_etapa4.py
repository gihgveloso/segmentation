"""Testes de Etapa 4: esticar extremos e dividir nós de grau 0 (Seção III-C)."""
import numpy as np
import pytest
from features.graph import Node, StructureGraph
from features.connections import esticar_extremos, dividir_grau_zero, contar_componentes, ConfigSubdivisao


def _grafo(*posicoes):
    g = StructureGraph()
    for i, (y, x) in enumerate(posicoes):
        g.add_node(Node(id=i, x=x, y=y, gray=5.0))
    return g


def _skel_linha(nrows, ncols, row, c_ini, c_fim):
    s = np.zeros((nrows, ncols), dtype=bool)
    s[row, c_ini:c_fim + 1] = True
    return s


def _relevo_uniforme(shape, valor=5.0):
    r = np.zeros(shape, dtype=np.float32)
    return r  # fundo 0; pixels de skel preenchidos fora


# ─── esticar_extremos ─────────────────────────────────────────────────────────

def test_esticar_move_terminal_horizontal():
    """Terminal em col 2 com esqueleto até col 4: deve mover para col 4. D4."""
    skel = _skel_linha(3, 6, 1, 0, 4)   # row 1, cols 0..4
    relevo = np.full((3, 6), 5.0, dtype=np.float32)

    g = _grafo((1, 0), (1, 2))          # vizinho=0, terminal=1
    g.add_edge(0, 1)
    n = esticar_extremos(g, skel, relevo)

    assert n == 1
    assert g.nodes[1].x == 4
    assert g.nodes[1].y == 1


def test_esticar_nao_move_interior():
    """Nó interior (grau 2) não deve ser esticado. D4."""
    skel = _skel_linha(3, 8, 1, 0, 7)
    relevo = np.full((3, 8), 5.0, dtype=np.float32)

    g = _grafo((1, 0), (1, 3), (1, 7))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    n = esticar_extremos(g, skel, relevo)

    assert g.nodes[1].x == 3   # interior não se move
    assert n == 0              # nenhum nó movido (ambos os terminais já estão nas bordas)


def test_esticar_terminal_ja_no_fim():
    """Terminal já no último pixel de esqueleto: nenhuma mudança."""
    skel = _skel_linha(3, 6, 1, 0, 4)
    relevo = np.full((3, 6), 5.0, dtype=np.float32)

    g = _grafo((1, 0), (1, 4))          # terminal já em col 4
    g.add_edge(0, 1)
    n = esticar_extremos(g, skel, relevo)

    assert n == 0
    assert g.nodes[1].x == 4


def test_esticar_atualiza_gray_d5():
    """D5: gray do nó esticado é atualizado para relevo[nova_pos]."""
    skel = _skel_linha(3, 6, 1, 0, 4)
    relevo = np.zeros((3, 6), dtype=np.float32)
    relevo[1, :] = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]

    g = _grafo((1, 0), (1, 2))
    g.nodes[0].gray = 10.0
    g.nodes[1].gray = 6.0
    g.add_edge(0, 1)

    esticar_extremos(g, skel, relevo)

    assert g.nodes[1].x == 4
    assert g.nodes[1].gray == pytest.approx(2.0)   # relevo na nova pos


def test_esticar_direcao_vertical():
    """Terminal em linha 2 com esqueleto descendo até linha 4: move para linha 4."""
    skel = np.zeros((6, 3), dtype=bool)
    skel[0:5, 1] = True   # col 1, rows 0..4
    relevo = np.full((6, 3), 5.0, dtype=np.float32)

    g = _grafo((0, 1), (2, 1))          # vizinho=row 0, terminal=row 2
    g.add_edge(0, 1)
    esticar_extremos(g, skel, relevo)

    assert g.nodes[1].y == 4
    assert g.nodes[1].x == 1


def test_esticar_direcao_diagonal():
    """Terminal em (2,2), vizinho em (0,0): direção (1,1) → anda na diagonal."""
    skel = np.zeros((5, 5), dtype=bool)
    for k in range(5):
        skel[k, k] = True   # diagonal principal
    relevo = np.full((5, 5), 5.0, dtype=np.float32)

    g = _grafo((0, 0), (2, 2))
    g.add_edge(0, 1)
    esticar_extremos(g, skel, relevo)

    assert g.nodes[1].y == 4
    assert g.nodes[1].x == 4


def test_esticar_sem_esqueleto_alem_nao_move():
    """Sem pixels de esqueleto além do terminal: não move."""
    skel = _skel_linha(3, 4, 1, 0, 2)   # row 1, cols 0..2
    relevo = np.full((3, 4), 5.0, dtype=np.float32)

    g = _grafo((1, 0), (1, 2))
    g.add_edge(0, 1)
    esticar_extremos(g, skel, relevo)

    assert g.nodes[1].x == 2   # col 3 não é esqueleto; fica em col 2


# ─── dividir_grau_zero ────────────────────────────────────────────────────────

def test_dividir_nao_age_em_componente_com_nos_conectados():
    """D13 fix: dividir_grau_zero não toca componente que já tem nó grau≥1."""
    # Esqueleto: row 1, cols 0-9 (10 px — acima de max_segmento_px=5).
    # Grafo tem nó 0 e nó 1 conectados + nó 2 de grau 0, todos no mesmo componente.
    skel = np.zeros((3, 12), dtype=bool)
    skel[1, 0:10] = True
    relevo = np.full((3, 12), 5.0, dtype=np.float32)

    g = StructureGraph()
    g.add_node(Node(id=0, x=3, y=1, gray=5.0))
    g.add_node(Node(id=1, x=7, y=1, gray=5.0))
    g.add_edge(0, 1)                             # ← nós conectados
    g.add_node(Node(id=2, x=1, y=1, gray=5.0))  # grau 0, mesma componente skel

    nos_antes = len(g.nodes)
    arestas_antes = len(g.edges)
    n = dividir_grau_zero(g, skel, relevo, cfg_sub=ConfigSubdivisao(max_segmento_px=5))

    assert n == 0, "componente com nó conectado não deve ser processada"
    assert len(g.nodes) == nos_antes, "nenhum nó deve ser adicionado ou removido"
    assert len(g.edges) == arestas_antes, "nenhuma aresta deve mudar"
    assert 1 in g.neighbors(0), "aresta original 0-1 deve permanecer"


def test_dividir_pixel_isolado_nao_divide():
    """D3: pixel isolado (componente de 1 pixel) → nó mantido, sem divisão."""
    skel = np.zeros((5, 5), dtype=bool)
    skel[2, 2] = True

    g = _grafo((2, 2))
    n = dividir_grau_zero(g, skel)

    assert n == 0
    assert len(g.nodes) == 1
    assert g.degree(0) == 0


def test_dividir_segmento_3px_em_dois_extremos():
    """Segmento de 3 px com nó de grau 0 no meio → 2 nós nos extremos, conectados."""
    skel = np.zeros((3, 5), dtype=bool)
    skel[1, 1:4] = True   # pixels (1,1),(1,2),(1,3)
    relevo = np.full((3, 5), 7.0, dtype=np.float32)

    g = _grafo((1, 2))    # nó no centro, grau 0
    comp_antes = contar_componentes(g)
    n = dividir_grau_zero(g, skel, relevo)

    assert n == 1
    assert len(g.nodes) == 2
    cols = {no.x for no in g.nodes.values()}
    assert cols == {1, 3}   # extremos do segmento
    # Extremos conectados entre si → 1 aresta, grau 1 em cada
    assert len(g.edges) == 1
    for nid in g.nodes:
        assert g.degree(nid) == 1
    assert contar_componentes(g) == comp_antes   # componentes não aumentam


def test_dividir_segmento_2px():
    """Segmento de 2 px → 2 nós nos extremos, conectados."""
    skel = np.zeros((3, 4), dtype=bool)
    skel[1, 1] = True
    skel[1, 2] = True
    relevo = np.full((3, 4), 5.0, dtype=np.float32)

    g = _grafo((1, 1))    # nó em um dos pixels
    comp_antes = contar_componentes(g)
    n = dividir_grau_zero(g, skel, relevo)

    assert n == 1
    assert len(g.nodes) == 2
    cols = {no.x for no in g.nodes.values()}
    assert cols == {1, 2}
    assert len(g.edges) == 1
    for nid in g.nodes:
        assert g.degree(nid) == 1
    assert contar_componentes(g) == comp_antes


def test_dividir_segmento_6px_1_componente():
    """Trecho isolado de 6 px → 2 nós conectados, componentes não aumentam."""
    skel = np.zeros((3, 8), dtype=bool)
    skel[1, 1:7] = True   # 6 pixels: (1,1)..(1,6)
    relevo = np.full((3, 8), 5.0, dtype=np.float32)

    g = _grafo((1, 3))    # nó interior, grau 0
    comp_antes = contar_componentes(g)
    n = dividir_grau_zero(g, skel, relevo)

    assert n == 1
    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    for nid in g.nodes:
        assert g.degree(nid) == 1
    assert contar_componentes(g) == comp_antes   # 1 → 1


def test_dividir_grau1_nao_e_dividido():
    """Nó com grau ≥ 1 não é dividido."""
    skel = _skel_linha(3, 6, 1, 0, 5)
    relevo = np.full((3, 6), 5.0, dtype=np.float32)

    g = _grafo((1, 0), (1, 3))
    g.add_edge(0, 1)
    n = dividir_grau_zero(g, skel, relevo)

    assert n == 0
    assert len(g.nodes) == 2


def test_dividir_atualiza_gray_d5():
    """D5: os dois novos nós têm gray = relevo na sua posição."""
    skel = np.zeros((3, 5), dtype=bool)
    skel[1, 1:4] = True
    relevo = np.zeros((3, 5), dtype=np.float32)
    relevo[1, 1] = 3.0
    relevo[1, 2] = 5.0
    relevo[1, 3] = 4.0

    g = _grafo((1, 2))    # nó no centro
    dividir_grau_zero(g, skel, relevo)

    grays = {no.gray for no in g.nodes.values()}
    assert 3.0 in grays    # extremo esquerdo
    assert 4.0 in grays    # extremo direito


def test_dividir_varios_nos_mesma_componente():
    """Dois nós de grau 0 na mesma componente → uma divisão, dois extremos conectados."""
    skel = np.zeros((3, 5), dtype=bool)
    skel[1, 1:4] = True
    relevo = np.full((3, 5), 5.0, dtype=np.float32)

    g = StructureGraph()
    g.add_node(Node(id=0, x=1, y=1, gray=5.0))
    g.add_node(Node(id=1, x=3, y=1, gray=5.0))

    comp_antes = contar_componentes(g)   # 2 (dois isolados)
    n = dividir_grau_zero(g, skel, relevo)

    assert len(g.nodes) == 2
    cols = {no.x for no in g.nodes.values()}
    assert cols == {1, 3}
    assert len(g.edges) == 1
    for nid in g.nodes:
        assert g.degree(nid) == 1
    # 2 nós isolados → 1 componente conectada (reduz)
    assert contar_componentes(g) == 1
    assert contar_componentes(g) < comp_antes


# ─── contar_componentes ───────────────────────────────────────────────────────

def test_contar_componentes_grafo_vazio():
    assert contar_componentes(StructureGraph()) == 0


def test_contar_componentes_isolados():
    g = _grafo((0, 0), (2, 2), (4, 4))
    assert contar_componentes(g) == 3


def test_contar_componentes_todos_conectados():
    g = _grafo((0, 0), (0, 1), (0, 2))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    assert contar_componentes(g) == 1


def test_contar_componentes_dois_grupos():
    g = _grafo((0, 0), (0, 1), (5, 5), (5, 6))
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    assert contar_componentes(g) == 2
