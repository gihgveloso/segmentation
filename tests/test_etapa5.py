"""Testes de Etapa 5: conexões candidatas (Seção III-C)."""
import numpy as np
import pytest
from features.graph import Node, StructureGraph
from features.connections import conectar_candidatas, ConfigConexao


def _grafo(*posicoes):
    g = StructureGraph()
    for i, (y, x) in enumerate(posicoes):
        g.add_node(Node(id=i, x=x, y=y, gray=5.0))
    return g


def _relevo_branco(shape, pixels_brancos):
    """Cria relevo com valor 1.0 nos pixels indicados e 0.0 no resto."""
    r = np.zeros(shape, dtype=np.float32)
    for (y, x) in pixels_brancos:
        r[y, x] = 1.0
    return r


# ─── Linha com lacuna ────────────────────────────────────────────────────────

def test_lacuna_3px_reconectada():
    """Dois terminais separados por 3 px de gap são conectados. III-C."""
    # Segmento esquerdo: nós 0(col 0) e 1(col 3); segmento direito: nós 2(col 6) e 3(col 9)
    # gap de 2 colunas (cols 4-5) sem esqueleto
    g = _grafo((5, 0), (5, 3), (5, 6), (5, 9))
    g.add_edge(0, 1)   # terminal = nó 1 (grau 1)
    g.add_edge(2, 3)   # terminal = nó 2 (grau 1)

    # Relevo com brancos ao longo da linha, incluindo o gap
    brancos = [(5, c) for c in range(10)]
    relevo = _relevo_branco((10, 10), brancos)

    cfg = ConfigConexao(dist_max=5.0, limiar_custo=10.0)
    n = conectar_candidatas(g, relevo, cfg)

    assert n >= 1, "ao menos uma candidata aceita"
    assert 2 in g.neighbors(1) or 1 in g.neighbors(2), \
        "terminais 1 e 2 devem estar conectados"


def test_lacuna_grande_nao_reconectada():
    """Terminais separados por mais que dist_max não são conectados."""
    g = _grafo((5, 0), (5, 2), (5, 20), (5, 22))
    g.add_edge(0, 1)
    g.add_edge(2, 3)

    relevo = np.zeros((10, 25), dtype=np.float32)
    cfg = ConfigConexao(dist_max=5.0, limiar_custo=10.0)
    n = conectar_candidatas(g, relevo, cfg)

    assert n == 0
    assert 2 not in g.neighbors(1)


def test_terminais_opostos_linha_reta_conectados():
    """Dois nós de grau 0 alinhados com relevo alto entre eles são conectados."""
    g = _grafo((3, 1), (3, 7))
    # sem arestas → ambos com grau 0

    brancos = [(3, c) for c in range(1, 8)]
    relevo = _relevo_branco((7, 9), brancos)

    cfg = ConfigConexao(dist_max=10.0, limiar_custo=20.0)
    n = conectar_candidatas(g, relevo, cfg)

    assert n >= 1
    assert 1 in g.neighbors(0)


# ─── Continuidade angular ─────────────────────────────────────────────────────

def test_continuidade_angular_favorece_alinhado():
    """Candidata alinhada com a aresta existente tem custo menor e é preferida."""
    # Segmento A: (5,0)-(5,3) horizontal.  Terminal = nó 1.
    # Candidato B alinhado: nó 2 em (5,6).
    # Candidato C desviado: nó 3 em (0,4).
    g = _grafo((5, 0), (5, 3), (5, 6), (0, 4))
    g.add_edge(0, 1)
    # nó 2 grau 0, nó 3 grau 0

    brancos = [(5, c) for c in range(7)] + [(r, 4) for r in range(6)]
    relevo = _relevo_branco((8, 8), brancos)

    cfg = ConfigConexao(dist_max=5.0, limiar_custo=20.0, peso_angular=2.0)
    conectar_candidatas(g, relevo, cfg)

    # Nó alinhado deve ter sido conectado ao terminal
    assert 2 in g.neighbors(1), "candidata alinhada deve ser conectada"


# ─── Grau máximo 4 ───────────────────────────────────────────────────────────

def test_grau_maximo_4_respeitado():
    """Nó com grau 4 não recebe nova aresta mesmo dentro do raio. III-C."""
    # Nó central com grau 4: conectado a 4 vizinhos
    g = StructureGraph()
    g.add_node(Node(id=0, x=5, y=5, gray=5.0))   # central
    for i in range(1, 5):
        g.add_node(Node(id=i, x=i, y=i, gray=5.0))
        g.add_edge(0, i)
    # Nó terminal próximo
    g.add_node(Node(id=5, x=6, y=5, gray=5.0))

    relevo = np.ones((10, 10), dtype=np.float32)
    cfg = ConfigConexao(dist_max=5.0, limiar_custo=100.0)
    conectar_candidatas(g, relevo, cfg)

    assert g.degree(0) == 4, "grau do nó central não deve ultrapassar 4"


# ─── Sem terminais ────────────────────────────────────────────────────────────

def test_sem_terminais_sem_candidatas():
    """Grafo sem terminais (todos grau 2+) não gera candidatas."""
    g = _grafo((0, 0), (0, 1), (0, 2))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)   # ciclo → todos grau 2

    relevo = np.ones((3, 3), dtype=np.float32)
    cfg = ConfigConexao(dist_max=5.0, limiar_custo=10.0)
    n = conectar_candidatas(g, relevo, cfg)

    assert n == 0


def test_grafo_vazio_sem_erro():
    """Grafo vazio não lança exceção."""
    g = StructureGraph()
    relevo = np.zeros((5, 5), dtype=np.float32)
    cfg = ConfigConexao()
    n = conectar_candidatas(g, relevo, cfg)
    assert n == 0


# ─── Custo via cinza médio ────────────────────────────────────────────────────

def test_custo_sem_branco_impede_conexao():
    """Candidata com relevo médio baixo (sem branco) tem custo acima do limiar."""
    g = _grafo((5, 0), (5, 2), (5, 6), (5, 8))
    g.add_edge(0, 1)
    g.add_edge(2, 3)

    # Sem brancos entre os terminais (cols 3-5): custo alto
    relevo = np.zeros((10, 10), dtype=np.float32)
    relevo[5, 0] = 1.0
    relevo[5, 2] = 1.0
    relevo[5, 6] = 1.0
    relevo[5, 8] = 1.0

    cfg = ConfigConexao(dist_max=10.0, limiar_custo=5.0, peso_cinza=3.0)
    n = conectar_candidatas(g, relevo, cfg)

    assert n == 0, "sem branco no gap o custo deve ultrapassar o limiar"


# ─── D12: laço intra-componente ──────────────────────────────────────────────

def test_laco_trivial_bloqueado():
    """Dois terminais do mesmo ramo com dist_grafo < dist_grafo_min não são ligados. D12."""
    # Cadeia: 0 - 1 - 2, terminais 0 e 2. dist_grafo=2, dist_eucl=2.
    g = _grafo((5, 0), (5, 1), (5, 2))
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    relevo = np.ones((10, 5), dtype=np.float32) * 5.0
    # dist_grafo_min=5 → 2 < 5 → trivial; fator=1.5 → 1.5*2=3 → 2 < 3 → não é laço grande
    cfg = ConfigConexao(dist_max=5.0, limiar_custo=100.0,
                        dist_grafo_min=5, fator_laco_grande=1.5)
    n = conectar_candidatas(g, relevo, cfg)

    assert n == 0, "laço trivial (dist_grafo=2 < min=5) não deve ser fechado"
    assert 2 not in g.neighbors(0)


def test_laco_grande_aceito():
    """Terminais de arco longo com dist_grafo >= fator*dist_eucl são ligados. D12."""
    # U-shape de 11 nós: (0,0)→(1,0)→...→(10,0)→(10,2)→...→(0,2)
    # Terminais em (0,0) e (0,2): dist_eucl=2, dist_grafo=12 hops.
    # 12 >= 1.5*2=3 → laço grande → aceito; mas dist_grafo_min=5 → 12 >= 5 → também não trivial.
    # Para testar especificamente o override: usar dist_grafo_min=20 (muito alto), fator=1.5.
    # Então: trivial=True (12<20), laco_grande=True (12>=3) → ACEITO pelo override.
    g = StructureGraph()
    # Braço esquerdo: col 0, rows 0..10
    for r in range(11):
        g.add_node(Node(id=r, x=0, y=r, gray=5.0))
        if r > 0:
            g.add_edge(r - 1, r)
    # Braço direito: col 2, rows 10..0
    for r in range(11):
        nid = 11 + r
        g.add_node(Node(id=nid, x=2, y=10 - r, gray=5.0))
        if r > 0:
            g.add_edge(nid - 1, nid)
    # Conectar base: (10, col0) e (10, col2) → fechar o U
    g.add_edge(10, 11)
    # Terminais: nó 0 em (0,0) e nó 21 em (0,2)
    # dist_eucl = 2; dist_grafo = 22 hops (0→1→...→10→11→...→21)

    relevo = np.ones((12, 4), dtype=np.float32) * 5.0

    # Usar dist_grafo_min muito alto para forçar uso do override por fator
    cfg = ConfigConexao(dist_max=5.0, limiar_custo=100.0,
                        dist_grafo_min=50, fator_laco_grande=1.5)
    n = conectar_candidatas(g, relevo, cfg)

    assert n >= 1, "laço grande (dist_grafo=22 >= fator*2=3) deve ser aceito"
    assert 21 in g.neighbors(0) or 0 in g.neighbors(21)
