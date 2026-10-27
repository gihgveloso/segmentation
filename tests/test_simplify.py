"""Testes para features/simplify.py (Etapa 6 — IV-A simplificação)."""
import math
import numpy as np
import pytest

from features.graph import Node, StructureGraph
from features.simplify import (
    ConfigSimplificacao,
    fundir_juncoes,
    fusao_juncoes_cluster,
    remover_diagonais_espurias,
    remover_espurios,
    remover_nos_curvatura_baixa,
    simplificar,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _grafo(*posicoes):
    """Cria StructureGraph com nós (id, x, y); gray padrão 0.5."""
    g = StructureGraph()
    for nid, x, y in posicoes:
        g.add_node(Node(id=nid, x=x, y=y, gray=0.5))
    return g


def _aresta_existe(g, a, b):
    return b in g.neighbors(a)


# ─── Test A: suavização — nó de grau 2 em linha quase reta é removido ─────────

def test_suavizacao_remove_no_reto():
    """IV-A: nó grau-2 colinear (ângulo≈180°) deve ser removido; edge A-B adicionada."""
    # Cadeia A(x=0,y=0) - N(x=0,y=5) - B(x=0,y=10): todos colineares
    g = _grafo((0, 0, 0), (1, 0, 5), (2, 0, 10))
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    cfg = ConfigSimplificacao(limiar_angulo_grau2=170.0)
    removidos = remover_nos_curvatura_baixa(g, cfg)

    assert removidos >= 1, "deve remover pelo menos o nó N"
    assert 1 not in g.nodes, "nó N deve ser removido"
    assert _aresta_existe(g, 0, 2), "aresta A-B deve existir"
    assert g.degree(0) == 1
    assert g.degree(2) == 1


# ─── Test B: espúrio curto some ────────────────────────────────────────────────

def test_espurio_curto_some():
    """IV-A: ramo espúrio de comprimento < limiar deve ser removido."""
    # Junção J(5,5) grau 3: braços A1(5,0), A2(0,5), folha L(5,7) comprimento=2px
    # J=id0(x=5,y=5), A1=id1(x=5,y=0), A2=id2(x=0,y=5), L=id3(x=5,y=7)
    g = _grafo((0, 5, 5), (1, 5, 0), (2, 0, 5), (3, 5, 7))
    g.add_edge(0, 1)  # J-A1 comprimento=5
    g.add_edge(0, 2)  # J-A2 comprimento=~7.07
    g.add_edge(0, 3)  # J-L  comprimento=2  (espúrio)

    cfg = ConfigSimplificacao(comprimento_espurio_px=5.0)
    removidos = remover_espurios(g, cfg)

    assert removidos >= 1, "pelo menos um ramo espúrio deve ser removido"
    assert 3 not in g.nodes, "folha L deve ter sido removida"
    # J permanece com grau 2 (braços A1 e A2 sobrevivem)
    assert g.degree(0) == 2
    assert 0 in g.nodes and 1 in g.nodes and 2 in g.nodes


# ─── Test C: T fica grau 3 após remover diagonais ─────────────────────────────

def test_t_grau3_pos_diagonais():
    """IV-A: arestas diagonais curtas em triângulo devem ser removidas."""
    # T-junction J(x=5,y=5) com braços N(x=4,y=5), W(x=5,y=4), E(x=5,y=6)
    # Arestas espúrias: N-W (dist=sqrt(2)) e N-E (dist=sqrt(2))
    g = _grafo((0, 5, 5), (1, 4, 5), (2, 5, 4), (3, 5, 6))
    g.add_edge(0, 1)  # J-N (distância=1)
    g.add_edge(0, 2)  # J-W (distância=1)
    g.add_edge(0, 3)  # J-E (distância=1)
    g.add_edge(1, 2)  # N-W espúria (dist=sqrt(2)≈1.41)
    g.add_edge(1, 3)  # N-E espúria (dist=sqrt(2)≈1.41)

    cfg = ConfigSimplificacao(limiar_diagonal_px=2.0)
    removidas = remover_diagonais_espurias(g, cfg)

    assert removidas >= 2, "deve remover pelo menos as 2 diagonais espúrias"
    assert not _aresta_existe(g, 1, 2), "aresta N-W deve ser removida"
    assert not _aresta_existe(g, 1, 3), "aresta N-E deve ser removida"
    # J ainda grau 3
    assert g.degree(0) == 3


# ─── Test D: duas junções coladas viram um nó de grau 4 ───────────────────────

def test_duas_juncoes_fundem():
    """IV-A: duas junções grau≥3 a distância < distancia_fusao_px devem ser fundidas."""
    # J1(x=5,y=5) grau 3: edges para A(x=5,y=0), B(x=0,y=5), J2
    # J2(x=5,y=6) grau 3: edges para C(x=5,y=11), D(x=10,y=5), J1
    g = _grafo(
        (0, 5, 5),   # J1
        (1, 5, 0),   # A
        (2, 0, 5),   # B
        (3, 5, 6),   # J2
        (4, 5, 11),  # C
        (5, 10, 5),  # D
    )
    g.add_edge(0, 1)  # J1-A
    g.add_edge(0, 2)  # J1-B
    g.add_edge(0, 3)  # J1-J2
    g.add_edge(3, 4)  # J2-C
    g.add_edge(3, 5)  # J2-D

    cfg = ConfigSimplificacao(distancia_fusao_px=3.0)
    fusoes = fundir_juncoes(g, skeleton=None, relevo=None, cfg=cfg)

    assert fusoes >= 1, "pelo menos uma fusão deve ocorrer"
    # Após fusão: um dos J é removido; o que sobrar tem grau 4 (A, B, C, D)
    sobreviventes = {nid for nid in (0, 3) if nid in g.nodes}
    assert len(sobreviventes) == 1, "um dos J deve ter sido removido"
    jid = sobreviventes.pop()
    assert g.degree(jid) == 4, f"junção fundida deve ter grau 4, tem {g.degree(jid)}"
    for arm in (1, 2, 4, 5):
        assert _aresta_existe(g, jid, arm), f"braço {arm} deve estar conectado à junção fundida"


# ─── Test E: suavização iterativa ─────────────────────────────────────────────

def test_suavizacao_cadeia_reta():
    """IV-A: suavização iterativa deve remover todos os nós intermediários colineares."""
    # Cadeia A(0,0)-N1(0,3)-N2(0,6)-B(0,9), todos colineares
    g = _grafo((0, 0, 0), (1, 0, 3), (2, 0, 6), (3, 0, 9))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)

    cfg = ConfigSimplificacao(limiar_angulo_grau2=170.0)
    removidos = remover_nos_curvatura_baixa(g, cfg)

    assert removidos == 2, "ambos N1 e N2 devem ser removidos"
    assert 1 not in g.nodes
    assert 2 not in g.nodes
    assert 0 in g.nodes and 3 in g.nodes
    assert _aresta_existe(g, 0, 3)


# ─── Test F: espúrio longo permanece ──────────────────────────────────────────

def test_espurio_longo_permanece():
    """IV-A: ramo espúrio com comprimento >= limiar não deve ser removido."""
    # J grau 3, braço longo de comprimento=6px, limiar=5 → braço fica
    g = _grafo((0, 5, 5), (1, 5, 0), (2, 0, 5), (3, 5, 11))
    g.add_edge(0, 1)  # J-A1 comprimento=5
    g.add_edge(0, 2)  # J-A2 comprimento=~7.07
    g.add_edge(0, 3)  # J-L  comprimento=6  (não-espúrio)

    cfg = ConfigSimplificacao(comprimento_espurio_px=5.0)
    removidos = remover_espurios(g, cfg)

    # Nenhum ramo deve ser removido: comprimentos são >=5
    assert 3 in g.nodes, "ramo longo deve permanecer"
    assert g.degree(0) == 3, "junção deve permanecer grau 3"


# ─── Test G: simplificar retorna dicionário com todas as chaves ───────────────

def test_simplificar_retorna_dict():
    """simplificar deve retornar dict com chaves 'diagonais', 'curvatura', 'espurios', 'fusoes'."""
    g = _grafo((0, 0, 0), (1, 0, 5), (2, 0, 10))
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    resultado = simplificar(g)

    assert isinstance(resultado, dict)
    for chave in ('diagonais', 'curvatura', 'espurios', 'fusoes'):
        assert chave in resultado, f"chave '{chave}' deve estar no resultado"
        assert isinstance(resultado[chave], int)


# ─── Test H: fusão de junções em cluster (D23) ────────────────────────────────

def test_fusao_cluster_dois_t_viram_x():
    """D23: dois T a 5px conectados por aresta direta fundem em X (grau 4)."""
    #   (5,10)      (20,10)
    #      |            |
    #   (10,10)-----(15,10)
    #      |            |
    #   (10,5)       (15,5)
    g = _grafo(
        (0, 10, 10),  # J1
        (1,  5, 10),  # braco J1-esq
        (2, 10,  5),  # braco J1-cima
        (3, 15, 10),  # J2
        (4, 20, 10),  # braco J2-dir
        (5, 15,  5),  # braco J2-cima
    )
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)
    g.add_edge(3, 4); g.add_edge(3, 5)

    cfg = ConfigSimplificacao(usar_fusao_juncoes=True, r_fusao_juncoes=6.0, max_arestas_fusao=8)
    fusoes = fusao_juncoes_cluster(g, skeleton=None, cfg=cfg)

    assert fusoes >= 1, "deve realizar pelo menos uma fusao"
    grau4 = [nid for nid in g.nodes if g.degree(nid) == 4]
    assert len(grau4) == 1, f"deve ter exatamente 1 no grau 4, tem {len(grau4)}"
    rep = grau4[0]
    # Todos os 4 bracos externos devem estar conectados ao representante
    vizinhos_rep = set(g.neighbors(rep))
    arm_grau1 = {nid for nid in g.nodes if g.degree(nid) == 1}
    assert arm_grau1.issubset(vizinhos_rep), \
        f"bracos {arm_grau1} devem conectar ao representante {rep}"


def test_fusao_cluster_t_distante_nao_funde():
    """D23: dois T a 20px (> r_fusao=6) nao sao fundidos."""
    g = _grafo(
        (0,  5, 10), (1,  0, 10), (2,  5,  5), (3,  5, 15),  # J1 e bracos
        (4, 25, 10), (5, 30, 10), (6, 25,  5), (7, 25, 15),  # J2 e bracos
    )
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)
    g.add_edge(4, 5); g.add_edge(4, 6); g.add_edge(4, 7)

    cfg = ConfigSimplificacao(usar_fusao_juncoes=True, r_fusao_juncoes=6.0)
    fusoes = fusao_juncoes_cluster(g, skeleton=None, cfg=cfg)

    assert fusoes == 0, "T distante nao deve ser fundido"
    assert g.degree(0) == 3
    assert g.degree(4) == 3


def test_fusao_cluster_t_isolado_continua_t():
    """D23: unico T no grafo continua T (nenhum outro no grau>=3 para fundir)."""
    g = _grafo(
        (0, 10, 10), (1,  5, 10), (2, 15, 10), (3, 10,  5),
    )
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)

    cfg = ConfigSimplificacao(usar_fusao_juncoes=True, r_fusao_juncoes=6.0)
    fusoes = fusao_juncoes_cluster(g, skeleton=None, cfg=cfg)

    assert fusoes == 0
    assert g.degree(0) == 3


# ─── Tests D27: fusão de par de T com barras colineares ───────────────────────

def test_t_par_oposto_vira_grau4():
    """D27: dois T com barras colineares e hastes em lados opostos viram X (grau=4).

    T1(10,10): barra A(0,10)–T2(20,10), haste C1(10,2) [acima].
    T2(20,10): barra T1(10,10)–D(30,10), haste C2(20,18) [abaixo].
    dist(T1,T2)=10px ≤ r=10; barras horizontais → colineares; hastes opostas.
    """
    from features.simplify import fusao_t_par, ConfigSimplificacao
    g = _grafo(
        (0, 10, 10),  # T1
        (1,  0, 10),  # A (braço esq de T1)
        (2, 20, 10),  # T2
        (3, 10,  2),  # C1 haste T1 — acima
        (4, 30, 10),  # D (braço dir de T2)
        (5, 20, 18),  # C2 haste T2 — abaixo
    )
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)
    g.add_edge(2, 4); g.add_edge(2, 5)

    cfg = ConfigSimplificacao(usar_fusao_t_par=True, r_fusao_t_par_px=12.0,
                              angulo_barra_t_par=160.0)
    fusoes = fusao_t_par(g, skeleton=None, cfg=cfg)

    assert fusoes >= 1, "deve realizar pelo menos uma fusão T-par"
    nos_grau4 = [nid for nid in g.nodes if g.degree(nid) == 4]
    assert len(nos_grau4) == 1, f"deve haver exatamente 1 nó de grau 4, tem {nos_grau4}"
    ext = set(g.neighbors(nos_grau4[0]))
    for arm in (1, 3, 4, 5):
        assert arm in ext, f"braço {arm} deve estar conectado ao X fundido"


def test_t_par_mesmo_lado_nao_funde_d27():
    """D27: T par com hastes no MESMO lado da barra não deve ser fundido."""
    from features.simplify import fusao_t_par, ConfigSimplificacao
    g = _grafo(
        (0, 10, 10),  # T1
        (1,  0, 10),  # A
        (2, 20, 10),  # T2
        (3, 10,  2),  # C1 haste T1 — acima
        (4, 30, 10),  # D
        (5, 20,  2),  # C2 haste T2 — TAMBÉM acima (mesmo lado)
    )
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)
    g.add_edge(2, 4); g.add_edge(2, 5)

    cfg = ConfigSimplificacao(usar_fusao_t_par=True, r_fusao_t_par_px=12.0,
                              angulo_barra_t_par=160.0)
    fusoes = fusao_t_par(g, skeleton=None, cfg=cfg)

    assert fusoes == 0, "hastes no mesmo lado: não deve fundir"
    assert g.degree(0) == 3 and g.degree(2) == 3


def test_t_isolado_d27_continua_t():
    """D27: T isolado (sem segundo T próximo) permanece T."""
    from features.simplify import fusao_t_par, ConfigSimplificacao
    g = _grafo(
        (0, 10, 10), (1, 0, 10), (2, 20, 10), (3, 10, 2),
    )
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)

    cfg = ConfigSimplificacao(usar_fusao_t_par=True, r_fusao_t_par_px=12.0)
    fusoes = fusao_t_par(g, skeleton=None, cfg=cfg)

    assert fusoes == 0
    assert g.degree(0) == 3
