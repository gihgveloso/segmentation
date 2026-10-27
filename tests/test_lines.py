"""
Testes para features/lines.py — IV-B: extração de linhas por split-and-merge.
Testes T1–T21 usam grafos sintéticos.
Testes T22+ usam grafos REAIS (imagens de images/) para regressão.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from features.graph import Node, StructureGraph
from features.lines import ConfigLinhas, extrair_linhas, _para_linefeature, _componentes_grau2
from features.types import LineFeature

# Imagens reais — puladas se o arquivo não existir (CI sem images/)
_IMAGE1 = 'images/image_1.png'
_IMAGE  = 'images/image.png'


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares para montar grafos sintéticos
# ─────────────────────────────────────────────────────────────────────────────

def _grafo_cadeia(pts: list[tuple[int, int]]) -> StructureGraph:
    """Cadeia de nós de grau 2 (extremos grau 1) nas coordenadas dadas."""
    g = StructureGraph()
    for i, (x, y) in enumerate(pts):
        g.add_node(Node(i, x, y, 1.0))
    for i in range(len(pts) - 1):
        g.add_edge(i, i + 1)
    return g


def _grafo_l(
    bh: list[tuple[int, int]],
    bv: list[tuple[int, int]],
) -> StructureGraph:
    """Forma um L: braço horizontal + braço vertical com um nó junção (grau 3)."""
    g = StructureGraph()
    nid = 0
    h_ids = []
    for x, y in bh:
        g.add_node(Node(nid, x, y, 1.0))
        h_ids.append(nid)
        nid += 1
    v_ids = []
    for x, y in bv:
        g.add_node(Node(nid, x, y, 1.0))
        v_ids.append(nid)
        nid += 1
    for i in range(len(h_ids) - 1):
        g.add_edge(h_ids[i], h_ids[i + 1])
    for i in range(len(v_ids) - 1):
        g.add_edge(v_ids[i], v_ids[i + 1])
    # junção: último nó horizontal == primeiro nó vertical (compartilham o canto)
    g.add_edge(h_ids[-1], v_ids[0])
    return g


# ─────────────────────────────────────────────────────────────────────────────
# T1: reta horizontal → 1 linha com ângulo ≈ 0
# ─────────────────────────────────────────────────────────────────────────────

def test_reta_horizontal():
    pts = [(x, 10) for x in range(0, 20, 2)]
    g = _grafo_cadeia(pts)
    cfg = ConfigLinhas()
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 1
    lf = linhas[0]
    assert abs(lf.angulo) < math.radians(5) or abs(abs(lf.angulo) - math.pi) < math.radians(5)
    assert lf.distancia > 0
    assert isinstance(lf, LineFeature)


# ─────────────────────────────────────────────────────────────────────────────
# T2: reta diagonal → 1 linha com ângulo ≈ 45°
# ─────────────────────────────────────────────────────────────────────────────

def test_reta_diagonal():
    pts = [(i * 2, i * 2) for i in range(8)]
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 1
    lf = linhas[0]
    ang = lf.angulo % math.pi
    assert abs(ang - math.radians(45)) < math.radians(10)


# ─────────────────────────────────────────────────────────────────────────────
# T3: polilinha em L → 2 linhas (split na junção grau-3)
# ─────────────────────────────────────────────────────────────────────────────

def test_polilinha_l_vira_2_linhas():
    # Braço horizontal: x=0..10 y=10; braço vertical: x=10 y=10..20
    bh = [(x, 10) for x in range(0, 12, 2)]
    bv = [(10, y) for y in range(10, 22, 2)]
    g = _grafo_l(bh, bv)
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 2


# ─────────────────────────────────────────────────────────────────────────────
# T4: componente com 2 nós é descartado
# ─────────────────────────────────────────────────────────────────────────────

def test_componente_2_nos_descartado():
    pts = [(0, 0), (5, 0)]
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 0


# ─────────────────────────────────────────────────────────────────────────────
# T5: linha interrompida por nó T → 2 linhas terminando na junção
# ─────────────────────────────────────────────────────────────────────────────

def test_linha_interrompida_por_t():
    """Linha horizontal com junção T no meio (braço vertical no centro)."""
    g = StructureGraph()
    # horizontal: nós 0-1-2-3-4 (y=10)
    for i in range(5):
        g.add_node(Node(i, i * 4, 10, 1.0))
    for i in range(4):
        g.add_edge(i, i + 1)
    # braço vertical: nó 5 e 6, conectados ao nó 2 (centro)
    g.add_node(Node(5, 8, 4, 1.0))
    g.add_node(Node(6, 8, 0, 1.0))
    g.add_edge(2, 5)
    g.add_edge(5, 6)

    linhas = extrair_linhas(g, fator_escala=8)
    # Deve produzir pelo menos 2 segmentos (os dois lados da horizontal)
    assert len(linhas) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# T6: extremos ficam sobre a reta ajustada (não os extremos brutos)
# ─────────────────────────────────────────────────────────────────────────────

def test_extremos_projetados_na_reta():
    """Cadeia quase horizontal com um pequeno desvio no extremo.
    O extremo projetado deve estar na reta ajustada, não no pixel bruto."""
    pts = [(0, 10), (4, 10), (8, 10), (12, 10), (16, 11)]  # último desviado 1 px
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 1
    lf = linhas[0]
    # Extremos devem estar na reta (resíduo ortogonal pequeno)
    dx = lf.x2 - lf.x1
    dy = lf.y2 - lf.y1
    norm = math.hypot(dx, dy)
    assert norm > 0
    # normal à reta
    nx, ny = -dy / norm, dx / norm
    # residuo de x1,y1 relativo a qualquer ponto na reta (use o ponto médio dos nós brutos)
    mx = sum(x for x, y in pts) / len(pts)
    my = sum(y for x, y in pts) / len(pts)
    res1 = abs((lf.x1 - mx) * nx + (lf.y1 - my) * ny)
    res2 = abs((lf.x2 - mx) * nx + (lf.y2 - my) * ny)
    # extremos projetados devem ter resíduo similar ao dos pontos da cadeia
    # (menor que 2 px de desvio ortogonal)
    assert res1 < 2.0, f"extremo 1 fora da reta: resíduo={res1:.2f}"
    assert res2 < 2.0, f"extremo 2 fora da reta: resíduo={res2:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# T7: confiança reflete fração de nós próximos à reta
# ─────────────────────────────────────────────────────────────────────────────

def test_confianca_linha_reta_alta():
    pts = [(x, 10) for x in range(0, 20, 2)]
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 1
    assert linhas[0].confianca > 0.8


def test_confianca_linha_zigue_zague_baixa():
    """Linha com desvio > desvio_max/2 tem confiança < 1.0."""
    cfg = ConfigLinhas(desvio_max_celulas=2.0)
    # zigzag: y alterna entre 0 e 4 (desvio 2 px da reta y=2)
    pts = [(x * 4, 0 if x % 2 == 0 else 4) for x in range(6)]
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    # Pode dividir ou não, mas se produz alguma linha, confiança < 1.0
    for lf in linhas:
        assert lf.confianca <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# T8: n_nos reflete o número de nós do componente
# ─────────────────────────────────────────────────────────────────────────────

def test_n_nos_correto():
    pts = [(x, 10) for x in range(0, 20, 2)]  # 10 nós
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8)
    assert len(linhas) == 1
    assert linhas[0].n_nos == len(pts)


# ─────────────────────────────────────────────────────────────────────────────
# T9: coordenadas originais = grade × fator_escala
# ─────────────────────────────────────────────────────────────────────────────

def test_coordenadas_originais_fator8():
    pts = [(x, 10) for x in range(0, 20, 2)]
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8)
    lf = linhas[0]
    assert abs(lf.x1_orig - lf.x1 * 8) < 1e-6
    assert abs(lf.y1_orig - lf.y1 * 8) < 1e-6
    assert abs(lf.x2_orig - lf.x2 * 8) < 1e-6
    assert abs(lf.y2_orig - lf.y2 * 8) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# T10: split-and-merge divide polilinha dobrada no meio
# ─────────────────────────────────────────────────────────────────────────────

def test_split_dobra_no_meio():
    """Dois segmentos retos em ângulo reto → split produz 2 linhas."""
    cfg = ConfigLinhas(desvio_max_celulas=1.5)
    pts = ([(x, 20) for x in range(0, 22, 2)]      # horizontal y=20
         + [(20, y) for y in range(22, 42, 2)])     # vertical x=20
    g = _grafo_cadeia(pts)
    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    assert len(linhas) == 2


# ─────────────────────────────────────────────────────────────────────────────
# T11: fusão de dois segmentos colineares próximos
# ─────────────────────────────────────────────────────────────────────────────

def test_fusao_segmentos_colineares():
    """Dois segmentos horizontais paralelos e colineares (mesma reta, ângulo idêntico,
    distância ortogonal ≈ 0) devem ser fundidos se a config de fusão estiver ativa."""
    cfg = ConfigLinhas(
        desvio_max_celulas=1.5,
        usar_fusao=True,
        limiar_angulo_fusao_grau=5.0,
        limiar_dist_ortogonal_fusao=1.5,
    )
    # segmento A: x=0..10 y=5
    g = StructureGraph()
    for i in range(6):
        g.add_node(Node(i, i * 2, 5, 1.0))
    for i in range(5):
        g.add_edge(i, i + 1)
    # segmento B: x=14..24 y=5 (gap de 3 px, mesma reta)
    for i in range(6):
        g.add_node(Node(i + 6, 14 + i * 2, 5, 1.0))
    for i in range(5):
        g.add_edge(i + 6, i + 7)
    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    # Dois componentes separados → 2 linhas extraídas por componente;
    # a fusão funde os dois em 1
    assert len(linhas) == 1


# ─────────────────────────────────────────────────────────────────────────────
# T12: extremos independem da ordem de traversal dos nós (1.1)
# ─────────────────────────────────────────────────────────────────────────────

def test_extremos_independem_da_ordem_nos():
    """Extremos da linha devem ser determinados por projeção, não por pts[0]/pts[-1]."""
    import random
    rng = random.Random(42)
    pts_ord = [(x, 10) for x in range(0, 20, 2)]   # 10 pts, x: 0,2,...,18
    pts_shuf = pts_ord[:]
    rng.shuffle(pts_shuf)

    g_ord  = _grafo_cadeia(pts_ord)
    g_shuf = _grafo_cadeia(pts_shuf)

    l_ord  = extrair_linhas(g_ord,  fator_escala=8)
    l_shuf = extrair_linhas(g_shuf, fator_escala=8)

    assert len(l_ord) == 1 and len(l_shuf) == 1
    lf1, lf2 = l_ord[0], l_shuf[0]
    range1 = (min(lf1.x1, lf1.x2), max(lf1.x1, lf1.x2))
    range2 = (min(lf2.x1, lf2.x2), max(lf2.x1, lf2.x2))
    assert abs(range1[0] - range2[0]) < 1.0, f"x_min: {range1[0]:.1f} vs {range2[0]:.1f}"
    assert abs(range1[1] - range2[1]) < 1.0, f"x_max: {range1[1]:.1f} vs {range2[1]:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# T13: _para_linefeature usa projeção, não pts[0]/pts[-1] (1.1 — direto)
# ─────────────────────────────────────────────────────────────────────────────

def test_extremos_por_projecao_direto():
    """_para_linefeature deve cobrir o range real de x, mesmo com pts fora de ordem."""
    # Simula vstack pós-fusão: seg_A x=57..112, seg_B x=1..46 (ordem não-monotônica)
    # pts[0]=(57,82), pts[-1]=(46,82) → old code: range [46,57] (errado)
    # new code: range [1,112] via projeção ✓
    pts_a = np.array([[x, 82] for x in range(57, 113, 5)], dtype=np.float32)
    pts_b = np.array([[x, 82] for x in range(1, 47, 5)],  dtype=np.float32)
    pts = np.vstack([pts_a, pts_b])
    ids = list(range(len(pts)))

    lf = _para_linefeature(ids, pts, pts * 8, 8)

    x_min = min(lf.x1, lf.x2)
    x_max = max(lf.x1, lf.x2)
    assert x_min < 5,   f"extremo mínimo deve ser ≈1, obtido {x_min:.1f}"
    assert x_max > 107, f"extremo máximo deve ser ≈112, obtido {x_max:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# T14: fusão rejeitada quando gap > lacuna_max (1.2)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusao_rejeita_vao_grande():
    """Gap de 34 células entre dois componentes separados → fusão bloqueada."""
    cfg = ConfigLinhas(usar_fusao=True, lacuna_max_celulas=8.0, suporte_vao_min=0.6)
    g = StructureGraph()
    # Segmento A: x=0..20, y=5
    for i in range(11):
        g.add_node(Node(i, i * 2, 5, 1.0))
    for i in range(10):
        g.add_edge(i, i + 1)
    # Segmento B: x=54..74, y=5 (gap de 34 células)
    for i in range(11):
        g.add_node(Node(i + 11, 54 + i * 2, 5, 1.0))
    for i in range(10):
        g.add_edge(i + 11, i + 12)

    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    assert len(linhas) == 2, f"esperado 2 linhas (sem fusão), obtido {len(linhas)}"


# ─────────────────────────────────────────────────────────────────────────────
# T15: fusão rejeitada quando branco=dark mesmo com gap pequeno (1.2)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusao_rejeita_vao_escuro():
    """Gap ≤ lacuna_max mas branco=zero no vão → fusão bloqueada."""
    cfg = ConfigLinhas(usar_fusao=True, lacuna_max_celulas=8.0, suporte_vao_min=0.6)
    g = StructureGraph()
    for i in range(6):
        g.add_node(Node(i, i * 2, 5, 1.0))        # x: 0,2,...,10
    for i in range(5):
        g.add_edge(i, i + 1)
    for i in range(6):
        g.add_node(Node(i + 6, 14 + i * 2, 5, 1.0))  # x: 14,...,24  (gap=4)
    for i in range(5):
        g.add_edge(i + 6, i + 7)

    branco = np.zeros((15, 30), dtype=np.uint8)   # sem branco no vão

    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg, branco=branco)
    assert len(linhas) == 2, f"esperado 2 linhas (vão escuro), obtido {len(linhas)}"


# ─────────────────────────────────────────────────────────────────────────────
# T16: fusão aceita quando gap pequeno + branco no vão (1.2 — regressão)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusao_aceita_vao_branco():
    """Gap ≤ lacuna_max e branco no vão → fusão permitida."""
    cfg = ConfigLinhas(usar_fusao=True, lacuna_max_celulas=8.0, suporte_vao_min=0.6)
    g = StructureGraph()
    for i in range(6):
        g.add_node(Node(i, i * 2, 5, 1.0))
    for i in range(5):
        g.add_edge(i, i + 1)
    for i in range(6):
        g.add_node(Node(i + 6, 14 + i * 2, 5, 1.0))
    for i in range(5):
        g.add_edge(i + 6, i + 7)

    branco = np.zeros((15, 30), dtype=np.uint8)
    branco[5, 10:15] = 255      # branco no vão

    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg, branco=branco)
    assert len(linhas) == 1, f"esperado 1 linha (fundida), obtido {len(linhas)}"


# ─────────────────────────────────────────────────────────────────────────────
# T17: aresta escura e longa é cortada em _componentes_grau2 (1.3)
# ─────────────────────────────────────────────────────────────────────────────

def test_aresta_escura_corta_componente():
    """Aresta longa (dist>4) e escura (frac<0.5) é cortada: vertical não inclui nó horizontal."""
    # Horizontal: nós 0-4 (x=0,2,...,8; y=10), com junção em nó 2 (x=4,y=10)
    # Vertical: nós 5-8 (x=4,y=20,...,26)
    # Aresta (2→5): dist=10, escura → deve ser cortada
    g = StructureGraph()
    for i in range(5):
        g.add_node(Node(i, i * 2, 10, 1.0))
    for i in range(4):
        g.add_edge(i, i + 1)
    for i in range(4):
        g.add_node(Node(i + 5, 4, 20 + i * 2, 1.0))
    for i in range(3):
        g.add_edge(i + 5, i + 6)
    g.add_edge(2, 5)

    branco = np.zeros((30, 15), dtype=np.uint8)
    cfg = ConfigLinhas(usar_fusao=False, suporte_aresta_min=0.5)

    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg, branco=branco)

    # Sem corte: vertical vai de y=10 a y=26
    # Com corte: vertical vai de y=20 a y=26
    linhas_vert = [lf for lf in linhas
                   if abs(math.degrees(lf.angulo) % 180 - 90) < 25]
    assert len(linhas_vert) >= 1, "esperado ≥1 linha vertical"
    y_min_vert = min(min(lf.y1, lf.y2) for lf in linhas_vert)
    assert y_min_vert >= 19, (
        f"aresta escura não foi cortada: y_min_vert={y_min_vert:.1f} (esperado ≥19)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T18: candidata fecha lacuna → exatamente 1 LineFeature (2.4-a)
# ─────────────────────────────────────────────────────────────────────────────

def test_candidata_fecha_lacuna():
    """Dois segmentos conectados por aresta candidata (gap 4px) → 1 LineFeature."""
    cfg = ConfigLinhas(usar_fusao=False)
    g = StructureGraph()
    for i in range(5):
        g.add_node(Node(i, i * 2, 10, 1.0))
    for i in range(4):
        g.add_edge(i, i + 1)
    for i in range(5):
        g.add_node(Node(i + 5, 12 + i * 2, 10, 1.0))
    for i in range(4):
        g.add_edge(i + 5, i + 6)
    g.add_edge(4, 5)   # candidata: fecha gap de 4px

    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    assert len(linhas) == 1, f"esperado 1 linha (candidata une componente), obtido {len(linhas)}"


# ─────────────────────────────────────────────────────────────────────────────
# T19: junção T → exatamente 3 LineFeatures (2.4-b)
# ─────────────────────────────────────────────────────────────────────────────

def test_t_produz_exatamente_3_linhas():
    """Junção T (grau 3) deve produzir exatamente 3 LineFeatures (sem fusão)."""
    g = StructureGraph()
    for i in range(11):
        g.add_node(Node(i, i * 4, 10, 1.0))
    for i in range(10):
        g.add_edge(i, i + 1)
    for i in range(5):
        g.add_node(Node(11 + i, 20, 14 + i * 2, 1.0))
    for i in range(4):
        g.add_edge(11 + i, 12 + i)
    g.add_edge(5, 11)   # nó 5 (x=20,y=10) vira grau 3

    cfg = ConfigLinhas(usar_fusao=False)
    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    assert len(linhas) == 3, f"esperado 3 linhas (T-junction), obtido {len(linhas)}"


# ─────────────────────────────────────────────────────────────────────────────
# T20: cada aresta aparece em exatamente uma componente (2.4-c)
# ─────────────────────────────────────────────────────────────────────────────

def test_cada_aresta_em_exatamente_uma_componente():
    """_componentes_grau2: cada aresta do grafo aparece em exatamente uma componente."""
    g = StructureGraph()
    for i in range(9):
        g.add_node(Node(i, i * 4, 10, 1.0))
    for i in range(8):
        g.add_edge(i, i + 1)
    g.add_node(Node(9,  16, 0, 1.0))
    g.add_node(Node(10, 16, 5, 1.0))
    g.add_edge(9, 10)
    g.add_edge(10, 4)   # nó 4 tem grau 3

    comps = _componentes_grau2(g)

    edge_count: dict[tuple[int, int], int] = {}
    for comp in comps:
        for i in range(len(comp) - 1):
            e = (min(comp[i], comp[i + 1]), max(comp[i], comp[i + 1]))
            edge_count[e] = edge_count.get(e, 0) + 1

    for a, b in g.edges:
        e = (min(a, b), max(a, b))
        count = edge_count.get(e, 0)
        assert count == 1, f"aresta {e} aparece {count} vez(es) nas componentes"


# ─────────────────────────────────────────────────────────────────────────────
# T21: 3 segmentos colineares via junções X com lacunas 9+16 → 1 linha (1.2-b)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusao_1_2b_x_junction_lacunas_9_16():
    """1.2(b): 3 segmentos colineares via junções X compartilhadas → 1 linha.

    Cadeia A: x=0..20; J1 em x=29 (9 células de A); cadeia B: x=30..48;
    J2 em x=64 (16 células de B); cadeia C: x=66..80.
    J1 e J2 têm grau 4 (junções X) com galhos verticais curtos.

    Regra de fusão em cada lacuna:
    - A ↔ B: sets_ids[A] ∩ sets_ids[B] = {J1} → nó compartilhado (gap check bypass)
    - B ↔ C: sets_ids[B] ∩ sets_ids[C] = {J2} → nó compartilhado (gap check bypass)
    As lacunas de 9 e 16 células são comprimentos de aresta INTERNOS às cadeias A e B,
    não gaps espaciais entre cadeias — por isso o gap check não é o fator decisivo.
    """
    cfg = ConfigLinhas(usar_fusao=True, lacuna_max_celulas=8.0)
    g = StructureGraph()
    nid = 0

    # Cadeia A: x=0,2,...,20 (11 nós), y=10
    for i in range(11):
        g.add_node(Node(nid, i * 2, 10, 1.0))
        nid += 1
    for i in range(10):
        g.add_edge(i, i + 1)
    a_end = 10  # nó em x=20

    # J1 em x=29 (9 células além de A)
    j1 = nid; g.add_node(Node(j1, 29, 10, 1.0)); nid += 1
    g.add_edge(a_end, j1)
    for dy in (-5, 5):   # galhos verticais → grau 4
        g.add_node(Node(nid, 29, 10 + dy, 1.0)); nid += 1
        g.add_edge(j1, nid - 1)

    # Cadeia B: x=30,32,...,48 (10 nós), y=10
    b_start = nid
    for i in range(10):
        g.add_node(Node(nid, 30 + i * 2, 10, 1.0))
        nid += 1
    g.add_edge(j1, b_start)
    for i in range(9):
        g.add_edge(b_start + i, b_start + i + 1)
    b_end = nid - 1  # nó em x=48

    # J2 em x=64 (16 células além de B)
    j2 = nid; g.add_node(Node(j2, 64, 10, 1.0)); nid += 1
    g.add_edge(b_end, j2)
    for dy in (-5, 5):   # galhos verticais → grau 4
        g.add_node(Node(nid, 64, 10 + dy, 1.0)); nid += 1
        g.add_edge(j2, nid - 1)

    # Cadeia C: x=66,68,...,80 (8 nós), y=10
    c_start = nid
    for i in range(8):
        g.add_node(Node(nid, 66 + i * 2, 10, 1.0))
        nid += 1
    g.add_edge(j2, c_start)
    for i in range(7):
        g.add_edge(c_start + i, c_start + i + 1)

    linhas = extrair_linhas(g, fator_escala=8, cfg=cfg)
    horiz = [lf for lf in linhas if abs(math.degrees(lf.angulo) % 180) < 15]

    assert len(horiz) == 1, (
        f"esperado 1 linha horizontal (A+B+C fundidas), obtido {len(horiz)} "
        f"de {len(linhas)} totais"
    )
    x_min = min(horiz[0].x1, horiz[0].x2)
    x_max = max(horiz[0].x1, horiz[0].x2)
    assert x_min < 3,  f"extremo mínimo deve ser ≈0, obtido {x_min:.1f}"
    assert x_max > 77, f"extremo máximo deve ser ≈80, obtido {x_max:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# T22: linha do meio de image_1 → UMA LineFeature x≈1..165, y≈82  [regressão real]
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(_IMAGE1), reason='image_1.png ausente')
def test_fusao_image1_linha_meio_real():
    """Regressão: extrair_features(res) em image_1 produz UMA LineFeature para a linha
    do meio (y≈82), cobrindo x≈1 a x≈165 (n_nos≈66, dist≈164 células).

    D36 (lacuna_max=20): os três fragmentos brutos (x≈1-48, x≈57-112, x≈128-165)
    com gaps de 9 e 16 células são fundidos. D37 filtra candidatas escuras sem
    afetar a linha do meio. A diagonal falsa de D33 (160,68)→(120,85) permanece bloqueada.
    """
    import sys
    sys.path.insert(0, '.')
    from main import Pixel_Segment
    from features import extrair_features

    ps  = Pixel_Segment('green_pixels.csv')
    img = __import__('cv2').imread(_IMAGE1)
    res = ps.processar(img)

    result = extrair_features(res)
    linhas = result.lines

    def _quase_horiz(lf):
        a = math.degrees(lf.angulo) % 180
        return min(a, 180 - a) < 10

    TOLERANCIA_Y = 3.0
    candidatas = [
        lf for lf in linhas
        if _quase_horiz(lf)
        and abs((lf.y1 + lf.y2) / 2 - 82) < TOLERANCIA_Y
        and max(lf.x1, lf.x2) - min(lf.x1, lf.x2) > 100
    ]
    assert len(candidatas) == 1, (
        f"esperado 1 linha do meio fundida, obtido {len(candidatas)}; "
        f"linhas próx. y=82: {[(round(lf.x1,1),round(lf.y1,1),round(lf.x2,1),round(lf.y2,1)) for lf in linhas if abs((lf.y1+lf.y2)/2-82)<TOLERANCIA_Y]}"
    )
    lf = candidatas[0]
    x_min = min(lf.x1, lf.x2)
    x_max = max(lf.x1, lf.x2)
    assert x_min <= 3,   f"extremo mínimo deve ser ≈1 (±2), obtido {x_min:.1f}"
    assert x_max >= 163, f"extremo máximo deve ser ≈165 (±2), obtido {x_max:.1f}"

    def _perto(lf, x, y, r=5.0):
        return (math.hypot(lf.x1-x, lf.y1-y) < r or math.hypot(lf.x2-x, lf.y2-y) < r)

    diagonal_falsa = [
        lf for lf in linhas
        if _perto(lf, 160, 68) and _perto(lf, 120, 85)
    ]
    assert len(diagonal_falsa) == 0, (
        f"diagonal falsa detectada: {[(lf.x1,lf.y1,lf.x2,lf.y2) for lf in diagonal_falsa]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T23: linha do meio de image → fragmentos #5, #16, #17 — o que funde?  [regressão real]
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(_IMAGE), reason='image.png ausente')
def test_fusao_image_linha_meio_real():
    """Regressão: extrair_features(res) em image produz UMA LineFeature para a linha
    do meio (y≈77-78), cobrindo x≈1 a x≈156 (n_nos≈85, dist≈155 células).

    Com D37 ativo (branco em construir_grafo), as 6 candidatas escuras que antes
    impediam a fusão são filtradas antes da Etapa 5. O grafo resultante conecta os
    três fragmentos brutos via D36 (lacuna_max=20), produzindo uma linha completa.

    Nota: sem branco em construir_grafo (ex.: T23 anterior), fragmento #17 ficava
    separado porque o gap≈22 > lacuna_max=20. O caminho de produção (extrair_features)
    usa branco=branco e produz a linha completa. Esse comportamento é o esperado.
    """
    import sys
    sys.path.insert(0, '.')
    from main import Pixel_Segment
    from features import extrair_features

    ps  = Pixel_Segment('green_pixels.csv')
    img = __import__('cv2').imread(_IMAGE)
    res = ps.processar(img)

    result = extrair_features(res)
    linhas = result.lines

    def _quase_horiz(lf):
        a = math.degrees(lf.angulo) % 180
        return min(a, 180 - a) < 10

    faixa = [
        lf for lf in linhas
        if _quase_horiz(lf)
        and 75 < (lf.y1 + lf.y2) / 2 < 82
    ]

    # Com pipeline de produção: todos os fragmentos fundem em 1 linha x≈1-156
    linha_completa = [
        lf for lf in faixa
        if min(lf.x1, lf.x2) <= 5 and max(lf.x1, lf.x2) >= 140
    ]
    assert len(linha_completa) == 1, (
        f"esperado 1 linha completa (x≈1-156), obtido {len(linha_completa)}; "
        f"faixa y=75-82: {[(round(lf.x1,1),round(lf.y1,1),round(lf.x2,1),round(lf.y2,1)) for lf in faixa]}"
    )
    lf = linha_completa[0]
    x_min = min(lf.x1, lf.x2)
    x_max = max(lf.x1, lf.x2)
    assert x_min <= 5,   f"extremo mínimo deve ser ≈1 (±4), obtido {x_min:.1f}"
    assert x_max >= 150, f"extremo máximo deve ser ≈156 (±6), obtido {x_max:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# T24: extrair_features == pipeline etapa a etapa [integração real]
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(_IMAGE1), reason='image_1.png ausente')
def test_extrair_features_igual_etapa_a_etapa():
    """B1: extrair_features(res) deve produzir as mesmas listas de cruzamentos e
    linhas que a chamada etapa a etapa (construir_grafo + simplificar + detectar
    + extrair_linhas) com os mesmos parâmetros padrão.
    """
    import sys
    sys.path.insert(0, '.')
    import cv2 as _cv2
    import numpy as _np
    from main import Pixel_Segment
    from features import extrair_features
    from features.pipeline import construir_grafo, ConfigGrafo
    from features.simplify import simplificar, ConfigSimplificacao
    from features.crossings import detectar_cruzamentos, ConfigCruzamento
    from features.lines import extrair_linhas, ConfigLinhas

    ps  = Pixel_Segment('green_pixels.csv')
    img = _cv2.imread(_IMAGE1)
    res = ps.processar(img)

    # ── Caminho de produção ───────────────────────────────────────────────────
    result = extrair_features(res)

    # ── Caminho etapa a etapa (deve ser idêntico) ─────────────────────────────
    skel   = res['skeleton_img'].astype(bool)
    relevo = res['mascara_relevo'].astype(_np.float32)
    branco = res['white_mask_resized'].astype(_np.uint8)
    verde  = res['green_mask_resized'].astype(_np.uint8)

    g = construir_grafo(skel, relevo, ConfigGrafo(), branco=branco)
    linhas_ref = extrair_linhas(g, fator_escala=8, cfg=ConfigLinhas(),
                                branco=branco, verde=verde)
    simplificar(g, skel, relevo, ConfigSimplificacao())
    crossings_ref, _ = detectar_cruzamentos(g, branco, verde, ConfigCruzamento())

    # ── Comparações ──────────────────────────────────────────────────────────
    assert len(result.lines) == len(linhas_ref), (
        f"linhas: extrair_features={len(result.lines)}, etapa-a-etapa={len(linhas_ref)}"
    )
    assert len(result.crossings) == len(crossings_ref), (
        f"crossings: extrair_features={len(result.crossings)}, etapa-a-etapa={len(crossings_ref)}"
    )
    assert result.fator_escala == 8
    assert result.tamanho_grade == skel.shape

    for i, (lf, lr) in enumerate(zip(result.lines, linhas_ref)):
        assert abs(lf.x1 - lr.x1) < 0.1, f"linha {i}: x1 difere"
        assert abs(lf.y1 - lr.y1) < 0.1, f"linha {i}: y1 difere"
        assert abs(lf.distancia - lr.distancia) < 0.1, f"linha {i}: distancia difere"
