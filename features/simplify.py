"""IV-A: simplificação do StructureGraph — suavização, espúrios, diagonais, fusão."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from features.graph import Node, StructureGraph


@dataclass
class ConfigSimplificacao:
    """Parâmetros de simplificação. D16, D17, D23, D27."""
    limiar_angulo_grau2: float = 170.0   # °; acima disto nó grau-2 é "quase reto" → remove
    comprimento_espurio_px: float = 5.0  # px; ramo <= este terminando em junção é espúrio
    limiar_diagonal_px: float = 2.5      # px; aresta <= este em triângulo → diagonal espúria
    distancia_fusao_px: float = 3.0      # px; junções conectadas a <= esta dist → fundir (D17)
    # D23: fusão de clusters de junções
    usar_fusao_juncoes: bool = True       # habilita fusao_juncoes_cluster
    r_fusao_juncoes: float = 6.0         # px; raio para cluster de grau>=3
    max_arestas_fusao: int = 8            # máx. arestas no caminho entre dois membros do cluster
    # D27: fusão de par de T em X
    usar_fusao_t_par: bool = True         # habilita fusao_t_par
    r_fusao_t_par_px: float = 10.0        # px; distância máxima entre os dois T
    angulo_barra_t_par: float = 160.0     # °; ângulo mínimo para considerar a barra colinear


# ─── IV-A: remoção de nós de baixa curvatura ─────────────────────────────────

def remover_nos_curvatura_baixa(g: StructureGraph, cfg: Optional[ConfigSimplificacao] = None) -> int:
    """IV-A: remove nós de grau 2 quase retos iterativamente.

    Para cada nó N de grau 2 com vizinhos A e B:
      - calcula o ângulo entre os vetores (A-N) e (B-N);
      - se ângulo > limiar_angulo_grau2 (em °): remove N, adiciona aresta A-B (se A≠B).
    Repete até nenhuma mudança. Retorna total de nós removidos.
    """
    if cfg is None:
        cfg = ConfigSimplificacao()
    limiar_rad = math.radians(cfg.limiar_angulo_grau2)
    total = 0
    while True:
        mudou = False
        for nid in list(g.nodes.keys()):
            if nid not in g.nodes:
                continue
            if g.degree(nid) != 2:
                continue
            vizinhos = g.neighbors(nid)
            a, b = vizinhos[0], vizinhos[1]
            if a not in g.nodes or b not in g.nodes:
                continue
            n = g.nodes[nid]
            na = g.nodes[a]
            nb = g.nodes[b]
            # Vetores de N para A e de N para B
            vay = na.y - n.y
            vax = na.x - n.x
            vby = nb.y - n.y
            vbx = nb.x - n.x
            norm_a = math.hypot(vay, vax)
            norm_b = math.hypot(vby, vbx)
            if norm_a == 0 or norm_b == 0:
                continue
            cos_theta = (vay * vby + vax * vbx) / (norm_a * norm_b)
            # Clamp para evitar domain error de acos
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angulo = math.acos(cos_theta)
            if angulo > limiar_rad and a != b:
                # Só remove se a aresta A-B ainda não existe
                g.remove_node(nid)
                if a in g.nodes and b in g.nodes:
                    g.add_edge(a, b)
                total += 1
                mudou = True
        if not mudou:
            break
    return total


# ─── IV-A: remoção de ramos espúrios ─────────────────────────────────────────

def remover_espurios(g: StructureGraph, cfg: Optional[ConfigSimplificacao] = None) -> int:
    """IV-A: remove ramos curtos que terminam em folha (grau 1) em direção a uma junção.

    Para cada folha (grau 1): percorre a cadeia de nós de grau 2 até encontrar
    uma junção (grau ≥ 3) ou outro extremo. Calcula o comprimento euclidiano total
    do ramo (soma de distâncias entre pares consecutivos). Se < comprimento_espurio_px,
    remove todos os nós do ramo exceto a junção.
    Repete até nenhuma mudança. Retorna total de ramos removidos.
    """
    if cfg is None:
        cfg = ConfigSimplificacao()
    total = 0
    while True:
        mudou = False
        for nid in list(g.nodes.keys()):
            if nid not in g.nodes:
                continue
            if g.degree(nid) != 1:
                continue
            # Percorrer o ramo a partir da folha
            caminho = [nid]
            prev = None
            cur = nid
            comprimento = 0.0
            while True:
                vizinhos = [v for v in g.neighbors(cur) if v != prev]
                if not vizinhos:
                    break  # extremo sem mais saída
                prox = vizinhos[0]
                if prox not in g.nodes:
                    break
                n_cur = g.nodes[cur]
                n_prox = g.nodes[prox]
                comprimento += math.hypot(n_prox.y - n_cur.y, n_prox.x - n_cur.x)
                if g.degree(prox) != 2:
                    # Chegamos ao fim do ramo: prox é junção (≥3) ou outro extremo (1 ou 0)
                    # Só removemos se termina em junção
                    if g.degree(prox) >= 3 and comprimento < cfg.comprimento_espurio_px:
                        # Remover todos exceto a junção
                        for n in caminho:
                            if n in g.nodes:
                                g.remove_node(n)
                        total += 1
                        mudou = True
                    break
                prev = cur
                cur = prox
                caminho.append(cur)
        if not mudou:
            break
    return total


# ─── IV-A: remoção de diagonais espúrias ─────────────────────────────────────

def remover_diagonais_espurias(g: StructureGraph, cfg: Optional[ConfigSimplificacao] = None) -> int:
    """IV-A: remove arestas diagonais espúrias que fecham triângulos.

    Para cada triângulo (a, b, c) no grafo: remove a aresta mais longa que for
    <= limiar_diagonal_px. Isso garante que a aresta diagonal de um T/X (criada
    por 8-conectividade entre braços vizinhos) é removida, não a aresta mais curta
    entre a junção e o braço. D8: as diagonais surgem em junções T/X por sobreposição
    de regiões 3×3; são redundantes pois o caminho pela junção as substitui.
    Retorna total de arestas removidas. D16.
    """
    if cfg is None:
        cfg = ConfigSimplificacao()

    # Encontrar todas as arestas candidatas a remoção:
    # Uma aresta (a, b) é candidata se existe c vizinho de ambos a e b,
    # e dist(a, b) é a mais longa das três arestas do triângulo, E dist(a, b) <= limiar.
    para_remover: set[tuple[int, int]] = set()

    for a, b in list(g.edges):
        if a not in g.nodes or b not in g.nodes:
            continue
        na = g.nodes[a]
        nb = g.nodes[b]
        dist_ab = math.hypot(na.y - nb.y, na.x - nb.x)
        if dist_ab > cfg.limiar_diagonal_px:
            continue
        # Verificar se há vizinho comum c que forme triângulo
        viz_a = set(g.neighbors(a))
        viz_b = set(g.neighbors(b))
        comuns = (viz_a & viz_b) - {a, b}
        for c in comuns:
            if c not in g.nodes:
                continue
            nc = g.nodes[c]
            dist_ac = math.hypot(na.y - nc.y, na.x - nc.x)
            dist_bc = math.hypot(nb.y - nc.y, nb.x - nc.x)
            # Remove (a, b) se for a aresta mais longa do triângulo
            if dist_ab >= dist_ac and dist_ab >= dist_bc:
                key = (min(a, b), max(a, b))
                para_remover.add(key)
                break  # basta um triângulo que a justifique

    for a, b in para_remover:
        g.remove_edge(a, b)
    return len(para_remover)


# ─── IV-A: fusão de junções próximas ─────────────────────────────────────────

def fundir_juncoes(
    g: StructureGraph,
    skeleton: Optional[np.ndarray] = None,
    relevo: Optional[np.ndarray] = None,
    cfg: Optional[ConfigSimplificacao] = None,
) -> int:
    """IV-A: funde pares de junções (grau ≥ 3) conectadas e próximas.

    Para cada aresta (a, b) onde ambos têm grau ≥ 3 e dist(a, b) < distancia_fusao_px:
      - mantém o nó com maior gray (ou `a` em caso de empate);
      - recabia todos os vizinhos do nó descartado para o nó mantido (sem self-loops);
      - remove o nó descartado.
    Reinicia após cada fusão. Retorna total de fusões.
    D17: skeleton e relevo podem ser None (não há restrição de colocação neste passo).
    """
    if cfg is None:
        cfg = ConfigSimplificacao()
    total = 0
    while True:
        fundiu = False
        for a, b in list(g.edges):
            if a not in g.nodes or b not in g.nodes:
                continue
            if g.degree(a) < 3 or g.degree(b) < 3:
                continue
            na = g.nodes[a]
            nb = g.nodes[b]
            dist = math.hypot(na.y - nb.y, na.x - nb.x)
            if dist >= cfg.distancia_fusao_px:
                continue
            # Decidir qual manter
            if nb.gray > na.gray:
                mantido, descartado = b, a
            else:
                mantido, descartado = a, b
            # Rewire: conectar vizinhos do descartado ao mantido
            for viz in list(g.neighbors(descartado)):
                if viz != mantido and viz in g.nodes:
                    g.add_edge(mantido, viz)
            g.remove_node(descartado)
            total += 1
            fundiu = True
            break  # reiniciar o loop
        if not fundiu:
            break
    return total


# ─── IV-A: fusão de clusters de junções (D23) ────────────────────────────────

def _caminho_existe(g: StructureGraph, origem: int, destino: int, max_arestas: int) -> bool:
    """IV-A: D23. Verifica se há caminho de `origem` a `destino` com ≤ max_arestas arestas."""
    from collections import deque
    if origem == destino:
        return True
    visitados: set[int] = {origem}
    fila: deque[tuple[int, int]] = deque([(origem, 0)])
    while fila:
        cur, saltos = fila.popleft()
        if saltos >= max_arestas:
            continue
        for viz in g.neighbors(cur):
            if viz == destino:
                return True
            if viz not in visitados:
                visitados.add(viz)
                fila.append((viz, saltos + 1))
    return False


def fusao_juncoes_cluster(
    g: StructureGraph,
    skeleton: Optional[np.ndarray] = None,
    cfg: Optional[ConfigSimplificacao] = None,
) -> int:
    """IV-A: funde pares de junções (grau ≥ 3) iterativamente. D23.

    A cada iteração, escolhe o par (ni, nj) com menor distância que satisfaça:
      (a) dist(ni, nj) ≤ r_fusao_juncoes
      (b) existe caminho com ≤ max_arestas_fusao arestas
      (c) grau resultante ≤ 4 (impede sobre-fusão em imagens reais)
    Representante = pixel de esqueleto mais próximo do centróide do par,
    ou nó com maior gray em empate. Reinicia após cada fusão.
    Retorna número de fusões realizadas.
    """
    if cfg is None:
        cfg = ConfigSimplificacao()

    total = 0
    while True:
        juncoes = [nid for nid in g.nodes if g.degree(nid) >= 3]
        if len(juncoes) < 2:
            break

        melhor: Optional[tuple[int, int, float]] = None
        melhor_dist = float('inf')

        for i, ni in enumerate(juncoes):
            for nj in juncoes[i + 1:]:
                n_i, n_j = g.nodes[ni], g.nodes[nj]
                dist = math.hypot(n_i.y - n_j.y, n_i.x - n_j.x)
                if dist > cfg.r_fusao_juncoes or dist >= melhor_dist:
                    continue
                if not _caminho_existe(g, ni, nj, cfg.max_arestas_fusao):
                    continue
                # Grau resultante = número de vizinhos externos únicos
                ext = (set(g.neighbors(ni)) | set(g.neighbors(nj))) - {ni, nj}
                if len(ext) > 4:
                    continue
                melhor = (ni, nj, dist)
                melhor_dist = dist

        if melhor is None:
            break

        ni, nj, _ = melhor
        n_i, n_j = g.nodes[ni], g.nodes[nj]

        # Representante: pixel de esqueleto mais próximo do centróide, ou maior gray
        cx = (n_i.x + n_j.x) / 2.0
        cy = (n_i.y + n_j.y) / 2.0
        if skeleton is not None and skeleton.ndim == 2:
            skel_ys, skel_xs = np.where(skeleton)
            if len(skel_ys) > 0:
                dists = np.hypot(skel_xs - cx, skel_ys - cy)
                idx = int(dists.argmin())
                px_x, px_y = int(skel_xs[idx]), int(skel_ys[idx])
                mantido = ni if math.hypot(n_i.x - px_x, n_i.y - px_y) <= \
                    math.hypot(n_j.x - px_x, n_j.y - px_y) else nj
            else:
                mantido = ni if n_i.gray >= n_j.gray else nj
        else:
            mantido = ni if n_i.gray >= n_j.gray else nj
        descartado = nj if mantido == ni else ni

        for viz in list(g.neighbors(descartado)):
            if viz != mantido and viz in g.nodes:
                g.add_edge(mantido, viz)
        g.remove_node(descartado)
        total += 1

    return total


# ─── D27: fusão de par de T com barras colineares e hastes opostas ───────────

def fusao_t_par(
    g: StructureGraph,
    skeleton: Optional[np.ndarray] = None,
    cfg: Optional[ConfigSimplificacao] = None,
) -> int:
    """D27: funde pares de grau=3 adjacentes com barras colineares e hastes opostas em grau=4.

    Condições: dois nós T diretamente conectados, distância ≤ r_fusao_t_par_px,
    cada um tem barra (par de braços com ângulo ≥ angulo_barra_t_par) que inclui
    o outro T, e as hastes ficam em lados opostos da direção T1→T2.
    Retorna o número de fusões realizadas.
    """
    if cfg is None:
        cfg = ConfigSimplificacao()
    barra_rad = math.radians(cfg.angulo_barra_t_par)
    total = 0

    while True:
        juncoes3 = [nid for nid in g.nodes if g.degree(nid) == 3]
        melhor: Optional[tuple[int, int]] = None
        melhor_dist = float('inf')

        for i, ni in enumerate(juncoes3):
            for nj in juncoes3[i + 1:]:
                n_i = g.nodes[ni]
                n_j = g.nodes[nj]
                dist = math.hypot(n_j.x - n_i.x, n_j.y - n_i.y)
                if dist > cfg.r_fusao_t_par_px or dist >= melhor_dist:
                    continue
                if nj not in g.neighbors(ni):
                    continue

                def _barra(nid_centro: int, must_include: int) -> Optional[tuple[int, int, int]]:
                    """Retorna (arm_a, arm_b, stem) se os vizinhos arm_a/arm_b formam barra
                    com ângulo ≥ barra_rad e must_include é um deles; None caso contrário."""
                    viz = g.neighbors(nid_centro)
                    nc = g.nodes[nid_centro]
                    for a in range(3):
                        for b in range(a + 1, 3):
                            va, vb = viz[a], viz[b]
                            if va != must_include and vb != must_include:
                                continue
                            na_n, nb_n = g.nodes[va], g.nodes[vb]
                            dy_a = na_n.y - nc.y; dx_a = na_n.x - nc.x
                            dy_b = nb_n.y - nc.y; dx_b = nb_n.x - nc.x
                            nm_a = math.hypot(dy_a, dx_a) or 1.0
                            nm_b = math.hypot(dy_b, dx_b) or 1.0
                            dot = (dy_a / nm_a) * (dy_b / nm_b) + (dx_a / nm_a) * (dx_b / nm_b)
                            ang = math.acos(max(-1.0, min(1.0, dot)))
                            if ang >= barra_rad:
                                return (va, vb, viz[3 - a - b])
                    return None

                bi = _barra(ni, nj)
                bj = _barra(nj, ni)
                if bi is None or bj is None:
                    continue

                stem_ni = bi[2]
                stem_nj = bj[2]
                ns_i = g.nodes[stem_ni]
                ns_j = g.nodes[stem_nj]

                d12_dx = n_j.x - n_i.x
                d12_dy = n_j.y - n_i.y
                nm12 = math.hypot(d12_dx, d12_dy) or 1.0
                d12_dx /= nm12; d12_dy /= nm12

                si_dx = ns_i.x - n_i.x; si_dy = ns_i.y - n_i.y
                nm_si = math.hypot(si_dx, si_dy) or 1.0
                si_dx /= nm_si; si_dy /= nm_si

                sj_dx = ns_j.x - n_j.x; sj_dy = ns_j.y - n_j.y
                nm_sj = math.hypot(sj_dx, sj_dy) or 1.0
                sj_dx /= nm_sj; sj_dy /= nm_sj

                # 2D cross product: sign tells which side of d12 the haste falls on
                cross_i = d12_dx * si_dy - d12_dy * si_dx
                cross_j = d12_dx * sj_dy - d12_dy * sj_dx
                if cross_i * cross_j >= 0:
                    continue  # hastes no mesmo lado

                melhor = (ni, nj)
                melhor_dist = dist

        if melhor is None:
            break

        ni, nj = melhor
        n_i, n_j = g.nodes[ni], g.nodes[nj]

        # Pick representative that gives the best X geometry (two opposite pairs).
        # The 4 external arms are the same set viewed from different positions.
        ext_i = [v for v in g.neighbors(ni) if v != nj]
        ext_j = [v for v in g.neighbors(nj) if v != ni]
        ext_all = ext_i + ext_j  # exactly 4 arms (no shared externals)

        def _score_x_rep(rep_nid: int) -> float:
            nr = g.nodes[rep_nid]
            vecs = []
            for arm in ext_all:
                na = g.nodes[arm]
                dy_, dx_ = na.y - nr.y, na.x - nr.x
                nm_ = math.hypot(dy_, dx_) or 1.0
                vecs.append((dy_ / nm_, dx_ / nm_))
            best_ = 0.0
            for a_, b_, c_, d_ in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
                d_ab = vecs[a_][0]*vecs[b_][0] + vecs[a_][1]*vecs[b_][1]
                d_cd = vecs[c_][0]*vecs[d_][0] + vecs[c_][1]*vecs[d_][1]
                score_ = min(math.acos(max(-1.0, min(1.0, d_ab))),
                             math.acos(max(-1.0, min(1.0, d_cd))))
                best_ = max(best_, score_)
            return best_

        score_i = _score_x_rep(ni)
        score_j = _score_x_rep(nj)
        mantido = ni if score_i >= score_j else nj
        descartado = nj if mantido == ni else ni

        for viz in list(g.neighbors(descartado)):
            if viz != mantido and viz in g.nodes:
                g.add_edge(mantido, viz)
        g.remove_node(descartado)
        total += 1

    return total


# ─── IV-A: pipeline de simplificação ─────────────────────────────────────────

def simplificar(
    g: StructureGraph,
    skeleton: Optional[np.ndarray] = None,
    relevo: Optional[np.ndarray] = None,
    cfg: Optional[ConfigSimplificacao] = None,
) -> dict:
    """IV-A: aplica todas as etapas de simplificação em ordem.

    Ordem: diagonais → curvatura → espurios → fusao.
    Retorna dict com contagens: {'diagonais': n, 'curvatura': n, 'espurios': n, 'fusoes': n}.
    """
    if cfg is None:
        cfg = ConfigSimplificacao()
    n_diag = remover_diagonais_espurias(g, cfg)
    n_curv = remover_nos_curvatura_baixa(g, cfg)
    n_espu = remover_espurios(g, cfg)
    n_fuso = fundir_juncoes(g, skeleton, relevo, cfg)
    n_cluster = fusao_juncoes_cluster(g, skeleton, cfg) if cfg.usar_fusao_juncoes else 0
    n_tpar = fusao_t_par(g, skeleton, cfg) if cfg.usar_fusao_t_par else 0
    return {
        'diagonais': n_diag,
        'curvatura': n_curv,
        'espurios': n_espu,
        'fusoes': n_fuso + n_cluster + n_tpar,
    }
