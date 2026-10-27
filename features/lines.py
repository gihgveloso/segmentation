"""IV-B: extração de linhas por split-and-merge. Schulz 2009, Seção IV-B.

Usa o grafo ANTES da simplificação (saída de construir_grafo / Etapa 5), de modo
que as linhas não dependam da qualidade do detector de cruzamentos.

Algoritmo:
  1. Componentes formadas por nós de grau 2 + suas extremidades de grau ≠ 2.
  2. Undistort (identidade sem calibração).
  3. Split-and-merge por mínimos quadrados totais (reta que minimiza dist ortogonal).
     Limiar: desvio_max_celulas em ConfigLinhas.
     Componentes com < min_nos_componente nós são descartados.
  4. Fusão de segmentos com ângulo próximo e distância ortogonal próxima.
  5. Extremos = projeção dos extremos brutos do componente sobre a reta ajustada.
  6. Saída: lista de LineFeature.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from features.graph import StructureGraph
from features.geometry import ConfigCamera, undistort_pontos
from features.types import LineFeature


@dataclass
class ConfigLinhas:
    """IV-B: parâmetros de extração de linhas."""
    desvio_max_celulas: float = 2.0       # limiar de split: desvio ortogonal máximo
    min_nos_componente: int = 3           # componentes com < N nós são descartados
    distancia_min_celulas: float = 1.0    # segmentos mais curtos que isso são descartados
    usar_fusao: bool = True               # fundir segmentos colineares próximos
    limiar_angulo_fusao_grau: float = 5.0 # diferença máxima de ângulo para fusão
    limiar_dist_ortogonal_fusao: float = 2.0  # distância ortogonal máxima para fusão
    lacuna_max_celulas: float = 20.0      # D36: gap máximo (células) para permitir fusão
    suporte_vao_min: float = 0.8          # D36: frac branco mínima no vão (se branco fornecido)
    suporte_aresta_min: float = 0.5       # frac branco mínima para aresta longa não ser cortada
    dist_aresta_min: float = 4.0          # comprimento mínimo de aresta para checar suporte
    projetar_no_chao: bool = False        # só ativa com calibração de câmera
    cfg_cam: Optional[ConfigCamera] = field(default=None, compare=False)


# ─────────────────────────────────────────────────────────────────────────────
# Estruturas internas
# ─────────────────────────────────────────────────────────────────────────────

class _Segmento:
    """IV-B: segmento de linha antes de ser convertido para LineFeature."""
    __slots__ = ('nos_ids', 'pts', 'reta')

    def __init__(self, nos_ids: list[int], pts: np.ndarray) -> None:
        self.nos_ids = nos_ids    # ids dos nós nesta ordem
        self.pts = pts            # (N,2) float32 (x,y) undistorted, grade
        self.reta: Optional[tuple[float, float, float, float]] = None
        # reta = (nx, ny, mx, my): normal unitária, ponto na reta (centroide)


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste de reta por mínimos quadrados totais (SVD)
# ─────────────────────────────────────────────────────────────────────────────

def _ajustar_reta(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Retorna (nx, ny, mx, my) onde (nx,ny) é a normal e (mx,my) é o centroide.

    IV-B: mínimos quadrados totais — minimiza soma de distâncias ortogonais ao quadrado.
    """
    mx, my = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    if len(pts) < 2:
        return 0.0, 1.0, mx, my
    centered = pts - np.array([mx, my], dtype=np.float32)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt[0] = direção de maior variância (ao longo da reta)
    dx, dy = float(Vt[0, 0]), float(Vt[0, 1])
    # normal = perpendicular
    nx, ny = -dy, dx
    return nx, ny, mx, my


def _dist_ortogonal(pts: np.ndarray, nx: float, ny: float, mx: float, my: float) -> np.ndarray:
    """IV-B: distância ortogonal de cada ponto à reta (nx,ny,mx,my)."""
    return np.abs((pts[:, 0] - mx) * nx + (pts[:, 1] - my) * ny)


def _projetar_ponto(x: float, y: float, nx: float, ny: float, mx: float, my: float) -> tuple[float, float]:
    """Projeta (x,y) sobre a reta (nx,ny,mx,my). IV-B: extremos por projeção."""
    dist = (x - mx) * nx + (y - my) * ny
    return x - dist * nx, y - dist * ny


def _frac_branco(x1: float, y1: float, x2: float, y2: float, branco: np.ndarray) -> float:
    """IV-B: D34. Fração de pixels brancos ao longo do segmento (x1,y1)→(x2,y2)."""
    n = max(int(math.hypot(x2 - x1, y2 - y1)) * 2 + 1, 2)
    xs = np.round(np.linspace(x1, x2, n)).astype(int)
    ys = np.round(np.linspace(y1, y2, n)).astype(int)
    H, W = branco.shape
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[m], ys[m]
    return float(np.count_nonzero(branco[ys, xs] > 0)) / len(xs) if len(xs) else 0.0


def _pontos_mais_proximos(
    pts_i: np.ndarray, pts_j: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """IV-B: D33. Par de endpoints mais próximos entre dois segmentos; retorna (pi, pj, dist)."""
    best_d = float('inf')
    best_pi, best_pj = pts_i[0], pts_j[0]
    for pi in (pts_i[0], pts_i[-1]):
        for pj in (pts_j[0], pts_j[-1]):
            d = math.hypot(float(pi[0] - pj[0]), float(pi[1] - pj[1]))
            if d < best_d:
                best_d, best_pi, best_pj = d, pi, pj
    return best_pi, best_pj, best_d


def _extremos_projetados(pts: np.ndarray, reta: tuple) -> np.ndarray:
    """IV-B: D32. Os dois extremos de pts projetados sobre a reta, invariante à ordem de traversal.

    Após np.vstack de segmentos fundidos, pts[0] e pts[-1] podem não ser os
    extremos espaciais. Usa projeção escalar para o gap check de _fundir_segmentos.
    """
    nx, ny, mx, my = reta
    dx_dir, dy_dir = -ny, nx   # direção ao longo da reta
    projs = (pts[:, 0] - mx) * dx_dir + (pts[:, 1] - my) * dy_dir
    i_min = int(np.argmin(projs))
    i_max = int(np.argmax(projs))
    p_min = _projetar_ponto(float(pts[i_min, 0]), float(pts[i_min, 1]), nx, ny, mx, my)
    p_max = _projetar_ponto(float(pts[i_max, 0]), float(pts[i_max, 1]), nx, ny, mx, my)
    return np.array([[p_min[0], p_min[1]], [p_max[0], p_max[1]]], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Split recursivo
# ─────────────────────────────────────────────────────────────────────────────

def _split(
    nos_ids: list[int],
    pts: np.ndarray,
    desvio_max: float,
    min_nos: int,
) -> list[tuple[list[int], np.ndarray]]:
    """Split-and-merge (só split). Retorna lista de (ids, pts) já ajustados.

    IV-B: o nó interior com maior desvio ortogonal define o ponto de divisão.
    Apenas pontos interiores (índices 1..N-2) são candidatos ao split, o que
    garante terminação: cada subproblema tem menos pontos que o pai.
    """
    if len(nos_ids) < min_nos:
        return []

    # Só busca o máximo no interior para garantir que os dois lados sejam menores
    if len(pts) <= 2:
        return [(nos_ids, pts)]

    nx, ny, mx, my = _ajustar_reta(pts)
    devios = _dist_ortogonal(pts, nx, ny, mx, my)

    interior = devios[1:-1]
    if len(interior) == 0:
        return [(nos_ids, pts)]

    idx_interior = int(np.argmax(interior))
    idx_max = idx_interior + 1          # índice real no array completo
    max_dev = float(interior[idx_interior])

    if max_dev <= desvio_max:
        return [(nos_ids, pts)]

    left_ids  = nos_ids[: idx_max + 1]
    right_ids = nos_ids[idx_max:]
    left_pts  = pts[: idx_max + 1]
    right_pts = pts[idx_max:]

    result = []
    if len(left_ids) >= min_nos:
        result += _split(left_ids, left_pts, desvio_max, min_nos)
    if len(right_ids) >= min_nos:
        result += _split(right_ids, right_pts, desvio_max, min_nos)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Extração de componentes de grau 2
# ─────────────────────────────────────────────────────────────────────────────

def _componentes_grau2(
    g: StructureGraph,
    branco: Optional[np.ndarray] = None,
    suporte_aresta_min: float = 0.5,
    dist_aresta_min: float = 4.0,
) -> list[list[int]]:
    """IV-B: cadeias de nós de grau 2 delimitadas por âncoras (grau ≠ 2).

    Usa travessia por ARESTA para não perder ramos de junções T/X.
    Quando `branco` é fornecido, cadeias são cortadas em arestas com
    comprimento > dist_aresta_min E frac_branco < suporte_aresta_min.
    """
    nodes = g.nodes
    if not nodes:
        return []

    def is_ancora(nid: int) -> bool:
        return g.degree(nid) != 2

    def walk_from(start: int, prev: int) -> list[int]:
        """Percorre cadeia grau-2 a partir de start, vindo de prev."""
        path = [start]
        cur, ant = start, prev
        while True:
            vizs = [v for v in g.neighbors(cur) if v != ant]
            if not vizs:
                break
            nxt = vizs[0]
            path.append(nxt)
            if is_ancora(nxt) or len(vizs) > 1:
                break
            ant, cur = cur, nxt
        return path

    def _cortar_arestas_ruins(chain: list[int]) -> list[list[int]]:
        """Divide cadeia em sub-cadeias excluindo arestas escuras e longas."""
        if branco is None or len(chain) < 2:
            return [chain]
        sub: list[list[int]] = []
        current = [chain[0]]
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            na, nb = nodes[a], nodes[b]
            d = math.hypot(nb.x - na.x, nb.y - na.y)
            if d > dist_aresta_min:
                fb = _frac_branco(na.x, na.y, nb.x, nb.y, branco)
                if fb < suporte_aresta_min:
                    sub.append(current)
                    current = [b]
                    continue
            current.append(b)
        sub.append(current)
        return [s for s in sub if len(s) >= 2]

    visited_edges: set[tuple[int, int]] = set()
    componentes: list[list[int]] = []

    def _add_chain(chain: list[int]) -> None:
        for sub in _cortar_arestas_ruins(chain):
            componentes.append(sub)

    # Semeia a partir de âncoras para capturar todos os ramos
    for nid in list(nodes.keys()):
        if not is_ancora(nid):
            continue
        for viz in g.neighbors(nid):
            edge = (min(nid, viz), max(nid, viz))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)

            if is_ancora(viz):
                _add_chain([nid, viz])
            else:
                chain = [nid] + walk_from(viz, nid)
                for i in range(len(chain) - 1):
                    visited_edges.add((min(chain[i], chain[i + 1]),
                                       max(chain[i], chain[i + 1])))
                _add_chain(chain)

    # Componentes inteiramente de grau-2 (laços ou cadeias sem âncoras)
    for a, b in g.edges:
        edge = (min(a, b), max(a, b))
        if edge in visited_edges:
            continue
        chain = [a] + walk_from(b, a)
        for i in range(len(chain) - 1):
            visited_edges.add((min(chain[i], chain[i + 1]),
                               max(chain[i], chain[i + 1])))
        _add_chain(chain)

    return componentes


# ─────────────────────────────────────────────────────────────────────────────
# Conversão para LineFeature
# ─────────────────────────────────────────────────────────────────────────────

def _para_linefeature(
    nos_ids: list[int],
    pts: np.ndarray,
    pts_orig: np.ndarray,
    fator_escala: int,
) -> LineFeature:
    """IV-B: constrói LineFeature a partir dos nós ajustados."""
    nx, ny, mx, my = _ajustar_reta(pts)

    # Extremos = projeção do ponto de menor e maior coordenada ao longo da reta.
    # Usa projeções escalar (não pts[0]/pts[-1]) para ser invariante à ordem da travessia.
    dx_dir, dy_dir = -ny, nx   # direção ao longo da reta (perpendicular à normal)
    projecoes = (pts[:, 0] - mx) * dx_dir + (pts[:, 1] - my) * dy_dir
    i_min = int(np.argmin(projecoes))
    i_max = int(np.argmax(projecoes))
    x1p, y1p = _projetar_ponto(float(pts[i_min, 0]), float(pts[i_min, 1]), nx, ny, mx, my)
    x2p, y2p = _projetar_ponto(float(pts[i_max, 0]), float(pts[i_max, 1]), nx, ny, mx, my)

    distancia = math.hypot(x2p - x1p, y2p - y1p)

    # Ângulo: direção da reta (Vt[0]) → atan2(dy, dx) em [-pi, pi]
    dx = x2p - x1p
    dy = y2p - y1p
    if distancia > 0:
        angulo = math.atan2(dy, dx)
    else:
        angulo = 0.0

    # Coordenadas originais: escala dos extremos projetados
    x1o, y1o = x1p * fator_escala, y1p * fator_escala
    x2o, y2o = x2p * fator_escala, y2p * fator_escala

    return LineFeature(
        x1=x1p, y1=y1p, x2=x2p, y2=y2p,
        x1_orig=x1o, y1_orig=y1o, x2_orig=x2o, y2_orig=y2o,
        distancia=distancia,
        angulo=angulo,
        n_nos=len(nos_ids),
        confianca=0.0,  # placeholder; extrair_linhas sobrescreve com cfg.desvio_max_celulas/2
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fusão de segmentos
# ─────────────────────────────────────────────────────────────────────────────

def _angulo_diff(a1: float, a2: float) -> float:
    """IV-B: não está no artigo. Diferença angular em [0, pi/2] (invariante a inversão)."""
    d = abs(a1 - a2) % math.pi
    return min(d, math.pi - d)


def _dist_ortogonal_retas(
    pts1: np.ndarray, reta1: tuple,
    pts2: np.ndarray, reta2: tuple,
) -> float:
    """IV-B: não está no artigo. Distância ortogonal média entre centroides de dois segmentos."""
    nx1, ny1, mx1, my1 = reta1
    nx2, ny2, mx2, my2 = reta2
    # Distância do centroide de seg2 à reta de seg1
    d1 = abs((mx2 - mx1) * nx1 + (my2 - my1) * ny1)
    # Distância do centroide de seg1 à reta de seg2
    d2 = abs((mx1 - mx2) * nx2 + (my1 - my2) * ny2)
    return (d1 + d2) / 2.0


def _fundir_segmentos(
    segs: list[tuple[list[int], np.ndarray]],
    cfg: ConfigLinhas,
    branco: Optional[np.ndarray] = None,
) -> list[tuple[list[int], np.ndarray]]:
    """IV-B: funde segmentos colineares próximos.

    Dois segmentos são fundidos se:
    1. ângulo diff < limiar_angulo_fusao_grau
    2. distância ortogonal < limiar_dist_ortogonal_fusao
    3. conectados (compartilham um nó) OU gap ≤ lacuna_max E (branco=None OU frac ≥ suporte_vao_min)
    """
    lim_ang  = math.radians(cfg.limiar_angulo_fusao_grau)
    lim_dist = cfg.limiar_dist_ortogonal_fusao

    def _reta_seg(seg_pts: np.ndarray) -> tuple:
        return _ajustar_reta(seg_pts)

    changed = True
    while changed:
        changed = False
        n = len(segs)
        merged = [False] * n
        novos: list[tuple[list[int], np.ndarray]] = []

        retas = [_reta_seg(s[1]) for s in segs]
        sets_ids = [set(s[0]) for s in segs]

        for i in range(n):
            if merged[i]:
                continue
            ids_i, pts_i = segs[i]
            set_i = sets_ids[i]
            r_i = retas[i]
            nx_i, ny_i = r_i[0], r_i[1]
            ang_i = math.atan2(ny_i, nx_i)

            for j in range(i + 1, n):
                if merged[j]:
                    continue
                ids_j, pts_j = segs[j]
                r_j = retas[j]
                nx_j, ny_j = r_j[0], r_j[1]
                ang_j = math.atan2(ny_j, nx_j)

                if _angulo_diff(ang_i, ang_j) > lim_ang:
                    continue
                if _dist_ortogonal_retas(pts_i, r_i, pts_j, r_j) > lim_dist:
                    continue

                # Verificação de gap: conectados pelo grafo (nó compartilhado) OU gap ≤ lacuna_max.
                # Usa extremos projetados (não pts[0]/pts[-1]) para ser invariante à ordem do
                # vstack após fusões encadeadas — D36 fix.
                if not (set_i & sets_ids[j]):
                    ext_i = _extremos_projetados(pts_i, r_i)
                    ext_j = _extremos_projetados(pts_j, r_j)
                    pi, pj, gap = _pontos_mais_proximos(ext_i, ext_j)
                    if gap > cfg.lacuna_max_celulas:
                        continue
                    if branco is not None:
                        if _frac_branco(float(pi[0]), float(pi[1]),
                                        float(pj[0]), float(pj[1]), branco) < cfg.suporte_vao_min:
                            continue

                pts_fundidos = np.vstack([pts_i, pts_j])
                ids_fundidos = ids_i + ids_j
                set_fused = set_i | sets_ids[j]
                merged[i] = merged[j] = True
                ids_i, pts_i = ids_fundidos, pts_fundidos
                set_i = set_fused
                r_i = _reta_seg(pts_i)
                nx_i, ny_i = r_i[0], r_i[1]
                ang_i = math.atan2(ny_i, nx_i)
                changed = True

            novos.append((ids_i, pts_i))
            sets_ids[i] = set_i

        segs = novos

    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Topo verde (para d_topo_celulas)
# ─────────────────────────────────────────────────────────────────────────────

def _y_topo_verde(verde: np.ndarray, corrida_min: int = 3) -> np.ndarray:
    """IV-B: não está no artigo. Y do topo do verde por coluna para d_topo_celulas.

    Usa corrida de `corrida_min` pixels consecutivos para estabilidade;
    colunas sem verde ficam com float('nan').
    """
    H, W = verde.shape
    y_topo = np.full(W, float('nan'), dtype=np.float64)
    for c in range(W):
        run = 0
        for r in range(H):
            if verde[r, c] > 0:
                run += 1
                if run >= corrida_min:
                    y_topo[c] = float(r - corrida_min + 1)
                    break
            else:
                run = 0
    return y_topo


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada público
# ─────────────────────────────────────────────────────────────────────────────

def extrair_linhas(
    g: StructureGraph,
    fator_escala: int = 8,
    cfg: Optional[ConfigLinhas] = None,
    branco: Optional[np.ndarray] = None,
    verde: Optional[np.ndarray] = None,
) -> list[LineFeature]:
    """IV-B: extrai LineFeatures do grafo de estrutura.

    Usa o grafo ANTES da simplificação (saída da Etapa 5).
    `branco`: máscara binária de pixels brancos (grade subamostrada).
      Se fornecida, ativa filtragem de arestas escuras e verificação de suporte no vão.
    `verde`: máscara binária de pixels verdes (grade subamostrada).
      Se fornecida, preenche `LineFeature.d_topo_celulas` (distância vertical do centro
      da linha ao topo do campo; negativa = linha acima do topo do verde).
    """
    if cfg is None:
        cfg = ConfigLinhas()

    nodes = g.nodes
    if not nodes:
        return []

    # 1. Componentes de grau-2
    componentes = _componentes_grau2(
        g, branco=branco,
        suporte_aresta_min=cfg.suporte_aresta_min,
        dist_aresta_min=cfg.dist_aresta_min,
    )

    # 2. Undistort (identidade sem calibração)
    # Coleta todos os pontos para undistort em lote
    segmentos_raw: list[tuple[list[int], np.ndarray, np.ndarray]] = []
    for ids in componentes:
        if len(ids) < cfg.min_nos_componente:
            continue
        pts = np.array([[nodes[nid].x, nodes[nid].y] for nid in ids], dtype=np.float32)
        pts_ud, pts_orig = undistort_pontos(pts, fator_escala, cfg.cfg_cam)
        segmentos_raw.append((ids, pts_ud, pts_orig))

    # 3. Split-and-merge
    segs_split: list[tuple[list[int], np.ndarray]] = []
    segs_pts_orig: list[np.ndarray] = []
    for ids, pts_ud, pts_orig in segmentos_raw:
        sub = _split(ids, pts_ud, cfg.desvio_max_celulas, cfg.min_nos_componente)
        for sub_ids, sub_pts in sub:
            segs_split.append((sub_ids, sub_pts))
            # pts_orig para os mesmos índices (alinhado pela posição na lista original)
            idx_map = {nid: k for k, nid in enumerate(ids)}
            orig_sub = np.array([pts_orig[idx_map[nid]] for nid in sub_ids], dtype=np.float32)
            segs_pts_orig.append(orig_sub)

    if not segs_split:
        return []

    # 4. Fusão (opcional)
    if cfg.usar_fusao and len(segs_split) > 1:
        segs_split = _fundir_segmentos(segs_split, cfg, branco=branco)
        # Recalcula pts_orig para segmentos fundidos (usa escala direta)
        segs_pts_orig = []
        for ids, pts in segs_split:
            orig = pts.copy() * fator_escala
            segs_pts_orig.append(orig)

    # 5. Converter para LineFeature
    y_topo: Optional[np.ndarray] = _y_topo_verde(verde) if verde is not None else None
    W_grade = verde.shape[1] if verde is not None else 0

    resultado: list[LineFeature] = []
    for (ids, pts), pts_orig in zip(segs_split, segs_pts_orig):
        if len(ids) < cfg.min_nos_componente:
            continue
        if len(pts) < 2:
            continue
        lf = _para_linefeature(ids, pts, pts_orig, fator_escala)
        # recalcula confiança com desvio_max real
        nx, ny, mx, my = _ajustar_reta(pts)
        devios = _dist_ortogonal(pts, nx, ny, mx, my)
        limiar = cfg.desvio_max_celulas / 2.0

        # d_topo_celulas (item 4): distância vertical do centro ao topo do verde
        d_topo: Optional[float] = None
        if y_topo is not None:
            cx = (lf.x1 + lf.x2) / 2.0
            cy = (lf.y1 + lf.y2) / 2.0
            col = min(int(round(cx)), W_grade - 1)
            yt = float(y_topo[col])
            if math.isfinite(yt):
                d_topo = cy - yt

        lf = LineFeature(
            x1=lf.x1, y1=lf.y1, x2=lf.x2, y2=lf.y2,
            x1_orig=lf.x1_orig, y1_orig=lf.y1_orig,
            x2_orig=lf.x2_orig, y2_orig=lf.y2_orig,
            distancia=lf.distancia,
            angulo=lf.angulo,
            n_nos=lf.n_nos,
            confianca=float(np.mean(devios < limiar)),
            d_topo_celulas=d_topo,
        )
        if lf.distancia >= cfg.distancia_min_celulas:
            resultado.append(lf)

    return resultado
