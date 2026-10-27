"""III-C: conexão de nós por vizinhança 3×3, esticar extremos, dividir grau 0, candidatas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from features.graph import Node, StructureGraph

# 8 direções — D2: 8-conectividade pura, sem poda de diagonais
_DIRS = ((-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1),  (1, 0),  (1, 1))


def connect_nodes(g: StructureGraph, skeleton: np.ndarray) -> None:
    """III-C: insere arestas entre nós cujas regiões 3×3 se sobrepõem ou tocam no esqueleto.

    Regra: dois nós A e B são conectados se existe algum pixel de esqueleto p em
    3×3(A) e algum pixel de esqueleto q em 3×3(B) tais que p e q são 8-adjacentes
    (inclui p == q, ou seja, overlap direto). Modifica g in-place.
    """
    h, w = skeleton.shape

    # pixel_to_nodes: cada pixel de esqueleto mapeado ao conjunto de nós
    # cujas regiões 3×3 o contêm.
    pixel_to_nodes: dict[tuple[int, int], set[int]] = {}

    for nid, node in g.nodes.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                py, px = node.y + dy, node.x + dx
                if 0 <= py < h and 0 <= px < w and skeleton[py, px]:
                    pixel_to_nodes.setdefault((py, px), set()).add(nid)

    # Conectar nós que compartilham o mesmo pixel (overlap)
    for nids in pixel_to_nodes.values():
        if len(nids) < 2:
            continue
        nids_list = sorted(nids)
        for i in range(len(nids_list)):
            for j in range(i + 1, len(nids_list)):
                g.add_edge(nids_list[i], nids_list[j])

    # Conectar nós cujos pixels cobertos são 8-adjacentes (touch)
    for (py, px), nids_a in pixel_to_nodes.items():
        for dy, dx in _DIRS:
            vizinho = (py + dy, px + dx)
            if vizinho not in pixel_to_nodes:
                continue
            for a in nids_a:
                for b in pixel_to_nodes[vizinho]:
                    if a != b:
                        g.add_edge(a, b)


# ─── Helpers internos ─────────────────────────────────────────────────────────

def _bfs_componente(skeleton: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    """III-C: D10. BFS 8-conexo a partir de (y, x) no esqueleto; retorna lista de pixels."""
    h, w = skeleton.shape
    visitados: set[tuple[int, int]] = {(y, x)}
    fila = [(y, x)]
    resultado = []
    while fila:
        cy, cx = fila.pop()
        resultado.append((cy, cx))
        for dy, dx in _DIRS:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and (ny, nx) not in visitados:
                visitados.add((ny, nx))
                fila.append((ny, nx))
    return resultado


def _dois_extremos(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    """III-C: D10. Dois pixels mais distantes de uma componente de esqueleto."""
    p1, p2 = comp[0], comp[-1]
    max_d = -1.0
    for i in range(len(comp)):
        for j in range(i + 1, len(comp)):
            d = np.hypot(comp[i][0] - comp[j][0], comp[i][1] - comp[j][1])
            if d > max_d:
                max_d = d
                p1, p2 = comp[i], comp[j]
    return p1, p2


# ─── Etapa 4: esticar extremos ────────────────────────────────────────────────

def esticar_extremos(
    g: StructureGraph,
    skeleton: np.ndarray,
    relevo: Optional[np.ndarray] = None,
) -> int:
    """III-C (D4): move nós terminais (grau 1) ao último pixel de esqueleto na direção da aresta.

    D5: gray é atualizado para relevo[nova_posição] se relevo fornecido.
    Retorna o número de nós efetivamente movidos.
    """
    h, w = skeleton.shape

    # Coletar terminais antes de modificar (grau 1 exato)
    terminais = [
        (nid, g.neighbors(nid)[0])
        for nid in list(g.nodes.keys())
        if g.degree(nid) == 1
    ]

    movidos = 0
    for nid_t, nid_v in terminais:
        if nid_t not in g.nodes or nid_v not in g.nodes:
            continue

        # Acessa o objeto Node diretamente pela referência (Node é mutável)
        node = g.nodes[nid_t]
        viz  = g.nodes[nid_v]

        dy = node.y - viz.y
        dx = node.x - viz.x
        sy, sx = int(np.sign(dy)), int(np.sign(dx))
        if sy == 0 and sx == 0:
            continue

        # Caminha na direção (sy, sx) enquanto há pixels de esqueleto
        cur_y, cur_x = node.y, node.x
        while True:
            ny, nx = cur_y + sy, cur_x + sx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                cur_y, cur_x = ny, nx
            else:
                break

        if cur_y == node.y and cur_x == node.x:
            continue  # já estava no fim

        # Modifica in-place (node é o mesmo objeto em g._nodes[nid_t])
        node.y = cur_y
        node.x = cur_x
        if relevo is not None:
            node.gray = float(relevo[cur_y, cur_x])
        movidos += 1

    return movidos


# ─── Etapa 4: subdivisão de arcos — helpers e config ─────────────────────────

@dataclass
class ConfigSubdivisao:
    """Parâmetros para subdivisão de arcos longos em dividir_grau_zero. D13."""
    max_segmento_px: int = 30
    # passo_override > 0: usa este valor fixo.
    # passo_override == 0: usa mediana das arestas existentes (auto, recomendado).
    passo_override: float = 0.0


def _mediana_espaco_nos(g: StructureGraph) -> float:
    """III-C: D13. Mediana do comprimento euclidiano das arestas; 0.0 se vazio."""
    lengths = [
        np.hypot(g.nodes[a].y - g.nodes[b].y, g.nodes[a].x - g.nodes[b].x)
        for a, b in g.edges
    ]
    return float(np.median(lengths)) if lengths else 0.0


def _bfs_caminho_esqueleto(
    skeleton: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """III-C: D13. Caminho mais curto (BFS 8-conexo) no esqueleto de start a end.

    Retorna None se não houver caminho.
    """
    if start == end:
        return [start]
    from collections import deque
    h, w = skeleton.shape
    prev: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    fila: deque[tuple[int, int]] = deque([start])
    while fila:
        cy, cx = fila.popleft()
        if (cy, cx) == end:
            path: list[tuple[int, int]] = []
            cur: Optional[tuple[int, int]] = end
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for dy, dx in _DIRS:
            ny, nx = cy + dy, cx + dx
            nb = (ny, nx)
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and nb not in prev:
                prev[nb] = (cy, cx)
                fila.append(nb)
    return None


# ─── Etapa 4: dividir nós de grau 0 ──────────────────────────────────────────

def dividir_grau_zero(
    g: StructureGraph,
    skeleton: np.ndarray,
    relevo: Optional[np.ndarray] = None,
    cfg_sub: Optional[ConfigSubdivisao] = None,
) -> int:
    """III-C (D3, D13): substitui nós de grau 0 por extremos do trecho de esqueleto.

    Para trechos com len(comp) <= 1: mantido (D3).
    Para trechos com len(comp) <= cfg_sub.max_segmento_px: 2 extremos + 1 aresta.
    Para trechos com len(comp) > cfg_sub.max_segmento_px: cadeia de nós ao longo do
    BFS-path p1→p2, espaçados pelo passo (D13).

    Passo = mediana das arestas existentes se cfg_sub.passo_override == 0,
    caso contrário cfg_sub.passo_override. Fallback: max_segmento_px // 5.
    D5: novos nós recebem gray = relevo[posição].
    Retorna o número de componentes processadas.
    """
    if cfg_sub is None:
        cfg_sub = ConfigSubdivisao()

    # Medir passo uma vez (antes do loop, para não contar as novas arestas)
    if cfg_sub.passo_override > 0:
        passo = cfg_sub.passo_override
    else:
        med = _mediana_espaco_nos(g)
        passo = med if med > 0 else float(cfg_sub.max_segmento_px) // 5

    visitados_skel: set[tuple[int, int]] = set()
    divididos = 0
    max_id_ever = max(g.nodes.keys(), default=-1)

    for nid in list(g.nodes.keys()):
        if nid not in g.nodes or g.degree(nid) != 0:
            continue

        node = g.nodes[nid]
        origem = (node.y, node.x)

        if origem in visitados_skel:
            continue

        comp = _bfs_componente(skeleton, node.y, node.x)
        visitados_skel.update(comp)

        if len(comp) <= 1:
            continue  # D3: pixel isolado mantido

        p1, p2 = _dois_extremos(comp)

        comp_set = set(comp)

        # D13 fix: só age se TODOS os nós nesta componente têm grau 0.
        # Se há qualquer nó conectado (grau≥1), a componente já foi coberta por
        # place_nodes + connect_nodes — não criar cadeia redundante.
        if any(g.degree(nid2) >= 1
               for nid2, nd2 in g.nodes.items()
               if (nd2.y, nd2.x) in comp_set):
            continue

        # Remover todos os nós de grau 0 desta componente
        for nid2 in list(g.nodes.keys()):
            n2 = g.nodes.get(nid2)
            if n2 is not None and (n2.y, n2.x) in comp_set and g.degree(nid2) == 0:
                g.remove_node(nid2)

        max_id_ever = max(max_id_ever, max(g.nodes.keys(), default=-1))

        if len(comp) > cfg_sub.max_segmento_px:
            # D13: cadeia de nós ao longo do BFS-path
            caminho = _bfs_caminho_esqueleto(skeleton, p1, p2)
            if caminho is None:
                caminho = [p1, p2]  # fallback improvável

            # Selecionar posições a cada `passo` de distância acumulada
            nos_pos: list[tuple[int, int]] = [caminho[0]]
            dist_acum = 0.0
            for i in range(1, len(caminho)):
                d = np.hypot(caminho[i][0] - caminho[i - 1][0],
                             caminho[i][1] - caminho[i - 1][1])
                dist_acum += d
                if dist_acum >= passo:
                    nos_pos.append(caminho[i])
                    dist_acum = 0.0
            if nos_pos[-1] != caminho[-1]:
                nos_pos.append(caminho[-1])

            # Criar nós e conectar em cadeia
            ids_novos: list[int] = []
            for (ey, ex) in nos_pos:
                gray = float(relevo[ey, ex]) if relevo is not None else 1.0
                max_id_ever += 1
                g.add_node(Node(id=max_id_ever, x=ex, y=ey, gray=gray))
                ids_novos.append(max_id_ever)
            for i in range(len(ids_novos) - 1):
                g.add_edge(ids_novos[i], ids_novos[i + 1])

        else:
            # Comportamento original: 2 extremos + 1 aresta
            ids_novos = []
            for (ey, ex) in (p1, p2):
                gray = float(relevo[ey, ex]) if relevo is not None else 1.0
                max_id_ever += 1
                g.add_node(Node(id=max_id_ever, x=ex, y=ey, gray=gray))
                ids_novos.append(max_id_ever)
            g.add_edge(ids_novos[0], ids_novos[1])

        divididos += 1

    return divididos


# ─── Utilitário: componentes conexas ─────────────────────────────────────────

def contar_componentes(g: StructureGraph) -> int:
    """Não está no artigo. Conta componentes conexas no grafo (BFS)."""
    visitados: set[int] = set()
    count = 0
    for nid in g.nodes:
        if nid not in visitados:
            count += 1
            fila = [nid]
            while fila:
                n = fila.pop()
                if n not in visitados:
                    visitados.add(n)
                    fila.extend(g.neighbors(n))
    return count


# ─── Etapa 5: conexões candidatas ────────────────────────────────────────────

@dataclass
class ConfigConexao:
    """Parâmetros para geração de conexões candidatas. III-C."""
    dist_max: float = 8.0          # distância máxima entre terminais para candidatura (px)
    limiar_custo: float = 10.0     # custo máximo para aceitar a candidata
    peso_angular: float = 1.0      # peso da penalidade angular no custo
    peso_cinza: float = 1.0        # peso do custo de cinza (1 - cinza_medio)
    dist_grafo_min: int = 5        # D12: min hops no grafo para aceitar laço intra-componente
    fator_laco_grande: float = 1.5 # D12: laço é "grande" se dist_grafo >= fator * dist_eucl
    filtrar_escuras: bool = False  # 1.5-relevo: rejeitar candidatas longas E escuras (relevo)
    dist_min_filtro: float = 4.0   # comprimento mínimo para acionar filtros de escuras
    cinza_min: float = 0.5         # relevo mínimo médio para candidata não ser rejeitada
    suporte_candidata_min: float = 0.5  # D37: frac branco mínima (quando branco fornecido)


def _direcao_entrada(g: StructureGraph, nid: int) -> Optional[tuple[float, float]]:
    """III-C: D11. Vetor unitário apontando para fora do nó terminal (grau 1)."""
    vizinhos = g.neighbors(nid)
    if len(vizinhos) != 1:
        return None
    nid_v = vizinhos[0]
    node = g.nodes[nid]
    viz  = g.nodes[nid_v]
    dy = node.y - viz.y
    dx = node.x - viz.x
    norm = np.hypot(dy, dx)
    if norm == 0:
        return None
    return (dy / norm, dx / norm)


def _cinza_medio_segmento(
    relevo: np.ndarray,
    y1: int, x1: int,
    y2: int, x2: int,
    n_amostras: int = 10,
) -> float:
    """III-C: D11. Cinza médio ao longo do segmento (y1,x1)→(y2,x2)."""
    h, w = relevo.shape
    ts = np.linspace(0.0, 1.0, max(n_amostras, 2))
    ys = np.clip(np.round(y1 + ts * (y2 - y1)).astype(int), 0, h - 1)
    xs = np.clip(np.round(x1 + ts * (x2 - x1)).astype(int), 0, w - 1)
    return float(relevo[ys, xs].mean())


def _frac_branco_candidata(
    branco: np.ndarray,
    y1: float, x1: float,
    y2: float, x2: float,
) -> float:
    """D37: fração de pixels brancos ao longo do segmento reto (nó a nó).

    Mesma métrica de D34 (frac_branco em lines.py), aplicada a candidatas.
    """
    dist = float(np.hypot(y2 - y1, x2 - x1))
    n = max(int(dist) * 2 + 1, 2)
    H, W = branco.shape
    ys = np.round(np.linspace(y1, y2, n)).astype(int)
    xs = np.round(np.linspace(x1, x2, n)).astype(int)
    m = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys, xs = ys[m], xs[m]
    if len(ys) == 0:
        return 0.0
    return float(np.count_nonzero(branco[ys, xs] > 0)) / len(ys)


def _custo_candidata(
    g: StructureGraph,
    relevo: np.ndarray,
    nid_a: int,
    nid_b: int,
    cfg: ConfigConexao,
) -> float:
    """III-C: D11. custo = dist + peso_cinza*(1-cinza) + peso_angular*penalidade.

    Penalidade angular: (1-cosθ)/2 por terminal de grau 1; nós de grau 0 sem penalidade.
    """
    na = g.nodes[nid_a]
    nb = g.nodes[nid_b]
    dist = np.hypot(na.y - nb.y, na.x - nb.x)

    cinza = _cinza_medio_segmento(relevo, na.y, na.x, nb.y, nb.x)
    custo_cinza = cfg.peso_cinza * (1.0 - cinza)

    # Vetor da candidata (de a para b), normalizado
    dy_ab = nb.y - na.y
    dx_ab = nb.x - na.x
    norm_ab = np.hypot(dy_ab, dx_ab)
    if norm_ab == 0:
        return float('inf')
    vy, vx = dy_ab / norm_ab, dx_ab / norm_ab

    penalidade = 0.0
    dir_a = _direcao_entrada(g, nid_a)
    dir_b = _direcao_entrada(g, nid_b)

    if dir_a is not None:
        # cos do ângulo entre a direção de saída de a e o vetor a→b
        cos_a = dir_a[0] * vy + dir_a[1] * vx
        penalidade += (1.0 - cos_a) / 2.0

    if dir_b is not None:
        # direção de saída de b aponta de b para a: vetor -v
        cos_b = dir_b[0] * (-vy) + dir_b[1] * (-vx)
        penalidade += (1.0 - cos_b) / 2.0

    custo_angular = cfg.peso_angular * penalidade
    return dist + custo_cinza + custo_angular


def _bfs_dist_grafo(g: StructureGraph, start: int, end: int) -> int:
    """III-C: D12. Distância em hops (BFS) entre start e end; retorna 10**9 se desconexos."""
    if start == end:
        return 0
    dist: dict[int, int] = {start: 0}
    fila = [start]
    while fila:
        n = fila.pop(0)
        for nb in g.neighbors(n):
            if nb not in dist:
                dist[nb] = dist[n] + 1
                if nb == end:
                    return dist[nb]
                fila.append(nb)
    return 10 ** 9


def conectar_candidatas(
    g: StructureGraph,
    relevo: np.ndarray,
    cfg: Optional[ConfigConexao] = None,
    _debug_info: Optional[list] = None,
    branco: Optional[np.ndarray] = None,
) -> int:
    """III-C: insere arestas candidatas entre terminais próximos.

    Regras:
    - Candidatos: nós com grau <= 1 (terminais e isolados).
    - Distância entre candidatos <= cfg.dist_max.
    - Restrição: grau máximo 4 em qualquer nó.
    - Custo = dist + peso_cinza*(1-cinza_medio) + peso_angular*penalidade.
    - Seleção gulosa pelo menor custo; só aceita se custo < cfg.limiar_custo.
    - D12: candidata intra-componente (pelo UF dinâmico) é bloqueada se
      dist_grafo < dist_grafo_min E NÃO for laço grande
      (dist_grafo < fator_laco_grande * dist_eucl).
    - D37: quando branco fornecido, candidata com dist > dist_min_filtro E
      frac_branco < suporte_candidata_min é bloqueada (mesma métrica D34).
    Retorna o número de arestas inseridas (inter + intra aceitas).

    Se _debug_info é uma lista, cada aresta inserida acrescenta um dict com:
    {'a': nid_a, 'b': nid_b, 'inter': bool, 'dist_eucl': float,
     'custo': float, 'dist_grafo': int_ou_None}
    inter=True significa que os nós estavam em componentes distintos no UF
    no momento da inserção (logo: reduz a contagem de componentes em 1).
    """
    if cfg is None:
        cfg = ConfigConexao()

    terminais = [nid for nid in g.nodes if g.degree(nid) <= 1]
    if len(terminais) < 2:
        return 0

    # Union-Find para rastrear componentes dinamicamente durante a seleção
    _pai: dict[int, int] = {nid: nid for nid in g.nodes}

    def _find(x: int) -> int:
        while _pai[x] != x:
            _pai[x] = _pai[_pai[x]]
            x = _pai[x]
        return x

    def _union(x: int, y: int) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            _pai[px] = py

    for a, b in g.edges:
        _union(a, b)

    # Gerar candidatos dentro de dist_max com custo abaixo do limiar
    candidatos: list[tuple[float, int, int, float]] = []
    for i in range(len(terminais)):
        for j in range(i + 1, len(terminais)):
            a, b = terminais[i], terminais[j]
            if b in g.neighbors(a):
                continue
            na, nb = g.nodes[a], g.nodes[b]
            dist_eucl = np.hypot(na.y - nb.y, na.x - nb.x)
            if dist_eucl > cfg.dist_max:
                continue
            if cfg.filtrar_escuras and dist_eucl > cfg.dist_min_filtro:
                cinza = _cinza_medio_segmento(relevo, na.y, na.x, nb.y, nb.x)
                if cinza < cfg.cinza_min:
                    continue
            # D37: filtro branco (mesma métrica de D34, aplicada às candidatas)
            if branco is not None and dist_eucl > cfg.dist_min_filtro:
                fb = _frac_branco_candidata(branco, na.y, na.x, nb.y, nb.x)
                if fb < cfg.suporte_candidata_min:
                    continue
            custo = _custo_candidata(g, relevo, a, b, cfg)
            if custo < cfg.limiar_custo:
                candidatos.append((custo, a, b, dist_eucl))

    candidatos.sort()

    inseridas = 0
    for custo, a, b, dist_eucl in candidatos:
        if g.degree(a) >= 4 or g.degree(b) >= 4:
            continue
        if b in g.neighbors(a):
            continue

        # Classificação live pelo UF: inter=True → vai reduzir contagem de componentes
        inter = (_find(a) != _find(b))
        dist_g: Optional[int] = None

        if not inter:
            # D12: verificar laço intra-componente
            dist_g = _bfs_dist_grafo(g, a, b)
            laco_grande = dist_g >= cfg.fator_laco_grande * dist_eucl
            trivial = dist_g <= cfg.dist_grafo_min   # <= inclui exatamente dist_grafo_min hops
            if trivial and not laco_grande:
                continue  # laço trivial: bloquear

        g.add_edge(a, b)
        _union(a, b)
        inseridas += 1

        if _debug_info is not None:
            _debug_info.append({
                'a': a, 'b': b,
                'inter': inter,
                'dist_eucl': dist_eucl,
                'custo': custo,
                'dist_grafo': dist_g,
            })

    return inseridas
