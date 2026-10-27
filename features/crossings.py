"""IV-A: detecção e verificação de cruzamentos L, T e X. D19, D20."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from features.graph import StructureGraph
from features.types import CrossingFeature


@dataclass
class ConfigCruzamento:
    """Parâmetros de detecção de cruzamentos. D19, D20, D21, D24, D29."""
    # Janela de ângulo para candidato L (D20): fora disto não é canto
    angulo_L_min_grau: float = 50.0
    angulo_L_max_grau: float = 130.0
    # D24: ângulo L medido via arco (0 = usar vizinho imediato)
    distancia_arco_l_px: int = 7
    # Verificação de geometria de T e X (D22)
    verificar_geometria_t: bool = True
    angulo_barra_t_grau: float = 140.0    # mínimo entre os dois braços da barra
    angulo_stem_t_min_grau: float = 50.0  # D26 revertido: mínimo stem (sem efeito com arco=0)
    angulo_stem_t_max_grau: float = 130.0 # D26 revertido: máximo stem (sem efeito com arco=0)
    # D26 (desligado por padrão): arco em geometry T/X; 0 = vizinho imediato
    distancia_arco_geom_px: int = 0
    verificar_geometria_x: bool = True
    angulo_par_x_grau: float = 140.0     # mínimo por par oposto no X
    # V1: consenso em múltiplos raios (D23)
    usar_consenso_v1: bool = True
    raios_circulo: list[int] = field(default_factory=lambda: [4, 5, 6, 8])
    tolerancia_segmentos: int = 1        # ±n no número de segmentos esperado
    min_raios_consenso: int = 2          # raios que devem concordar para V1=0.5
    # V2: BFS em pixels brancos do esqueleto
    max_passos_v2: int = 60              # px; comprimento máximo do caminho BFS
    # Aceitação e supressão de duplicatas
    limiar_confianca: float = 0.5
    r_fusao_px: float = 3.0             # px; candidatos a < r_fusao são fundidos
    # D29: descartar candidatos com círculo fora da grade
    verificar_borda: bool = True
    # IV-A: amostras no círculo V1 (D19); 36 = 10° por ponto
    n_amostras_circulo: int = 36


# ─── IV-A: helpers de ângulo e orientação ────────────────────────────────────

def _caminhar_ate_arco(g: StructureGraph, nid: int, viz_start: int, d_arco: float) -> int:
    """IV-A: D24. Percorre a cadeia a partir de viz_start em direção oposta a nid por ~d_arco px.

    Para ao encontrar: (a) nó de grau ≠ 2 (extremo ou junção), ou (b) acumulado ≥ d_arco.
    Retorna o último nó visitado.
    """
    if d_arco <= 0:
        return viz_start
    prev = nid
    cur = viz_start
    accumulated = math.hypot(
        g.nodes[cur].y - g.nodes[nid].y,
        g.nodes[cur].x - g.nodes[nid].x,
    )
    if accumulated >= d_arco:
        return cur
    while True:
        vizs = [v for v in g.neighbors(cur) if v != prev]
        if not vizs or g.degree(cur) != 2:
            return cur
        nxt = vizs[0]
        step = math.hypot(
            g.nodes[nxt].y - g.nodes[cur].y,
            g.nodes[nxt].x - g.nodes[cur].x,
        )
        accumulated += step
        prev = cur
        cur = nxt
        if accumulated >= d_arco:
            return cur
    return cur  # unreachable


def _angulo_entre_vizinhos(g: StructureGraph, nid: int) -> float:
    """IV-A: D19. Ângulo em radianos entre os dois vizinhos de um nó de grau 2."""
    vizinhos = g.neighbors(nid)
    n  = g.nodes[nid]
    na = g.nodes[vizinhos[0]]
    nb = g.nodes[vizinhos[1]]
    vay, vax = na.y - n.y, na.x - n.x
    vby, vbx = nb.y - n.y, nb.x - n.x
    norm_a = math.hypot(vay, vax)
    norm_b = math.hypot(vby, vbx)
    if norm_a == 0 or norm_b == 0:
        return math.pi
    cos_t = (vay * vby + vax * vbx) / (norm_a * norm_b)
    return math.acos(max(-1.0, min(1.0, cos_t)))


def _angulo_arco_l(g: StructureGraph, nid: int, d_arco: float) -> float:
    """IV-A: D24. Ângulo L medido via pontos a distância de arco d_arco ao longo do grafo.

    Substitui _angulo_entre_vizinhos quando distancia_arco_l_px > 0.
    """
    vizinhos = g.neighbors(nid)
    far0 = _caminhar_ate_arco(g, nid, vizinhos[0], d_arco)
    far1 = _caminhar_ate_arco(g, nid, vizinhos[1], d_arco)
    n  = g.nodes[nid]
    f0 = g.nodes[far0]
    f1 = g.nodes[far1]
    vay, vax = f0.y - n.y, f0.x - n.x
    vby, vbx = f1.y - n.y, f1.x - n.x
    norm_a = math.hypot(vay, vax) or 1.0
    norm_b = math.hypot(vby, vbx) or 1.0
    cos_t = (vay * vby + vax * vbx) / (norm_a * norm_b)
    return math.acos(max(-1.0, min(1.0, cos_t)))


def _orientacao_l(g: StructureGraph, nid: int) -> float:
    """IV-A: D19. Bissetriz apontando para o interior do canto L."""
    vizinhos = g.neighbors(nid)
    n  = g.nodes[nid]
    na = g.nodes[vizinhos[0]]
    nb = g.nodes[vizinhos[1]]
    vay, vax = na.y - n.y, na.x - n.x
    vby, vbx = nb.y - n.y, nb.x - n.x
    norm_a = math.hypot(vay, vax) or 1.0
    norm_b = math.hypot(vby, vbx) or 1.0
    by = vay / norm_a + vby / norm_b
    bx = vax / norm_a + vbx / norm_b
    return math.atan2(by, bx)


def _orientacao_t(g: StructureGraph, nid: int) -> float:
    """IV-A: D19. Direção da haste do T.

    Par de braços mais antiparalelos forma a "barra"; o terceiro é a haste.
    """
    vizinhos = g.neighbors(nid)
    n = g.nodes[nid]
    vecs = []
    for v in vizinhos:
        nv = g.nodes[v]
        dy, dx = nv.y - n.y, nv.x - n.x
        norm = math.hypot(dy, dx) or 1.0
        vecs.append((dy / norm, dx / norm))

    best_score = -2.0
    stem_i = 0
    for i in range(3):
        for j in range(i + 1, 3):
            score = -(vecs[i][0] * vecs[j][0] + vecs[i][1] * vecs[j][1])
            if score > best_score:
                best_score = score
                stem_i = 3 - i - j

    nv = g.nodes[vizinhos[stem_i]]
    return math.atan2(nv.y - n.y, nv.x - n.x)


# ─── IV-A: verificação de geometria T e X (D22) ──────────────────────────────

def _verifica_geometria_t(g: StructureGraph, nid: int, cfg: ConfigCruzamento) -> bool:
    """IV-A: não está no artigo (D22, D26). True se os 3 braços têm geometria de T.

    Exige: um par de braços com ângulo ≥ angulo_barra_t_grau (a barra); o terceiro
    (haste) faz ângulo em [angulo_stem_t_min_grau, angulo_stem_t_max_grau] com cada
    braço da barra. D26: usa arco distancia_arco_geom_px se > 0.
    """
    vizinhos = g.neighbors(nid)
    n = g.nodes[nid]
    d_arco = cfg.distancia_arco_geom_px
    vecs = []
    for v in vizinhos:
        if d_arco > 0:
            far = _caminhar_ate_arco(g, nid, v, float(d_arco))
            nf = g.nodes[far]
        else:
            nf = g.nodes[v]
        dy, dx = nf.y - n.y, nf.x - n.x
        norm = math.hypot(dy, dx) or 1.0
        vecs.append((dy / norm, dx / norm))

    barra_rad = math.radians(cfg.angulo_barra_t_grau)
    stem_min = cfg.angulo_stem_t_min_grau
    stem_max = cfg.angulo_stem_t_max_grau

    for i in range(3):
        for j in range(i + 1, 3):
            dot_ij = vecs[i][0] * vecs[j][0] + vecs[i][1] * vecs[j][1]
            if math.acos(max(-1.0, min(1.0, dot_ij))) < barra_rad:
                continue
            k = 3 - i - j
            stem_ok = True
            for bar in (i, j):
                d = vecs[k][0] * vecs[bar][0] + vecs[k][1] * vecs[bar][1]
                ang = math.degrees(math.acos(max(-1.0, min(1.0, d))))
                if not (stem_min <= ang <= stem_max):
                    stem_ok = False
                    break
            if stem_ok:
                return True
    return False


def _verifica_geometria_x(g: StructureGraph, nid: int, cfg: ConfigCruzamento) -> bool:
    """IV-A: não está no artigo (D22, D26). True se os 4 braços formam dois pares opostos.

    Exige que exista uma partição dos 4 braços em 2 pares onde cada par tem ângulo
    ≥ angulo_par_x_grau (as duas "linhas" que se cruzam). D26: usa arco distancia_arco_geom_px se > 0.
    """
    vizinhos = g.neighbors(nid)
    n = g.nodes[nid]
    d_arco = cfg.distancia_arco_geom_px
    vecs = []
    for v in vizinhos:
        if d_arco > 0:
            far = _caminhar_ate_arco(g, nid, v, float(d_arco))
            nf = g.nodes[far]
        else:
            nf = g.nodes[v]
        dy, dx = nf.y - n.y, nf.x - n.x
        norm = math.hypot(dy, dx) or 1.0
        vecs.append((dy / norm, dx / norm))

    par_rad = math.radians(cfg.angulo_par_x_grau)

    def _ang(i: int, j: int) -> float:
        d = vecs[i][0] * vecs[j][0] + vecs[i][1] * vecs[j][1]
        return math.acos(max(-1.0, min(1.0, d)))

    for a, b, c, d in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
        if _ang(a, b) >= par_rad and _ang(c, d) >= par_rad:
            return True
    return False


# ─── IV-A: classificação de candidatos ───────────────────────────────────────

def classificar_candidatos(
    g: StructureGraph,
    cfg: Optional[ConfigCruzamento] = None,
) -> list[CrossingFeature]:
    """IV-A: classifica nós do grafo em candidatos L, T ou X. D20, D22.

    - grau 2, ângulo ∈ [min, max]: candidato L (D20)
    - grau 3, geometria T válida (se verificar_geometria_t): candidato T (D22)
    - grau 4, dois pares opostos (se verificar_geometria_x): candidato X (D22)
    """
    if cfg is None:
        cfg = ConfigCruzamento()

    min_rad = math.radians(cfg.angulo_L_min_grau)
    max_rad = math.radians(cfg.angulo_L_max_grau)
    candidatos: list[CrossingFeature] = []

    d_arco = cfg.distancia_arco_l_px if cfg is not None else 0

    for nid, nd in g.nodes.items():
        deg = g.degree(nid)
        if deg == 2:
            if d_arco > 0:
                angulo = _angulo_arco_l(g, nid, float(d_arco))
            else:
                angulo = _angulo_entre_vizinhos(g, nid)
            if min_rad <= angulo <= max_rad:
                ori = _orientacao_l(g, nid)
                candidatos.append(CrossingFeature(
                    tipo='L', x=nd.x, y=nd.y,
                    orientacao=ori, confianca=1.0,
                ))
        elif deg == 3:
            if cfg.verificar_geometria_t and not _verifica_geometria_t(g, nid, cfg):
                continue
            ori = _orientacao_t(g, nid)
            candidatos.append(CrossingFeature(
                tipo='T', x=nd.x, y=nd.y,
                orientacao=ori, confianca=1.0,
            ))
        elif deg == 4:
            if cfg.verificar_geometria_x and not _verifica_geometria_x(g, nid, cfg):
                continue
            candidatos.append(CrossingFeature(
                tipo='X', x=nd.x, y=nd.y,
                orientacao=0.0, confianca=1.0,
            ))

    return candidatos


# ─── IV-A: verificação V1 — círculo de varredura ─────────────────────────────

def _amostrar_circulo(
    cx: int, cy: int, raio: int,
    branco: np.ndarray, verde: np.ndarray,
    n_amostras: int = 36,
) -> list[tuple[bool, bool]]:
    """IV-A: V1 (D19). Amostra n_amostras pontos uniformes no círculo centrado em (cx, cy)."""
    h, w = branco.shape
    amostras = []
    for i in range(n_amostras):
        theta = 2.0 * math.pi * i / n_amostras
        px = int(round(cx + raio * math.cos(theta)))
        py = int(round(cy + raio * math.sin(theta)))
        if 0 <= py < h and 0 <= px < w:
            amostras.append((bool(branco[py, px] > 0), bool(verde[py, px] > 0)))
        else:
            amostras.append((False, False))
    return amostras


def _contar_segmentos(amostras: list[tuple[bool, bool]]) -> tuple[int, int]:
    """IV-A: V1 (D19). Conta segmentos brancos e verdes (transições fundo→sinal) no círculo."""
    n = len(amostras)
    n_b = n_v = 0
    for i in range(n):
        if not amostras[(i - 1) % n][0] and amostras[i][0]:
            n_b += 1
        if not amostras[(i - 1) % n][1] and amostras[i][1]:
            n_v += 1
    return n_b, n_v


def _verificacao1_circulo(
    tipo: str,
    cx: int, cy: int,
    raio: int,
    branco: np.ndarray, verde: np.ndarray,
    tolerancia: int = 1,
    n_amostras: int = 36,
) -> tuple[float, int, int]:
    """IV-A: V1 a raio fixo (D19). Retorna (score, n_branco, n_verde).

    Score: 0.0 se verde=0 (sem campo). Caso contrário:
    0.5*(branco_ok) + 0.5*(verde_ok), onde *_ok=1 se |n − esperado| ≤ tolerancia.
    """
    n_esp = {'L': 2, 'T': 3, 'X': 4}.get(tipo, 0)
    amostras = _amostrar_circulo(cx, cy, raio, branco, verde, n_amostras)
    n_b, n_v = _contar_segmentos(amostras)

    if n_v == 0:
        return 0.0, n_b, n_v

    s_b = 0.5 if abs(n_b - n_esp) <= tolerancia else 0.0
    s_v = 0.5 if abs(n_v - n_esp) <= tolerancia else 0.0
    return s_b + s_v, n_b, n_v


def _verificacao1_multi_raio(
    tipo: str,
    cx: int, cy: int,
    raios: list[int],
    branco: np.ndarray, verde: np.ndarray,
    tolerancia: int = 1,
    usar_consenso: bool = True,
    min_raios_consenso: int = 2,
    n_amostras: int = 36,
) -> tuple[float, int, int, int]:
    """IV-A: V1 em múltiplos raios. D21, D23.

    Modo consenso (usar_consenso=True, padrão):
      Conta em quantos raios branco_count e verde_count estão dentro da tolerância.
      Exige ≥ min(min_raios_consenso, len(raios)) concordâncias para cada metade.
      Score = 0.5*(branco_consenso) + 0.5*(verde_consenso).
      Isso evita aceitar cruzamentos que passam em apenas um raio "sortudo".

    Modo legado (usar_consenso=False):
      Escolhe o raio com maior score individual (melhor caso).

    Retorna (score, n_branco, n_verde, raio_usado).
    """
    n_esp = {'L': 2, 'T': 3, 'X': 4}.get(tipo, 0)
    if not raios:
        return 0.0, 0, 0, 5

    if not usar_consenso:
        best_score, best_nb, best_nv, best_raio = -1.0, 0, 0, raios[0]
        for raio in raios:
            s, nb, nv = _verificacao1_circulo(tipo, cx, cy, raio, branco, verde, tolerancia, n_amostras)
            err = abs(nb - n_esp)
            best_err = abs(best_nb - n_esp)
            if s > best_score or (s == best_score and err < best_err):
                best_score, best_nb, best_nv, best_raio = s, nb, nv, raio
        return best_score, best_nb, best_nv, best_raio

    # Consenso: conta raios com contagem correta
    branco_ok_count = 0
    verde_ok_count = 0
    algum_verde = False
    best_nb, best_nv, best_raio = 0, 0, raios[0]
    best_err = float('inf')

    for raio in raios:
        _, nb, nv = _verificacao1_circulo(tipo, cx, cy, raio, branco, verde, tolerancia, n_amostras)
        if nv > 0:
            algum_verde = True
        if abs(nb - n_esp) <= tolerancia:
            branco_ok_count += 1
        if nv > 0 and abs(nv - n_esp) <= tolerancia:
            verde_ok_count += 1
        err = abs(nb - n_esp)
        if err < best_err:
            best_err, best_nb, best_nv, best_raio = err, nb, nv, raio

    if not algum_verde:
        return 0.0, best_nb, best_nv, best_raio

    min_ok = min(min_raios_consenso, len(raios))
    s_b = 0.5 if branco_ok_count >= min_ok else 0.0
    s_v = 0.5 if verde_ok_count >= min_ok else 0.0
    return s_b + s_v, best_nb, best_nv, best_raio


# ─── IV-A: verificação V2 — caminho branco até vizinhos ──────────────────────

def _verificacao2_caminho(
    nid: int,
    g: StructureGraph,
    branco: np.ndarray,
    max_passos: int,
) -> float:
    """IV-A: V2 (D21). BFS em pixels brancos do nó até cada vizinho.

    Segue a linha branca do esqueleto (incluindo curvas), não a reta até o vizinho.
    O nó de partida não precisa ser branco; o BFS parte dele e explora apenas
    pixels brancos adjacentes. Score = fração de vizinhos alcançados.
    """
    nd = g.nodes[nid]
    h, w = branco.shape
    vizinhos = g.neighbors(nid)
    if not vizinhos:
        return 0.0

    if not (0 <= nd.y < h and 0 <= nd.x < w):
        return 0.0

    targets = frozenset(
        (g.nodes[v].y, g.nodes[v].x)
        for v in vizinhos
        if 0 <= g.nodes[v].y < h and 0 <= g.nodes[v].x < w
    )
    if not targets:
        return 0.0

    start = (nd.y, nd.x)
    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int, int]] = deque([(nd.y, nd.x, 0)])
    found: set[tuple[int, int]] = set()

    while queue:
        y, x, steps = queue.popleft()
        if (y, x) in targets:
            found.add((y, x))
            if len(found) == len(targets):
                break
        if steps >= max_passos:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h and 0 <= nx < w
                    and (ny, nx) not in visited
                    and branco[ny, nx] > 0
                ):
                    visited.add((ny, nx))
                    queue.append((ny, nx, steps + 1))

    ok = sum(1 for pos in targets if pos in found)
    return ok / len(vizinhos)


# ─── IV-A: supressão de duplicatas ───────────────────────────────────────────

def _suprimir_duplicatas(
    candidatos: list[CrossingFeature],
    r_fusao: float,
) -> tuple[list[CrossingFeature], list[CrossingFeature]]:
    """Não está no artigo (D21). Remove candidatos redundantes a < r_fusao px entre si.

    Mantém o de maior grau de tipo (X>T>L); em empate, maior confiança.
    Retorna (mantidos, suprimidos).
    """
    if r_fusao <= 0 or len(candidatos) < 2:
        return list(candidatos), []

    tipo_grau = {'L': 2, 'T': 3, 'X': 4}
    n = len(candidatos)
    suprimido = [False] * n

    for i in range(n):
        if suprimido[i]:
            continue
        for j in range(i + 1, n):
            if suprimido[j]:
                continue
            ci, cj = candidatos[i], candidatos[j]
            if math.hypot(ci.x - cj.x, ci.y - cj.y) >= r_fusao:
                continue
            gi = tipo_grau.get(ci.tipo, 0)
            gj = tipo_grau.get(cj.tipo, 0)
            if gi > gj or (gi == gj and ci.confianca >= cj.confianca):
                suprimido[j] = True
            else:
                suprimido[i] = True
                break  # i foi suprimido; passa para próximo i

    mantidos  = [c for c, s in zip(candidatos, suprimido) if not s]
    suprimidos = [c for c, s in zip(candidatos, suprimido) if s]
    return mantidos, suprimidos


# ─── IV-A: verificação pública ───────────────────────────────────────────────

def verificar_cruzamento(
    candidato: CrossingFeature,
    g: StructureGraph,
    branco: np.ndarray,
    verde: np.ndarray,
    cfg: Optional[ConfigCruzamento] = None,
) -> CrossingFeature:
    """IV-A: aplica V1 (multi-raio) e V2 (BFS) e devolve confiança atualizada."""
    if cfg is None:
        cfg = ConfigCruzamento()

    nid = next(
        (nid for nid, nd in g.nodes.items()
         if nd.x == candidato.x and nd.y == candidato.y),
        None,
    )
    if nid is None:
        return CrossingFeature(
            tipo=candidato.tipo, x=candidato.x, y=candidato.y,
            orientacao=candidato.orientacao, confianca=0.0,
        )

    s1, _, _, _ = _verificacao1_multi_raio(
        candidato.tipo, candidato.x, candidato.y,
        cfg.raios_circulo, branco, verde, cfg.tolerancia_segmentos,
        usar_consenso=cfg.usar_consenso_v1,
        min_raios_consenso=cfg.min_raios_consenso,
        n_amostras=cfg.n_amostras_circulo,
    )
    s2 = _verificacao2_caminho(nid, g, branco, cfg.max_passos_v2)
    return CrossingFeature(
        tipo=candidato.tipo, x=candidato.x, y=candidato.y,
        orientacao=candidato.orientacao, confianca=(s1 + s2) / 2.0,
    )


# ─── IV-A: pipeline completo ─────────────────────────────────────────────────

def detectar_cruzamentos(
    g: StructureGraph,
    branco: np.ndarray,
    verde: np.ndarray,
    cfg: Optional[ConfigCruzamento] = None,
    _debug_info: Optional[list] = None,
) -> tuple[list[CrossingFeature], list[CrossingFeature]]:
    """IV-A: classifica, verifica, suprime duplicatas e filtra cruzamentos.

    Retorna (aceitos, rejeitados). Rejeitados incluem candidatos abaixo do limiar
    de confiança e candidatos suprimidos por duplicata.

    _debug_info: se fornecido, cada elemento recebe
    {'tipo', 'x', 'y', 'angulo_grau'(L), 'v1', 'v2', 'n_branco', 'n_verde',
     'raio_usado', 'confianca', 'aceito', 'motivo'}.
    Candidatos suprimidos têm 'motivo' sufixado com ',suprimido'.
    """
    if cfg is None:
        cfg = ConfigCruzamento()

    candidatos = classificar_candidatos(g, cfg)
    aceitos_raw:    list[CrossingFeature] = []
    rejeitados_raw: list[CrossingFeature] = []
    info_map: dict[tuple[str, int, int], dict] = {}

    for c in candidatos:
        nid = next(
            (nid for nid, nd in g.nodes.items()
             if nd.x == c.x and nd.y == c.y),
            None,
        )
        if nid is None:
            rej = CrossingFeature(
                tipo=c.tipo, x=c.x, y=c.y,
                orientacao=c.orientacao, confianca=0.0,
            )
            rejeitados_raw.append(rej)
            if _debug_info is not None:
                entry: dict = {
                    'tipo': c.tipo, 'x': c.x, 'y': c.y,
                    'v1': 0.0, 'v2': 0.0, 'n_branco': 0, 'n_verde': 0,
                    'raio_usado': cfg.raios_circulo[0] if cfg.raios_circulo else 5,
                    'confianca': 0.0, 'aceito': False, 'motivo': 'no_node',
                }
                if c.tipo == 'L':
                    entry['angulo_grau'] = math.degrees(_angulo_entre_vizinhos(g, nid)) if nid else 0.0
                _debug_info.append(entry)
                info_map[(c.tipo, c.x, c.y)] = entry
            continue

        # D29: rejeitar candidatos cujo círculo de verificação sai da grade
        if cfg.verificar_borda and cfg.raios_circulo:
            H, W = branco.shape[:2]
            r_max = max(cfg.raios_circulo)
            if (c.x - r_max < 0 or c.x + r_max >= W
                    or c.y - r_max < 0 or c.y + r_max >= H):
                rej = CrossingFeature(
                    tipo=c.tipo, x=c.x, y=c.y,
                    orientacao=c.orientacao, confianca=0.0,
                )
                rejeitados_raw.append(rej)
                if _debug_info is not None:
                    entry = {
                        'tipo': c.tipo, 'x': c.x, 'y': c.y,
                        'v1': 0.0, 'v2': 0.0, 'n_branco': 0, 'n_verde': 0,
                        'raio_usado': r_max, 'confianca': 0.0,
                        'aceito': False, 'motivo': 'fora_borda',
                    }
                    _debug_info.append(entry)
                    info_map[(c.tipo, c.x, c.y)] = entry
                continue

        s1, n_b, n_v, raio_usado = _verificacao1_multi_raio(
            c.tipo, c.x, c.y,
            cfg.raios_circulo, branco, verde, cfg.tolerancia_segmentos,
            usar_consenso=cfg.usar_consenso_v1,
            min_raios_consenso=cfg.min_raios_consenso,
            n_amostras=cfg.n_amostras_circulo,
        )
        s2 = _verificacao2_caminho(nid, g, branco, cfg.max_passos_v2)
        conf = (s1 + s2) / 2.0
        c_ver = CrossingFeature(
            tipo=c.tipo, x=c.x, y=c.y,
            orientacao=c.orientacao, confianca=conf,
        )

        if _debug_info is not None:
            n_esp = {'L': 2, 'T': 3, 'X': 4}.get(c.tipo, 0)
            motivos = []
            if n_v == 0:
                motivos.append('sem_verde')
            else:
                if abs(n_b - n_esp) > cfg.tolerancia_segmentos:
                    motivos.append(f'branco={n_b}≠{n_esp}')
                if abs(n_v - n_esp) > cfg.tolerancia_segmentos:
                    motivos.append(f'verde={n_v}≠{n_esp}')
            if s2 < 0.5:
                motivos.append(f'V2={s2:.2f}')

            entry = {
                'tipo': c.tipo, 'x': c.x, 'y': c.y,
                'v1': s1, 'v2': s2,
                'n_branco': n_b, 'n_verde': n_v,
                'raio_usado': raio_usado,
                'confianca': conf,
                'aceito': conf >= cfg.limiar_confianca,
                'motivo': ','.join(motivos) if motivos else 'ok',
            }
            if c.tipo == 'L':
                entry['angulo_grau'] = math.degrees(_angulo_entre_vizinhos(g, nid))
            _debug_info.append(entry)
            info_map[(c.tipo, c.x, c.y)] = entry

        if conf >= cfg.limiar_confianca:
            aceitos_raw.append(c_ver)
        else:
            rejeitados_raw.append(c_ver)

    # Supressão de duplicatas entre aceitos
    aceitos, suprimidos = _suprimir_duplicatas(aceitos_raw, cfg.r_fusao_px)

    if _debug_info is not None and suprimidos:
        suprimidos_key = {(c.tipo, c.x, c.y) for c in suprimidos}
        for entry in _debug_info:
            if (entry['tipo'], entry['x'], entry['y']) in suprimidos_key:
                entry['aceito'] = False
                motivo_atual = entry.get('motivo', 'ok')
                entry['motivo'] = (motivo_atual + ',suprimido') if motivo_atual != 'ok' \
                    else 'suprimido'

    rejeitados = rejeitados_raw + suprimidos
    return aceitos, rejeitados
