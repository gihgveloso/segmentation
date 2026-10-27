"""III-C: c(x,y) e colocação de nós no esqueleto."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from features.graph import Node, StructureGraph


@dataclass
class ConfigNos:
    """Parâmetros de colocação de nós. D1, D7."""
    usar_relevo: bool = True   # D1: True=Opção A (relevo float), False=Opção B (binário)
    distancia_minima: float = 2.0  # D7: dist mínima entre nós de crista (pixels de grade)


# ─── c(x,y) ──────────────────────────────────────────────────────────────────

def neighbor_count(
    relevo: Optional[np.ndarray],
    skeleton: np.ndarray,
    config: Optional[ConfigNos] = None,
) -> np.ndarray:
    """III-C: c(x,y) para todos os pixels do esqueleto; 0 fora.

    Opção A (config.usar_relevo=True): c = nº de 8-vizinhos com relevo >= central.
    Opção B (usar_relevo=False): c = nº de 8-vizinhos que são pixels do esqueleto.
    Retorna array uint8 com shape igual a skeleton.
    """
    if config is None:
        config = ConfigNos()

    if config.usar_relevo and relevo is not None:
        return _c_relevo(relevo, skeleton)
    return _c_binario(skeleton)


def _c_relevo(relevo: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """III-C: D1 Opção A. c(x,y) via mapa de relevo; 8-vizinhança pura."""
    r = relevo.astype(np.float32)
    pad = np.pad(r, 1, mode='constant', constant_values=0.0)
    vizinhos = np.stack([
        pad[:-2, 1:-1],  # N
        pad[2:,  1:-1],  # S
        pad[1:-1, 2:],   # E
        pad[1:-1, :-2],  # W
        pad[:-2, 2:],    # NE
        pad[2:,  2:],    # SE
        pad[:-2, :-2],   # NW
        pad[2:,  :-2],   # SW
    ])
    c = (vizinhos >= r[np.newaxis]).sum(axis=0).astype(np.uint8)
    return np.where(skeleton, c, np.uint8(0))


def _c_binario(skeleton: np.ndarray) -> np.ndarray:
    """III-C: D1 Opção B. c(x,y) = nº de vizinhos-8 que são pixels do esqueleto; 8-viz pura."""
    s = skeleton.astype(np.uint8)
    pad = np.pad(s, 1, mode='constant', constant_values=0)
    vizinhos = np.stack([
        pad[:-2, 1:-1],
        pad[2:,  1:-1],
        pad[1:-1, 2:],
        pad[1:-1, :-2],
        pad[:-2, 2:],
        pad[2:,  2:],
        pad[:-2, :-2],
        pad[2:,  :-2],
    ])
    c = vizinhos.sum(axis=0).astype(np.uint8)
    return np.where(skeleton, c, np.uint8(0))


# ─── Colocação de nós (Etapa 2) ──────────────────────────────────────────────

def place_nodes(
    skeleton: np.ndarray,
    relevo: Optional[np.ndarray],
    c_xy: np.ndarray,
    config: Optional[ConfigNos] = None,
) -> StructureGraph:
    """III-C: coloca nós no esqueleto.

    Regras (D3, D7):
    1. c_xy==0 (picos/junções): nó incondicional.
    2. c_binario==1 (extremidades topológicas): nó incondicional — garante "extremos cobertos".
    3. c_binario==2 (pixels de passagem): nó guloso por relevo decrescente, dist >= dist_min.
    """
    if config is None:
        config = ConfigNos()

    usar_relevo = config.usar_relevo and relevo is not None
    gray_map = relevo.astype(np.float32) if usar_relevo else np.ones(skeleton.shape, np.float32)
    dist_min = config.distancia_minima

    # c_binario sempre calculado para identificar extremidades (D7)
    c_bin = _c_binario(skeleton)

    g = StructureGraph()
    nid = 0

    # Mapa de proximidade: True onde algum nó já foi colocado dentro de dist_min
    ocupado = np.zeros(skeleton.shape, dtype=bool)

    def _marcar(gy: int, gx: int) -> None:
        r = int(np.ceil(dist_min))
        h, w = skeleton.shape
        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
        ys_idx, xs_idx = np.ogrid[y0:y1, x0:x1]
        mask = (ys_idx - gy) ** 2 + (xs_idx - gx) ** 2 < dist_min ** 2
        ocupado[y0:y1, x0:x1] |= mask

    def _add(gy: int, gx: int) -> None:
        nonlocal nid
        g.add_node(Node(id=nid, x=int(gx), y=int(gy), gray=float(gray_map[gy, gx])))
        _marcar(gy, gx)
        nid += 1

    # 1. Picos (c_xy=0): incondicional
    for gy, gx in zip(*np.where(skeleton & (c_xy == 0))):
        _add(int(gy), int(gx))

    # 2. Extremidades binárias (c_bin=1): incondicional (D7)
    for gy, gx in zip(*np.where(skeleton & (c_bin == 1))):
        if not ocupado[gy, gx]:
            _add(int(gy), int(gx))

    # 3. Pixels de crista/passagem: guloso por relevo decrescente
    # D1 Opção A (usar_relevo=True): usa c_xy (= c_relevo) ≥ 1, que cobre c_bin∈{1,2,3+}.
    #   O critério anterior c_bin==2 excluía pixels com c_bin≥3 (junções binárias), criando
    #   lacunas de cobertura em trechos curvos onde múltiplos ramos se tocam diagonalmente.
    #   Com c_xy≥1 os picos (c_xy=0, já colocados no passo 1) ficam de fora; `ocupado`
    #   impede duplicatas dos passos 1 e 2.
    # D1 Opção B (usar_relevo=False): mantém c_bin==2 como fallback (sem mapa de relevo).
    if usar_relevo:
        mask_passagem = skeleton & (c_xy >= 1)
    else:
        mask_passagem = skeleton & (c_bin == 2)
    ys_p, xs_p = np.where(mask_passagem)
    if len(ys_p) > 0:
        grays = gray_map[ys_p, xs_p]
        for i in np.argsort(-grays):
            gy, gx = int(ys_p[i]), int(xs_p[i])
            if not ocupado[gy, gx]:
                _add(gy, gx)

    return g
