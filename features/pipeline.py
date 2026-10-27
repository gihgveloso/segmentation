"""Pipeline: constrói StructureGraph a partir de skeleton + relevo (Etapas 1–5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from features.graph import StructureGraph
from features.nodes import ConfigNos, neighbor_count, place_nodes
from features.connections import (
    ConfigConexao,
    ConfigSubdivisao,
    connect_nodes,
    conectar_candidatas,
    dividir_grau_zero,
    esticar_extremos,
)


@dataclass
class ConfigGrafo:
    """Parâmetros para construção do StructureGraph (Etapas 1–5)."""
    nos: ConfigNos = field(default_factory=ConfigNos)
    sub: ConfigSubdivisao = field(default_factory=ConfigSubdivisao)
    con: ConfigConexao = field(default_factory=ConfigConexao)


def construir_grafo(
    skeleton: np.ndarray,
    relevo: np.ndarray,
    cfg: Optional[ConfigGrafo] = None,
    branco: Optional[np.ndarray] = None,
) -> StructureGraph:
    """III-C: encadeia Etapas 1–5 e retorna o StructureGraph.

    1. c_xy = neighbor_count(relevo, skeleton, cfg.nos)
    2. g = place_nodes(skeleton, relevo, c_xy, cfg.nos)
    3. connect_nodes(g, skeleton)
    4a. esticar_extremos(g, skeleton, relevo)
    4b. dividir_grau_zero(g, skeleton, relevo, cfg_sub=cfg.sub)
    5. conectar_candidatas(g, relevo, cfg.con, branco=branco)

    `branco`: máscara binária de pixels brancos (grade subamostrada).
      Se fornecida, ativa o filtro D37 em conectar_candidatas: candidatas com
      comprimento > dist_min_filtro e frac_branco < suporte_candidata_min são rejeitadas.
    """
    if cfg is None:
        cfg = ConfigGrafo()

    c_xy = neighbor_count(relevo, skeleton, cfg.nos)
    g = place_nodes(skeleton, relevo, c_xy, cfg.nos)
    connect_nodes(g, skeleton)
    esticar_extremos(g, skeleton, relevo)
    dividir_grau_zero(g, skeleton, relevo, cfg_sub=cfg.sub)
    conectar_candidatas(g, relevo, cfg.con, branco=branco)
    return g
