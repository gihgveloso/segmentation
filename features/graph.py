"""III-C: nós e grafo de estrutura do esqueleto."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class Node:
    """Nó do grafo de estrutura. Coordenadas (x, y) na grade subamostrada."""
    id: int
    x: int     # coluna
    y: int     # linha
    gray: float  # mascara_relevo[y, x] no momento da criação/movimentação (D5)


class StructureGraph:
    """Grafo não-dirigido de nós do esqueleto. III-C."""

    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._adj: Dict[int, Set[int]] = {}

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        self._adj.setdefault(node.id, set())

    def add_edge(self, id_a: int, id_b: int) -> None:
        """Adiciona aresta não-dirigida; idempotente."""
        self._adj[id_a].add(id_b)
        self._adj[id_b].add(id_a)

    def remove_edge(self, id_a: int, id_b: int) -> None:
        """Remove aresta não-dirigida; silencioso se não existir."""
        self._adj.get(id_a, set()).discard(id_b)
        self._adj.get(id_b, set()).discard(id_a)

    def remove_node(self, node_id: int) -> None:
        for viz in list(self._adj.get(node_id, [])):
            self._adj[viz].discard(node_id)
        self._adj.pop(node_id, None)
        self._nodes.pop(node_id, None)

    def degree(self, node_id: int) -> int:
        return len(self._adj.get(node_id, set()))

    def neighbors(self, node_id: int) -> List[int]:
        return list(self._adj.get(node_id, set()))

    @property
    def nodes(self) -> Dict[int, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> List[Tuple[int, int]]:
        seen: Set[Tuple[int, int]] = set()
        result: List[Tuple[int, int]] = []
        for a, vizinhos in self._adj.items():
            for b in vizinhos:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result
