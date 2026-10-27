"""Tipos de saída para o filtro de partículas (Seção IV)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrossingFeature:
    """IV-A: cruzamento L, T ou X detectado no esqueleto."""
    tipo: str        # 'L', 'T' ou 'X'
    x: int           # coluna na grade subamostrada
    y: int           # linha na grade subamostrada
    orientacao: float   # ângulo em radianos da aresta principal
    confianca: float    # 0.0–1.0


@dataclass
class LineFeature:
    """IV-B: segmento de linha ajustado por split-and-merge."""
    x1: float        # extremo 1, coluna (grade)
    y1: float        # extremo 1, linha (grade)
    x2: float        # extremo 2, coluna (grade)
    y2: float        # extremo 2, linha (grade)
    x1_orig: float   # extremo 1, coluna (imagem original, fator 8)
    y1_orig: float   # extremo 1, linha (imagem original)
    x2_orig: float   # extremo 2, coluna (imagem original)
    y2_orig: float   # extremo 2, linha (imagem original)
    distancia: float   # comprimento em pixels de grade
    angulo: float      # ângulo em radianos
    n_nos: int         # número de nós do componente ajustado
    confianca: float   # fração de nós a < desvio_max/2 da reta
    d_topo_celulas: Optional[float] = field(default=None, compare=False)
    # distância vertical do centro da linha ao y_topo_verde (negativa = acima do verde)
