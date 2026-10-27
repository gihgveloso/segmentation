"""III-C + IV: ponto de entrada único da pipeline de features.

Interface para o futuro filtro de partículas: chamar extrair_features com o
dict devolvido por Pixel_Segment.processar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from features.types import CrossingFeature, LineFeature
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao
from features.crossings import detectar_cruzamentos, ConfigCruzamento
from features.lines import extrair_linhas, ConfigLinhas

_FATOR_ESCALA = 8   # fator fixo do Pixel_Segment (contrato_entrada.md)


@dataclass
class ConfigFeatures:
    """Parâmetros para todas as etapas da pipeline de features."""
    grafo:  ConfigGrafo          = field(default_factory=ConfigGrafo)
    simp:   ConfigSimplificacao  = field(default_factory=ConfigSimplificacao)
    cruz:   ConfigCruzamento     = field(default_factory=ConfigCruzamento)
    linhas: ConfigLinhas         = field(default_factory=ConfigLinhas)


@dataclass
class FeaturesResult:
    """Resultado da pipeline de features.

    Campos:
        crossings      — lista de CrossingFeature (L/T/X) na grade subamostrada.
        lines          — lista de LineFeature na grade subamostrada.
        tamanho_grade  — (H, W) da grade subamostrada.
        fator_escala   — fator de downsampling Pixel_Segment → grade (sempre 8).
    """
    crossings:     list[CrossingFeature]
    lines:         list[LineFeature]
    tamanho_grade: tuple[int, int]
    fator_escala:  int


def extrair_features(
    resultado_processar: dict,
    cfg: Optional[ConfigFeatures] = None,
) -> FeaturesResult:
    """III-C + IV-A + IV-B: extrai CrossingFeature e LineFeature.

    Encadeia:
      construir_grafo (III-C, Etapas 1-5)
      → extrair_linhas (IV-B, grafo pré-simplificação)
      → simplificar   (IV-A, Etapa 6)
      → detectar_cruzamentos (IV-A, Etapa 7)

    Parâmetro:
        resultado_processar: dict devolvido por Pixel_Segment.processar.
        cfg: config unificado; usa defaults se None.
    """
    if cfg is None:
        cfg = ConfigFeatures()

    skeleton = resultado_processar['skeleton_img'].astype(bool)
    relevo   = resultado_processar['mascara_relevo'].astype(np.float32)
    branco   = resultado_processar['white_mask_resized'].astype(np.uint8)
    verde    = resultado_processar['green_mask_resized'].astype(np.uint8)

    H, W = skeleton.shape

    if skeleton.sum() == 0:
        return FeaturesResult(
            crossings=[], lines=[],
            tamanho_grade=(H, W),
            fator_escala=_FATOR_ESCALA,
        )

    # Etapas 1-5: grafo conectado
    g = construir_grafo(skeleton, relevo, cfg.grafo, branco=branco)

    # IV-B: linhas no grafo PRÉ-simplificação (D31)
    lines = extrair_linhas(
        g, fator_escala=_FATOR_ESCALA,
        cfg=cfg.linhas, branco=branco, verde=verde,
    )

    # IV-A: simplificação (Etapa 6) + cruzamentos (Etapa 7)
    simplificar(g, skeleton, relevo, cfg.simp)
    crossings, _ = detectar_cruzamentos(g, branco, verde, cfg.cruz)

    return FeaturesResult(
        crossings=crossings,
        lines=lines,
        tamanho_grade=(H, W),
        fator_escala=_FATOR_ESCALA,
    )
