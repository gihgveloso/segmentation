"""IV-B: undistort e projeção no chão. Seção IV do artigo Schulz 2009.

Calibração de câmera é OPCIONAL. Sem calibração, todas as funções retornam
coordenadas na grade subamostrada (ou imagem original) sem modificação.
O contrato de entrada NÃO inclui matriz de câmera — não existe calibração
disponível no pipeline atual (Pixel_Segment não a expõe). A projeção no chão
fica atrás da flag `projetar_no_chao=False` em ConfigLinhas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ConfigCamera:
    """Parâmetros opcionais de calibração de câmera.

    Todos os campos são None por padrão → sem calibração → identidade.
    """
    camera_matrix: Optional[np.ndarray] = None    # shape (3,3)
    dist_coeffs: Optional[np.ndarray] = None      # shape (1,5) ou (1,4)
    altura_m: float = 0.0    # altura da câmera em metros
    pitch_rad: float = 0.0   # inclinação em radianos
    roll_rad: float = 0.0    # rotação em radianos


def undistort_pontos(
    pts_grade: np.ndarray,
    fator_escala: int,
    cfg_cam: Optional[ConfigCamera] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IV-B: remove distorção de lente dos pontos da grade.

    Args:
        pts_grade: array (N, 2) de (x, y) na grade subamostrada.
        fator_escala: fator de subsampling (tipicamente 8).
        cfg_cam: parâmetros de câmera; None ou sem matrix → identidade.

    Returns:
        (pts_grade_ud, pts_orig_ud):
          - pts_grade_ud: (N,2) coordenadas undistorted na grade
          - pts_orig_ud : (N,2) coordenadas undistorted na imagem original
    """
    if pts_grade.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    pts_orig = pts_grade.astype(np.float32) * fator_escala

    if cfg_cam is None or cfg_cam.camera_matrix is None:
        return pts_grade.astype(np.float32), pts_orig

    import cv2
    pts_orig_ud = cv2.undistortPoints(
        pts_orig.reshape(-1, 1, 2),
        cfg_cam.camera_matrix,
        cfg_cam.dist_coeffs,
        P=cfg_cam.camera_matrix,
    ).reshape(-1, 2)
    pts_grade_ud = pts_orig_ud / fator_escala
    return pts_grade_ud, pts_orig_ud


def projetar_no_chao(
    pts_grade: np.ndarray,
    fator_escala: int,
    cfg_cam: ConfigCamera,
) -> np.ndarray:
    """Projeção perspectiva inversa dos pontos da grade no plano do chão.

    Só chamada quando ConfigLinhas.projetar_no_chao=True E cfg_cam.camera_matrix
    não é None. Retorna coordenadas (x, y) em metros no referencial da câmera.
    """
    if cfg_cam.camera_matrix is None:
        raise ValueError("projetar_no_chao requer camera_matrix")

    pts_orig = pts_grade.astype(np.float64) * fator_escala
    fx = cfg_cam.camera_matrix[0, 0]
    fy = cfg_cam.camera_matrix[1, 1]
    cx = cfg_cam.camera_matrix[0, 2]
    cy = cfg_cam.camera_matrix[1, 2]
    h  = cfg_cam.altura_m
    cp, sp = np.cos(cfg_cam.pitch_rad), np.sin(cfg_cam.pitch_rad)
    cr, sr = np.cos(cfg_cam.roll_rad),  np.sin(cfg_cam.roll_rad)

    resultado = np.zeros((len(pts_orig), 2), dtype=np.float64)
    for i, (px, py) in enumerate(pts_orig):
        # raio normalizado
        xn = (px - cx) / fx
        yn = (py - cy) / fy
        # projeção simples com pitch (roll ignorado nesta implementação básica)
        denom = cp - yn * sp
        if abs(denom) < 1e-9:
            resultado[i] = [0.0, 0.0]
            continue
        Xw = h * xn / denom
        Zw = h * sp / denom + h * cp / denom  # simplificado
        resultado[i] = [Xw, Zw]
    return resultado
