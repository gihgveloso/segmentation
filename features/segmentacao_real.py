"""Subclasse de Pixel_Segment configurada para imagens reais (calibração externa).

O caminho padrão Webots (Pixel_Segment direto) NÃO é alterado.
Use PixelSegmentReal SOMENTE quando a calibração real for necessária.

Uso:
    from features.segmentacao_real import PixelSegmentReal
    ps = PixelSegmentReal.from_json('calibracao/real_provisoria_n2.json',
                                     'green_pixels.csv')
    res = ps.processar(img)   # branco revertido ao sair, seguro para uso misto
"""
from __future__ import annotations
import contextlib
import json
import os
import sys

import io
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

with contextlib.redirect_stdout(io.StringIO()):
    from main import Pixel_Segment  # noqa: E402


@contextlib.contextmanager
def _patch_branco(lower, up):
    """Monkeypatcha def_white_threshold temporariamente; reverte ao sair."""
    import def_white_threshold as _m
    _old_lower = _m.lower.copy() if isinstance(_m.lower, np.ndarray) else _m.lower
    _old_up    = _m.up.copy()    if isinstance(_m.up,    np.ndarray) else _m.up
    _m.lower = np.array(lower, dtype=np.int32)
    _m.up    = np.array(up,    dtype=np.int32)
    try:
        yield
    finally:
        _m.lower = _old_lower
        _m.up    = _old_up


class PixelSegmentReal(Pixel_Segment):
    """Pixel_Segment com calibração para imagens reais (campo real, iluminação natural).

    Branco e verde são configurados via dicionário de calibração; o fator de
    subamostração (8) é herdado do pai sem alteração.

    Parameters
    ----------
    calibracao : dict
        Dicionário com chaves 'branco' e 'verde', cada um com
        h_min/h_max/s_min/s_max/v_min/v_max.
    lut_csv : str
        Caminho para green_pixels.csv (lido pelo pai; o LUT verde é sobrescrito
        por _criar_lut_verde antes de ser usado).
    """

    def __init__(self, calibracao: dict, lut_csv: str,
                 tpix: int = 20, trow: int = 10000, twin: int = 3000) -> None:
        v = calibracao['verde']
        b = calibracao['branco']

        # Deve ser definido ANTES de super().__init__, que chama _criar_lut_verde.
        self._calib_verde = v
        self._white_lower = [b['h_min'], b['s_min'], b['v_min']]
        self._white_upper = [b['h_max'], b['s_max'], b['v_max']]

        super().__init__(lut_csv, tpix, trow, twin)

    def _criar_lut_verde(self) -> np.ndarray:
        """III-A: LUT verde construída a partir de faixas HSV da calibração."""
        v = self._calib_verde
        lut = np.zeros((180, 256, 256), dtype=np.uint8)
        for h in range(v['h_min'], v['h_max'] + 1):
            lut[h, v['s_min']:v['s_max'] + 1, v['v_min']:v['v_max'] + 1] = 255
        return lut

    def processar(self, frame: np.ndarray) -> dict:
        """III-A/III-B: pipeline com calibração real; branco revertido ao sair."""
        with _patch_branco(self._white_lower, self._white_upper):
            return super().processar(frame)

    @classmethod
    def from_json(cls, path: str, lut_csv: str, **kw) -> 'PixelSegmentReal':
        """Instancia a partir de arquivo de calibração JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            calib = json.load(f)
        return cls(calib, lut_csv, **kw)
