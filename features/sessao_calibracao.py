"""Sessão de calibração de cor sem dependência de GUI.

Gerencia o fluxo: fase inicial → estimação → rodadas corretivas.
Sem janela, sem cv2.imshow, sem cv2.namedWindow.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from features.calibracao_cor import (
    Aviso,
    ParametrosEstimacao,
    avaliar_calibracao,
    estimar_calibracao,
    montar_json,
    reamostrar,
    salvar_json,
    verificar_sanidade,
)

# ---------------------------------------------------------------------------
# Constantes — provisório, Rodada 6 (D47)
# ---------------------------------------------------------------------------

MAX_RODADAS_CORRETIVAS: int = 1       # provisório, Rodada 6 (D47)
LIMIAR_SALTO_VERDE_PP: float = 5.0    # provisório, Rodada 6 (D47)
LIMIAR_H_DESVIO: int = 12             # provisório, Rodada 6 (D47)
LIMIAR_FRAC_VERDE_LINHA: float = 0.50 # provisório, Rodada 6 (D47)


# ---------------------------------------------------------------------------
# Exceções
# ---------------------------------------------------------------------------

class SessaoNaoProntaError(RuntimeError):
    """Estimação solicitada antes de ter specs suficientes."""


class LimiteRodadasError(RuntimeError):
    """Tentativa de rodada corretiva extra sem confirmar_extra=True."""


class SaltoVerdeGrandeError(RuntimeError):
    """Δ(frV) > LIMIAR_SALTO_VERDE_PP; aguarda confirmar_salto() ou rejeitar_salto()."""

    def __init__(self, delta_pp: float, calibracao_proposta: dict):
        super().__init__(
            f"Δ(fracao_verde)={delta_pp:.1f}pp > {LIMIAR_SALTO_VERDE_PP}pp. "
            "Chame confirmar_salto() para aceitar ou rejeitar_salto() para cancelar. "
            "Aviso: alargamento pode vazar para parede/rede/robôs."
        )
        self.delta_pp = delta_pp
        self.calibracao_proposta = calibracao_proposta


class SemPropostaError(RuntimeError):
    """confirmar_salto/rejeitar_salto chamado sem proposta pendente."""


class SalvarSemConfirmacaoError(RuntimeError):
    """salvar() chamado com confirmado=False."""


# ---------------------------------------------------------------------------
# SessaoCalibracao
# ---------------------------------------------------------------------------

class SessaoCalibracao:
    """Sessão de calibração de cor sem dependência de GUI.

    Fluxo: adicionar_carpete/adicionar_linha (>= 4 carpet + >= 1 linha)
    → estimar() → (rodadas corretivas opcionais) → salvar().
    """

    def __init__(
        self,
        imagens_por_nome: Dict[str, np.ndarray],
        params: Optional[ParametrosEstimacao] = None,
    ):
        self._imagens = imagens_por_nome
        self._params = params if params is not None else ParametrosEstimacao()
        self._specs_carp: List[dict] = []
        self._specs_lin: List[dict] = []
        self._historico: List[dict] = []
        self._n_rodadas: int = 0
        self._calibracao: Optional[dict] = None
        self._fracao_verde_atual: Optional[float] = None
        self._proposta: Optional[dict] = None

    # ---- estado ----

    def pronto(self) -> bool:
        """True se há specs suficientes para estimar (>= 4 carpete, >= 1 linha)."""
        return len(self._specs_carp) >= 4 and len(self._specs_lin) >= 1

    @property
    def calibracao(self) -> Optional[dict]:
        return self._calibracao

    @property
    def n_rodadas(self) -> int:
        return self._n_rodadas

    # ---- undo / reset ----

    def _snapshot(self, tipo: str = 'spec') -> dict:
        return {
            'tipo': tipo,
            'specs_carp': copy.deepcopy(self._specs_carp),
            'specs_lin': copy.deepcopy(self._specs_lin),
            'n_rodadas': self._n_rodadas,
            'calibracao': copy.deepcopy(self._calibracao),
            'fracao_verde_atual': self._fracao_verde_atual,
        }

    def _push(self, tipo: str = 'spec') -> None:
        self._historico.append(self._snapshot(tipo))

    def _restaurar(self, s: dict) -> None:
        self._specs_carp = s['specs_carp']
        self._specs_lin = s['specs_lin']
        self._n_rodadas = s['n_rodadas']
        self._calibracao = s['calibracao']
        self._fracao_verde_atual = s['fracao_verde_atual']
        self._proposta = None

    def desfazer(self) -> bool:
        """Desfaz a última ação (adição de spec ou estimação).

        Returns True se havia algo para desfazer.
        """
        if not self._historico:
            return False
        self._restaurar(self._historico.pop())
        return True

    def reiniciar(self) -> None:
        """Reinicia a sessão por completo."""
        self._specs_carp = []
        self._specs_lin = []
        self._historico = []
        self._n_rodadas = 0
        self._calibracao = None
        self._fracao_verde_atual = None
        self._proposta = None

    # ---- adição de amostras ----

    def adicionar_carpete(self, spec: dict) -> List[str]:
        """Adiciona spec de carpete e retorna lista de avisos suaves.

        Avisos: (b) H_med difere >LIMIAR_H_DESVIO de existentes;
                (c) todas as amostras em terço único de imagem única.
        """
        avisos: List[str] = []

        # aviso (b): H_med difere de existentes
        if self._specs_carp:
            pxs_novos = reamostrar([spec], self._imagens)
            h_med_novo = float(np.median(pxs_novos[:, 0]))
            pxs_exist = reamostrar(self._specs_carp, self._imagens)
            h_med_exist = float(np.median(pxs_exist[:, 0]))
            delta_h = abs(h_med_novo - h_med_exist)
            if delta_h > LIMIAR_H_DESVIO:
                avisos.append(
                    f"AVISO_H_DESVIO: H_med do novo patch ({h_med_novo:.0f}) difere "
                    f"{delta_h:.0f} unidades do conjunto existente ({h_med_exist:.0f}) "
                    f"> {LIMIAR_H_DESVIO}. Pode indicar iluminação inconsistente."
                )

        self._push()
        self._specs_carp.append(spec)

        # aviso (c): todas as amostras em terço único de imagem única (mín. 2 specs)
        if len(self._specs_carp) >= 2:
            aviso_teco = self._checar_teco_unico()
            if aviso_teco:
                avisos.append(aviso_teco)

        return avisos

    def adicionar_linha(self, spec: dict) -> List[str]:
        """Adiciona spec de linha e retorna lista de avisos suaves.

        Aviso (a): >LIMIAR_FRAC_VERDE_LINHA dos pixels da linha na caixa verde.
        """
        avisos: List[str] = []

        # aviso (a): pixels na caixa verde
        if self._calibracao is not None:
            pxs = reamostrar([spec], self._imagens)
            frac_verde = _frac_em_caixa(pxs, self._calibracao['verde'])
            if frac_verde > LIMIAR_FRAC_VERDE_LINHA:
                avisos.append(
                    f"AVISO_LINHA_NO_VERDE: {frac_verde * 100:.0f}% dos pixels da linha "
                    f"estão na caixa verde (limiar={LIMIAR_FRAC_VERDE_LINHA * 100:.0f}%). "
                    "Verifique se o clique foi em uma linha real."
                )

        self._push()
        self._specs_lin.append(spec)
        return avisos

    def _checar_teco_unico(self) -> Optional[str]:
        """Aviso (c): todas as amostras de carpete em um único terço de imagem única."""
        imagens_usadas = {s['imagem'] for s in self._specs_carp}
        if len(imagens_usadas) != 1:
            return None

        nome_img = next(iter(imagens_usadas))
        h_img = self._imagens[nome_img].shape[0]

        ys = []
        for s in self._specs_carp:
            if s['tipo'] == 'patch':
                ys.append(s['y'])
            elif s['tipo'] == 'poligono':
                ys.append(float(np.mean([p[1] for p in s['pontos']])))

        if not ys:
            return None

        teco_min = int(min(ys) * 3 / h_img)
        teco_max = int(max(ys) * 3 / h_img)

        if teco_min == teco_max:
            nomes_tecos = ['superior', 'médio', 'inferior']
            nome_teco = nomes_tecos[min(teco_min, 2)]
            return (
                f"AVISO_TECO_UNICO: todas as amostras de carpete estão no terço "
                f"{nome_teco} de '{nome_img}'. "
                "Distribua os cliques para melhor cobertura."
            )
        return None

    # ---- estimação ----

    def estimar(self, confirmar_extra: bool = False) -> Tuple[dict, List[Aviso]]:
        """Estima calibração a partir das specs acumuladas.

        Na primeira chamada: fase inicial (n_rodadas não incrementa).
        Nas seguintes: rodada corretiva (máx MAX_RODADAS_CORRETIVAS por padrão).

        Raises
        ------
        SessaoNaoProntaError   se specs insuficientes
        LimiteRodadasError     se n_rodadas >= MAX e confirmar_extra=False
        SaltoVerdeGrandeError  se Δ(frV) > LIMIAR_SALTO_VERDE_PP
        """
        if not self.pronto():
            raise SessaoNaoProntaError(
                f"Specs insuficientes: carpete={len(self._specs_carp)}/4, "
                f"linha={len(self._specs_lin)}/1."
            )

        is_corretiva = (self._calibracao is not None)

        if is_corretiva and self._n_rodadas >= MAX_RODADAS_CORRETIVAS and not confirmar_extra:
            raise LimiteRodadasError(
                f"Já foram feitas {self._n_rodadas} rodadas corretivas "
                f"(máx={MAX_RODADAS_CORRETIVAS}). "
                "Use confirmar_extra=True para continuar. "
                "Aviso: alargamento pode vazar para parede/rede/robôs."
            )

        carpete_hsv = reamostrar(self._specs_carp, self._imagens)
        linha_hsv = reamostrar(self._specs_lin, self._imagens)
        nova_calib = estimar_calibracao(carpete_hsv, linha_hsv, self._params)

        avisos_total, fracao_verde_nova = self._avaliar_e_coletar(
            nova_calib, carpete_hsv, linha_hsv
        )

        # growth guard
        if self._fracao_verde_atual is not None:
            delta_pp = (fracao_verde_nova - self._fracao_verde_atual) * 100.0
            if delta_pp > LIMIAR_SALTO_VERDE_PP:
                self._proposta = {
                    'calibracao': nova_calib,
                    'fracao_verde': fracao_verde_nova,
                    'avisos': avisos_total,
                }
                raise SaltoVerdeGrandeError(delta_pp, nova_calib)

        self._push(tipo='estimacao')
        self._calibracao = nova_calib
        self._fracao_verde_atual = fracao_verde_nova
        if is_corretiva:
            self._n_rodadas += 1

        return nova_calib, avisos_total

    def confirmar_salto(self) -> Tuple[dict, List[Aviso]]:
        """Aceita a proposta pendente de grande salto de verde."""
        if self._proposta is None:
            raise SemPropostaError(
                "Nenhuma proposta pendente. Chame estimar() antes."
            )
        is_corretiva = (self._calibracao is not None)
        nova_calib = self._proposta['calibracao']
        fracao_verde_nova = self._proposta['fracao_verde']
        avisos_total = self._proposta['avisos']

        self._push(tipo='estimacao')
        self._calibracao = nova_calib
        self._fracao_verde_atual = fracao_verde_nova
        if is_corretiva:
            self._n_rodadas += 1
        self._proposta = None

        return nova_calib, avisos_total

    def rejeitar_salto(self) -> None:
        """Rejeita a proposta pendente; estado não muda."""
        if self._proposta is None:
            raise SemPropostaError("Nenhuma proposta pendente.")
        self._proposta = None

    def _avaliar_e_coletar(
        self,
        calibracao: dict,
        carpete_hsv: np.ndarray,
        linha_hsv: np.ndarray,
    ) -> Tuple[List[Aviso], float]:
        """Avalia calibracao em todas as imagens e agrega avisos e frV média."""
        avisos_total: List[Aviso] = []
        fracao_verde_soma = 0.0
        n = len(self._imagens)
        for frame_bgr in self._imagens.values():
            resultado = avaliar_calibracao(
                frame_bgr, calibracao,
                carpete=carpete_hsv,
                linha=linha_hsv,
            )
            avisos_total.extend(verificar_sanidade(resultado))
            fracao_verde_soma += resultado.fracao_verde
        fracao_verde_media = fracao_verde_soma / n if n > 0 else 0.0
        return avisos_total, fracao_verde_media

    # ---- salvar ----

    def salvar(
        self,
        path: str,
        *,
        confirmado: bool = False,
        sobrescrever: bool = False,
    ) -> None:
        """Salva calibração em JSON.

        Raises SalvarSemConfirmacaoError se confirmado=False.
        Raises SessaoNaoProntaError se estimar() nunca foi chamado.
        Raises FileExistsError se path existe e sobrescrever=False.
        """
        if not confirmado:
            raise SalvarSemConfirmacaoError(
                "Use confirmado=True para confirmar que verificou a calibração."
            )
        if self._calibracao is None:
            raise SessaoNaoProntaError(
                "Chame estimar() antes de salvar()."
            )

        amostras = {
            'carpete': copy.deepcopy(self._specs_carp),
            'linha': copy.deepcopy(self._specs_lin),
            'negativos': [],
        }
        params_dict = {k: v for k, v in vars(self._params).items()}
        dados = montar_json(
            self._calibracao,
            amostras=amostras,
            parametros_estimacao=params_dict,
            validada_visualmente=False,
        )
        salvar_json(path, dados, sobrescrever=sobrescrever)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _frac_em_caixa(pixels_hsv: np.ndarray, caixa: dict) -> float:
    """Fração de pixels dentro da caixa HSV (inclusive)."""
    H = pixels_hsv[:, 0].astype(int)
    S = pixels_hsv[:, 1].astype(int)
    V = pixels_hsv[:, 2].astype(int)
    return float(np.mean(
        (H >= caixa['h_min']) & (H <= caixa['h_max']) &
        (S >= caixa['s_min']) & (S <= caixa['s_max']) &
        (V >= caixa['v_min']) & (V <= caixa['v_max'])
    ))
