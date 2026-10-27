"""Visualizador de debug para a Etapa 7 (IV-A: cruzamentos L/T/X)."""
from __future__ import annotations

import os

import cv2
import numpy as np

from main import Pixel_Segment
from features.pipeline import construir_grafo, ConfigGrafo
from features.simplify import simplificar, ConfigSimplificacao
from features.crossings import detectar_cruzamentos, ConfigCruzamento


_COR_TIPO = {
    'L': (255, 120,  30),   # laranja
    'T': ( 50, 220,  50),   # verde
    'X': (  0, 220, 255),   # ciano
}
_COR_REJEITADO = (100, 100, 100)


def processar_imagem(
    img_path: str,
    ps: Pixel_Segment,
    cfg_grafo: ConfigGrafo,
    cfg_simp: ConfigSimplificacao,
    cfg_cruz: ConfigCruzamento,
    scale: int = 8,
) -> None:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Não foi possível carregar {img_path}")
        return

    res = ps.processar(img)
    skeleton = res['skeleton_img'].astype(bool)
    relevo   = res['mascara_relevo'].astype(np.float32)
    branco   = res['branco'].astype(np.uint8) if 'branco' in res else res.get('white_mask')
    verde    = res['verde'].astype(np.uint8)  if 'verde'  in res else res.get('green_mask')

    # Tentar chaves alternativas
    if branco is None:
        for k in res:
            if 'branco' in k.lower() or 'white' in k.lower():
                branco = res[k].astype(np.uint8)
                break
    if verde is None:
        for k in res:
            if 'verde' in k.lower() or 'green' in k.lower():
                verde = res[k].astype(np.uint8)
                break

    if skeleton.sum() == 0:
        print(f"  {basename}: sem esqueleto, pulando.")
        return

    # Grafo
    g = construir_grafo(skeleton, relevo, cfg_grafo)
    simplificar(g, skeleton, relevo, cfg_simp)

    # Grade subamostrada branco/verde
    if branco is None or verde is None:
        h_s, w_s = skeleton.shape
        # Fallback: usar relevo como proxy de branco, zeros para verde
        branco_sub = (relevo > 0).astype(np.uint8)
        verde_sub  = np.zeros_like(branco_sub, dtype=np.uint8)
        print(f"  {basename}: aviso — máscaras branco/verde não encontradas, usando fallback")
    else:
        # Subamostrar para a grade do esqueleto se necessário
        h_s, w_s = skeleton.shape
        if branco.shape != (h_s, w_s):
            branco_sub = cv2.resize(branco, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
        else:
            branco_sub = branco
        if verde.shape != (h_s, w_s):
            verde_sub = cv2.resize(verde, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
        else:
            verde_sub = verde

    debug_info: list = []
    aceitos, rejeitados = detectar_cruzamentos(
        g, branco_sub, verde_sub, cfg_cruz, _debug_info=debug_info
    )

    # ── Estatísticas ──────────────────────────────────────────────────────────
    tipos = ('L', 'T', 'X')
    n_cand  = {t: sum(1 for d in debug_info if d['tipo'] == t) for t in tipos}
    n_ace   = {t: sum(1 for d in debug_info if d['tipo'] == t and d['aceito']) for t in tipos}
    n_rej   = {t: n_cand[t] - n_ace[t] for t in tipos}

    print(f"=== {basename} ===")
    total_c = sum(n_cand.values())
    total_a = sum(n_ace.values())
    total_r = sum(n_rej.values())
    print(f"  Candidatos: L={n_cand['L']} T={n_cand['T']} X={n_cand['X']} total={total_c}")
    print(f"  Aceitos:    L={n_ace['L']}  T={n_ace['T']}  X={n_ace['X']}  total={total_a}")
    print(f"  Rejeitados: L={n_rej['L']}  T={n_rej['T']}  X={n_rej['X']}  total={total_r}")

    print(f"  Aceitos (detalhes):")
    for d in debug_info:
        if not d['aceito']:
            continue
        ang = f"  ang={d['angulo_grau']:.1f}°" if 'angulo_grau' in d else ''
        print(f"    [{d['tipo']}] ({d['x']},{d['y']}){ang}"
              f"  V1={d['v1']:.2f}  V2={d['v2']:.2f}"
              f"  r={d['raio_usado']}"
              f"  branco={d['n_branco']}  verde={d['n_verde']}"
              f"  conf={d['confianca']:.2f}")
    print(f"  Rejeitados (detalhes):")
    for d in debug_info:
        if d['aceito']:
            continue
        ang = f"  ang={d['angulo_grau']:.1f}°" if 'angulo_grau' in d else ''
        print(f"    [{d['tipo']}] ({d['x']},{d['y']}){ang}"
              f"  V1={d['v1']:.2f}  V2={d['v2']:.2f}"
              f"  r={d['raio_usado']}"
              f"  branco={d['n_branco']}  verde={d['n_verde']}"
              f"  motivo={d['motivo']}")
    print()

    # ── Canvas ────────────────────────────────────────────────────────────────
    h, w = skeleton.shape
    canvas = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    # Fundo: esqueleto
    rel_max = relevo.max() + 1e-5
    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                v = int(relevo[y, x] / rel_max * 60) + 20
                canvas[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = [v, v, v]

    # Rejeitados (cinza com motivo)
    for d in debug_info:
        if d['aceito']:
            continue
        cx = d['x'] * scale + scale // 2
        cy = d['y'] * scale + scale // 2
        cv2.circle(canvas, (cx, cy), 4, _COR_REJEITADO, 1)
        motivo_short = d['motivo'][:10]
        cv2.putText(canvas, motivo_short, (cx + 5, cy - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.6, _COR_REJEITADO, 1)

    # Aceitos (cor por tipo)
    for d in debug_info:
        if not d['aceito']:
            continue
        cx = d['x'] * scale + scale // 2
        cy = d['y'] * scale + scale // 2
        cor = _COR_TIPO[d['tipo']]
        cv2.circle(canvas, (cx, cy), 5, cor, -1)
        cv2.putText(canvas, d['tipo'], (cx + 5, cy - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, cor, 1)

    out_path = f"debug_out/etapa7_{basename}.png"
    cv2.imwrite(out_path, canvas)
    print(f"  Salvo: {out_path}")


def main() -> None:
    os.makedirs("debug_out", exist_ok=True)
    ps = Pixel_Segment('green_pixels.csv')
    cfg_grafo = ConfigGrafo()
    cfg_simp  = ConfigSimplificacao()
    cfg_cruz  = ConfigCruzamento()

    images = [
        'images/image_1.png',
        'images/image_2.png',
        'images/image.png',
    ]

    for path in images:
        processar_imagem(path, ps, cfg_grafo, cfg_simp, cfg_cruz)


if __name__ == '__main__':
    main()
