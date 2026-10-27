"""Diagnóstico da Etapa 8: D1–D4.
Sem modificar nenhum arquivo de features/.
"""
from __future__ import annotations
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from main import Pixel_Segment
from features.graph import StructureGraph, Node
from features.nodes import ConfigNos, neighbor_count, place_nodes
from features.connections import (ConfigConexao, ConfigSubdivisao,
    connect_nodes, conectar_candidatas, dividir_grau_zero, esticar_extremos)
from features.pipeline import ConfigGrafo
from features.lines import extrair_linhas, _componentes_grau2, ConfigLinhas

LUT = 'green_pixels.csv'
ps  = Pixel_Segment(LUT)

# ─── Utilitários ─────────────────────────────────────────────────────────────

def _frac_branco(x1, y1, x2, y2, branco):
    """Fração de pixels da reta (x1,y1)-(x2,y2) que estão na máscara branco."""
    n = max(int(math.hypot(x2-x1, y2-y1)) * 2 + 1, 2)
    xs = np.round(np.linspace(x1, x2, n)).astype(int)
    ys = np.round(np.linspace(y1, y2, n)).astype(int)
    H, W = branco.shape
    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[mask], ys[mask]
    if len(xs) == 0:
        return 0.0
    return float(np.count_nonzero(branco[ys, xs] > 0)) / len(xs)

def _origem_aresta(a, b, e3, e4, e5):
    """Classifica a origem de uma aresta (min(a,b), max(a,b)) nos três snapshots."""
    k = (min(a,b), max(a,b))
    if k in e3: return 'adj3x3'
    if k in e4: return 'etapa4'
    if k in e5: return 'candidata'
    return '?'

def _build_step_by_step(skeleton, relevo, cfg=None):
    """Constrói grafo passo a passo e captura snapshot de arestas após cada etapa."""
    if cfg is None:
        cfg = ConfigGrafo()
    c_xy = neighbor_count(relevo, skeleton, cfg.nos)
    g = place_nodes(skeleton, relevo, c_xy, cfg.nos)
    connect_nodes(g, skeleton)
    edges_after3 = set((min(a,b), max(a,b)) for a,b in g.edges)
    esticar_extremos(g, skeleton, relevo)
    dividir_grau_zero(g, skeleton, relevo, cfg_sub=cfg.sub)
    edges_after4 = set((min(a,b), max(a,b)) for a,b in g.edges)
    conectar_candidatas(g, relevo, cfg.con)
    edges_after5 = set((min(a,b), max(a,b)) for a,b in g.edges)
    new4 = edges_after4 - edges_after3
    new5 = edges_after5 - edges_after4
    return g, edges_after3, new4, new5

# ─── Processar imagem ────────────────────────────────────────────────────────

def processar(nome, path):
    frame = __import__('cv2').imread(path)
    res   = ps.processar(frame)
    skel  = res['skeleton_img'].astype(bool)
    relevo= res['mascara_relevo'].astype(np.float32)
    branco= res['white_mask_resized'].astype(np.uint8)
    verde = res['green_mask_resized'].astype(np.uint8)
    H, W  = skel.shape
    g, e3, e4, e5 = _build_step_by_step(skel, relevo)
    linhas = extrair_linhas(g, fator_escala=8)
    nodes  = g.nodes

    print(f"\n{'='*70}")
    print(f"  IMAGEM: {nome}  [{W}×{H} grade]")
    print(f"{'='*70}")

    # ─── D1: reta longa falsa ─────────────────────────────────────────────
    if nome in ('image_1',):
        print("\n── D1: reta longa falsa (x≈120-160, y≈68-85) ──")
        alvo = [lf for lf in linhas
                if min(lf.x1,lf.x2) > 100 and min(lf.y1,lf.y2) < 75
                   and abs(lf.angulo % math.pi - math.radians(157)) < math.radians(20)]
        if not alvo:
            # Broader search
            alvo = [lf for lf in linhas
                    if (lf.x1 > 100 or lf.x2 > 100)
                    and abs(lf.y1 - 76) < 12 or abs(lf.y2 - 76) < 12]
            alvo = [lf for lf in linhas if lf.distancia > 30 and
                    max(lf.x1, lf.x2) > 120 and min(lf.y1, lf.y2) < 80]
        print(f"  Candidatas: {len(alvo)}")
        for lf in alvo:
            print(f"  LF: ({lf.x1:.1f},{lf.y1:.1f})-({lf.x2:.1f},{lf.y2:.1f})"
                  f"  dist={lf.distancia:.1f}  n_nos={lf.n_nos}"
                  f"  ang={math.degrees(lf.angulo):.1f}°  conf={lf.confianca:.2f}")

        # Componente que produziu essa linha (busca pelos nós mais próximos)
        # LineFeature #1 da tabela anterior: (160,68)-(120,85) dist=43.8 n_nos=11
        target_lf = None
        for lf in linhas:
            if (abs(lf.x1-160.4) < 5 or abs(lf.x2-160.4) < 5):
                target_lf = lf
                break
        if target_lf is None:
            target_lf = max(linhas, key=lambda l: l.distancia) if linhas else None

        if target_lf:
            print(f"\n  LineFeature alvo: ({target_lf.x1:.1f},{target_lf.y1:.1f})"
                  f"-({target_lf.x2:.1f},{target_lf.y2:.1f})"
                  f"  dist={target_lf.distancia:.1f}  n_nos={target_lf.n_nos}")
            # Encontrar todos os componentes e identificar o que contém nós nessa região
            comps = _componentes_grau2(g)
            comp_alvo = None
            for comp in comps:
                xs = [nodes[nid].x for nid in comp]
                ys = [nodes[nid].y for nid in comp]
                if (max(xs) > 115 and min(ys) < 75 and max(ys) > 80 and len(comp) >= 5):
                    comp_alvo = comp
                    break
            if comp_alvo:
                print(f"\n  Componente ({len(comp_alvo)} nós):")
                print(f"  {'nid':>5}  {'x':>4}  {'y':>4}  {'grau':>5}")
                for nid in comp_alvo:
                    n = nodes[nid]
                    print(f"  {nid:>5}  {n.x:>4}  {n.y:>4}  {g.degree(nid):>5}")
                # Maior espaço entre nós consecutivos
                max_gap = 0.0
                max_gap_pair = None
                for i in range(len(comp_alvo)-1):
                    a, b = comp_alvo[i], comp_alvo[i+1]
                    na, nb = nodes[a], nodes[b]
                    d = math.hypot(nb.x-na.x, nb.y-na.y)
                    if d > max_gap:
                        max_gap = d
                        max_gap_pair = (a, b, na.x, na.y, nb.x, nb.y)
                print(f"\n  Maior espaço entre nós consecutivos: {max_gap:.2f} px")
                if max_gap_pair:
                    a, b, ax, ay, bx, by = max_gap_pair
                    ek = (min(a,b), max(a,b))
                    orig = _origem_aresta(a, b, e3, e4, e5)
                    fb = _frac_branco(ax, ay, bx, by, branco)
                    print(f"  Entre nó {a}({ax},{ay}) e nó {b}({bx},{by})"
                          f"  origem={orig}  frac_branco={fb:.2f}")

                # Verificar se veio de fusão
                cfg_fusao = ConfigLinhas(usar_fusao=True)
                cfg_sem_fusao = ConfigLinhas(usar_fusao=False)
                linhas_com = extrair_linhas(g, fator_escala=8, cfg=cfg_fusao)
                linhas_sem = extrair_linhas(g, fator_escala=8, cfg=cfg_sem_fusao)
                n_com = len(linhas_com)
                n_sem = len(linhas_sem)
                print(f"\n  Linhas com fusão={n_com}, sem fusão={n_sem}")
                if n_sem != n_com:
                    print("  → linha veio (parcialmente) de FUSÃO de segmentos")
                    # Identificar segmentos que foram fundidos
                    from features.lines import _split, _componentes_grau2 as cg2
                    from features.geometry import undistort_pontos
                    cfg_check = ConfigLinhas(usar_fusao=False)
                    segs_sem = []
                    for ids in cg2(g):
                        if len(ids) < cfg_check.min_nos_componente:
                            continue
                        pts = np.array([[nodes[nid].x, nodes[nid].y] for nid in ids], dtype=np.float32)
                        pts_ud, _ = undistort_pontos(pts, 8, None)
                        sub = _split(ids, pts_ud, cfg_check.desvio_max_celulas, cfg_check.min_nos_componente)
                        segs_sem.extend(sub)
                    # Quais segmentos sem fusão mapeiam para a região da reta falsa?
                    cands = []
                    for seg_ids, seg_pts in segs_sem:
                        xs = seg_pts[:,0]
                        ys = seg_pts[:,1]
                        if max(xs) > 115 and min(ys) < 75:
                            cands.append((seg_ids, seg_pts))
                    print(f"  Segmentos sem fusão na região: {len(cands)}")
                    for seg_ids, seg_pts in cands:
                        x1, y1 = seg_pts[0]
                        x2, y2 = seg_pts[-1]
                        print(f"    seg ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
                              f"  n_nos={len(seg_ids)}")
                else:
                    print("  → linha veio de cadeia única (sem fusão)")

    # ─── D2: linha do meio fora do círculo ────────────────────────────────
    print(f"\n── D2: linha do meio fora do círculo ──")
    # Região esquerda: x=0..42, y≈83 (image_1) / image usa y≈83 também
    # Região direita: x≈127..167, y≈83
    for regiao, x_min, x_max, y_ref in [
        ('esquerda', 0, 45, 83),
        ('direita', 120, 170, 83),
    ]:
        nos_regiao = [(nid, n) for nid, n in nodes.items()
                      if x_min <= n.x <= x_max and abs(n.y - y_ref) <= 8]
        print(f"\n  Região {regiao} (x=[{x_min},{x_max}], y≈{y_ref}±8): {len(nos_regiao)} nós")
        if nos_regiao:
            print(f"  {'nid':>5}  {'x':>4}  {'y':>4}  {'grau':>5}  skel?")
            for nid, n in sorted(nos_regiao, key=lambda t: t[1].x):
                sk = '✓' if (0 <= n.y < H and 0 <= n.x < W and skel[n.y, n.x]) else '✗'
                print(f"  {nid:>5}  {n.x:>4}  {n.y:>4}  {g.degree(nid):>5}  {sk}")
            # Componente que contém esses nós
            comps = _componentes_grau2(g)
            nids_regiao = set(t[0] for t in nos_regiao)
            comp_hit = None
            for comp in comps:
                if nids_regiao & set(comp):
                    comp_hit = comp
                    break
            if comp_hit:
                print(f"  → faz parte de componente de {len(comp_hit)} nós")
                cfg_check = ConfigLinhas()
                if len(comp_hit) < cfg_check.min_nos_componente:
                    print(f"  → DESCARTADO: len({len(comp_hit)}) < min_nos={cfg_check.min_nos_componente}"
                          f"  (features/lines.py:extrair_linhas, filtro min_nos_componente)")
                else:
                    # Verificar se passou pelo split
                    from features.lines import _split as _sp
                    from features.geometry import undistort_pontos as _ud
                    pts = np.array([[nodes[nid].x, nodes[nid].y] for nid in comp_hit], dtype=np.float32)
                    pts_ud, _ = _ud(pts, 8, None)
                    sub = _sp(comp_hit, pts_ud, cfg_check.desvio_max_celulas, cfg_check.min_nos_componente)
                    print(f"  → split produz {len(sub)} sub-segmentos")
                    for seg_ids, seg_pts in sub:
                        x1, y1 = float(seg_pts[0,0]), float(seg_pts[0,1])
                        x2, y2 = float(seg_pts[-1,0]), float(seg_pts[-1,1])
                        d = math.hypot(x2-x1, y2-y1)
                        print(f"     ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
                              f"  n={len(seg_ids)}  dist={d:.1f}")
            else:
                print("  → NÃO faz parte de nenhum componente de grau-2")
                print("    (nós de grau≠2 — âncoras — não geram componente próprio)")
                # Verificar grau dos nós
                for nid, n in sorted(nos_regiao, key=lambda t: t[1].x):
                    deg = g.degree(nid)
                    if deg != 2:
                        print(f"    nó {nid} grau={deg} → âncora (excluído de cadeias de grau-2)")
        else:
            # Verificar se há pixels de esqueleto na região sem nó
            ys_reg = range(max(0,y_ref-8), min(H, y_ref+9))
            xs_reg = range(x_min, min(W, x_max+1))
            skel_px = [(y,x) for y in ys_reg for x in xs_reg if skel[y,x]]
            print(f"  pixels de esqueleto sem nó: {len(skel_px)}")
            if skel_px:
                print(f"  primeiros: {skel_px[:5]}")
                print("  → esses pixels não foram cobertos por nenhum nó do grafo")

    # ─── D3: arestas com frac_branco < 0.5 e dist > 4 ────────────────────
    print(f"\n── D3: arestas frac_branco<0.5 e dist>4 ──")
    bad = []
    for a, b in g.edges:
        na, nb = nodes[a], nodes[b]
        d = math.hypot(nb.x-na.x, nb.y-na.y)
        if d <= 4.0:
            continue
        fb = _frac_branco(na.x, na.y, nb.x, nb.y, branco)
        if fb < 0.5:
            orig = _origem_aresta(a, b, e3, e4, e5)
            bad.append((a, b, na.x, na.y, nb.x, nb.y, d, fb, orig))
    bad.sort(key=lambda r: r[7])
    print(f"  Total arestas ruins: {len(bad)}")
    print(f"  {'a':>5}  {'b':>5}  {'(ax,ay)':>9}  {'(bx,by)':>9}  "
          f"{'dist':>6}  {'frac':>5}  {'orig':>10}")
    for a, b, ax, ay, bx, by, d, fb, orig in bad:
        print(f"  {a:>5}  {b:>5}  ({ax:>3},{ay:>3})  ({bx:>3},{by:>3})  "
              f"{d:>6.1f}  {fb:>5.2f}  {orig:>10}")

    # ─── D4: falsas verticais em image e reta da bola ─────────────────────
    if nome == 'image':
        print(f"\n── D4: verticais falsas e reta da bola ──")
        # vertical x≈75, y=68-75
        # vertical x≈99, y=67-76
        # bola x≈153-166, y≈76
        regioes_d4 = [
            ('vert_75',  73, 77, 66, 77),
            ('vert_99',  97,101, 65, 78),
            ('bola_153',148,168, 74, 79),
        ]
        for label, xmin, xmax, ymin, ymax in regioes_d4:
            print(f"\n  {label} (x=[{xmin},{xmax}], y=[{ymin},{ymax}])")
            nos = [(nid, n) for nid, n in nodes.items()
                   if xmin <= n.x <= xmax and ymin <= n.y <= ymax]
            if not nos:
                print("  (sem nós)")
                continue
            print(f"  {'nid':>5}  {'x':>4}  {'y':>4}  {'grau':>5}")
            for nid, n in sorted(nos, key=lambda t: t[1].y):
                print(f"  {nid:>5}  {n.x:>4}  {n.y:>4}  {g.degree(nid):>5}")
            # Arestas dentro da região
            nids_set = set(t[0] for t in nos)
            arestas_reg = [(a,b) for a,b in g.edges if a in nids_set or b in nids_set]
            print(f"  arestas da região: {len(arestas_reg)}")
            for a, b in arestas_reg:
                na, nb = nodes[a], nodes[b]
                d  = math.hypot(nb.x-na.x, nb.y-na.y)
                fb = _frac_branco(na.x, na.y, nb.x, nb.y, branco)
                orig = _origem_aresta(a, b, e3, e4, e5)
                sk_a = '✓' if skel[na.y, na.x] else '✗'
                sk_b = '✓' if skel[nb.y, nb.x] else '✗'
                print(f"  {a}({na.x},{na.y})skel{sk_a} – {b}({nb.x},{nb.y})skel{sk_b}"
                      f"  d={d:.1f}  fb={fb:.2f}  orig={orig}")


# ─── Rodar ────────────────────────────────────────────────────────────────────
for nome, path in [('image_1','images/image_1.png'),
                   ('image_2','images/image_2.png'),
                   ('image',  'images/image.png')]:
    processar(nome, path)
