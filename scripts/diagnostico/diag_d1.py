"""D1 detalhado: componente da reta falsa de image_1."""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from main import Pixel_Segment
from features.graph import StructureGraph
from features.nodes import ConfigNos, neighbor_count, place_nodes
from features.connections import (ConfigConexao, ConfigSubdivisao,
    connect_nodes, conectar_candidatas, dividir_grau_zero, esticar_extremos)
from features.pipeline import ConfigGrafo
from features.lines import (extrair_linhas, _componentes_grau2,
    _split, _ajustar_reta, _dist_ortogonal, _para_linefeature, ConfigLinhas)
from features.geometry import undistort_pontos

def _frac(x1,y1,x2,y2,branco):
    n = max(int(math.hypot(x2-x1,y2-y1))*2+1,2)
    xs = np.round(np.linspace(x1,x2,n)).astype(int)
    ys = np.round(np.linspace(y1,y2,n)).astype(int)
    H,W = branco.shape
    m = (xs>=0)&(xs<W)&(ys>=0)&(ys<H)
    xs,ys=xs[m],ys[m]
    if len(xs)==0: return 0.0
    return float(np.count_nonzero(branco[ys,xs]>0))/len(xs)

ps = Pixel_Segment('green_pixels.csv')
frame = __import__('cv2').imread('images/image_1.png')
res = ps.processar(frame)
skel   = res['skeleton_img'].astype(bool)
relevo = res['mascara_relevo'].astype(np.float32)
branco = res['white_mask_resized'].astype(np.uint8)

cfg = ConfigGrafo()
c_xy = neighbor_count(relevo, skel, cfg.nos)
g = place_nodes(skel, relevo, c_xy, cfg.nos)
connect_nodes(g, skel)
e3 = set((min(a,b),max(a,b)) for a,b in g.edges)
esticar_extremos(g, skel, relevo)
dividir_grau_zero(g, skel, relevo, cfg_sub=cfg.sub)
e4 = set((min(a,b),max(a,b)) for a,b in g.edges) - e3
conectar_candidatas(g, relevo, cfg.con)
e5 = set((min(a,b),max(a,b)) for a,b in g.edges) - e3 - e4

def orig(a,b):
    k=(min(a,b),max(a,b))
    if k in e3: return 'adj3x3'
    if k in e4: return 'etapa4'
    if k in e5: return 'candidata'
    return '?'

nodes = g.nodes
cfg_L = ConfigLinhas(usar_fusao=False)
cfg_LF= ConfigLinhas(usar_fusao=True)

# Lista TODOS os componentes e os sub-segmentos que produzem
comps = _componentes_grau2(g)
print(f"Total componentes: {len(comps)}")

resultados = []
for comp in comps:
    if len(comp) < cfg_L.min_nos_componente:
        continue
    pts = np.array([[nodes[nid].x, nodes[nid].y] for nid in comp], dtype=np.float32)
    pts_ud,_ = undistort_pontos(pts, 8, None)
    subs = _split(comp, pts_ud, cfg_L.desvio_max_celulas, cfg_L.min_nos_componente)
    for sub_ids, sub_pts in subs:
        lf = _para_linefeature(sub_ids, sub_pts, sub_pts*8, 8)
        # recalcula confiança
        nx,ny,mx,my = _ajustar_reta(sub_pts)
        devios = _dist_ortogonal(sub_pts, nx,ny,mx,my)
        conf = float(np.mean(devios < cfg_L.desvio_max_celulas/2))
        resultados.append((comp, sub_ids, sub_pts, lf, conf))

print(f"Total sub-segmentos (sem fusão): {len(resultados)}")

# Identificar o que mais se parece com a reta falsa
alvo = None
for comp, sub_ids, sub_pts, lf, conf in resultados:
    if abs(lf.distancia - 43.8) < 5 and len(sub_ids) >= 9:
        alvo = (comp, sub_ids, sub_pts, lf, conf)
        break

if alvo is None:
    # Busca pelo ângulo
    for comp, sub_ids, sub_pts, lf, conf in resultados:
        if lf.distancia > 35 and abs(math.degrees(lf.angulo) % 180 - 157) < 15:
            alvo = (comp, sub_ids, sub_pts, lf, conf)
            break

if alvo is None:
    # Busca por posição dos extremos
    for comp, sub_ids, sub_pts, lf, conf in resultados:
        if (max(lf.x1,lf.x2) > 140 and abs(min(lf.y1,lf.y2) - 68) < 8):
            alvo = (comp, sub_ids, sub_pts, lf, conf)
            break

if alvo is None:
    print("\n  [alvo não encontrado — listando candidatos dist>30]")
    for comp, sub_ids, sub_pts, lf, conf in sorted(resultados, key=lambda r: -r[3].distancia)[:6]:
        print(f"  ({lf.x1:.0f},{lf.y1:.0f})-({lf.x2:.0f},{lf.y2:.0f})"
              f"  dist={lf.distancia:.1f}  n={len(sub_ids)}  ang={math.degrees(lf.angulo)%180:.0f}°")
else:
    comp, sub_ids, sub_pts, lf, conf = alvo
    print(f"\n=== LineFeature alvo ===")
    print(f"  ({lf.x1:.1f},{lf.y1:.1f})-({lf.x2:.1f},{lf.y2:.1f})"
          f"  dist={lf.distancia:.1f}  n_nos={len(sub_ids)}"
          f"  ang={math.degrees(lf.angulo)%180:.1f}°")
    print(f"  componente total: {len(comp)} nós")
    print(f"  sub-segmento: {len(sub_ids)} nós")

    print(f"\n  Nós do sub-segmento (em ordem):")
    print(f"  {'nid':>5}  {'x':>4}  {'y':>4}  {'grau':>5}  skel?")
    for nid in sub_ids:
        n = nodes[nid]
        sk = '✓' if skel[n.y, n.x] else '✗'
        print(f"  {nid:>5}  {n.x:>4}  {n.y:>4}  {g.degree(nid):>5}  {sk}")

    print(f"\n  Arestas do sub-segmento:")
    print(f"  {'a':>5}  {'b':>5}  {'(ax,ay)':>9}  {'(bx,by)':>9}  "
          f"{'dist':>5}  {'fb':>5}  {'orig':>10}")
    max_gap = 0.0; max_gap_info = None
    for i in range(len(sub_ids)-1):
        a,b = sub_ids[i], sub_ids[i+1]
        # verificar se são vizinhos no grafo
        e_grafo = b in g.neighbors(a)
        na,nb = nodes[a], nodes[b]
        d = math.hypot(nb.x-na.x, nb.y-na.y)
        if d > max_gap: max_gap=d; max_gap_info=(a,b,na,nb)
        fb = _frac(na.x,na.y,nb.x,nb.y,branco)
        o = orig(a,b) if e_grafo else 'NÃO-ADJ'
        print(f"  {a:>5}  {b:>5}  ({na.x:>3},{na.y:>3})  ({nb.x:>3},{nb.y:>3})"
              f"  {d:>5.1f}  {fb:>5.2f}  {o:>10}")
    print(f"\n  Maior espaço: {max_gap:.1f} px")
    if max_gap_info:
        a,b,na,nb = max_gap_info
        print(f"  Entre nó {a}({na.x},{na.y}) e nó {b}({nb.x},{nb.y})")

    # Verificar com fusão
    linhas_sem = extrair_linhas(g, fator_escala=8, cfg=ConfigLinhas(usar_fusao=False))
    linhas_com = extrair_linhas(g, fator_escala=8, cfg=ConfigLinhas(usar_fusao=True))
    print(f"\n  Linhas sem fusão: {len(linhas_sem)}")
    print(f"  Linhas com fusão: {len(linhas_com)}")
    if len(linhas_sem) != len(linhas_com):
        print("  → contagem diferente: a linha veio (ao menos parcialmente) de FUSÃO")
    else:
        print("  → contagem igual: linha vem de cadeia única (fusão não alterou contagem)")
        # Verificar se a linha aparece em ambos
        match_sem = [lf2 for lf2 in linhas_sem if abs(lf2.distancia - lf.distancia) < 3]
        match_com = [lf2 for lf2 in linhas_com if abs(lf2.distancia - lf.distancia) < 3]
        print(f"  Match dist≈{lf.distancia:.0f} sem fusão: {len(match_sem)}, com fusão: {len(match_com)}")
