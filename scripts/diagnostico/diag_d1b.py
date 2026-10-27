"""D1b: identificar a fusão que gera a reta diagonal de image_1."""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from main import Pixel_Segment
from features.pipeline import ConfigGrafo
from features.nodes import neighbor_count, place_nodes
from features.connections import (connect_nodes, conectar_candidatas,
    dividir_grau_zero, esticar_extremos)
from features.lines import (_componentes_grau2, _split, _para_linefeature,
    _ajustar_reta, _dist_ortogonal, _angulo_diff, _dist_ortogonal_retas,
    ConfigLinhas, extrair_linhas)
from features.geometry import undistort_pontos
import math

def _frac(x1,y1,x2,y2,branco):
    n = max(int(math.hypot(x2-x1,y2-y1))*2+1,2)
    xs = np.round(np.linspace(x1,x2,n)).astype(int)
    ys = np.round(np.linspace(y1,y2,n)).astype(int)
    H,W=branco.shape; m=(xs>=0)&(xs<W)&(ys>=0)&(ys<H)
    xs,ys=xs[m],ys[m]
    return float(np.count_nonzero(branco[ys,xs]>0))/len(xs) if len(xs) else 0.0

ps = Pixel_Segment('green_pixels.csv')
frame = __import__('cv2').imread('images/image_1.png')
res = ps.processar(frame)
skel  = res['skeleton_img'].astype(bool)
relv  = res['mascara_relevo'].astype(np.float32)
bco   = res['white_mask_resized'].astype(np.uint8)

cfg = ConfigGrafo()
c_xy = neighbor_count(relv, skel, cfg.nos)
g = place_nodes(skel, relv, c_xy, cfg.nos)
connect_nodes(g, skel)
esticar_extremos(g, skel, relv)
dividir_grau_zero(g, skel, relv, cfg_sub=cfg.sub)
conectar_candidatas(g, relv, cfg.con)
nodes = g.nodes

cfgL = ConfigLinhas()

# ── Todos os sub-segmentos antes da fusão ────────────────────────────────────
comps = _componentes_grau2(g)
segs_pre = []  # (ids, pts, lf_approx)
for comp in comps:
    if len(comp) < cfgL.min_nos_componente:
        continue
    pts = np.array([[nodes[nid].x, nodes[nid].y] for nid in comp], dtype=np.float32)
    pts_ud,_ = undistort_pontos(pts, 8, None)
    subs = _split(comp, pts_ud, cfgL.desvio_max_celulas, cfgL.min_nos_componente)
    for sub_ids, sub_pts in subs:
        lf = _para_linefeature(sub_ids, sub_pts, sub_pts*8, 8)
        segs_pre.append((sub_ids, sub_pts, lf))

print(f"Sub-segmentos pré-fusão: {len(segs_pre)}")
print(f"{'idx':>4}  {'(x1,y1)':>10}  {'(x2,y2)':>10}  {'dist':>6}  {'ang°':>6}  {'n':>4}")
for i, (ids, pts, lf) in enumerate(segs_pre):
    x1,y1=float(pts[0,0]),float(pts[0,1])
    x2,y2=float(pts[-1,0]),float(pts[-1,1])
    ang=math.degrees(lf.angulo)%180
    print(f"{i:>4}  ({x1:>5.0f},{y1:>4.0f})  ({x2:>5.0f},{y2:>4.0f})"
          f"  {lf.distancia:>6.1f}  {ang:>6.1f}  {len(ids):>4}")

# ── LineFeatures pós-fusão ────────────────────────────────────────────────────
linhas = extrair_linhas(g, fator_escala=8, cfg=cfgL)
print(f"\nLineFeatures pós-fusão: {len(linhas)}")
print(f"{'idx':>4}  {'(x1,y1)':>12}  {'(x2,y2)':>12}  {'dist':>6}  {'ang°':>6}  {'n':>4}")
for i, lf in enumerate(linhas):
    ang=math.degrees(lf.angulo)%180
    print(f"{i:>4}  ({lf.x1:>5.1f},{lf.y1:>4.1f})  ({lf.x2:>5.1f},{lf.y2:>4.1f})"
          f"  {lf.distancia:>6.1f}  {ang:>6.1f}  {lf.n_nos:>4}")

# ── Quais pares de pré-fusão têm ang_diff e dist_ort dentro dos limiares? ────
print(f"\nPares elegíveis para fusão (ang<{cfgL.limiar_angulo_fusao_grau}°,"
      f" dOrt<{cfgL.limiar_dist_ortogonal_fusao}):")
lim_ang = math.radians(cfgL.limiar_angulo_fusao_grau)
lim_d   = cfgL.limiar_dist_ortogonal_fusao
from features.lines import _ajustar_reta as _ajr

def reta_seg(pts):
    return _ajr(pts)

printed_any = False
for i in range(len(segs_pre)):
    for j in range(i+1, len(segs_pre)):
        ids_i, pts_i, lf_i = segs_pre[i]
        ids_j, pts_j, lf_j = segs_pre[j]
        r_i = reta_seg(pts_i)
        r_j = reta_seg(pts_j)
        nx_i,ny_i = r_i[0],r_i[1]
        nx_j,ny_j = r_j[0],r_j[1]
        ang_i = math.atan2(ny_i,nx_i)
        ang_j = math.atan2(ny_j,nx_j)
        adiff = _angulo_diff(ang_i, ang_j)
        if adiff > lim_ang:
            continue
        # distância ortogonal entre as retas (centroide-a-reta)
        from features.lines import _dist_ortogonal_retas as _dor
        d_ort = _dor(pts_i, r_i, pts_j, r_j)
        if d_ort > lim_d:
            continue
        printed_any = True
        print(f"  seg{i}({pts_i[0,0]:.0f},{pts_i[0,1]:.0f})-({pts_i[-1,0]:.0f},{pts_i[-1,1]:.0f})"
              f"  n={len(ids_i)}"
              f"  ×  seg{j}({pts_j[0,0]:.0f},{pts_j[0,1]:.0f})-({pts_j[-1,0]:.0f},{pts_j[-1,1]:.0f})"
              f"  n={len(ids_j)}"
              f"  ang_diff={math.degrees(adiff):.2f}°  d_ort={d_ort:.2f}")
        # Distância entre centros
        cx_i = pts_i[:,0].mean(); cy_i = pts_i[:,1].mean()
        cx_j = pts_j[:,0].mean(); cy_j = pts_j[:,1].mean()
        print(f"    gap centros: Δx={abs(cx_i-cx_j):.1f} Δy={abs(cy_i-cy_j):.1f}")
if not printed_any:
    print("  nenhum par elegível dentro dos limiares")
    print(f"\n  Limiares atuais: ang<{cfgL.limiar_angulo_fusao_grau}°, dOrt<{cfgL.limiar_dist_ortogonal_fusao}")
    print("  Verificando com limiares relaxados (ang<10°, dOrt<20):")
    for i in range(len(segs_pre)):
        for j in range(i+1, len(segs_pre)):
            ids_i,pts_i,lf_i = segs_pre[i]
            ids_j,pts_j,lf_j = segs_pre[j]
            r_i = reta_seg(pts_i); r_j = reta_seg(pts_j)
            nx_i,ny_i=r_i[0],r_i[1]; nx_j,ny_j=r_j[0],r_j[1]
            ang_i=math.atan2(ny_i,nx_i); ang_j=math.atan2(ny_j,nx_j)
            adiff=_angulo_diff(ang_i,ang_j)
            if adiff > math.radians(10): continue
            from features.lines import _dist_ortogonal_retas as _dor
            d_ort=_dor(pts_i,r_i,pts_j,r_j)
            if d_ort > 20: continue
            print(f"    seg{i}({pts_i[0,0]:.0f},{pts_i[0,1]:.0f})→({pts_i[-1,0]:.0f},{pts_i[-1,1]:.0f})"
                  f" n={len(ids_i)}"
                  f" × seg{j}({pts_j[0,0]:.0f},{pts_j[0,1]:.0f})→({pts_j[-1,0]:.0f},{pts_j[-1,1]:.0f})"
                  f" n={len(ids_j)}"
                  f" ang={math.degrees(adiff):.1f}° d_ort={d_ort:.1f}")
