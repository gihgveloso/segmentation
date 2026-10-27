"""Testes de Etapa 7: detecção e verificação de cruzamentos (IV-A)."""
import math
import numpy as np
import pytest

from features.graph import Node, StructureGraph
from features.types import CrossingFeature
from features.crossings import (
    ConfigCruzamento,
    classificar_candidatos,
    verificar_cruzamento,
    detectar_cruzamentos,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _grafo_l(jx=10, jy=10):
    """L: nó central grau 2, braços esquerdo e acima → ângulo 90°."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=jx,     y=jy,     gray=5.0))
    g.add_node(Node(id=1, x=jx - 5, y=jy,     gray=5.0))
    g.add_node(Node(id=2, x=jx,     y=jy - 5, gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    return g


def _grafo_l_angulo(jx=10, jy=10, angulo_grau=90):
    """L com ângulo arbitrário; braço 1 vai para a esquerda, braço 2 em angulo."""
    g = StructureGraph()
    ang = math.radians(angulo_grau)
    g.add_node(Node(id=0, x=jx, y=jy, gray=5.0))
    g.add_node(Node(id=1, x=jx - 5, y=jy, gray=5.0))  # braço esquerdo
    g.add_node(Node(id=2,
                    x=int(round(jx + 5 * math.cos(ang))),
                    y=int(round(jy + 5 * math.sin(ang))),
                    gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    return g


def _grafo_t(jx=10, jy=10):
    """T: nó central grau 3, braços esquerdo, direito e cima."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=jx,     y=jy,     gray=5.0))
    g.add_node(Node(id=1, x=jx - 5, y=jy,     gray=5.0))
    g.add_node(Node(id=2, x=jx + 5, y=jy,     gray=5.0))
    g.add_node(Node(id=3, x=jx,     y=jy - 5, gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    return g


def _grafo_x(jx=10, jy=10):
    """X: nó central grau 4."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=jx,     y=jy,     gray=5.0))
    g.add_node(Node(id=1, x=jx - 5, y=jy,     gray=5.0))
    g.add_node(Node(id=2, x=jx + 5, y=jy,     gray=5.0))
    g.add_node(Node(id=3, x=jx,     y=jy - 5, gray=5.0))
    g.add_node(Node(id=4, x=jx,     y=jy + 5, gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    return g


def _mascara_l(shape=(25, 25), jx=10, jy=10):
    """Branco no braço horizontal e vertical; verde no restante."""
    branco = np.zeros(shape, dtype=np.uint8)
    verde  = np.zeros(shape, dtype=np.uint8)
    branco[jy, :jx + 1] = 255
    branco[:jy + 1, jx] = 255
    verde[:, :] = 255
    verde[branco > 0] = 0
    return branco, verde


def _mascara_t(shape=(25, 25), jx=10, jy=10):
    """Branco: linha horizontal + haste vertical acima; verde: restante."""
    branco = np.zeros(shape, dtype=np.uint8)
    verde  = np.zeros(shape, dtype=np.uint8)
    branco[jy, :] = 255
    branco[:jy + 1, jx] = 255
    verde[:, :] = 255
    verde[branco > 0] = 0
    return branco, verde


def _mascara_x(shape=(25, 25), jx=10, jy=10):
    """Branco: cruz completa; verde: restante."""
    branco = np.zeros(shape, dtype=np.uint8)
    verde  = np.zeros(shape, dtype=np.uint8)
    branco[jy, :] = 255
    branco[:, jx] = 255
    verde[:, :] = 255
    verde[branco > 0] = 0
    return branco, verde


def _mascara_t_falso(shape=(25, 25), jx=10, jy=10):
    """T falso: forma branca correta mas verde = 0 em toda a imagem."""
    branco = np.zeros(shape, dtype=np.uint8)
    verde  = np.zeros(shape, dtype=np.uint8)
    branco[jy, :] = 255
    branco[:jy + 1, jx] = 255
    return branco, verde   # verde inteiramente 0


# ─── Test A: classificação L/T/X ─────────────────────────────────────────────

def test_classificar_l():
    """Nó grau 2 com ângulo 90° → tipo 'L'."""
    g = _grafo_l()
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    candidatos = classificar_candidatos(g, cfg)
    assert any(c.tipo == 'L' for c in candidatos), "esperado L"


def test_classificar_t():
    """Nó grau 3 → tipo 'T'."""
    g = _grafo_t()
    candidatos = classificar_candidatos(g, ConfigCruzamento())
    assert any(c.tipo == 'T' for c in candidatos)


def test_classificar_x():
    """Nó grau 4 → tipo 'X'."""
    g = _grafo_x()
    candidatos = classificar_candidatos(g, ConfigCruzamento())
    assert any(c.tipo == 'X' for c in candidatos)


def test_grau2_reto_nao_e_l():
    """Grau 2 com ângulo ≈180° (linha reta) não é candidato L."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=0,  y=10, gray=5.0))
    g.add_node(Node(id=1, x=10, y=10, gray=5.0))
    g.add_node(Node(id=2, x=20, y=10, gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    assert not any(c.tipo == 'L' for c in classificar_candidatos(g, cfg))


# ─── Test B: janela de ângulo para L (D20) ────────────────────────────────────

def test_l_90_graus_aceito():
    """L de 90° (perpendicular) → candidato L com janela 50–130°."""
    g = _grafo_l()
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    assert any(c.tipo == 'L' for c in classificar_candidatos(g, cfg))


def test_l_perspectiva_60_aceito():
    """L de 60° (perspectiva aguda) → candidato L com janela 50–130°."""
    # braço 1 aponta para direita (0°): (20, 10); braço 2 aponta ~61° acima: (15, 19)
    # ângulo real: acos(50 / (10 * sqrt(106))) ≈ 60.9° ∈ [50°, 130°]
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    g.add_node(Node(id=1, x=20, y=10, gray=5.0))  # braço direito (0°)
    g.add_node(Node(id=2, x=15, y=19, gray=5.0))  # braço ~61° acima-direita
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    candidatos = classificar_candidatos(g, cfg)
    l_cands = [c for c in candidatos if c.tipo == 'L']
    assert l_cands, "L de ~60° deve ser candidato"


def test_l_perspectiva_120_aceito():
    """L de 120° (perspectiva obtusa dentro da janela) → candidato L."""
    # braço 1: esquerdo (180°); braço 2: 60° (acima-direito) → ângulo entre eles = 120°
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    g.add_node(Node(id=1, x=5,  y=10, gray=5.0))  # esquerdo (braço na dir 180°)
    # vetor 60°: dx=cos(60°)*5=2.5→3, dy=-sin(60°)*5=-4.3→-4 (acima-direito)
    g.add_node(Node(id=2, x=13, y=6, gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    candidatos = classificar_candidatos(g, cfg)
    l_cands = [c for c in candidatos if c.tipo == 'L']
    assert l_cands, "L de ~120° deve ser candidato"


@pytest.mark.parametrize("angulo_grau", [150, 155, 160, 165, 170])
def test_grau2_quase_reto_nao_e_l(angulo_grau):
    """Vértices de poligonal circular com ângulos 150-170° NÃO viram L (D20)."""
    # Nó central, braço esquerdo e braço em `angulo_grau` do eixo horizontal
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    g.add_node(Node(id=1, x=5,  y=10, gray=5.0))  # braço esquerdo
    # O segundo braço aponta de forma que o ângulo ENTRE os dois braços = angulo_grau
    # braço 1 aponta para 180°; braço 2 aponta para (180° - angulo_grau) para formar angulo
    ang2 = math.radians(180.0 - angulo_grau)  # direção do segundo braço
    x2 = int(round(10 + 5 * math.cos(ang2)))
    y2 = int(round(10 + 5 * math.sin(ang2)))
    g.add_node(Node(id=2, x=x2, y=y2, gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    l_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'L']
    assert not l_cands, f"ângulo {angulo_grau}° não deve ser candidato L (janela 50-130°)"


# ─── Test C: orientação ───────────────────────────────────────────────────────

def test_orientacao_l_aponta_para_dentro():
    """Orientação do L aponta para o interior do canto."""
    g = _grafo_l(jx=10, jy=10)
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0)
    candidatos = classificar_candidatos(g, cfg)
    l_cands = [c for c in candidatos if c.tipo == 'L']
    assert l_cands
    assert isinstance(l_cands[0].orientacao, float)


def test_orientacao_t_haste():
    """Orientação do T aponta para a haste (≈ -π/2 para haste acima)."""
    g = _grafo_t(jx=10, jy=10)
    cfg = ConfigCruzamento()
    t_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'T']
    assert t_cands
    ang = t_cands[0].orientacao
    assert abs(ang - (-math.pi / 2)) < 0.3 or abs(ang - (3 * math.pi / 2)) < 0.3, \
        f"orientação da haste deveria ser ≈-π/2, got {ang:.3f}"


def test_orientacao_x_zero():
    """X não tem orientação preferencial → 0.0."""
    g = _grafo_x()
    x_cands = [c for c in classificar_candidatos(g, ConfigCruzamento()) if c.tipo == 'X']
    assert x_cands
    assert x_cands[0].orientacao == 0.0


# ─── Test D: verificação com máscaras sintéticas ─────────────────────────────

def test_verificacao_l_aceito():
    """L sintético com máscaras corretas → confiança > 0.5."""
    g = _grafo_l(jx=10, jy=10)
    branco, verde = _mascara_l(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0,
                           raios_circulo=[5])
    l_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'L']
    assert l_cands
    c = verificar_cruzamento(l_cands[0], g, branco, verde, cfg)
    assert c.confianca > 0.5, f"L bem formado → confiança > 0.5, got {c.confianca}"


def test_verificacao_t_aceito():
    """T sintético com máscaras corretas → confiança > 0.5."""
    g = _grafo_t(jx=10, jy=10)
    branco, verde = _mascara_t(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(raios_circulo=[5])
    t_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'T']
    assert t_cands
    c = verificar_cruzamento(t_cands[0], g, branco, verde, cfg)
    assert c.confianca > 0.5


def test_verificacao_x_aceito():
    """X sintético com máscaras corretas → confiança > 0.5."""
    g = _grafo_x(jx=10, jy=10)
    branco, verde = _mascara_x(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(raios_circulo=[5])
    x_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'X']
    assert x_cands
    c = verificar_cruzamento(x_cands[0], g, branco, verde, cfg)
    assert c.confianca > 0.5


def test_t_falso_rejeitado():
    """T falso (verde=0 em todo o círculo) → V1=0 → confiança ≤ limiar."""
    g = _grafo_t(jx=10, jy=10)
    branco, verde = _mascara_t_falso(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(raios_circulo=[5], limiar_confianca=0.5)
    t_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'T']
    assert t_cands
    c = verificar_cruzamento(t_cands[0], g, branco, verde, cfg)
    assert c.confianca <= cfg.limiar_confianca, \
        f"T falso deve ter confiança ≤ {cfg.limiar_confianca}, got {c.confianca}"


# ─── Test E: X no cruzamento círculo-reta é aceito ───────────────────────────

def test_x_circulo_reta_aceito():
    """X onde círculo encontra linha horizontal: 4 braços, verde em 4 setores → aceito."""
    g = _grafo_x(jx=10, jy=10)
    branco, verde = _mascara_x(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(raios_circulo=[5], limiar_confianca=0.5)
    x_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'X']
    assert x_cands
    c = verificar_cruzamento(x_cands[0], g, branco, verde, cfg)
    assert c.confianca >= cfg.limiar_confianca, \
        f"X bem formado deve ser aceito, confiança={c.confianca}"


# ─── Test F: cruzamento no limite da imagem não quebra ───────────────────────

def test_cruzamento_no_limite_nao_quebra():
    """Cruzamento no canto da imagem não deve levantar exceção."""
    g = _grafo_l(jx=1, jy=1)
    branco = np.zeros((5, 5), dtype=np.uint8)
    verde  = np.zeros((5, 5), dtype=np.uint8)
    cfg = ConfigCruzamento(raios_circulo=[4], angulo_L_min_grau=50.0,
                           angulo_L_max_grau=130.0)
    for c in classificar_candidatos(g, cfg):
        verificar_cruzamento(c, g, branco, verde, cfg)  # não deve levantar


# ─── Test G: detectar_cruzamentos ────────────────────────────────────────────

def test_detectar_cruzamentos_aceita_limiar():
    """detectar_cruzamentos filtra por limiar_confianca."""
    g = _grafo_t(jx=10, jy=10)
    branco, verde = _mascara_t(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(raios_circulo=[5], limiar_confianca=0.0)
    aceitos, rejeitados = detectar_cruzamentos(g, branco, verde, cfg)
    assert len(aceitos) + len(rejeitados) >= 1


def test_detectar_debug_info():
    """_debug_info deve conter detalhes de cada candidato, incluindo raio_usado."""
    g = _grafo_t(jx=10, jy=10)
    branco, verde = _mascara_t(shape=(25, 25), jx=10, jy=10)
    cfg = ConfigCruzamento(raios_circulo=[5], limiar_confianca=0.0)
    info = []
    aceitos, rejeitados = detectar_cruzamentos(g, branco, verde, cfg, _debug_info=info)
    assert len(info) >= 1
    for d in info:
        for chave in ('tipo', 'x', 'y', 'v1', 'v2', 'n_branco', 'n_verde',
                      'raio_usado', 'confianca', 'aceito', 'motivo'):
            assert chave in d, f"chave '{chave}' ausente em debug_info"


# ─── Test H: triângulo na junção é desfeito (Etapa 6 diagonal) ───────────────

def test_triangulo_juncao_desfeito():
    """Etapa 6: triângulo de 3 nós em junção desfeito por remover_diagonais_espurias."""
    from features.simplify import remover_diagonais_espurias, ConfigSimplificacao

    g = StructureGraph()
    g.add_node(Node(id=0, x=5, y=5, gray=5.0))  # J
    g.add_node(Node(id=1, x=4, y=5, gray=5.0))  # N (braço esquerdo)
    g.add_node(Node(id=2, x=5, y=4, gray=5.0))  # W (braço cima)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)  # diagonal espúria

    cfg = ConfigSimplificacao(limiar_diagonal_px=2.0)
    removidas = remover_diagonais_espurias(g, cfg)

    assert removidas >= 1, "diagonal N-W deve ser removida"
    assert 2 not in g.neighbors(1)
    assert g.degree(0) == 2
    assert all(n in g.nodes for n in (0, 1, 2))


# ─── Test I: verificação de geometria T e X (D22) ────────────────────────────

def test_t_simetrico_nao_e_candidato():
    """T com três braços a 120° (sem par oposto) não é candidato com geometry check."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    for i, ang_deg in enumerate([0, 120, 240]):
        ang = math.radians(ang_deg)
        g.add_node(Node(id=i + 1,
                        x=int(round(10 + 5 * math.cos(ang))),
                        y=int(round(10 + 5 * math.sin(ang))), gray=5.0))
        g.add_edge(0, i + 1)
    cfg = ConfigCruzamento(verificar_geometria_t=True)
    t_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'T']
    assert not t_cands, "T com braços simétricos a 120° não deve ser candidato"


def test_t_simetrico_aceito_sem_geometria():
    """Sem geometry check, o mesmo nó vira candidato T (verificar_geometria_t=False)."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    for i, ang_deg in enumerate([0, 120, 240]):
        ang = math.radians(ang_deg)
        g.add_node(Node(id=i + 1,
                        x=int(round(10 + 5 * math.cos(ang))),
                        y=int(round(10 + 5 * math.sin(ang))), gray=5.0))
        g.add_edge(0, i + 1)
    cfg = ConfigCruzamento(verificar_geometria_t=False)
    t_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'T']
    assert t_cands, "Sem geometry check, nó grau 3 deve ser candidato T"


def test_x_falso_nao_e_candidato():
    """X com 4 braços em 0°,60°,120°,180° (só 1 par oposto) não é candidato."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    for i, ang_deg in enumerate([0, 60, 120, 180]):
        ang = math.radians(ang_deg)
        g.add_node(Node(id=i + 1,
                        x=int(round(10 + 5 * math.cos(ang))),
                        y=int(round(10 + 5 * math.sin(ang))), gray=5.0))
        g.add_edge(0, i + 1)
    cfg = ConfigCruzamento(verificar_geometria_x=True, angulo_par_x_grau=140.0)
    x_cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'X']
    assert not x_cands, "X sem dois pares opostos não deve ser candidato"


# ─── Test I: V2 por BFS segue braços curvilíneos ────────────────────────────

def test_x_braco_curvo_v2_aceito():
    """X com braço curvilíneo: BFS encontra o vizinho mesmo sem linha reta branca.

    O caminho horizontal y=10, x=11..19 é deliberadamente escuro;
    o vizinho direito só é acessível via arco branco.
    """
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))   # junção
    g.add_node(Node(id=1, x=5,  y=10, gray=5.0))   # braço esquerdo (reto)
    g.add_node(Node(id=2, x=10, y=5,  gray=5.0))   # braço superior (reto)
    g.add_node(Node(id=3, x=10, y=15, gray=5.0))   # braço inferior (reto)
    g.add_node(Node(id=4, x=20, y=10, gray=5.0))   # braço direito (via arco)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(0, 4)

    branco = np.zeros((25, 25), dtype=np.uint8)
    verde  = np.zeros((25, 25), dtype=np.uint8)

    # Braços retos: esq, cima, baixo
    branco[10, 5:11] = 255    # esquerdo
    branco[5:11, 10] = 255    # cima
    branco[10:16, 10] = 255   # baixo

    # Arco curvo para (x=20,y=10): não passa por y=10,x=11..19 (mantidos escuros)
    for y, x in [(10, 10), (9, 11), (8, 12), (8, 13), (8, 14),
                 (8, 15), (9, 16), (10, 17), (10, 18), (10, 19), (10, 20)]:
        branco[y, x] = 255

    verde[:, :] = 255
    verde[branco > 0] = 0

    cfg = ConfigCruzamento(limiar_confianca=0.5)
    aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)
    assert any(c.tipo == 'X' for c in aceitos), \
        "X com braço curvilíneo deve ser aceito via BFS V2"


# ─── Test J: supressão de duplicatas ─────────────────────────────────────────

def test_suprimir_duplicatas_dois_l():
    """Dois candidatos L a < r_fusao px: só um permanece em aceitos."""
    g = StructureGraph()
    # L1 at (10, 10): braços esq (5,10) e cima (10,5)
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    g.add_node(Node(id=1, x=5,  y=10, gray=5.0))
    g.add_node(Node(id=2, x=10, y=5,  gray=5.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    # L2 at (12, 10): distância 2 px < r_fusao=3; braços dir (17,10) e cima (12,5)
    g.add_node(Node(id=3, x=12, y=10, gray=5.0))
    g.add_node(Node(id=4, x=17, y=10, gray=5.0))
    g.add_node(Node(id=5, x=12, y=5,  gray=5.0))
    g.add_edge(3, 4)
    g.add_edge(3, 5)

    branco = np.zeros((25, 25), dtype=np.uint8)
    verde  = np.zeros((25, 25), dtype=np.uint8)
    branco[10, 5:18] = 255    # linha horizontal
    branco[5:11, 10] = 255    # braço cima do L1
    branco[5:11, 12] = 255    # braço cima do L2
    verde[:, :] = 255
    verde[branco > 0] = 0

    cfg = ConfigCruzamento(
        r_fusao_px=3.0,
        angulo_L_min_grau=50.0, angulo_L_max_grau=130.0,
        limiar_confianca=0.0,
    )
    aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)
    l_aceitos = [c for c in aceitos if c.tipo == 'L']
    assert len(l_aceitos) == 1, \
        f"dois L a 2 px com r_fusao=3 → deve sobrar 1 L, sobrou {len(l_aceitos)}"


# ─── Test K: angulo L por arco (D24) ─────────────────────────────────────────

def test_l_arredondado_detectado_por_arco():
    """D24: canto L distribuido em 5 nos — arco detecta onde imediato rejeita.

    Cadeia: (0,10)-(5,10)-(8,8)-(10,5)-(10,0).
    No (8,8) tem angulo imediato ~157 graus (rejeitado), arco ~118 graus (aceito).
    """
    g = StructureGraph()
    for nid, x, y in [(0, 0, 10), (1, 5, 10), (2, 8, 8), (3, 10, 5), (4, 10, 0)]:
        g.add_node(Node(id=nid, x=x, y=y, gray=5.0))
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 3); g.add_edge(3, 4)

    cfg_imediato = ConfigCruzamento(
        angulo_L_min_grau=50.0, angulo_L_max_grau=130.0,
        distancia_arco_l_px=0,
    )
    cfg_arco = ConfigCruzamento(
        angulo_L_min_grau=50.0, angulo_L_max_grau=130.0,
        distancia_arco_l_px=8,
    )

    cands_im = [c for c in classificar_candidatos(g, cfg_imediato) if c.tipo == 'L']
    cands_arco = [c for c in classificar_candidatos(g, cfg_arco) if c.tipo == 'L']

    assert not any(c.x == 8 and c.y == 8 for c in cands_im), \
        "medicao imediata deve rejeitar no (8,8): angulo ~157 graus"
    assert any(c.tipo == 'L' for c in cands_arco), \
        "medicao por arco deve detectar canto arredondado"


def test_linha_reta_com_ruido_nao_e_l():
    """D24: linha quasi-reta com 1 px de ruido nao e candidato L com medicao por arco.

    Cadeia: (0,10)-(5,11)-(10,10)-(15,11)-(20,10); angulo no (5,11) ~168 graus.
    """
    g = StructureGraph()
    for nid, x, y in [(0, 0, 10), (1, 5, 11), (2, 10, 10), (3, 15, 11), (4, 20, 10)]:
        g.add_node(Node(id=nid, x=x, y=y, gray=5.0))
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 3); g.add_edge(3, 4)

    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0,
                           distancia_arco_l_px=7)
    cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'L']
    assert not cands, "linha com ruido de 1 px nao deve ser candidato L"


def test_arco_suave_nao_e_l():
    """D24: arco suave de circulo (raio~20) nao e candidato L.

    Cadeia: (20,0)-(19,5)-(17,10)-(14,14)-(10,17).
    Todos os nos internos tem angulo de arco > 130 graus.
    """
    g = StructureGraph()
    for nid, x, y in [(0, 20, 0), (1, 19, 5), (2, 17, 10), (3, 14, 14), (4, 10, 17)]:
        g.add_node(Node(id=nid, x=x, y=y, gray=5.0))
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 3); g.add_edge(3, 4)

    cfg = ConfigCruzamento(angulo_L_min_grau=50.0, angulo_L_max_grau=130.0,
                           distancia_arco_l_px=7)
    cands = [c for c in classificar_candidatos(g, cfg) if c.tipo == 'L']
    assert not cands, "arco suave de circulo nao deve ser candidato L"


# ─── Test L: D26 — arco no verificar_geometria_t ──────────────────────────────

def test_t_haste_45_detectado_com_d26():
    """D26 explícito: aceita T com haste a 45°; padrão (min=50°) rejeita.

    Barra A(0,10)–J(10,10)–B(20,10) (180°). Haste J→C(17,17): ângulo=45° com B, 135° com A.
    Padrão (min=50°,max=130°): 45°<50° e 135°>130° → rejeitado.
    D26 explícito (min=35°,max=145°): 45°≥35° ✓ e 135°≤145° ✓ → aceito.
    """
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))  # J
    g.add_node(Node(id=1, x=0,  y=10, gray=5.0))  # A
    g.add_node(Node(id=2, x=20, y=10, gray=5.0))  # B
    g.add_node(Node(id=3, x=17, y=17, gray=5.0))  # C haste 45°
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)

    cfg_d26    = ConfigCruzamento(angulo_stem_t_min_grau=35.0, angulo_stem_t_max_grau=145.0)
    cfg_padrao = ConfigCruzamento()  # min=50°, max=130°

    cands_d26 = [c for c in classificar_candidatos(g, cfg_d26) if c.tipo == 'T']
    cands_pad = [c for c in classificar_candidatos(g, cfg_padrao) if c.tipo == 'T']

    assert cands_d26, "D26 (min=35°,max=145°): haste 45° deve ser aceita"
    assert not cands_pad, "padrão (min=50°,max=130°): haste 45° deve ser rejeitada"


def test_t_haste_curva_arco_detectado_d26():
    """D26: haste curvada detectada via arco+ângulos D26 mas não via padrão.

    Barra A(0,10)–J(10,10)–B(20,10). Haste curva: J→(12,11)→(15,13)→(17,16).
    Padrão (arc=0, min=50°): imediato≈27°<50° → falha.
    D26 explícito (arc=7px, min=35°, max=145°): arco≈41°≥35° e 139°≤145° → passa.
    """
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))  # J
    g.add_node(Node(id=1, x= 0, y=10, gray=5.0))  # A
    g.add_node(Node(id=2, x=20, y=10, gray=5.0))  # B
    g.add_node(Node(id=3, x=12, y=11, gray=5.0))  # N1 grau=2
    g.add_node(Node(id=4, x=15, y=13, gray=5.0))  # N2 grau=2
    g.add_node(Node(id=5, x=17, y=16, gray=5.0))  # N3 endpoint
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)
    g.add_edge(3, 4); g.add_edge(4, 5)

    cfg_arco_d26 = ConfigCruzamento(distancia_arco_geom_px=7,
                                     angulo_stem_t_min_grau=35.0,
                                     angulo_stem_t_max_grau=145.0)
    cfg_padrao   = ConfigCruzamento()  # arc=0, min=50°

    cands_arco = [c for c in classificar_candidatos(g, cfg_arco_d26) if c.tipo == 'T']
    cands_pad  = [c for c in classificar_candidatos(g, cfg_padrao)   if c.tipo == 'T']

    assert cands_arco, "D26 (arc=7, min=35°): arco≈41°≥35° deve detectar T"
    assert not cands_pad, "padrão (arc=0, min=50°): imediato≈27°<50° não detecta T"


# ─── Test M: D29 — candidatos com círculo fora da grade ───────────────────────

def test_borda_rejeita_candidato_circulo_cortado():
    """D29: candidato cujo círculo de verificação sai da grade deve ser rejeitado."""
    # Grade 20x20; T em (3,10); raio máximo=8 → x-8=-5 < 0 → fora
    g = StructureGraph()
    g.add_node(Node(id=0, x=3,  y=10, gray=5.0))  # T perto da borda esquerda
    g.add_node(Node(id=1, x=0,  y=10, gray=5.0))  # braço esquerdo (sai da borda)
    g.add_node(Node(id=2, x=8,  y=10, gray=5.0))  # braço direito
    g.add_node(Node(id=3, x=3,  y=5,  gray=5.0))  # haste (acima)
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)

    branco = np.zeros((20, 20), dtype=np.uint8)
    verde  = np.zeros((20, 20), dtype=np.uint8)
    branco[10, 0:9] = 255
    branco[5:11, 3] = 255
    verde[:, :] = 255; verde[branco > 0] = 0

    cfg = ConfigCruzamento(raios_circulo=[4, 5, 6, 8], verificar_borda=True,
                           limiar_confianca=0.0)
    aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)
    assert not any(c.x == 3 and c.y == 10 for c in aceitos), \
        "T perto da borda (círculo sai) deve ser rejeitado"


def test_interior_aceita_candidato_longe_borda():
    """D29: candidato com círculo dentro da grade não é descartado por borda."""
    g = StructureGraph()
    g.add_node(Node(id=0, x=10, y=10, gray=5.0))
    g.add_node(Node(id=1, x=0,  y=10, gray=5.0))
    g.add_node(Node(id=2, x=20, y=10, gray=5.0))
    g.add_node(Node(id=3, x=10, y=0,  gray=5.0))
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3)

    branco = np.zeros((25, 25), dtype=np.uint8)
    verde  = np.zeros((25, 25), dtype=np.uint8)
    branco[10, 0:21] = 255
    branco[0:11, 10] = 255
    verde[:, :] = 255; verde[branco > 0] = 0

    cfg = ConfigCruzamento(raios_circulo=[4, 5, 6, 8], verificar_borda=True,
                           limiar_confianca=0.0)
    aceitos, _ = detectar_cruzamentos(g, branco, verde, cfg)
    assert any(c.x == 10 and c.y == 10 for c in aceitos), \
        "T bem no interior (10px de distância à borda ≥ raio=8) deve ser aceito"
