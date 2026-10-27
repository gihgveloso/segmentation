# features/ — Seções III-C e IV (Schulz et al., 2009)

Documentação da minha parte do repositório Segmentation_for_robocup.
A parte III-A e III-B foi implementada pelo colega em `main.py`; esta documentação
cobre apenas os módulos que escrevi.

---

## Comandos

```bash
# Visualizador interativo (a/d=navegar, q=sair, n/g/c/l/t=camadas)
python run_features.py

# Modo sem display: salva PNGs em debug_out/
python run_features.py --salvar

# Imagem específica
python run_features.py images/image_1.png

# Avaliação quantitativa contra gabarito (requer tests/gabarito_cruzamentos.json)
python avaliar_cruzamentos.py --tolerancia 6

# Anotar gabarito interativamente
python anotar_cruzamentos.py

# Debug detalhado de extração de linhas (tabela por imagem)
python run_etapa8.py

# Testes
pytest
```

---

## API

### `extrair_features(resultado_processar, cfg=None) -> FeaturesResult`

Ponto de entrada único. Encadeia todas as etapas.

```python
from main import Pixel_Segment
from features import extrair_features

ps  = Pixel_Segment('green_pixels.csv')
img = cv2.imread('images/image_1.png')
res = ps.processar(img)          # dict do colega (III-A/III-B)

result = extrair_features(res)
print(result.crossings)          # list[CrossingFeature]
print(result.lines)              # list[LineFeature]
print(result.tamanho_grade)      # (H, W) da grade subamostrada
print(result.fator_escala)       # 8 (sempre)
```

**Encadeamento interno:**

```
resultado_processar
  ↓ skeleton_img, mascara_relevo, white_mask_resized, green_mask_resized
  ↓
construir_grafo (III-C, Etapas 1–5)   ← branco ativo (D37)
  ├──→ extrair_linhas (IV-B, grafo pré-simplificação)  →  LineFeature[]
  └──→ simplificar (IV-A, Etapa 6)
         └──→ detectar_cruzamentos (IV-A, Etapa 7)     →  CrossingFeature[]
```

As linhas usam o grafo **antes** da simplificação (D31); cruzamentos usam o grafo
**após** a simplificação.

---

## Dicionário de CrossingFeature

Definição em `features/types.py`.

| Campo | Tipo | Unidade | Descrição |
|---|---|---|---|
| `tipo` | `str` | — | `'L'`, `'T'` ou `'X'` |
| `x` | `int` | célula (grade) | Coluna na grade subamostrada |
| `y` | `int` | célula (grade) | Linha na grade subamostrada |
| `orientacao` | `float` | radianos | Direção da haste (T), bissetriz do interior (L), 0.0 (X) |
| `confianca` | `float` | 0.0–1.0 | Média de V1 (círculo) e V2 (caminho branco) |

**Converter para coordenadas da imagem original:** `x_orig = x * fator_escala`.

---

## Dicionário de LineFeature

Definição em `features/types.py`.

| Campo | Tipo | Unidade | Descrição |
|---|---|---|---|
| `x1`, `y1` | `float` | célula (grade) | Extremo 1 (projeção na reta ajustada) |
| `x2`, `y2` | `float` | célula (grade) | Extremo 2 (projeção na reta ajustada) |
| `x1_orig`, `y1_orig` | `float` | pixel (imagem) | Extremo 1 escalado por `fator_escala` |
| `x2_orig`, `y2_orig` | `float` | pixel (imagem) | Extremo 2 escalado por `fator_escala` |
| `distancia` | `float` | célula (grade) | Comprimento do segmento ajustado |
| `angulo` | `float` | radianos | `atan2(dy, dx)` em `[-π, π]` |
| `n_nos` | `int` | — | Número de nós do componente ajustado |
| `confianca` | `float` | 0.0–1.0 | Fração de nós com desvio ortogonal < `desvio_max/2` |
| `d_topo_celulas` | `float` ou `None` | célula (grade) | Distância vertical do centro ao `y_topo_verde`; negativo = acima do verde |

---

## Parâmetros de configuração

Todos os parâmetros estão em dataclasses exportadas pelos módulos `features/`.
Passe um `ConfigFeatures` para `extrair_features` para ajustar qualquer parâmetro.

### ConfigNos (`features/nodes.py`)

| Parâmetro | Padrão | Onde é usado |
|---|---|---|
| `dist_min_nos` | 2 | III-C: distância mínima entre nós de crista |
| `usar_relevo` | `True` | III-C: usa relevo float para c(x,y) (D1) |

### ConfigSubdivisao (`features/connections.py`)

| Parâmetro | Padrão | Onde é usado |
|---|---|---|
| `max_segmento_px` | 30 | III-C: comprimento máximo antes de dividir em cadeia (D13) |
| `passo_override` | 0.0 | III-C: passo fixo (0 = auto por mediana de arestas) |

### ConfigConexao (`features/connections.py`)

| Parâmetro | Padrão | Onde é usado |
|---|---|---|
| `dist_max` | 8.0 | III-C: distância máxima para candidatura (Etapa 5) |
| `limiar_custo` | 10.0 | III-C: custo máximo para aceitar candidata |
| `peso_angular` | 1.0 | III-C: peso da penalidade angular no custo (D11) |
| `peso_cinza` | 1.0 | III-C: peso do custo de cinza |
| `dist_grafo_min` | 5 | III-C: mín. hops para aceitar laço intra-componente (D12) |
| `fator_laco_grande` | 1.5 | III-C: exceção para laços grandes (D12) |
| `filtrar_escuras` | `False` | III-C: filtro por custo de cinza mínimo (desligado) |
| `cinza_min` | 0.5 | III-C: cinza mínimo para aceitar candidata quando `filtrar_escuras=True` |
| `suporte_candidata_min` | 0.5 | III-C: frac branco mínima para candidata longa (D37) |
| `dist_min_filtro` | 4.0 | III-C: comprimento mínimo para acionar D37 |

### ConfigSimplificacao (`features/simplify.py`)

| Parâmetro | Padrão | Onde é usado |
|---|---|---|
| `limiar_diagonal_px` | 2.5 | IV-A: distância máxima para remover diagonal espúria (D16-1) |
| `limiar_angulo_grau2` | 170.0 | IV-A: ângulo mínimo para remover nó de grau 2 (D16-2) |
| `comprimento_espurio_px` | 5.0 | IV-A: ramo espúrio menor que isso é removido (D16-3) |
| `distancia_fusao_px` | 3.0 | IV-A: distância máxima para fundir junções (D17) |
| `usar_fusao_juncoes` | `True` | IV-A: fundir clusters de junções (D23) |
| `r_fusao_juncoes` | 6.0 | IV-A: raio de busca para fusão de clusters (D23) |
| `usar_fusao_t_par` | `True` | IV-A: fundir par de T colineares em X (D27) |
| `r_fusao_t_par_px` | 10.0 | IV-A: distância máxima para fusão de par T (D27) |
| `angulo_barra_t_par` | 160.0 | IV-A: ângulo mínimo para considerar braço como "barra" no par T (D27) |
| `max_arestas_fusao` | 8 | IV-A: máximo de arestas em nó para aceitar fusão de cluster (D23) |

### ConfigCruzamento (`features/crossings.py`)

| Parâmetro | Padrão | Onde é usado |
|---|---|---|
| `angulo_L_min_grau` | 50.0 | IV-A: ângulo mínimo para candidato L (D20) |
| `angulo_L_max_grau` | 130.0 | IV-A: ângulo máximo para candidato L (D20) |
| `distancia_arco_l_px` | 7 | IV-A: distância de arco para medir ângulo L (D24) |
| `verificar_geometria_t` | `True` | IV-A: verifica geometria do candidato T (D22) |
| `angulo_barra_t_grau` | 140.0 | IV-A: ângulo mínimo para braço "barra" do T (D22) |
| `angulo_stem_t_min_grau` | 50.0 | IV-A: ângulo mínimo para haste do T (D22) |
| `angulo_stem_t_max_grau` | 130.0 | IV-A: ângulo máximo para haste do T (D22) |
| `distancia_arco_geom_px` | 0 | IV-A: raio de arco para geometria T/X; 0 = desligado (D26 revertido) |
| `verificar_geometria_x` | `True` | IV-A: verifica geometria do candidato X (D22) |
| `angulo_par_x_grau` | 140.0 | IV-A: ângulo mínimo para par oposto no X (D22) |
| `usar_consenso_v1` | `True` | IV-A: exige V1 > 0 para aceitar (desligado por D28) |
| `raios_circulo` | `[4,5,6,8]` | IV-A: raios para V1 multi-raio (D21) |
| `tolerancia_segmentos` | 1 | IV-A: tolerância em nº de segmentos no círculo V1 |
| `min_raios_consenso` | 2 | IV-A: mín. raios com score > 0 para aceitar V1 |
| `max_passos_v2` | 60 | IV-A: passos máximos no BFS V2 por direção |
| `n_amostras_circulo` | 36 | IV-A: pontos amostrados no círculo de V1 |
| `limiar_confianca` | 0.5 | IV-A: confiança mínima para aceitar cruzamento |
| `r_fusao_px` | 3.0 | IV-A: raio para suprimir duplicatas (D21) |
| `verificar_borda` | `True` | IV-A: descartar candidatos com círculo fora da grade (D29) |

### ConfigLinhas (`features/lines.py`)

| Parâmetro | Padrão | Onde é usado |
|---|---|---|
| `desvio_max_celulas` | 2.0 | IV-B: limiar de split por desvio ortogonal |
| `min_nos_componente` | 3 | IV-B: mínimo de nós para formar linha |
| `distancia_min_celulas` | 1.0 | IV-B: descarta segmentos degenerados |
| `usar_fusao` | `True` | IV-B: fundir segmentos colineares após split |
| `limiar_angulo_fusao_grau` | 5.0 | IV-B: ângulo máximo para fusão |
| `limiar_dist_ortogonal_fusao` | 2.0 | IV-B: dist. ortogonal máxima para fusão |
| `lacuna_max_celulas` | 20.0 | IV-B: gap máximo para permitir fusão (D36) |
| `suporte_vao_min` | 0.8 | IV-B: frac branco mínima no vão (D36) |
| `suporte_aresta_min` | 0.5 | IV-B: frac branco mínima para aresta longa (D34) |
| `dist_aresta_min` | 4.0 | IV-B: comprimento mínimo para checar suporte de aresta |
| `projetar_no_chao` | `False` | IV-B: projeção perspectiva (requer calibração) |

---

## Limitações conhecidas (com números medidos)

### Detector de cruzamentos (Etapa 7)

Avaliação com `python3 avaliar_cruzamentos.py --tolerancia 6` nas 3 imagens de `images/`
(GT=12, Det=27, config padrão, medido em 27/10/2026). Ver D37 para comparação sem filtro de branco (Det=28).

**Totais por imagem (localização, qualquer tipo):**

| Imagem | GT | Det | TP | FP | FN |
|--------|----|----:|---:|---:|---:|
| image_1 | 4 | 2 | 1 | 1 | 3 |
| image_2 | 6 | 16 | 3 | 13 | 3 |
| image | 2 | 9 | 2 | 7 | 0 |
| **Total** | **12** | **27** | **6** | **21** | **6** |

Localização (qualquer tipo): RecA = 50% (6/12), PrecA = 22% (6/27).
Tipo correto: RecA = 25% (3/12), PrecA = 11% (3/27).

**Principais causas de erro não corrigíveis sem alterar III-B:**
- 2 T perdidos: junção inexistente no esqueleto entregue pelo colega
- 3 L perdidos: braços muito curtos (<2 px na grade subamostrada)
- 13 FP em image_2 (concentrados na faixa do fundo do campo, y ≤ 36)

### Extração de linhas (Etapa 8)

Resultado com config padrão (branco passado, lacuna_max=20):

| Imagem | Linhas | Descartados (<3 nós) |
|--------|-------:|---------------------:|
| image_1 | 11 | 11 |
| image_2 | 23 | 21 |
| image | 10 | 11 |

A linha do meio de image_1 (y≈82, dist≈164 células, n_nos=66) e de image (y≈77, dist≈155 células, n_nos=85) emergem como segmentos únicos após D36 (lacuna_max=20).

---

## O que depende do esqueleto III-B (colega)

Não é possível corrigir os pontos abaixo sem alterar `main.py`:

1. **T(153,53) e T(9,45) em image_2**: grau=2 antes da Etapa 6 — a junção T não existe no esqueleto.
2. **X(110,78) em image**: grau=2 antes e depois da Etapa 6 — a 4ª aresta do X não existe.
3. **L(62,68) e L(99,67) em image_1**: braços com < 2 px de comprimento na grade subamostrada.
4. **FP L em curvas**: o esqueleto subamostrado tem nós deg=2 com ângulo na janela [50°, 130°] em regiões sem cruzamentos — artefatos de resolução.

---

## Conformidade com o artigo (B4)

Texto do artigo extraído de `docs/paper.pdf` (pypdf). Seções III-C, IV-A e IV-B lidas na íntegra.
**17 Match · 3 Divergência · 9 Extensão.**

### III-C — Placement and Connection of Nodes

| Função | Status | Trecho do artigo |
|--------|--------|-----------------|
| `neighbor_count` / `_c_relevo` / `_c_binario` | ✓ Match | "the number c(x,y) of neighboring pixels (8-neighborhood) having an equal or higher gray-level is computed and if this number is less than three the central pixel is added to the skeleton" |
| `place_nodes` — picos (c=0) | ✓ Match | "Nodes are placed starting at peaks of the skeleton (c(x,y) = 0)" |
| `place_nodes` — crestas (c=1,2) | ✓ Match | "nodes are placed at least two pixels apart at pixels belonging to ridges (c(x,y) = 1 and c(x,y) = 2)" |
| `connect_nodes` (3×3) | ✓ Match | "inserting connections where 3×3 regions of nodes overlap or touch on the skeleton" |
| `esticar_extremos` | ✓ Match | "Lines are stretched by moving end-nodes to the last pixel of the skeleton" |
| `dividir_grau_zero` | ~ Divergência (D13) | "degree-0 nodes are split to fill in the gaps" — artigo não especifica mecanismo; código usa BFS ao longo do esqueleto com espaçamento pela mediana (D13) |
| `conectar_candidatas` | ✓ Match | "candidate connections are created and evaluated according to continuity, closure and simplicity. Specifically, the distance between nodes should be small and the grayness on the line between nodes should be similar to grayness of nodes" |
| `_frac_branco_candidata` (filtro suporte) | ~ Extensão (D37) | Não está no artigo. Rejeita candidata longa com fração branca insuficiente. |

### IV-A — Line Crossing Detection

| Função | Status | Trecho do artigo |
|--------|--------|-----------------|
| `remover_nos_curvatura_baixa` | ✓ Match | "the lines are smoothed and nodes are removed at locations of low curvature" |
| `remover_espurios` | ✓ Match | "Short lines ending in junctions are eliminated" |
| `fundir_juncoes` / `fusao_juncoes_cluster` | ✓ Match | "junctions that are close together and connected are merged to form a crossing. When merging and moving nodes, we make sure that the new nodes are placed on the skeleton" |
| `fusao_t_par` | ~ Extensão (D27) | O trecho acima cobre apenas fusão de junções próximas. A fusão de dois T colineares em X (barras opostas ≤ `r_fusao_t_par_px`) vai além do artigo. |
| `classificar_candidatos` | ✓ Match | "A degree-2 node with certain angle of edges connected to it is an L-crossing candidate, a degree-3 node is a T-crossing candidate, and a degree-4 node is an X-crossing candidate" |
| V1: `_verificacao1_circulo` / `_amostrar_circulo` / `_contar_segmentos` | ✓ Match | "we first use a state machine to check out the green-white and white-green color transition along a circle centered at the crossing" |
| V2: `_verificacao2_caminho` | ✓ Match | "we check whether there is a white path from the crossing to the next neighbors in each direction. Both checks are performed in the sub-sampled color-images" |
| `_verificacao1_multi_raio` | ~ Extensão (D21/D23) | Artigo usa raio único; código tenta múltiplos raios e retém o melhor score. |
| `_suprimir_duplicatas` | ~ Extensão (D21) | Não está no artigo. |
| `remover_diagonais_espurias` | ~ Extensão (D22) | Não está no artigo. |
| `_verifica_geometria_t` / `_verifica_geometria_x` | ~ Extensão (D19) | Artigo: "certain angle" sem critério; critério de ângulo mínimo é nossa interpretação. |

### IV-B — Field Line Extraction

| Função | Status | Trecho do artigo |
|--------|--------|-----------------|
| `_componentes_grau2` | ✓ Match | "a field line is a connected set of nodes with degree two and nodes connected directly to the set with different degree" |
| `_ajustar_reta` (SVD/TLS) | ~ Divergência (D38) | "we fit a line li to ci using least squares regression [11] on the coordinates of the nodes" — variante não especificada no artigo; usamos TLS (D38) |
| `_split` (split-and-merge) | ✓ Match | "The node n\* = argmax n∈ci dist(n,li) with the largest distance to the fitted line defines a splitting point. We split ci into two components if the node-line distance is sufficiently large and recursively process the resulting components" |
| descarte < 3 nós | ✓ Match | "Components containing less than three nodes are discarded" |
| `_fundir_segmentos` | ✓ Match | "During the merging phase, we merge components ci and cj if the parameters of li and lj are sufficiently similar" |
| `_extremos_projetados` | ✓ Match | "The final field line in image coordinates is determined by projecting the end-points of the component onto the fitted line. We do not use the end-points directly, since these can represent part of another line" |
| `_para_linefeature` (projeção no chão) | ~ Divergência (parcial) | "by projecting both end-points to the floor, we can infer the line parameters (distance and angle)" — artigo faz sempre; código só executa quando `projetar_no_chao=True` (padrão `False`). |
| `_angulo_diff` / `_dist_ortogonal_retas` | ~ Extensão (D33) | Não estão no artigo. Métricas para critério de fusão. |
| `_frac_branco` (gap-check) | ~ Extensão (D33/D36) | Não está no artigo. Gap-check por suporte branco. |
| `_y_topo_verde` | ~ Extensão | Não está no artigo. |

---

## Crédito

O código de segmentação nas Seções III-A e III-B (`main.py`, `test.py`, `def_white_threshold.py`,
`unique_table_values.py`) foi desenvolvido pelo colega no repositório original
[Segmentation_for_robocup](https://github.com/ivan-josef/Segmentation_for_robocup).
Esta documentação cobre apenas as Seções III-C e IV, implementadas nesta branch.
