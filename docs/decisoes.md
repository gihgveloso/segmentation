# Decisões de Implementação

Cada entrada registra a interpretação escolhida para trechos ambíguos do artigo
Schulz et al. (2009), com justificativa e impacto no código.

---

## D1 — c(x,y): relevo float como padrão, com fallback binário

**Decisão:** usar `mascara_relevo` (float32) para calcular c(x,y):
`c(x,y) = nº de 8-vizinhos onde relevo[viz] >= relevo[x,y]`, restrito a pixels do esqueleto.
Fallback binário disponível via `ConfigNos(usar_relevo=False)`:
`c(x,y) = nº de 8-vizinhos que também são pixels do esqueleto`.

**Por quê:** Opção A é fiel ao artigo. `mascara_relevo` está exposto no dict de `processar`.
O fallback existe para testes com esqueletos sintéticos sem mapa de relevo e para configuração.

**Impacto:** `features/nodes.py::neighbor_count(relevo, skeleton, config)`.

---

## D2 — 8-conectividade pura, sem poda de diagonais

**Decisão:** usar as 8 direções sem poda.

**Por quê:** O artigo usa 8-vizinhança sem restrições. A poda do colega
em `_extrair_segmentos_conectados` é uma escolha interna daquele método para evitar
caminhos redundantes, mas não afeta o esqueleto em si. Minha `StructureGraph` é
independente, e a 8-conectividade pura é mais simples e consistente com o artigo.

**Nota:** o esqueleto recebido (`skeleton_img`) já foi gerado com c_xy calculado em
8-conectividade pura (sem poda). A poda do colega só existe na etapa de extração de
caminhos, portanto o esqueleto em si não vem podado.

**Impacto:** `features/nodes.py::_c_relevo` e `_c_binario` usam stack de 8 direções.

---

## D3 — Nós de grau 0 (c_relevo=0 / isolados) mantidos no grafo

**Decisão:** pixels isolados (c_binario=0, nenhum vizinho no esqueleto) recebem
um nó e ficam no grafo. Eles **não são descartados** na Etapa 1 nem nas Etapas 2–5.
O descarte de nós de grau 0 (sem arestas) acontece na Etapa 7 (simplificação, IV-A),
junto com a remoção de espúrios curtos.

**Por quê:** o artigo menciona que nós de grau 0 são "divididos" (split), citando
Behnke 1997, sem detalhar o mecanismo. Manter no grafo e descartar tarde é conservador:
preserva a informação até que a simplificação global decida o que fazer.

**Impacto:** `place_nodes` coloca nó em qualquer pixel com c_relevo=0 (incluindo
isolados). `StructureGraph` permite nós sem arestas. Etapa 7 filtrará.

---

## D4 — Esticar extremos: até o último pixel do esqueleto

**Decisão:** um nó terminal (grau 1 no grafo) é movido ao longo da direção da sua
única aresta até o último pixel do esqueleto existente nessa direção.
Não é extrapolado para além do esqueleto nem até a borda de `mascara_convexa`.

**Por quê:** extrapolar além do esqueleto introduziria pontos sem suporte de dados.
A borda do campo é uma informação de segundo nível; o esqueleto é a fonte de verdade.

**Impacto:** `features/connections.py::esticar_extremos` percorre pixels do esqueleto
na direção do vetor (nó_terminal − nó_vizinho).

---

## D5 — Atributo `gray` do Node

**Decisão:** `gray = float(mascara_relevo[gy, gx])` no momento da criação do nó.
Quando um nó é movido (esticar extremos, Etapa 4), `gray` é atualizado para o valor
de relevo na nova posição.

**Por quê:** o relevo é o dado contínuo usado para ordenar e para verificações
qualitativas (cinza médio na verificação de conexões candidatas, Etapa 5).
Se D1=binário, `gray` recebe 1.0 por convenção (flag `usar_relevo=False`).

**Impacto:** `Node.gray` sempre reflete o relevo atual do nó. `place_nodes` lê
`relevo[gy, gx]` diretamente.

---

## D6 — StructureGraph é independente de `polylines_linhas`/`segmentos_linhas`

**Decisão:** `StructureGraph` é construída exclusivamente a partir de `skeleton`
(bool) e `mascara_relevo` (float32). Nunca lê `polylines_linhas` nem `segmentos_linhas`.

**Por quê:** as estruturas do colega existem para visualização/saída e usam
`approxPolyDP` com um epsilon que não existe no artigo. A minha parte implementa o
grafo do artigo do zero.

**Restrição de teste:** `tests/test_nodes.py` e `tests/test_graph.py` devem ter
um teste que verifique que nenhum módulo de `features/` importa `polylines_linhas`
ou `segmentos_linhas` (grep no código-fonte, não no runtime).

**Impacto:** `run_features.py` pode usar `segmentos_linhas` para comparação visual,
mas a pipeline `features/` não.

---

## D7 — Extremidades binárias sempre recebem nós (garantia de cobertura)

**Decisão (interpretação própria, não explícita no artigo):**
pixels com c_binario=1 (exatamente um vizinho no esqueleto — extremidade topológica)
recebem sempre um nó, mesmo quando estão dentro da distância mínima de outro nó.
Pixels com c_binario=2 (interior) usam a verificação de distância mínima.

**Por quê:** o PLANO.md exige "extremos cobertos" (Etapa 2). Com o algoritmo guloso
por relevo decrescente, a extremidade de uma linha — que tem o menor relevo — é
processada por último e pode ser bloqueada pelo nó anterior. Forçar um nó na
extremidade binária garante que cada ramo do esqueleto tem os dois extremos
representados no grafo, o que é essencial para as Etapas 3–8 (conexão, simplificação,
extração de cruzamentos e linhas).

**Impacto:** `place_nodes` calcula c_binario internamente (mesmo quando D1=relevo)
e aplica a regra: c_binario=1 → nó incondicional; c_binario=2 → nó com dist_min.

---

## D8 — "Regiões se tocam": 8-adjacência entre pixels cobertos por nós diferentes

**Decisão:** dois nós A e B são conectados se existe pixel de esqueleto p em 3×3(A)
e pixel de esqueleto q em 3×3(B) tais que p e q são 8-adjacentes (inclui p == q,
i.e., overlap direto). Modifica `g` in-place.

**Consequência conhecida:** em junções (T, X), a adjacência diagonal cria arestas
espúrias entre braços opostos (ex.: braço esquerdo e braço superior num T).
Os pixels de esqueleto a 1 passo da junção são sempre diagonalmente adjacentes entre
si, o que faz com que arm nodes próximos se conectem diretamente além da junção.
Essas arestas espúrias são aceitas aqui e **removidas na Etapa 6** (simplificação,
IV-A), que descarta arestas redundantes quando há caminho de menor custo passando
pela junção.

**Alternativa descartada:** conexão por sobreposição (shared pixel) apenas, sem toque.
Perde conexões legítimas entre nós espaçados 3 px quando há gap de 1 px entre as
regiões 3×3.

**Impacto:** `features/connections.py::connect_nodes`. Testes de T e X verificam
que a junção alcança todos os braços (`>=`) e que extremos de braços opostos distantes
não se conectam diretamente.

---

## D9 — esticar_extremos: direção discretizada por sign(dy, dx)

**Decisão:** a direção de estiramento de um nó terminal é `(sign(T.y−N.y), sign(T.x−N.x))`,
onde N é o único vizinho do terminal T. O nó avança passo a passo nessa direção
enquanto encontrar pixels de esqueleto; para no último pixel válido.

**Limitação conhecida:** para direções oblíquas (ex: dy=3, dx=1), o sign arredonda
para (1,1) em vez de (1,0), podendo desviar da trajetória real da linha.
Aceito para Etapa 4; se causar problemas em IV-A/B, refatorar para usar o ângulo
real do vetor e escolher o passo 8-conectado mais próximo de `arctan2(dy, dx)`.

**Impacto:** `features/connections.py::esticar_extremos`.

---

## D10 — dividir_grau_zero: extremos = dois pixels mais distantes da componente

**Decisão:** para dividir uma componente de esqueleto sem nós conectados, os dois
novos nós são colocados nos dois pixels da componente com maior distância euclidiana
entre si, calculada por busca exaustiva O(n²) na componente.

**Alternativa descartada:** usar c_binario=1 dentro da componente para identificar
extremidades topológicas. Falha em ângulos de 8-conectividade onde diagonais criam
"junções falsas" (ex.: L-shape com 8-conexo resulta em c=3 no canto, obscurecendo
o extremo real).

**D3 refinamento:** componentes de 1 pixel isolado são mantidas como nó único de
grau 0 sem divisão. A divisão só ocorre para componentes com ≥ 2 pixels.

**Resultado na imagem real (image_1.png):**  
- Antes da Etapa 4: 155 nós, 131 arestas, 24 componentes conexas.  
- Após estirar + dividir: 161 nós, 131 arestas, 30 componentes.  
- 6 nós esticados, 6 componentes divididas (Δ +6 componentes).  
Os nós cinzas (grau 0) no debug são os novos extremos das componentes divididas.

**Impacto:** `features/connections.py::dividir_grau_zero`. IDs dos nós novos são
sempre maiores que qualquer ID existente no início da chamada, evitando colisão
com a lista de iteração.

---

## D11 — Custo de candidatas: dist + cinza + ângulo; seleção gulosa

**Decisão:** o custo de uma aresta candidata entre terminais A e B é:
`custo = dist(A,B) + peso_cinza*(1 - cinza_medio_AB) + peso_angular*penalidade_angular`

Penalidade angular: para cada terminal com grau 1, mede o cosseno entre a direção
de saída da aresta existente e o vetor A→B; penalidade = (1 − cos θ) / 2 ∈ [0, 1].
Terminais de grau 0 não têm penalidade angular.
Seleção gulosa por custo crescente; só aceita se custo < `limiar_custo` e grau < 4.

**Por quê:** o artigo cita "continuidade, fechamento e simplicidade" sem fórmula.
A decomposição acima é a mais natural: dist penaliza distância grande, cinza penaliza
trechos sem branco (baixo suporte do relevo), e ângulo penaliza desvios de direção.
Parâmetros em `ConfigConexao` para ajuste posterior.

**Resultado na imagem real (image_1.png) — antes de D12:**
- Antes da Etapa 5: 161 nós, 137 arestas, 24 componentes.
- Após (sem filtro D12): 161 nós, 183 arestas, 7 componentes (+46 candidatas, Δ−17 componentes).
  - 17 eram inter-componente (fundem); 29 eram intra-componente (laços triviais).

Ver D12 para resultado final com filtro de laços.

**Impacto:** `features/connections.py::ConfigConexao`, `conectar_candidatas`.

---

## D12 — Laços intra-componente: bloquear triviais, aceitar grandes

**Decisão:** uma candidata entre nós do mesmo componente é bloqueada se:
`dist_grafo < dist_grafo_min` E `dist_grafo < fator_laco_grande * dist_eucl`.
O segundo critério (exceção) permite fechar um arco longo cujo atalho euclidiano é
muito curto relativo ao percurso real no grafo (ex.: fechamento de círculo).

Parâmetros: `ConfigConexao.dist_grafo_min=5` e `ConfigConexao.fator_laco_grande=1.5`.
Implementação: Union-Find para rastrear componentes dinamicamente durante a seleção gulosa;
BFS para calcular dist_grafo apenas para pares intra-componente.

**Por quê:** sem essa regra, 29 dos 46 candidatos aceitos eram laços triviais —
nós já vizinhos ou a 2-3 hops — criando redundância desnecessária antes da
simplificação. Com D12, as 46 candidatas se reduzem a 25, todas inter-componente.

**Detalhe de implementação:** `trivial = dist_g <= dist_grafo_min` (uso de `<=`, não `<`).
O `<` deixa passar exatamente `dist_grafo_min` hops sem ser classificado como trivial; o `<=`
fecha esse caso de borda.

O classificador de inter/intra usa o Union-Find **dinâmico** (atualizado a cada inserção),
não o snapshot de componentes antes da Etapa 5. O snapshot static superestima "inter" porque
arestas inseridas entre componentes já fundidas são contadas como inter pelo snapshot mas
não reduzem o contador de componentes.

**Resultado na imagem real (image_1.png) — com D12 (fix <=):**
- Inseridas: 24 (17 inter + 7 intra-laços grandes).
- Δ−17 componentes (24→7); inter=17 == Δcomp=17 ✓.
- 7 laços grandes aceitos: ratios dist_grafo/dist_eucl entre 1.50 e 14.76 (todos ≥ fator=1.5).

**Impacto:** `ConfigConexao.dist_grafo_min`, `ConfigConexao.fator_laco_grande`,
`features/connections.py::_bfs_dist_grafo`, `conectar_candidatas` (parâmetro opcional
`_debug_info` para diagnóstico).

---

## D13 — dividir_grau_zero: cadeia BFS para arcos longos

**Decisão:** quando uma componente de esqueleto sem nós conectados tem mais de
`max_segmento_px` pixels (padrão 30), em vez de criar apenas dois extremos ligados por
uma aresta reta, cria-se uma cadeia de nós ao longo do caminho BFS (esqueleto,
8-conexo) entre p1 e p2, com espaçamento igual à mediana das arestas existentes no
grafo antes da chamada.

Passo mediano medido em `image_1.png`: **2.00 px** (mediana), 2.21 px (média).
Fallback quando o grafo não tem arestas: `max_segmento_px // 5`.
Parâmetros em `ConfigSubdivisao(max_segmento_px=30, passo_override=0.0)`.

**Por quê:** sem essa regra, um arco isolado de 80 px gerava uma única aresta reta
ligando os dois extremos — no caso de `image_1.png`, a aresta 163-164 de 76 px
cruzava o interior do círculo central. Com a cadeia BFS, os nós seguem o contorno
real do arco e nenhuma aresta excede ~`passo+1` px.

**Causa raiz do nó isolado (node 30 em image_1.png):**
O pixel (83,46) tem `c_bin=3` (junção 8-conexa), portanto não é processado por
nenhum dos três critérios de `place_nodes` (c_xy==0, c_bin==1, c_bin==2). Sem nó
em (83,46), o pixel (83,44) fica fora de qualquer região 3×3 de nó conectado → grau 0.
A correção em `place_nodes` para cobrir pixels com c_bin≥3 está pendente de aprovação.

**Resultado em `image_1.png` após D13:**
- 3 arcos divididos (os 3 nós de grau 0 com componentes > 1 px).
- Maior aresta antes: 76 px (edge 163-164). Após: 7.1 px.
- Arestas > 15 px: 1 → **0**.
- Cobertura (fração de pixels de esqueleto a >6 px de qualquer nó): 0.3% → 0.3%
  (1 pixel em (71,82), independente do tamanho das arestas).

**Métrica de cobertura em todas as imagens (R=6 px, após Etapas 1-4):**

| Imagem | Skel px | Antes fix | Após fix | Δ |
|--------|---------|-----------|----------|---|
| image_1.png | 344 | 0.3% | 0.3% | 0 |
| image_2.png | 462 | 0.0% | 0.0% | 0 |
| image.png   | 296 | 0.0% | 0.0% | 0 |
| bitbots (spl) | 0 | — | — | — |
| eliza | 0 | — | — | — |

As duas últimas imagens não produzem esqueleto no pipeline atual (campo não detectado).
A cobertura já era excelente porque `place_nodes` coloca nós a cada ~2 px no esqueleto.

**Impacto:** `features/connections.py::ConfigSubdivisao`, `_mediana_espaco_nos`,
`_bfs_caminho_esqueleto`, `dividir_grau_zero` (novo parâmetro `cfg_sub`).
Testes: `tests/test_subdivisao.py` (6 testes, todos passando).

**Revisão pós-D14:** Com a correção de `place_nodes` (D14), o arco (83,44)→(86,120) passa
a ter todos os nós conectados após a Etapa 3. `dividir_grau_zero` (D13 fix) detecta que a
componente já tem nós de grau≥1 e não age — a cadeia redundante deixa de ser criada.
O nó (83,44) tem grau=1 após Etapa 3.

---

## D14 — place_nodes: critério de crista usa c_rel≥1, não c_bin==2

**Decisão:** o passo guloso de `place_nodes` (passo 3) usa `c_xy ≥ 1` quando
`usar_relevo=True` (`c_xy` = mapa de c_relevo passado como argumento). Quando
`usar_relevo=False` (fallback binário), mantém `c_bin == 2`.

**Por que c_bin==2 estava errado:** o critério `c_bin == 2` aceita apenas pixels com
exatamente 2 vizinhos de esqueleto ("passagem"), excluindo pixels com `c_bin ≥ 3`
("junções binárias"). Na Etapa 3, `connect_nodes` usa regiões 3×3: dois nós A e B só
se conectam se seus 3×3 se tocam em algum pixel de esqueleto. Pixels com c_bin≥3 que
ficam sem nó criam "pontes cegas" — pixels de esqueleto que nenhum nó cobre — fazendo
que connect_nodes não possa cruzar esse trecho. No caso concreto: pixels (83,45)–(85,48)
tinham c_bin∈{3,4} e ficavam sem nó; isso isolava o nó 30 em (83,44) do restante do arco
de 80 px, criando a aresta reta longa de 76 px na Etapa 4.

**Por que c_rel corrige:** `c_rel` conta vizinhos com relevo ≥ central, sem restrição
topológica. Um pixel com c_bin=3 tipicamente tem c_rel∈{1,2} (só 1 ou 2 vizinhos têm
relevo maior); com `c_rel ≥ 1`, ele entra no passo guloso e recebe nó.

**Resultado em `image_1.png` após D14:**
- Etapa 2: 166 nós (↑7 vs. 159 antes). Cobertura 3×3 = 0% ✓.
- Etapa 3: 138 arestas, 11 nós de grau 0 (vs. 3 antes — os 7 novos nós de c_bin≥3 podem
  ficar isolados se seus 3×3 ainda não tocam outro nó; são gerenciados na Etapa 5).
  **Nó (83,44) tem grau=1** (antes: grau 0) ✓.
- Etapa 4: `dividir_grau_zero` não processa o arco 80px (tem nós conectados — D13 fix).
  5 divisões de componentes puramente isoladas. Arestas >15px: 0 ✓.
- Cobertura 3×3 (fração de px de esqueleto fora da 3×3 de qualquer nó): 0% em todas etapas.

**Impacto:** `features/nodes.py::place_nodes` (passo 3).
Teste: `tests/test_nodes.py::test_arco_com_cbin3_fica_conexo`.

---

## D15 — dividir_grau_zero: não age em componentes com nós já conectados

**Decisão:** `dividir_grau_zero` verifica, antes de subdividir, se a componente de
esqueleto encontrada pelo BFS contém algum nó com grau ≥ 1. Se sim, pula a componente
inteiramente (não remove nós, não cria cadeia). Só age quando TODOS os nós da componente
têm grau 0 (componente verdadeiramente isolada do grafo).

**Por que era necessário:** antes da correção D14, o arco (83,44)→(86,120) tinha 1 nó
isolado (grau 0) mas 33 nós conectados na mesma componente de esqueleto. `dividir_grau_zero`
encontrava a componente inteira (80 px), removia o nó isolado e criava uma cadeia BFS de
39 nós sobreposta aos 33 existentes — duas cadeias paralelas no mesmo arco.

**Impacto:** `features/connections.py::dividir_grau_zero` (guarda nova verificação antes
de comp_set). Teste: `tests/test_etapa4.py::test_dividir_nao_age_em_componente_com_nos_conectados`.

---

## D16 — Etapa 6: suavização, espúrios e diagonais espúrias

**Decisão:** a simplificação aplica quatro passos em ordem, parametrizados em
`ConfigSimplificacao`. Cada passo opera in-place sobre o `StructureGraph`.

**Ordem:** (1) diagonais espúrias → (2) curvatura baixa → (3) espúrios curtos → (4) fusão.
A ordem importa: remover diagonais antes de suavizar evita que nós intermediários
sejam removidos antes de desfazer triângulos espúrios.

**D16-1: remover_diagonais_espurias(limiar_diagonal_px=2.5)**
Uma aresta (a, b) é "diagonal espúria" se: (a) existe pelo menos um vizinho comum c
formando triângulo e (b) dist(a,b) ≤ limiar_diagonal_px e (c) dist(a,b) é a maior
das três arestas do triângulo (i.e., é a aresta "mais diagonal", não a ligação normal
ao nó de junção). Sem triangulo → aresta segura. Esta regra corrige as arestas
braço-a-braço criadas por D8 em junções T/X.

Observação prática: após a correção D14 (c_xy≥1 em place_nodes), as imagens reais
não contêm mais triângulos no grafo — os nós adicionais nos pixels c_bin≥3 quebram
as sobreposições de 3×3 que causavam as diagonais. O passo continua no pipeline para
robustez com configurações diferentes de ConfigNos.

**D16-2: remover_nos_curvatura_baixa(limiar_angulo_grau2=170°)**
Para cada nó N de grau 2 com vizinhos A e B: calcula o ângulo entre os vetores (A−N)
e (B−N); se ângulo > 170° (quase reto), remove N e adiciona aresta A−B. Itera até
convergência. Limiar de 170° preserva curvas genuínas mas remove ruído de posicionamento
ao longo de retos.

**D16-3: remover_espurios(comprimento_espurio_px=5.0)**
Uma folha (grau 1) que percorre uma cadeia de grau-2 até uma junção (grau≥3) forma
um "ramo espúrio" se o comprimento euclidiano total < 5 px. Esses ramos surgem de
artefatos de esqueletização (branching em cantos). Não remove ramos que terminam em
outra folha (sem junção) — esses são extremos legítimos de linha.

**Resultado em image_1.png:** 107 nós removidos por curvatura, 3 ramos espúrios,
1 fusão de junção. Grafo: 171→60 nós, 170→59 arestas, 7 componentes (inalterado),
grau: 1=14→11, 2=144→39, 3=10→7, 4=2→2.

**Impacto:** `features/simplify.py::remover_diagonais_espurias`, `remover_nos_curvatura_baixa`,
`remover_espurios`, `simplificar`.
Testes: `tests/test_simplify.py` (7 testes).

---

## D17 — Etapa 6: fusão de junções próximas

**Decisão:** duas junções (grau ≥ 3) conectadas entre si por uma aresta de comprimento
< `distancia_fusao_px=3.0` são fundidas em um único nó. Mantém-se o nó com maior `gray`
(relevo mais alto); todos os vizinhos do nó descartado são recabiados para o mantido,
exceto self-loops. O passo itera até nenhuma fusão possível.

**Por quê:** junções adjacentes a 1-2 px de distância são artefatos de esqueletização:
cruzamentos T/X reais têm uma única posição, mas o esqueleto pode gerar dois pixels
adjacentes com 3+ vizinhos cada. Fundir os dois produz o nó de junção semântica correto.

**Restrição de colocação:** `skeleton` e `relevo` são aceitos como parâmetros mas não
são usados para restringir a posição do nó fundido — o nó mantido já está no esqueleto.
Caso futuro: mover o nó fundido para o pixel de maior relevo na vizinhança.

**Resultado em image_1.png:** 1 fusão. Em image_2.png: 3 fusões.

**Impacto:** `features/simplify.py::fundir_juncoes`.
Teste: `tests/test_simplify.py::test_duas_juncoes_fundem`.

---

## D18 — Módulo pipeline.py: construir_grafo encadeia Etapas 1-5

**Decisão:** criar `features/pipeline.py` com `ConfigGrafo` (dataclass unificando
`ConfigNos`, `ConfigSubdivisao`, `ConfigConexao`) e `construir_grafo(skeleton, relevo, cfg)`
que executa as Etapas 1-5 em sequência e retorna o `StructureGraph`.

**Motivação:** evitar repetição do encadeamento em scripts de debug, testes de integração
e no pipeline principal (Etapas 7-8).

**Impacto:** `features/pipeline.py`. `run_features.py` usa `construir_grafo`.

---

## D19 — Etapa 7: detecção e verificação de cruzamentos L/T/X

### Classificação de candidatos

**Regras:**
- Grau 2, ângulo entre braços ∈ [`angulo_L_min_grau`, `angulo_L_max_grau`] = [50°, 130°]: candidato **L**.
  Ângulo fora dessa janela → não é L (ver D20).
- Grau 3: candidato **T**.
- Grau 4: candidato **X**.

### Orientação (convenção em coordenadas da imagem — y cresce para baixo)

**L:** bissetriz dos dois braços, apontando para o interior do canto.
Cálculo: soma dos vetores unitários junção→vizinho (normalizada). Essa direção aponta
para o "interior" do L (o quadrante preenchido). Conversão egocêntrica fica para Etapa 9.

**T:** direção da haste. Heurística: o par de braços mais antiparalelos (maior `−dot(vi, vj)`)
forma a "barra"; o terceiro é a haste. Retorna `atan2(dy, dx)` do vetor junção→extremo da haste.
Evita o caso degenerado onde a soma dos vetores da barra cancela a zero (barra perfeitamente
horizontal ou vertical).

**X:** sem orientação preferencial → `orientacao = 0.0`.

### Verificação 1 — Círculo de varredura

Raio `raio_circulo=5` px. Amostra 36 pontos uniformes no círculo.
Conta segmentos brancos (transições fundo→branco) e segmentos verdes (transições fundo→verde).

Score V1 = 0.5 × (branco_ok) + 0.5 × (verde_ok), onde *_ok = 1.0 se |n_segmentos − esperado| ≤ 1.
**Regra especial:** se verde=0 em toda a amostra → V1 = 0.0 (cruzamento falso sem campo).

Esperado por tipo: L=2, T=3, X=4 (braços = segmentos brancos = setores verdes).

### Verificação 2 — Caminho branco até os vizinhos

Rasteriza o segmento reto junção→vizinho (Bresenham simples) e verifica se todos os
pixels são brancos. Score V2 = fração de vizinhos com caminho completamente branco.

### Confiança e limiar

`confianca = (V1 + V2) / 2`. Limiar padrão `limiar_confianca=0.5`.

### Resultado nas imagens reais (com janela [50°, 130°] — D20)

| Imagem | Cand L | Cand T | Cand X | Aceitos L | Aceitos T | Aceitos X | Rej total |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| image_1.png |  2 |  7 | 2 |  1 |  6 | 1 |  3 |
| image_2.png | 14 | 19 | 3 |  8 | 11 | 2 | 15 |
| image.png   |  9 | 11 | 2 |  5 |  7 | 0 | 10 |

**Impacto:** `features/crossings.py` (`ConfigCruzamento`, `classificar_candidatos`,
`verificar_cruzamento`, `detectar_cruzamentos`).
Testes: `tests/test_crossings.py` (18 testes, incluindo janela de ângulo, T falso e borda).
Diagnóstico em `debug_out/etapa7_*.png`: aceitos coloridos por tipo (L=laranja, T=verde,
X=ciano), rejeitados em cinza com motivo.

---

## D20 — Janela de ângulo para candidato L: [50°, 130°]

**Problema:** usando apenas um limiar superior (160°), os vértices da poligonal que aproxima
o círculo central do campo (ângulos tipicamente 150–165°) eram aceitos como candidatos L —
31 de 33 candidatos em image_1 eram falsos positivos desse tipo.

**Decisão:** L só é candidato se `angulo_L_min_grau ≤ ângulo ≤ angulo_L_max_grau`
(padrão 50° a 130°). O artigo define L como um "canto" de linha com aproximadamente 90°;
perspectiva projeta esse ângulo para o intervalo [50°, 130°] em câmeras típicas do SPL.

- **Limite inferior 50°:** ângulos muito agudos (<50°) são provavelmente artefatos de
  esqueletização (spike, T muito estreito) ou ruído, não cantos reais.
- **Limite superior 130°:** acima de 130° a curvatura visual se reduz a um arco suave;
  os vértices do círculo central (150–165°) ficam fora.

**Impacto:** `features/crossings.py::ConfigCruzamento` (campos `angulo_L_min_grau`,
`angulo_L_max_grau`); `features/crossings.py::classificar_candidatos`.
Testes: `test_grau2_quase_reto_nao_e_l` (150–170° rejeitados),
`test_l_perspectiva_60_aceito` e `test_l_perspectiva_120_aceito` (aceitos).

---

## D21 — V2 por BFS, V1 multi-raio e supressão de duplicatas

### V2: BFS em pixels brancos em vez de linha reta

**Problema:** V2 interpolava a reta da junção ao vizinho e verificava se todos os pixels
são brancos. Em cruzamentos onde um braço segue um arco (e.g., círculo central × linha do meio),
a reta entre junção e vizinho-arco atravessa o interior escuro — V2 retornava 0.

**Decisão:** BFS 8-conectividade sobre pixels com `branco > 0`, partindo da junção,
até `max_passos_v2=60` saltos. O nó de partida pode não ser branco (e.g., pixel de junção
exato não está na máscara branco); o BFS parte dele e só avança para pixels brancos.
Score = fração de vizinhos cujo pixel foi alcançado pelo BFS.

**Resultado:** X(50,81) em image_1 (cruzamento círculo × linha) passou de V2=0 para V2=0.50,
e o cruzamento foi aceito com confiança=0.75. Ambos os X de image_1 são agora aceitos.

### V1: escolha do melhor raio em lista configurável

**Problema:** um raio fixo pode ser pequeno demais (poucos pixels da linha atingidos) ou
grande demais (captura estruturas adjacentes). O artigo não fixa o raio.

**Decisão:** testar lista `raios_circulo=[4, 5, 6, 8]`; usar o raio que maximiza o score.
Em empate de score, o raio com menor erro em contagem de branco é preferido.
O `raio_usado` é registrado em `_debug_info`.

**Resultado:** X(50,81) aceito com r=6 (branco=5≈4, verde=5≈4); X(56,80) com r=4.

### Supressão de duplicatas

**Decisão:** após verificação, pares de candidatos aceitos a < `r_fusao_px=3.0` px entre si
são fundidos: mantém o de maior grau de tipo (X>T>L); em empate, maior confiança.
Suprimidos são movidos para rejeitados com `motivo=suprimido`.

**Resultado nas imagens reais (com todas as melhorias):**

| Imagem | Cand L | Cand T | Cand X | Aceitos L | Aceitos T | Aceitos X | Rej total |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| image_1.png | 2 | 7 | 2 | 1 | 5 | 2 | 3 (suprimidos) |
| image_2.png | 14 | 19 | 3 | 8 | 19 | 3 | 6 (suprimidos) |
| image.png | 9 | 11 | 2 | 5 | 10 | 0 | 7 |

**X rejeitados em image.png:** X(44,79) e X(77,78) têm branco=1–2 em todos os raios
(esperado=4) — esses nós de grau 4 estão numa região da imagem onde o arco do círculo e a
linha do meio são muito próximos, mas o interior do círculo permanece escuro na máscara.
V2=0.25 indica que apenas 1 de 4 vizinhos é alcançável por caminho branco; provavelmente
são nós de grau 4 por artefato de conexão, não junções reais.

**Impacto:** `features/crossings.py` (`_verificacao1_multi_raio`, `_verificacao2_caminho`,
`_suprimir_duplicatas`, `ConfigCruzamento`). Testes: `test_x_braco_curvo_v2_aceito`,
`test_suprimir_duplicatas_dois_l`.

---

## D22 — Verificação de geometria T e X

### Problema

O artigo (IV-A) classifica nós em L, T ou X pelo grau, mas não especifica testes
geométricos complementares para confirmar a forma. Nós de grau 3 com ângulos ruins
(três braços em leque, não em T) e nós de grau 4 sem dois pares opostos geravam FP.

### Decisão

Adicionar dois predicados de verificação, ativados por flags em `ConfigCruzamento`:

- **`verificar_geometria_t`** (padrão: `True`): para um candidato T, encontra o par de
  braços mais colinear (ângulo entre eles ≥ `angulo_barra_t_grau=140°`); exige que o
  terceiro braço (haste) forme ângulo ≥ `angulo_haste_t_grau=50°` com cada braço da barra.
  Se nenhum par de barra válido for encontrado, o candidato é rejeitado.

- **`verificar_geometria_x`** (padrão: `True`): para um candidato X, tenta formar dois
  pares de braços opostos em que cada par tem ângulo ≥ `angulo_par_x_grau=140°`.
  Se nenhuma partição em dois pares válidos existir, o candidato é rejeitado.

### Escolha dos limiares

`angulo_barra_t_grau=140°` e `angulo_par_x_grau=140°` foram escolhidos a priori por
simetria: ambos exigem que braços "opostos" formem ângulo próximo de 180° (colinear).
O limiar de 140° para T foi registrado como D19; o de X espelha esse valor (ver nota
"Origem de `angulo_par_x_grau=140°`" na seção Limitações).

**Impacto código:** `features/crossings.py` — `_verifica_geometria_t`,
`_verifica_geometria_x`, `classificar_candidatos`, `ConfigCruzamento`.
Testes: `test_crossings.py` — Grupo I (verificação de geometria T e X).

---

## D23 — Fusão de clusters de junções (simplify.py)

**Problema:** Em cruzamentos reais a esqueletização 8-conexa distribui a valência do cruzamento
em 2–4 nós de grau ≥ 3 dentro de ~6 px, produzindo candidatos T onde deveriam surgir X.
Medição na Parte A: image\_2 X(67,48) e X(107,51) apresentam 4 ramos no esqueleto em r=6 e r=8.

**Decisão:** Adicionar `fusao_juncoes_cluster` em `simplify.py`, executada ao final de
`simplificar` quando `usar_fusao_juncoes=True`. Usa fusão **par-a-par iterativa gulosa** (não
union-find com transitividide plena) para evitar criar nós de grau > 4 em cruzamentos reais.

**Critérios para fundir par (ni, nj):**
- (a) dist(ni, nj) ≤ `r_fusao_juncoes` (padrão 6.0 px)
- (b) existe caminho no grafo com ≤ `max_arestas_fusao` (padrão 8) arestas
- (c) grau resultante = |vizinhos externos únicos| ≤ 4

**Por que limitar a 4:** union-find puro cria nós de grau 5+ quando clusters com 4 nós de
grau=3 têm 5 arestas externas distintas (observado em image\_1 e image\_2). A limitação a 4
evita sobre-fusão e preserva a semântica de cruzamento de campo (máximo 4 braços).

**Representante:** quando `skeleton` está disponível, pixel de esqueleto mais próximo do
centróide do par; caso contrário, nó com maior `gray`.

**Parâmetros novos em `ConfigSimplificacao`:** `usar_fusao_juncoes=True`,
`r_fusao_juncoes=6.0`, `max_arestas_fusao=8`.

**Impacto medido (tolerância 6.0 px):**
| Métrica | Antes D23/D24 | Depois D23/D24 |
|---------|:---:|:---:|
| A: TP localização total | 5/32 det | 6/29 det |
| A: Precisão | 16% | 21% |
| A: Revocação | 42% | 50% |
| B: Acurácia classificação | 2/5 = 40% | 2/6 = 33% |

**Impacto:** `features/simplify.py` (`fusao_juncoes_cluster`, `ConfigSimplificacao`,
`simplificar`). Testes: `test_fusao_cluster_dois_t_viram_x`,
`test_fusao_cluster_t_distante_nao_funde`, `test_fusao_cluster_t_isolado_continua_t`.

---

## D24 — Ângulo L medido via arco do grafo (crossings.py)

**Problema:** `_angulo_entre_vizinhos` usa o vizinho imediato no grafo (1 salto), o que em
cantos arredondados pode dar ângulo fora da janela [50°, 130°] mesmo quando o canto real é ~90°.
Exemplos sintéticos: cadeia (0,10)-(5,10)-(8,8)-(10,5)-(10,0) tem ângulo imediato ~157° em
(8,8) mas ângulo via arco ~118° com d=8 px.

**Decisão:** Adicionar `_caminhar_ate_arco` e `_angulo_arco_l` em `crossings.py`. Quando
`distancia_arco_l_px > 0` (padrão 7), `classificar_candidatos` usa o ponto a ~d_arco px de
distância acumulada ao longo de cada braço para medir a direção, em vez do vizinho imediato.
O caminho segue apenas nós de grau 2 (não cruza junções).

**Quando usar vizinho imediato vs. arco:**
- `distancia_arco_l_px = 0`: vizinho imediato (retrocompatível com grafos onde os braços são
  nós isolados grau=1 a <7 px)
- `distancia_arco_l_px = 7` (padrão): percorre a cadeia de grau=2 até acumular ≥7 px.
  Se o braço termina antes (nó grau=1 ou grau≥3), usa o último nó alcançado.

**Compatibilidade com testes existentes:** todos os 103 testes anteriores continuam passando
porque os grafos sintéticos usam nós grau=1 a 5 px de distância. Com d_arco=7, o caminho
para na extremidade grau=1 (5 px < 7), dando resultado idêntico ao imediato.

**Casos que NÃO melhoram com arco:** GT L(62,68) e GT L(99,67) em image\_1 — os nós do grafo
nessa região têm ângulo < 50° tanto imediato quanto via arco, pois estão em segmentos quase
retos (linha reta, não canto).

**Parâmetro novo em `ConfigCruzamento`:** `distancia_arco_l_px=7`.

**Impacto:** `features/crossings.py` (`_caminhar_ate_arco`, `_angulo_arco_l`,
`classificar_candidatos`, `ConfigCruzamento`). Testes: `test_l_arredondado_detectado_por_arco`,
`test_linha_reta_com_ruido_nao_e_l`, `test_arco_suave_nao_e_l`.

---

## D25 — Filtro de espessura por largura de arco (NÃO implementado)

**Proposta avaliada:** Rejeitar candidatos cujo maior segmento branco no círculo de raio 6
tenha largura de arco > `largura_max_segmento_celulas` (para filtrar FP em linhas largas).

**Medição (Parte A, avaliador v1 antes de D23/D24):**
- Grupo casados com GT (n=5): min=2.48, mediana=6.45, max=17.36 células
- Grupo não casados (n=27): min=0.99, mediana=6.45, max=27.78 células
- Melhor limiar sem perder TP: 17.36 → elimina apenas 2/27 FP

**Decisão:** **Não implementar.** Os dois grupos se sobrepõem completamente
(ambos têm mediana=6.45 e valores em toda a faixa 0.99–27.78). Um limiar útil
eliminaria ≤7% dos FP sem garantia de não perder TPs futuros.

---

## D26 — Direções por arco em _verifica_geometria_t e _verifica_geometria_x

**Proposta:** Usar `_caminhar_ate_arco(g, nid, v, d_arco)` para calcular os vetores de
direção de cada braço em vez do vizinho imediato. Parâmetro `distancia_arco_geom_px`
controla a distância de caminhada (0 = modo imediato, desligado).

**Medição M2 (pré-implementação, image_2):**
- nó 90 (102,51): barra=170.5° (arco ≈ imediato), stem_min=35.5° (arco; imediato=63.4°)
- nó 91 (110,51): barra=177.3°, stem=45° (arco ≈ imediato)
- Com modo imediato e angulo_stem_t_min=50°: nó 90 falha (35.5° < 50°) → T(107,51) não detectado
- Com arco 7px e angulo_stem_t_min=35°: nó 90 passa → T(107,51) detectado como T

**Parâmetros derivados de M2:**
- `angulo_stem_t_min_grau = 35.0` (margem 0.5° abaixo do mínimo medido 35.5°)
- `angulo_stem_t_max_grau = 145.0` (margem de 10° acima do máximo bar-stem medido 135°)
- `distancia_arco_geom_px = 7` (mesmo valor que D24 `distancia_arco_l_px`)

**Impacto ablação (+I1 sobre BASE, tol=6):** dets 29→31, TP=6=6, FP 23→25, PrecA 21%→19%,
RecA 50%=50%, TypeAccB 33%=33%. D26 sozinho não melhora recall/precisão.

**Ablação de controle (I2+I4 sem I1):** Idêntico ao +I1+I2+I4 (dets=28, TP=6, FP=22, PrecA=21%,
RecA=50%, TypeAccB=50%). D26 tem efeito zero quando D27+D29 estão ativos: D27 funde os T a nível
de grafo (antes de classificar_candidatos) e o X resultante em (102,51) passa _verifica_geometria_x
com vizinhos imediatos (pares 180° e 175.6°, ambos ≥ 140°), sem necessitar de arco.

**Decisão: REVERTIDO.** Defaults restaurados para `angulo_stem_t_min_grau=50.0`,
`angulo_stem_t_max_grau=130.0`, `distancia_arco_geom_px=0`. O código das funções
`_verifica_geometria_t` e `_verifica_geometria_x` permanece com suporte ao arco (habilitável
por config); os testes D26 usam parâmetros explícitos.

**Impacto código:** `features/crossings.py` — `_verifica_geometria_t` e `_verifica_geometria_x`
suportam `distancia_arco_geom_px > 0` (desligado por padrão).

---

## D27 — Fusão de par de T com barras colineares e hastes opostas

**Proposta:** Dois nós de grau=3 adjacentes, distância ≤ `r_fusao_t_par_px`, cada um com barra
(ângulo ≥ `angulo_barra_t_par`) que inclui o outro como braço de barra, e hastes em lados
opostos da direção T1→T2 (teste de produto vetorial 2D) → fundidos em grau=4.

**Motivação:** X(107,51) em image_2 é representado por dois nós T adjacentes nó90(102,51) e
nó91(110,51). Sem fusão, detectados como T ou não detectados; com fusão correta, detectados
como X melhorando type accuracy.

**Escolha do representante:** Em vez de "maior gray vence", calcula o score X de cada candidato
a representante (max sobre as 3 partições do min par-angle entre os 4 braços externos) e
escolhe o de maior score. Crítico: nó91(110,51) como rep falha geometry_x (braços ao mesmo
lado), nó90(102,51) como rep passa (pares 180° e 175.6°).

**Lados opostos:** sinal do produto vetorial 2D `d12 × stem_i`: se `cross_i * cross_j < 0`
os dois stems ficam em lados opostos da linha T1→T2.

**Impacto ablação (+I2 sobre +I1, tol=6):** dets 31=31, TP=6=6, FP 25=25, PrecA 19%=19%,
RecA 50%=50%, TypeAccB 33%→50% (2/6→3/6). Recall e precisão inalterados; type accuracy +17%.

**Impacto código:** `features/simplify.py` — nova função `fusao_t_par(g, skeleton, cfg)`;
`simplificar()` a chama ao final; `ConfigSimplificacao` ganhou campos `usar_fusao_t_par`,
`r_fusao_t_par_px`, `angulo_barra_t_par`.

---

## D28 — Rejeitar candidatos com V1=0 (NÃO implementado)

**Proposta avaliada:** Eliminar candidatos aceitos com `v1 == 0.0` pois indicam ausência de
consenso no círculo de verificação.

**Medição M3 (6 TPs da métrica A, tol=6, 3 imagens):**
- image X→L(45,81): V1=0.0
- image X→T(108,80): V1=0.0
- Total: 2/6 TPs têm V1=0

**Decisão:** **Não implementar.** Eliminaria 2 dos 6 TPs (33% de queda em recall).
O valor V1=0 ocorre em TPs próximos à borda da grade de imagem onde o círculo de raio
menor tem poucas células; esses candidatos são salvos pelo V2 (BFS em esqueleto branco).

---

## D29 — Descartar candidatos cujo círculo de verificação sai da grade

**Proposta:** Rejeitar um candidato C(x,y) se `max(cfg.raios_circulo)` pixels a partir de
(x,y) extrapola a grade (i.e., `x-r<0 OR x+r≥W OR y-r<0 OR y+r≥H`). Controle por
`verificar_borda` em `ConfigCruzamento`.

**Motivação:** O círculo de V1 (`_verificacao1_multi_raio`) acessa pixels fora da grade quando
o candidato está muito perto da borda, podendo contar pixels inexistentes como brancos ou verdes
e introduzir FP espúrios.

**Segurança (M3):** nenhum TP (0/5 no estado final com D37 ativo; 0/6 no estado de ablação
pré-D37) está a menos de `max(raios_circulo)=8` px da borda → D29 não perde nenhum TP.

**Impacto ablação (+I4 sobre +I2, tol=6):** dets 31→28, TP=6=6, FP 25→22, PrecA 19%→21%,
RecA 50%=50%, TypeAccB 50%=50%. Remove 3 FP de borda sem custo em recall.

**Impacto código:** `features/crossings.py` — `detectar_cruzamentos` verifica `verificar_borda`
antes de chamar V1/V2; `ConfigCruzamento` ganhou campo `verificar_borda: bool = True`.

---

## Limitações conhecidas do detector de cruzamentos (Etapa 7)

Medições realizadas em 27/10/2026, config padrão (I2=D27 ativo, I4=D29 ativo, I1=D26 desligado),
tol=6.0 px subamostrados, 3 imagens (GT=12, Det=27). Comando: `python3 avaliar_cruzamentos.py`.
Código idêntico ao commitado; discrepância vs. medição de 26/10 documentada abaixo.

### Métricas por tipo do GT

Matching um-para-um guloso (menor distância, tol=6.0, qualquer tipo).

| GT-tipo | GT | Localiz. | Loc.+Tipo | FN | Recall(loc) | Recall(tipo) |
|---------|----|---------:|----------:|----|:-----------:|:------------:|
| L       |  4 |        1 |         1 |  3 |     25%     |      25%     |
| T       |  2 |        0 |         0 |  2 |      0%     |       0%     |
| X       |  6 |        5 |         2 |  1 |     83%     |      33%     |
| **Total** | **12** | **6** | **3** | **6** | **50%** | **25%** |

Localização: 6/12 = **50%** recall, 6/27 = **22%** precisão.
Tipo correto: 3/12 = **25%** recall, 3/27 = **11%** precisão.

### Métricas por tipo da DETECÇÃO

Configuração: D27 ativo, D29 ativo, D37 ativo (branco passado). Sem D37: Det=28, FP=22 (ver D37).
"TP (loc)" = localização correta (tol=6 px, qualquer tipo). "TP (tipo)" = localização + tipo correto.

| Det-tipo | Dets | TP (loc) | TP (tipo) | FP | Dets |
|----------|-----:|---------:|----------:|----|-----:|
| L        |   15 |        3 |         1 | 12 |   15 |
| T        |    8 |        1 |         0 |  7 |    8 |
| X        |    4 |        2 |         2 |  2 |    4 |
| **Total** | **27** | **6** | **3** | **21** | **27** |

### Matriz de confusão

Linhas = tipo GT, colunas = tipo detectado (incluindo não detectados = "—").

| GT \ Det | L | T | X | — | Total-GT |
|----------|--:|--:|--:|--:|:--------:|
| L        | 1 | 0 | 0 | 3 |    4     |
| T        | 0 | 0 | 0 | 2 |    2     |
| X        | 2 | 1 | 2 | 1 |    6     |
| Σ(par.)  | 3 | 1 | 4 | 6 |   12     |

FP não aparecem na matriz (não têm par de GT): L=12, T=7, X=2 → total 21.

### Meta combinada — NÃO atingida

- X recall(localização) = **83%**. Meta (X recall ≥ 80% com tipo correto) NÃO atingida: recall(tipo) = 33%. Localização isolada (qualquer tipo): 83%.
- X recall(tipo correto) = 33% < 80% ✗
- L precisão = 1/15 = **7%** < 60% ✗
- T precisão = 0/8 = **0%** < 60% ✗

### Causas medidas por ponto

**GT-X mal classificados ou perdidos (4 de 6); 2 de 6 com tipo correto:**

- *image_1 GT-X(124,82)*: FN. Nó mais próximo deg=2 em (126,83); o T(122,84) que poderia
  casar não é detectado com stem_min=50° (stem imediato cai fora da janela).
- *image_1 GT-X(46,82)*: TP — casado com det-X(50,81) d=4.1 px, tipo correto.
- *image_2 GT-X(67,48)*: casado com det-L(70,45) dist=4.2 px. O nó deg=4 mais próximo
  (68,49) foi fundido em D23 e o representante ficou em (68,49) com 3 braços; o deg=2
  adjacente em (70,45) gerou candidato L que "roubou" a posição.
- *image GT-X(44,79)*: casado com det-L(45,81) dist=2.2 px. Idem acima — esqueleto só
  tem deg=2 no raio 4; nó deg=3 em (108,80) distante 64 px.
- *image GT-X(110,78)*: casado com det-T(108,80) dist=2.8 px. O nó mais próximo é deg=2
  antes E depois da Etapa 6 — a 4ª aresta que formaria X nunca está no esqueleto.

**GT-T perdidos (2 de 2):**

- *image_2 T(153,53)*: nó mais próximo deg=2 (154,53) antes da Etapa 6; junção ausente
  no esqueleto desde a saída do III-B.
- *image_2 T(9,45)*: nó mais próximo deg=2 (9,45) antes da Etapa 6; 6 nós removidos em
  raio 6 px pela Etapa 6 (espurios), restando só folha deg=1 a 3.2 px.

**GT-L perdidos (3 de 4):**

- *image_1 L(62,68)*: nó deg=2 em (63,67) dist=1.4 px; ângulo imediato=arco=26.6° < 50°.
  Braços terminam em folha a < 2 px — arco D24 não consegue alcançar 7 px, retorna mesmo
  nó do vizinho imediato.
- *image_1 L(99,67)*: nó deg=2 em (100,68) dist=1.4 px; ângulo imediato=arco=11.3° < 50°.
  Mesma causa: braços muito curtos.
- *image_2 L(21,22)*: candidato X@(25,18) a 5.7 px (dentro da tol), mas rejeitado por
  `_verifica_geometria_x`: melhor par oposto = 135° < `angulo_par_x_grau=140°`. Abaixar
  para 135°: Det=28, TP=7 (tipo errado: GT-L ← Det-X, tipo incorreto).

### Discrepância entre medições (não explicada)

Em 26/10/2026 o avaliador registrou Det=26, TP=5 (image_1: Det=1, TP=0).
Em 27/10/2026, com o código **sem alterações detectáveis** (nenhum arquivo de features/
modificado, ambiente estável desde 16/10, código 100% determinístico em 10/10 rodadas),
o avaliador registrou Det=27, TP=6 (image_1: Det=2, TP=1; X em (50,81), d=4.1 px do
GT-X(46,82), tipo correto). **Causa: não identificada.**

Inspeção de todos os 123 blobs dangling do repositório: nenhum blob de features/ corresponde
a um estado intermediário que explique Det=26 com image_1=1. O commit base `eb35c39` é da
parte do colega e não contém features/. Nenhuma evidência de instalação ou atualização de
pacotes entre as duas sessões (pip list estável desde 16/10).

O estado de referência é o da medição de 27/10/2026, idêntico ao código commitado.

### Origem de `angulo_par_x_grau=140°` e `verificar_geometria_x`

**D-número:** referenciado no código como "D22" nos docstrings de `classificar_candidatos`
e `_verifica_geometria_x`/`_verifica_geometria_t`; documentado na seção ## D22 acima.
O parâmetro está em `features/crossings.py::ConfigCruzamento` (linha `angulo_par_x_grau: float = 140.0`).

**Escolha do valor:** 140° espelha `angulo_barra_t_grau=140°` (barra do T em D19) — ambos
exigem que braços "opostos" formem ângulo ≥ 140° (próximo de 180° = colinear). Não há
evidência nos docs de que esse valor foi ajustado observando o gabarito; a escolha parece
a priori por simetria com o threshold do T. Não foi documentada a data ou contexto exato.

### Concentração dos 21 FP

- **image_2**: 13 FP (62%) concentrados na faixa superior da imagem (y ≤ 36), onde a
  grade do campo tem linhas espessas e próximas que geram múltiplos candidatos L e T com
  V1 ≥ 0.5 por acidente geométrico.
- **image**: 7 FP (33%), maioria L em torno da elipse central (y 67–81), onde curvas do
  esqueleto têm ângulo dentro da janela [50°, 130°] mas não correspondem a cruzamentos reais.
- **image_1**: 1 FP (X@(56,80)).
- A confiança atual não discrimina FP de TP: FP têm conf 0.50–1.00 com distribuição
  sobreposta à dos TP (medido com curva de confiança: melhora em precisão não acima de
  posição de corte 0.5 sem perda de recall).

### Distribuição medida dos 21 FP (Fase 1 do recorte verde)

Medida com `fase1_topo_verde.py` (topo verde por coluna, corrida≥3, mediana 5 cols):

- **13 na faixa do fundo do campo** (|y − y_topo_verde| ≤ 7): 11 em image_2 e 2 em image.
- **6 na elipse do círculo central**: 2 em image_2 e 4 em image.
- **1 FP em image_1**: det-X(56,80) a 10.2 px do GT-X(46,82) — fora da tolerância.
- **1 vizinho do buraco preto**: det-L(153,78) em image.

Dois GT ficam na mesma faixa do fundo: L(21,22) e L(118,29) em image_2 (este o único L
localização-correto no resultado atual). Um corte por região perderia recall.

Ver D30.

### O que depende do esqueleto III-B (colega)

Os pontos a seguir não podem ser corrigidos na Etapa 7 sem alterar a saída do III-B:

1. **T(153,53) e T(9,45) em image_2**: grau=2 antes da Etapa 6 — a junção T não existe
   no esqueleto entregue pelo III-B. Não há nó de grau ≥ 3 que o detector possa usar.
2. **X(110,78) em image**: grau=2 antes e depois da Etapa 6 — a 4ª aresta do X não existe
   no esqueleto. O detector consegue no máximo um T (3 braços), nunca um X.
3. **L(62,68) e L(99,67) em image_1**: braços do deg=2 têm < 2 px de comprimento por
   quantização da grade subamostrada. A geometria do canto não é representada no esqueleto.
4. **Todos os FP L em curvas**: o esqueleto subamostrado tem nós deg=2 com ângulo na janela
   [50°, 130°] em regiões que não são cruzamentos. São artefatos de resolução do esqueleto,
   não erros do detector.

---

## D30 — Recorte de FP acima do topo verde (NÃO implementado)

**Proposta:** rejeitar detecções com `y < y_topo_verde − margem` (acima da linha verde),
onde `y_topo_verde` é estimado por `fase1_topo_verde.py` (corrida≥3, mediana 5 colunas).

**Resultado:** apenas 5/21 FP ficam acima da linha `y_topo_verde − 2` (critério exigia ≥ 11
para ser útil). O corte não reduziria os FP na faixa do fundo (13 FP com
`|y − y_topo_verde| ≤ 7`) nem os da elipse central (6 FP). Dois GT verdadeiros também
ficam nessa faixa: L(21,22) e L(118,29) em image_2 — um corte por região perderia recall.

**Decisão:** não implementado. O `main.py` continua intacto.

---

## D31 — Etapa 8: extração de linhas por split-and-merge (IV-B)

### Grafo de entrada

**Decisão:** usar o grafo **antes** da simplificação (saída da Etapa 5, `construir_grafo`).

**Por quê:** o artigo (Fig. 2c) aplica extração de linhas no grafo conectado antes de simplificar.
Usar o grafo pré-simplificação desacopla linhas de cruzamentos: erros de grau (junção
faltando no esqueleto do colega) não afetam a extração de linhas.

### Componentes de grau 2

**Decisão:** usar travessia por aresta (não por nó visitado).

**Por quê:** travessia por nó visitado perde ramos de junções T/X — ao visitar o nó de junção
no primeiro ramo, ele entra em `visitados` e os demais ramos nunca são explorados. A travessia
por aresta marca cada aresta como percorrida e semeia todos os ramos independentemente a partir
de cada âncora (nó de grau ≠ 2).

### Split-and-merge

**Decisão:** ajuste por mínimos quadrados totais (SVD); split somente no ponto interior de
maior desvio ortogonal (índices 1..N−2 excluídos os extremos).

**Por quê (excluir extremos):** com TLS, o ponto de maior desvio é frequentemente um extremo
da cadeia (e.g., num L a reta diagonal tem desvio máximo nos vértices do canto, mas o primeiro
extremo empata ou supera o desvio do canto). Incluir extremos como candidatos ao split causa
recursão infinita: `right_ids = nos_ids[0:]` é idêntico ao input. A exclusão dos extremos garante
que cada subproblema tenha estritamente menos pontos que o pai.

**Componentes < `min_nos_componente=3` nós:** descartados (não geram `LineFeature`).

**Distância mínima `distancia_min_celulas=1.0`:** segmentos degenerados (4 nós todos projetados
no mesmo ponto após SVD) produzem `distancia=0.0`; são descartados.

### Undistort (features/geometry.py)

**Decisão:** identidade por padrão (sem calibração). O pipeline atual (`Pixel_Segment`) não expõe
matriz de câmera. `undistort_pontos` retorna as coordenadas da grade sem modificação quando
`cfg_cam=None`. A projeção no chão fica atrás de `ConfigLinhas.projetar_no_chao=False`.

### Parâmetros em ConfigLinhas

| Parâmetro | Padrão | Significado |
|-----------|--------|-------------|
| `desvio_max_celulas` | 2.0 | Limiar de split ortogonal |
| `min_nos_componente` | 3 | Mínimo de nós para formar uma linha |
| `distancia_min_celulas` | 1.0 | Descarta segmentos degenerados |
| `usar_fusao` | True | Fundir segmentos colineares pós-split |
| `limiar_angulo_fusao_grau` | 5.0 | Ângulo máximo para fusão |
| `limiar_dist_ortogonal_fusao` | 2.0 | Distância ortogonal máxima para fusão |
| `projetar_no_chao` | False | Projeção perspectiva (requer calibração) |

### Resultado nas imagens reais (grafo pré-simp, config padrão)

| Imagem | Linhas | Descartados (<3 nós) | Mín dist | Med dist | Máx dist |
|--------|-------:|---------------------:|--------:|---------:|---------:|
| image_1 | 9 | 11 | 2.2 | 34.0 | 72.0 |
| image_2 | 24 | 23 | 1.6 | 9.2 | 136.2 |
| image | 17 | 10 | 2.1 | 9.2 | 64.0 |

**Interpretação da coluna `d_topo` (distância do centro da linha a y_topo_verde):**
Linhas com `d_topo < 5` células estão na faixa do fundo do campo; linhas com `d_topo > 10`
estão claramente dentro do campo útil. A maioria dos segmentos compridos (dist > 30) tem
`d_topo > 10`, confirmando que são estruturas do campo, não do fundo.

**Impacto código:** `features/lines.py` (`ConfigLinhas`, `extrair_linhas`, `_split`,
`_componentes_grau2`, `_ajustar_reta`, `_fundir_segmentos`);
`features/geometry.py` (`ConfigCamera`, `undistort_pontos`, `projetar_no_chao`);
`features/types.py` (`LineFeature` estendido com `x1_orig`, `y1_orig`, `x2_orig`, `y2_orig`,
`n_nos`, `confianca`).
Testes: `tests/test_lines.py` (12 testes).

---

## D32 — Extremos de LineFeature por projeção (não pts[0]/pts[-1])

**Problema:** `_para_linefeature` usava `pts[0]` e `pts[-1]` para definir os extremos da reta.
Após a fusão de segmentos via `np.vstack`, os pontos acumulados estão em ordem de traversal
(não espacial): `pts[-1]` pode ser um ponto interior ao range real, causando extremos truncados.
Exemplo diagnóstico D2: fusão de seg9(x=57..112) + seg12(x=1..48) → `pts[-1]=(48,82)` em vez de
`(1,82)` → x2=48 em vez de 1, perdendo ~47 células à esquerda.

**Decisão:** calcular a projeção escalar de TODOS os pontos sobre a direção da reta
(vetor `(-ny, nx)` perpendicular à normal) e usar os índices de projeção mínima e máxima.
Invariante à ordem de traversal: qualquer shuffle dos nós produz os mesmos extremos.

**Parâmetros envolvidos:** nenhum novo parâmetro.

**Impacto código:** `features/lines.py::_para_linefeature` (4 linhas substituídas).
Testes: `test_extremos_independem_da_ordem_nos`, `test_extremos_por_projecao_direto` (T12–T13).

---

## D33 — Fusão com verificação de gap e suporte branco

**Problema:** `_fundir_segmentos` fundia segmentos com diferença de ângulo e distância ortogonal
dentro dos limiares, sem verificar o gap físico entre eles. Diagnóstico D1: seg0(160,67)-(155,69)
e seg10(128,82)-(120,85) tinham ang_diff=1.19° e d_ort=0.84 — dentro dos limiares — mas
distância entre os endpoints mais próximos ≈ 30 células, criando uma linha diagonal falsa de 43 px.

**Decisão:** após a verificação de ângulo e distância ortogonal, exigir um dos dois:
1. **Conectividade pelo grafo**: os dois segmentos compartilham pelo menos um ID de nó
   (endpoints vindos do mesmo nó de junção após `_split`). Nesse caso, o gap é necessariamente 0.
2. **Gap ≤ lacuna_max_celulas**: distância mínima entre endpoints dos dois segmentos.
   Se `branco` fornecido: também exige frac_branco no vão ≥ `suporte_vao_min`.

**⚠ Parâmetros PARCIALMENTE SUBSTITUÍDOS por D36:**
Os valores iniciais de `lacuna_max_celulas` (8.0) e `suporte_vao_min` (0.6) foram ajustados
em D36. Esta entrada documenta apenas o mecanismo de gap-check. Para os valores atuais, ver D36.

**Impacto código:** `features/lines.py::_fundir_segmentos` (nova lógica de gap);
`_pontos_mais_proximos` (nova função auxiliar); `extrair_linhas` aceita `branco`.
Testes: `test_fusao_rejeita_vao_grande` (T14), `test_fusao_rejeita_vao_escuro` (T15),
`test_fusao_aceita_vao_branco` (T16).

---

## D34 — Filtragem de arestas escuras longas em _componentes_grau2

**Problema:** `_componentes_grau2` incluía nas cadeias arestas candidatas com frac_branco baixa
e comprimento alto (ex.: aresta 26→148 com fb=0.09 e aresta 32→146 com fb=0.13 em `image`),
gerando cadeias que atravessavam regiões escuras do campo. Diagnóstico D3/D4: linhas verticais
e linha da bola eram produto de candidatas escuras incorporadas às componentes.

**Decisão:** após construir cada cadeia, percorre-a par a par. Se uma aresta tiver
`dist > dist_aresta_min` (padrão 4.0 células) **e** `frac_branco < suporte_aresta_min`
(padrão 0.5), a cadeia é cortada nesse ponto. Sub-cadeias resultantes com comprimento < 2 nós
são descartadas; as demais são retornadas como componentes independentes.
Sem efeito quando `branco=None` (comportamento original).

**Novos parâmetros em `ConfigLinhas`:**
| Parâmetro | Padrão | Significado |
|-----------|--------|-------------|
| `suporte_aresta_min` | 0.5 | Frac branco mínima para aresta longa não ser cortada |
| `dist_aresta_min` | 4.0 | Comprimento mínimo para aplicar o check de suporte |

**Impacto código:** `features/lines.py::_componentes_grau2` (nova assinatura e pós-processamento);
`_frac_branco` (nova função auxiliar); `extrair_linhas` passa `branco` e os dois parâmetros.
`run_etapa8.py` passa `branco=res['white_mask_resized']` para ativar os filtros.
Testes: `test_aresta_escura_corta_componente` (T17).

---

## D35 — Distribuição de confiança nas 3 imagens (Etapa 8)

**Medição** (config padrão com branco passado, após D32–D34):
- image_1: confianças = {1.00: 10×, 0.96: 1×, 0.60: 1×, 0.46: 1×}
- image_2: confianças ∈ [0.57, 1.00]; distribuição contínua, maioria ≥ 0.80
- image: confianças ∈ [0.00, 1.00]; conf=0.00 em 1 linha (nós todos a desvio > limiar)

**Conclusão:** não há separação bimodal clara entre "linhas boas" (conf ≥ limiar) e
"linhas ruins". Linhas com conf baixa incluem tanto segmentos espúrios quanto segmentos
curvos (arco do círculo) que são geometricamente válidos. **Limiar de confiança não implementado.**

---

## D36 — Ajuste dos parâmetros de gap-check (lacuna_max 8→20, suporte_vao 0.6→0.8)

**Contexto:** D33 introduziu o gap-check com `lacuna_max_celulas=8.0` e `suporte_vao_min=0.6`.
Com esses valores, a linha do meio de `image_1` (y≈82) não era fusada: o gap entre dois
segmentos adjacentes era ≈22 células, superior ao limiar de 8, causando 3 fragmentos distintos
em vez de 1 linha contínua. O mesmo ocorreu em `image` (y≈77): 3 fragmentos com gap≈22.

**Decisão:**
- `lacuna_max_celulas`: 8.0 → **20.0** (permite gaps até 20 células quando o vão for branco).
- `suporte_vao_min`: 0.6 → **0.8** (exige 80% de branco no vão para fusões com gap > 0; maior
  limiar compensa o aumento do raio de busca — evita fusões espúrias através de regiões escuras).

**Evidência medida:**
- `image_1` linha do meio (y≈82): com `lacuna_max=8` → 3 fragmentos; com `lacuna_max=20` → 1
  segmento contínuo (dist≈164 células, n_nos=66). O gap medido era ≈22 células, todo ele branco.
- `image` linha do meio (y≈77): com `lacuna_max=8` → 3 fragmentos; com `lacuna_max=20` → 1
  segmento contínuo (dist≈155 células, n_nos=85). Mesmo gap característico.

**Parâmetros atuais em `ConfigLinhas`:**
| Parâmetro | Padrão | Significado |
|-----------|--------|-------------|
| `lacuna_max_celulas` | 20.0 | Gap máximo (células) entre endpoints para permitir fusão |
| `suporte_vao_min` | 0.8 | Frac branco mínima no vão (quando branco fornecido) |

**Por que o gap existe:** a linha do meio de `image_1` tem uma interseção com o arco do círculo
central. O nó de junção (grau≥3) é uma âncora em `_componentes_grau2`: a cadeia é cortada na
junção, gerando dois segmentos com um espaço entre eles. Com D36, o BFS de suporte confirma
que o vão é branco e a fusão é permitida.

**Impacto:** `features/lines.py::ConfigLinhas`. Não há novos parâmetros de código; apenas os
valores padrão foram alterados em relação ao que D33 introduziu.

---

## D37 — Filtro de candidatas por suporte branco (Etapa 5)

**Problema:** algumas conexões candidatas geradas na Etapa 5 ligavam terminais do grafo
passando por regiões escuras (baixo suporte branco), criando arestas espúrias que introduziam
cruzamentos falsos na Etapa 7. Em `image`, 6 candidatas com frac_branco ≈ 0.02–0.30 e
comprimento ≥ 5 px foram aceitas; em `image_2`, 2.

**Decisão:** quando `branco` é fornecido a `conectar_candidatas`, rejeitar toda candidata
com `dist > dist_min_filtro` **E** `frac_branco < suporte_candidata_min`. O critério é
idêntico ao D34 (fração medida por rasterização linear entre os dois nós).

**Parâmetros em `ConfigConexao`:**
| Parâmetro | Padrão | Significado |
|-----------|--------|-------------|
| `suporte_candidata_min` | 0.5 | Frac branco mínima para candidata longa ser aceita |
| `dist_min_filtro` | 4.0 | Comprimento mínimo para acionar o filtro |

**Resultado da ablação medido (`python3 medir_b.py`, tol=6.0, 3 imagens, config padrão completa):**

| Métrica | Sem D37 (`suporte_min=0.0`) | Com D37 (`suporte_min=0.5`) | Δ |
|---------|:---------------------------:|:---------------------------:|:---:|
| Det total | 28 | 27 | −1 |
| TP (localização) | 6 | 6 | 0 |
| FP | 22 | 21 | −1 |
| FN | 6 | 6 | 0 |
| RecA | 50% | 50% | 0 |
| PrecA | 21% | 22% | +1% |

Detecções extras sem D37 (ambas FP): `image:L@(100,69)` e `image_2:X@(91,45)`.
Detecção presente com D37 mas não sem (FP): `image:L@(75,75)`.
D37 muda o grafo e afeta indiretamente quais candidatos emergem; efeito líquido: −1 FP.

**Decisão: MANTIDO.** Remove 1 FP líquido sem custo em recall.

**Limitação conhecida:** o filtro só está ativo quando `branco` é passado para `construir_grafo`
(via `extrair_features` na pipeline padrão). Quando `branco=None`, todas as candidatas são
avaliadas apenas pelo custo D11.

**Impacto código:** `features/connections.py::ConfigConexao` (campos `suporte_candidata_min`,
`dist_min_filtro`, `filtrar_escuras`); `conectar_candidatas` (bloco D37, linhas 535-539);
`_frac_branco_candidata` (nova função auxiliar).

---

## D38 — Divergência: TLS em vez de OLS para ajuste de reta (IV-B)

**O que o artigo diz:** Seção IV-B — "we fit a line li to ci using least squares regression [11]
on the coordinates of the nodes." — O artigo escreve "least squares regression" sem qualificativo;
a variante (OLS, TLS ou outra) não está especificada. A referência [11] é Behnke 1997, que não
detalha o método de ajuste.

**O que o código faz:** `_ajustar_reta` usa SVD (decomposição em valores singulares) para
encontrar a direção de maior variância — o que corresponde a TLS (Total Least Squares,
mínimos quadrados totais: minimiza a soma dos quadrados das distâncias ortogonais).

**Por que TLS:** TLS é invariante a rotação — trata x e y simetricamente — e é o método
adequado para ajuste de linhas em coordenadas de imagem onde não há "variável dependente"
natural. OLS pressupõe uma variável dependente (ex.: y = a·x + b), falhando para linhas
verticais (coeficiente angular indefinido) e introduzindo viés para ângulos > 45°.

**Efeito esperado:** para linhas horizontais (ângulo ≈ 0°) a diferença OLS×TLS é desprezível.
Para linhas verticais ou diagonais (ângulo ≈ ±45° a ±90°), TLS produz fits mais precisos.
Nas imagens do campo de futebol, a linha do meio (quase horizontal) não é afetada; linhas de
fundo com maior ângulo beneficiam-se do TLS.

**Decisão: MANTIDO.** TLS é tecnicamente superior para o caso de uso; a desvio do artigo é
justificado. Registrado como D38 para transparência.

**Impacto código:** `features/lines.py::_ajustar_reta`.

---

## D39 — angulo_stem_t_max_grau=150 nas imagens reais (PROVISÓRIA, 2026-10-28)

**Contexto:** o filtro `_verifica_geometria_t` usa `angulo_stem_t_max_grau=130°` (default).
Nas imagens reais bitbots e eliza, os dois nós T do gabarito falham com 130° (stem vinculante:
bitbots 146,3°; eliza 135,0°). A varredura em `docs/fase1_imagens_reais.md §5.5.1` mostrou
que `stem_max=150, arco=0` detecta ambos os T reais sem introduzir T FP reais.

**Decisão PROVISÓRIA:** gravar `angulo_stem_t_max_grau=150` e `distancia_arco_geom_px=0`
na seção `"cruzamentos"` de `calibracao/real_provisoria_n2.json`, lida por
`avaliar_cruzamentos.py --reais` via `_cfg_cruzamento_from_calib`. O default do Webots
(`ConfigCruzamento().angulo_stem_t_max_grau=130`) não é alterado.

**Justificativa:** única combinação que recupera os 2 T reais sem T FP reais e sem degradar
Webots. Mecanismo: os ângulos de stem (146,3° e 135,0°) excedem 130° porque a barra do T
forma ângulo < 180° (perspectiva / geometria real), não por ruído de direção.

**Ressalvas:**
1. **In-sample:** limiar escolhido nas mesmas 2 imagens em que é avaliado — recall T=100% é tautológico.
2. **n=2 imagens, 2 T:** sem validade estatística.
3. **Filtro praticamente desligado:** stem_max=150° aceita hastes fazendo até 150° com um braço da barra; junções Y ou artefatos similares podem passar.
4. **Origem das imagens desconhecida:** calibração pode não generalizar.

**Impacto código:**
- `calibracao/real_provisoria_n2.json`: seção `"cruzamentos"` adicionada.
- `avaliar_cruzamentos.py`: função `_cfg_cruzamento_from_calib(calib_path)` lê a seção e aplica ao `ConfigCruzamento` no caminho `--reais`.
- `tests/test_fluxo_reais.py`: 3 testes de regressão verificam que default=130, real=150, e JSON sem seção usa default.
- Webots invariante: Det=27, TP_loc=6, TP_tipo=3, FP=21 (verificado 2026-10-28).

---
