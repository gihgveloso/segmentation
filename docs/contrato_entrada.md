# Contrato de Entrada — Saída de `Pixel_Segment.processar`

> Etapa 0 do PLANO.md. Somente leitura; nenhum código foi escrito.
> Fonte: `main.py` lido integralmente.

---

## 1. Chaves do dicionário devolvido por `processar(frame)`

O método mescla quatro sub-dicionários com `{**a, **b, **c, **d}`, produzindo 18 chaves:

### 1a. `_masks_and_resize` — resolução **original**

| Chave | Dtype | Shape | Descrição |
|---|---|---|---|
| `white_mask` | `uint8` | `(H, W)` | Máscara binária branca na resolução original |
| `green_mask` | `uint8` | `(H, W)` | Máscara binária verde na resolução original |
| `white_mask_resized` | `float32` | `(H/8, W/8)` | Máscara branca reduzida (INTER_AREA) |
| `green_mask_resized` | `float32` | `(H/8, W/8)` | Máscara verde reduzida (INTER_AREA) |

### 1b. `_binarization` — resolução **grade**

| Chave | Dtype | Shape | Descrição |
|---|---|---|---|
| `uniq_img` | `uint8` | `(Hg, Wg)` | Fusão branco×0.5 + verde |
| `borda_binaria` | `uint8` | `(Hg, Wg)` | Borda binarizada do campo (0 ou 255) |
| `histogram` | `list[list]` | `Wg × 2` | `[[y, col], …]` — borda por coluna |
| `y_gauss` | `float32` | `(Wg,)` | Borda suavizada (mediana + Gauss) |
| `contour_points` | `int32` | `(N, 2)` | Polígono superior do campo em coords de grade |
| `mascara_convexa` | `uint8` | `(Hg, Wg)` | Máscara preenchida do campo (0 ou 255) |
| `campo_detectado` | `bool` | — | `False` se poucas colunas válidas |
| `trow_efetivo` | `float` | — | Limiar de linha escalado pela largura real |

### 1c. `_skeletonization_and_connect` — resolução **grade**

| Chave | Dtype | Shape | Descrição |
|---|---|---|---|
| `mascara_campo_original` | `uint8` | `(H, W)` | `mascara_convexa` reescalada p/ resolução original |
| `field_line_mask` | `uint8` | `(Hg, Wg)` | `white_mask_resized` AND `mascara_convexa` |
| `segmentacao_linhas` | `uint8` | `(H, W)` | `white_mask` AND `mascara_campo_original` |
| `mascara_relevo` | `float32` | `(Hg, Wg)` | Gauss(5×5) de `field_line_mask` — **mapa de relevo** |
| `skeleton` | `bool` | `(Hg, Wg)` | Esqueleto — `True` onde é crista do relevo |
| `skeleton_img` | `uint8` | `(Hg, Wg)` | Esqueleto como imagem (0 ou 255) |

### 1d. `_extrair_segmentos_conectados` — coords **originais (float)**

| Chave | Tipo | Descrição |
|---|---|---|
| `polylines_linhas` | `list[dict]` | `{'pontos': [(x,y),…], 'fechada': bool}` — polilinhas simplificadas (approxPolyDP, ε=0.8) |
| `segmentos_linhas` | `list[dict]` | `{'p1': (x,y), 'p2': (x,y)}` — pares de pontos na resolução original |

---

## 2. O esqueleto

A **chave de entrada da minha parte** é `skeleton` (bool) ou equivalentemente `skeleton_img` (uint8 0/255), ambas em resolução de grade.

A chave `mascara_relevo` (float32, mesma grade) também está disponível e é o mapa de relevo contínuo do qual o esqueleto foi derivado.

### Como o esqueleto foi calculado (para referência)

```
field_line_mask  →  GaussianBlur(5×5)  →  mascara_relevo
c_xy[p] = nº de 8-vizinhos de p onde mascara_relevo[viz] >= mascara_relevo[p]
skeleton[p] = (mascara_relevo[p] > 0) AND (c_xy[p] < 3)
```

`c_xy` **não é exposto** no dicionário de saída, mas `mascara_relevo` sim.

---

## 3. Resolução da grade e escala

```python
grid_height = height // 8   # divisão inteira
grid_width  = width  // 8

escala_x = width  / grid_width   # ≈ 8.0 (pode diferir em 1 px por arredondamento)
escala_y = height / grid_height
```

Conversão de pixel de grade `(gx, gy)` para pixel original `(ox, oy)`:

```python
ox = (gx + 0.5) * escala_x   # +0.5 = centro da célula (convenção usada em _extrair_segmentos_conectados)
oy = (gy + 0.5) * escala_y
```

Toda lógica de nós e grafo opera **na grade**. A conversão para original só ocorre na saída final das features.

---

## 4. O esqueleto é binário ou tem cinza?

**O esqueleto (`skeleton`, `skeleton_img`) é binário** — valores {True/False} ou {0, 255}.

Porém o **mapa de relevo `mascara_relevo` é contínuo** (float32, aprox. 0–255), disponível no dict. Ele representa a "altitude" de cada ponto: pixels de linha branca espessa têm valor alto, bordas têm valor baixo.

Consequência para c(x,y):

- **Opção A — paper-fiel** (possível porque `mascara_relevo` está no dict):  
  `c(x,y) = nº de 8-vizinhos onde mascara_relevo[viz] >= mascara_relevo[x,y]`,  
  restrito a pixels onde `mascara_relevo > 0`.

- **Opção B — adaptação binária** (se preferir não depender do relevo):  
  `c(x,y) = nº de 8-vizinhos onde skeleton[viz] == True`.

A Opção A é mais fiel ao artigo e não tem custo extra (dado já existe). **Pendente: decisão abaixo (D1).**

---

## 5. Como calcular c(x,y) a partir do que já existe

### Opção A (recomendada se D1 aprovar)

```python
relevo = resultado['mascara_relevo']   # float32, shape (Hg, Wg)
skeleton = resultado['skeleton']       # bool, shape (Hg, Wg)

padded = np.pad(relevo, 1, mode='constant', constant_values=0)
vizinhos_8 = np.stack([
    padded[:-2, 1:-1], padded[2:, 1:-1],
    padded[1:-1, 2:],  padded[1:-1, :-2],
    padded[:-2, 2:],   padded[2:, 2:],
    padded[:-2, :-2],  padded[2:, :-2]
])
c_xy = (vizinhos_8 >= relevo[np.newaxis]).sum(axis=0).astype(np.uint8)
# c_xy fora do esqueleto não é usado; usar máscara skeleton
```

### Opção B (fallback binário)

```python
skel = resultado['skeleton_img'].astype(np.uint8) // 255
padded = np.pad(skel, 1)
vizinhos_8 = np.stack([
    padded[:-2, 1:-1], padded[2:, 1:-1],
    padded[1:-1, 2:],  padded[1:-1, :-2],
    padded[:-2, 2:],   padded[2:, 2:],
    padded[:-2, :-2],  padded[2:, :-2]
])
c_xy = vizinhos_8.sum(axis=0).astype(np.uint8)
c_xy = np.where(skel, c_xy, 0)
```

Em ambos os casos: pixel com c=1 é extremidade, c=2 é passagem, c≥3 é junção, c=0 é isolado.

---

## 6. Notas de implementação

- **`_extrair_segmentos_conectados` é do colega** e não deve ser alterado nem substituído.  
  Minha `StructureGraph` é uma estrutura **paralela e independente**, construída a partir do `skeleton`, para uso exclusivo das etapas IV-A e IV-B.

- **Entrada mínima da minha parte:**
  ```python
  resultado = segmentador.processar(frame)
  esqueleto   = resultado['skeleton']          # bool, grade
  relevo      = resultado['mascara_relevo']    # float32, grade
  campo_mask  = resultado['mascara_convexa']   # uint8, grade — para "esticar extremos"
  white_grade = resultado['white_mask_resized']  # float32, grade — para verificação IV-A
  green_grade = resultado['green_mask_resized']  # float32, grade — para verificação IV-A
  ```

- **Calibração de câmera:** nenhum arquivo encontrado no repositório. A etapa de undistortion (Etapa 8) fica atrás de flag. `geometry.py` pode ficar vazio até que haja calibração.

---

## 7. Dúvidas que precisam de decisão

### D1 — c(x,y): Opção A (relevo float) ou Opção B (esqueleto binário)?

**Contexto:** `mascara_relevo` está disponível no dict, então a Opção A é factível e fiel ao artigo.  
Porém a Opção B é mais simples e desacopla minha parte do mapa de relevo (que é um detalhe interno do III-B).  
**Pergunta:** usar o relevo float para c(x,y) (Opção A) ou só o esqueleto binário (Opção B)?

---

### D2 — Poda de diagonais no grafo

**Contexto:** `_extrair_segmentos_conectados` usa uma vizinhança 8-conectada **com poda**: se uma diagonal existe mas também existe o vizinho vertical ou horizontal no caminho, a diagonal é ignorada. Isso reduz o grau em junções.  
Minha `features/connections.py` deve usar a mesma poda para ser consistente, ou usar 8-conectividade pura como no artigo?  
**Pergunta:** poda de diagonais sim ou não?

---

### D3 — "Degree-0 nodes are split" (Seção III-C, via Behnke 1997)

**Contexto:** Pixel de esqueleto isolado (sem nenhum vizinho no esqueleto) tem c=0. O artigo diz que esses nós são "divididos", mas não detalha o mecanismo.  
Interpretações possíveis:  
  (a) O pixel vira dois nós sobrepostos formando um micro-segmento de comprimento zero.  
  (b) O pixel é descartado (muito curto para ser uma linha).  
  (c) O pixel vira um único nó de grau 0, mantido no grafo como anomalia.  
**Pergunta:** qual interpretação adotar? (Se não souber, registro (b) em `docs/decisoes.md` por ser a mais conservadora.)

---

### D4 — "Esticar extremos" até qual fronteira?

**Contexto:** A Seção III-C diz que nós terminais (grau 1) são "stretched to the end of the skeleton."  
Interpretação A: mover o nó terminal até o último pixel do esqueleto existente na direção da aresta (simplesmente verificar quantos pixels de esqueleto existem além do nó atual na mesma direção).  
Interpretação B: extrapolá-lo até a borda do campo (`mascara_convexa`), mesmo que não haja esqueleto lá.  
**Pergunta:** qual interpretação?

---

### D5 — Atributo `gray` do Node

**Contexto:** PLANO.md define `Node(id, x, y, gray)`. O `gray` seria o valor de `mascara_relevo` no pixel do nó?  
Isso só faz sentido se a D1 escolher Opção A.  
**Pergunta:** confirma que `gray = mascara_relevo[gy, gx]`? Se D1 = Opção B, o atributo some?

---

### D6 — Lógica de cruzamento do `_extrair_segmentos_conectados` existente

**Contexto:** O colega já usa `_extrair_segmentos_conectados` que identifica `nos` (pixels com vizinhos ≠ 2) e extrai caminhos. Isso é funcionalmente similar à colocação de nós da Seção III-C.  
Minha parte constrói um grafo independente (paralelo), mas ambos trabalham no mesmo `skeleton_img`.  
**Confirmação pedida:** está correto que minha `StructureGraph` é **completamente independente** de `polylines_linhas` e `segmentos_linhas`? Não devo reaproveitar esses resultados, só o `skeleton` bruto?

---

*Fim do contrato de entrada. Próximo passo: aguardar resposta às dúvidas D1–D6 antes de iniciar a Etapa 1.*
