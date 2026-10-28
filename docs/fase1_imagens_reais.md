# Fase 1 — Imagens reais da RoboCup: fatos medidos

Data da medição: 2026-10-27.
Todos os valores abaixo foram produzidos pelos scripts indicados, sem ajuste de parâmetros
do detector; a calibração foi escolhida entre candidatos por métrica (ver Seção 8).

---

## 1. Imagens disponíveis

| Nome-chave | Arquivo | Dimensões | Aspecto |
|-----------|---------|-----------|---------|
| `bitbots` | `images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg` | 640×640 px | 1:1 |
| `eliza`   | `images/eliza_nat_light_20001_png_b1.0_s1.2_k3.jpg` | 640×640 px | 1:1 |

- Nenhuma barra preta detectada (imagens quadradas sem letterbox).
- Origem das imagens: desconhecida; não verificada contra fonte pública.
- Os sufixos `b0.8 s1.0 k0` e `b1.0 s1.2 k3` nos nomes de arquivo **não foram interpretados**;
  pode ser brilho/saturação/contraste de pré-processamento, mas não confirmado.
- V mediano do campo (verde, dentro da máscara convexa) no bitbots ≈ 62; no Webots e eliza ≈ 137–146.
  O sufixo `b0.8` (se brilho) sozinho não explica a diferença de ~75 pontos.
- Grade subamostrada (fator 8, herdado de `main.py`): 80×80 células para ambas.

Fonte: `cv2.imread` + verificação de dimensões.

---

## 2. Larguras das linhas brancas (FWHM de perfis perpendiculares)

Medição com perfis perpendiculares aos segmentos Hough sobre o canal V (HSV).
Fonte: `scripts/diagnostico/diag_b2_perp.py`.
Imagens de debug: `debug_out/bitbots_b3_perp.png`, `debug_out/eliza_b3_perp.png`,
`debug_out/image_1_b3_perp.png`.

| Imagem | Seg | FWHM_med (px orig) | FWHM_min | FWHM_max | FWHM_grade (px_grade) |
|--------|-----|-------------------|---------|---------|----------------------|
| bitbots | seg1 | 16.2 px | 12.6 px | 25.7 px | 2.03 |
| bitbots | seg2 | 24.2 px | 20.7 px | 28.2 px | 3.03 |
| bitbots | seg3 | 26.4 px | 20.9 px | 30.3 px | 3.30 |
| eliza   | seg1 |  6.5 px |  5.8 px |  8.8 px | 0.81 |
| eliza   | seg2 | 10.8 px |  9.4 px | 11.2 px | 1.35 |
| eliza   | seg3 |  9.7 px |  8.3 px | 11.0 px | 1.21 |

**Resumo:** bitbots 16–26 px de mediana; eliza 6.5–11 px de mediana.
Em pixels de grade (÷8): bitbots 2.0–3.3 cel, eliza 0.8–1.4 cel.

Para image_1 (Webots, 1340×891 px): Hough encontra 3 segmentos; somente seg3 corresponde
a uma linha do campo (`debug_out/image_1_b3_perp.png`). Largura das linhas Webots:
30–34 px (medição de sessão anterior, seg3 image_1 = 4.5 px med, provavelmente linha
vista de longe). Sem confirmar valor único para Webots nesta sessão.

---

## 3. Fator de subamostração: 8 — não configurável

O fator 8 é fixado em `main.py` (linhas 44–45: `grid_height = height // 8`) e herdado por
`PixelSegmentReal` sem alteração. Os kernels do `main.py` são definidos em pixels de grade
e **não escalam** com o fator:

| Operação | Kernel / parâmetro | Arquivo | Linha (aprox.) |
|---------|-------------------|---------|---------------|
| `cv2.GaussianBlur` na máscara_relevo | 5×5 px_grade | `main.py` | 360 |
| `cv2.filter2D` (janela de soma verde) | 4×8 px_grade | `main.py` | 128–136 |
| `scipy.signal.medfilt` no y_gauss | kernel_size=3 | `main.py` | 241–243 |
| `gaussian_filter1d` no y_gauss | sigma=1.0 | `main.py` | 248 |

Hipótese não medida: em fator 2, o GaussianBlur 5×5 (em px de grade) cria um plateau na
linha e c_xy < 3 descarta o interior. Em fator 8 o esqueleto cobre 93–96% das linhas
anotadas (Seção 5.2).

---

## 4. Calibração provisória real (`calibracao/real_provisoria_n2.json`)

### Parâmetros HSV

| Canal | Parâmetro | Valor | Alterado em |
|-------|-----------|-------|-------------|
| Branco (B) | H | 0–179 | — |
| Branco (B) | S | 0–60 | — |
| Branco (B) | V | 165–255 | — |
| Verde LUT-B | H | **45–75** | 2026-10-28 (antes: 35–100) |
| Verde LUT-B | S | 40–255 | — |
| Verde LUT-B | V | 40–180 | — |

Implementação: `features/segmentacao_real.py` — subclasse `PixelSegmentReal`.
Branco aplicado via monkeypatch reversível (`_patch_branco`); verde via override de
`_criar_lut_verde`. O caminho Webots (`Pixel_Segment` direto) não é alterado.

**Validada visualmente pelo usuário em 28/10/2026 (debug_out/*_h4575_verde_convexa.png): eliza OK (convexa acompanha a borda do carpete; fundo escuro fora). bitbots: duas ressalvas visíveis: (1) uma cunha de carpete escuro (V<40, sombra) no centro-direita ficou fora da convexa, até y≈570; (2) a convexa sobe até y≈95 na rede branca do gol, à esquerda. Mantidas; hipótese: V_min 40 elevado demais agora que o H separa as classes, não testada.**
Escolha baseada exclusivamente em percentis de H medidos em n=2 imagens (ver abaixo).

### Fundamento da nova faixa de H (2026-10-28)

Diagnóstico (`diag_verde_fundo_escuro.py` e `diag_h_variantes.py`): com H 35–100, pixels
de fundo escuro da imagem eliza (parede/ambiente com tonalidade azul-esverdeada, H≈87–98)
satisfaziam a calibração e expandiam a `mascara_convexa` até y=0, causando `y_gauss` min=0
nas duas imagens.

Distribuição de H dos pixels verdes (máscara base H 35–100, S≥40, V 40–180):

| Imagem | Classe | N_px | p01 | p05 | p50 | p95 | p99 |
|--------|--------|------|-----|-----|-----|-----|-----|
| eliza  | carpete (linhas 400–640) | 148 976 | 52 | 54 | **58** | 63 | **65** |
| eliza  | fundo escuro (linhas 0–160) | 60 504 | 43 | 65 | **88** | 99 | 100 |
| bitbots | carpete (linhas 400–640) | 116 854 | 56 | 61 | **65** | 69 | **72** |
| bitbots | fundo (linhas 0–160) | 11 926 | 35 | 39 | 69 | 90 | 97 |

**Critério de seleção de H_max=75:** maior H_max que reduz a faixa 0–160 do eliza sem
reduzir mais de 3 pp as faixas 320–640 em nenhuma das duas imagens.
- H=75 cobre p99 do carpete bitbots (H=72) com margem de 3 unidades.
- H=75 exclui ≈92% dos pixels de fundo do eliza (p50=88, p05=65 ficam fora).
- H_min=45: escolhido abaixo de p01 do carpete eliza (H=52) com margem de 7 unidades.

**Margem estreita — mediana do carpete: eliza H=58, bitbots H=65.
H_max=75 fica a 3 unidades do p99 do carpete bitbots (H=72).
A separação de classes depende de calibração no local; pode variar com iluminação.**

Verificação: `pytest` 145/145 e avaliador Webots (Det=27 TP(loc)=6 TP(tipo)=3) sem
alteração após a mudança do JSON (o caminho Webots usa `green_pixels.csv`, não o JSON).

### Qualidade da máscara branca com convexa H 45–75

Medição: `scripts/avaliar_mascara_branco.py` e `diag_h4575_pipeline.py`.
Parâmetros: bitbots `d=15 tol=3`; eliza `d=8 tol=3`.

| Imagem | Critério | Rec_mc | Prec_mc | F1_mc | Nota |
|--------|---------|--------|---------|-------|------|
| bitbots | B∩convexa (H 45–75) | 98.5% | 48.7% | 0.652 | — |
| bitbots | top-hat s40 (ref. anterior) | — | — | 0.709 | melhor F1, depende de morfologia |
| eliza   | B∩convexa (H 45–75) | 98.1% | 56.1% | **0.714** | melhora vs. 0.463 anterior |
| eliza   | top-hat s20 (ref. anterior) | — | — | 0.468 | ref., H=35–100 |

A melhora de F1_mc do eliza (0.463 → 0.714) deve-se à correção da convexa: com H=35–100
a convexa estendia ao topo (y_gauss=0) e incluía FP do fundo; com H=45–75 y_gauss_min=20.01.

Rec por polilinha (B∩convexa): eliza p0=97% p1=100% p2=92%; bitbots p0=97% p1=100% p2=100%.

**Avisos sobre as métricas:**
- `F1_mc` = métrica com convexa: considera só pixels dentro da máscara convexa do campo.
- Precisão é subestimada: gabarito cobre apenas 3 polilinhas por imagem (não todas as linhas
  visíveis); pixels brancos fora das polilinhas anotadas são contados como FP.
- n=2 imagens: calibração provisória, não representativa do conjunto real.

---

## 5. Pipeline nas imagens reais

Calibração ativa: verde H 45–75 S≥40 V 40–180; branco V≥165 S≤60.
Fonte: `scripts/diagnostico/diag_h4575_pipeline.py` (2026-10-28).
Sobreposições: `debug_out/{nome}_h4575_verde_convexa.png`, `debug_out/{nome}_h4575_features.png`.

### 5.1 Etapas de segmentação

| Etapa | bitbots | eliza |
|-------|---------|-------|
| white_mask | 54 859 px (13.4%) | 43 557 px (10.6%) |
| green_mask | 220 145 px (53.7%) | 247 271 px (60.4%) |
| campo_detectado | True | True |
| y_gauss min (cel_grade) | 9.02 | **20.01** |
| mascara_convexa | 4 288 células (**67.0%** da grade) | 4 403 células (**68.8%**) |
| mascara_relevo | max e média > 0 | max e média > 0 |
| skeleton | **151** px de grade | **215** px de grade |

**Nota (H 45–75 vs. H 35–100):**
- eliza `y_gauss` min corrigido de 0 → 20.01: pixels de fundo escuro (H≈87–98) não
  entram mais na máscara verde; a convexa recua de 91.9% → 68.8% da grade.
- Eliza green_mask: 322 667 px (78.8%) → 247 271 px (60.4%); a redução vem do topo da imagem.
- Skeleton eliza: 300 → 215 px de grade; skeleton bitbots: 160 → 151 px de grade.
  A perda do esqueleto não afeta a cobertura do gabarito (ver 5.2).

### 5.2 Cobertura do gabarito de linhas pelo esqueleto

Tolerância: 3 px originais. Fonte: `diag_h4575_pipeline.py`.

| Polilinha | bitbots rec | eliza rec |
|-----------|------------|----------|
| p1 | 88% | 96% |
| p2 | 100% | 100% |
| p3 | 94% | 53% |
| **Total** | **93%** | **96%** |

Cobertura idêntica à medição anterior (H 35–100): os pixels do esqueleto perdidos com H 45–75
estavam fora das polilinhas do gabarito. eliza p3 (53%) reflete polilinha curta (4.3 cel / 34 px
na imagem original).

### 5.3 Features extraídas

Fonte: `diag_h4575_pipeline.py` e `diag_fragmentacao.py`.

| Métrica | bitbots | eliza |
|---------|---------|-------|
| Nós antes da simplificação | 76 | 101 |
| Nós depois da simplificação | 39 | 56 |
| Cruzamentos L | 10 | 4 |
| Cruzamentos T | 0 | 0 |
| Cruzamentos X | 0 | 1 |
| Total cruzamentos | **10** | **5** |
| Linhas extraídas | 15 | 14 |

Não há gabarito de cruzamentos para imagens reais.

#### 5.3.1 Classificação por fração de comprimento

**Definição:** uma linha é associada a uma polilinha se ≥50% dos pontos amostrados
(grade) estão a ≤3,0 células da polilinha rasterizada na grade. Ângulo entre a linha
e a direção da polilinha (ponta-a-ponta) ≤20° → **fragmento principal**; >20° →
**ramo de junção**; sem polilinha com fração ≥50% → **sem correspondência**.
(`TOL_GRADE=3.0`, `FRAC_MATCH=0.50`, `ANGULO_RAMO_GR=20.0`; `diag_fragmentacao.py`.)

| Categoria | bitbots (H 45–75) | eliza (H 45–75) |
|-----------|-------------------|-----------------|
| (a) fragmento principal | 4 linhas / 99.4 cel / 795 px | 3 linhas / 89.6 cel / 717 px |
| (b) ramo de junção | 5 linhas / 30.7 cel / 246 px | 3 linhas / 56.7 cel / 454 px |
| (c) sem correspondência | 6 linhas / ~45.4 cel / ~363 px | 8 linhas / ~81.4 cel / ~651 px |

Comparação com H 35–100 na mesma definição:

| Categoria | bitbots H 35–100 | bitbots H 45–75 | Δ | eliza H 35–100 | eliza H 45–75 | Δ |
|-----------|-----------------|-----------------|---|----------------|---------------|---|
| (a) fragmento principal | 4 | 4 | 0 | 3 | 3 | 0 |
| (b) ramo de junção | 5 | 5 | 0 | 2 | 3 | +1 |
| (c) sem correspondência | 8 | 6 | −2 | 17 | 8 | −9 |

A redução em (c) no eliza (17→8) é esperada: com H 35–100, pixels do fundo escuro
criavam linhas espúrias no topo da imagem que desapareceram com H 45–75.
A linha extra em (b) no eliza (2→3) é o fragmento de 36.3 cel de poly_2 (21.0°,
marginal; o limiar é 20°).

Linhas de categoria (c) **não foram inspecionadas visualmente**.

#### 5.3.2 Classificação por proximidade ao gabarito (d≤25 px)

**Definição:** uma linha é associada à polilinha mais próxima cujo extremo está a
≤25 px originais de um extremo da linha. Sem classificação por ângulo.
Fonte: `diag_h4575_pipeline.py`.

| Métrica | bitbots | eliza |
|---------|---------|-------|
| → (a) polilinha 0 do gabarito (d≤25px) | 6 (62.9 cel / 503 px) | 5 (110.1 cel / 881 px) |
| → (b) polilinhas 1 e 2 do gabarito (d≤25px) | 2 (62.4 cel / 499 px) | 1 (36.3 cel / 291 px) |
| → (c) sem correspondência no gabarito | 7 (50.2 cel / 402 px) | 8 (81.3 cel / 651 px) |

Cobertura do gabarito pelas linhas extraídas (d≤25px, poly_0 / poly_1 / poly_2):
- bitbots: 98.7% / 100.0% / 100.0%
- eliza:   100.0% / 78.9% / 94.4%  (eliza poly_1: polilinha curva, sub-cobertura esperada)

---

## 6. Diagnóstico de fragmentação

Fonte: `scripts/diagnostico/diag_fragmentacao.py`.
Imagens de debug: `debug_out/{nome}_fragmentacao.png`.
Limiar de classificação como ramo: ângulo > 20° em relação à direção principal da polilinha.

### 6.1 Ramos de junção identificados (não são fragmentações da linha)

| Imagem | p1 | p2 | p3 | Total | Comprimentos | Nó de bif. mais próximo |
|--------|----|----|----|----|----|----|
| bitbots | 4 | 1 | 1 | **6** | 3–8 cel (25–66 px) | grau 3/4 a 1.6–4.1 cel |
| eliza   | 1 | 2 | 0 | **3** | 3–10 cel (29–80 px) | grau 3/4 a 0.8–17 cel |
| image_1 | 1 | — | — | **1** | 8.5 cel (68 px) | grau 4 a 2.0 cel |

Nota eliza p2: o fragmento de 36.3 cel (291 px) foi classificado como ramo com 21.0° —
marginal. A polilinha p2 é curva e a direção foi medida de ponta a ponta; o limiar de 20°
pode ser inadequado para polilinhas curvas.

### 6.2 Fragmentos principais e critérios de merge (todos os 4 critérios)

Defaults de `ConfigLinhas`: ang ≤ 5°, orto ≤ 2.0 cel, vão ≤ 20 cel, sup ≥ 0.8.

**bitbots p1** (56.8 cel / 454 px):

| Fragmento | len | ang_vs_poly | cobertura |
|-----------|-----|------------|----------|
| f1 | 4.1 cel (33 px) | 8.8° | 100% |
| f2 | 39.0 cel (312 px) | 1.2° | 100% |
| cobertura conjunta | | | **92%** |

Par f1↔f2:

| Critério | Valor | Limiar | Resultado |
|---------|-------|--------|---------|
| ANG | 9.9° | ≤ 5.0° | **✗ FALHA** |
| ORTO | 2.2 cel | ≤ 2.0 cel | **✗ FALHA** |
| VÃO | 3.5 cel | ≤ 20.0 cel | ✓ OK |
| SUP | 1.00 | ≥ 0.8 | ✓ OK |

Nó no vão: bifurcação nó 19 grau 3 em (8, 42) (dist_borda=8 cel, zona BORDA).

**eliza p1** (83.1 cel / 665 px):

| Fragmento | len | ang_vs_poly | cobertura |
|-----------|-----|------------|----------|
| f1 | 78.4 cel (628 px) | 0.0° | 100% |
| f2 | 7.6 cel (61 px) | 8.1° | 100% |
| f3 | 3.6 cel (29 px) | 17.8° | 100% |
| cobertura conjunta | | | **99%** |

Par f1↔f2 (f2 está **dentro da zona coberta por f1**; gap=33.6 cel é entre o extremo
direito de f1 e f2, não uma lacuna real na detecção):

| Critério | Valor | Limiar | Resultado |
|---------|-------|--------|---------|
| ANG | 8.1° | ≤ 5.0° | **✗ FALHA** |
| ORTO | 0.5 cel | ≤ 2.0 cel | ✓ OK |
| VÃO | 33.6 cel | ≤ 20.0 cel | **✗ FALHA** |
| SUP | 1.00 | ≥ 0.8 | ✓ OK |

Para fundir este par, seria preciso relaxar ANG (8.1° → >5°) **e** VÃO
(33.6 cel → >20 cel) simultaneamente. Relaxar apenas o ângulo não basta.

Par f2↔f3:

| Critério | Valor | Limiar | Resultado |
|---------|-------|--------|---------|
| ANG | 9.6° | ≤ 5.0° | **✗ FALHA** |
| ORTO | 0.5 cel | ≤ 2.0 cel | ✓ OK |
| VÃO | 0.1 cel | ≤ 20.0 cel | ✓ OK |
| SUP | 1.00 | ≥ 0.8 | ✓ OK |

### 6.3 Bifurcações na borda da grade — bitbots p1

Bifurcações próximas ao início da polilinha p1 e distância à borda da grade 80×80:

| Posição (x,y) | dist_borda | Zona |
|--------------|-----------|------|
| (6, 42) | 6 cel | **BORDA** (< 10) |
| (8, 42) | 8 cel | **BORDA** |
| (9, 40) | 9 cel | **BORDA** |
| (10, 43) | 10 cel | interior |
| (19, 36) | 19 cel | interior |
| (22, 34) | 22 cel | interior |

Densidade de nós grau≥3 por célula de esqueleto — faixa de borda (< 10 cel) vs interior:

| Imagem | borda | interior | razão |
|--------|-------|---------|-------|
| bitbots | 0.214 bif/cel | 0.096 bif/cel | **2.2×** mais densa na borda |
| eliza   | 0.068 bif/cel | 0.071 bif/cel | ≈igual |
| image_1 | 0.028 bif/cel | 0.036 bif/cel | borda menos densa |

Somente bitbots mostra acúmulo de bifurcações na faixa de borda. Coincide com o início
da polilinha p1 passando pela margem esquerda da imagem.
n=1 imagem; limite de 10 células escolhido arbitrariamente.

---

## 7. Lacunas — o que NÃO foi medido nesta fase

1. **Horizonte não anotado.** `scripts/anotar_horizonte.py` foi criado mas não executado
   (requer display). Sem horizonte, não é possível medir fração de branco acima do campo
   (`scripts/medir_branco_acima_horizonte.py`) nem avaliar falsos positivos acima do campo.

2. **eliza p2 precisa ser re-anotada como duas retas.** A polilinha p2 atual é curva;
   o maior fragmento detectado (36.3 cel) é classificado como ramo com 21.0° quando se usa
   a direção ponta-a-ponta. eliza poly_1 fica com cobertura de 78.9% pelas linhas extraídas.

3. **eliza p3 sob dúvida.** A polilinha p3 (4.3 cel / 34 px) não produziu nenhum fragmento
   detectado. Pode ser uma linha real muito curta ou um artefato de anotação.

4. **Gabarito de cruzamentos reais inexistente.** Os cruzamentos detectados (bitbots: 10,
   eliza: 5 com H 45–75) não puderam ser avaliados como TP/FP.

5. **n=2 imagens.** Insuficiente para calibração definitiva ou avaliação estatística.
   Mais imagens com condições de iluminação e distâncias variadas são necessárias.

6. **Origem das imagens não verificada.** Os sufixos dos nomes de arquivo não foram
   interpretados; não se sabe de que dataset ou campeonato as imagens foram extraídas.

7. **Vazamento do esqueleto não medido.** A fração do esqueleto que cai fora das 3 polilinhas
   anotadas (categoria (c): bitbots 7 linhas / 50.2 cel; eliza 8 linhas / 81.3 cel) não foi
   inspecionada visualmente. Pode incluir linhas do campo não anotadas no gabarito.

8. **Margem de H estreita para bitbots.** H_max=75 cobre p99 do carpete bitbots (H=72) com
   margem de apenas 3 unidades. Sob iluminação diferente da medição (eliza iluminação natural,
   bitbots outdoor), o carpete pode deslocar-se para H>75 e ser excluído da máscara verde.
   A calibração depende de medição no local de cada campo.

9. **H 45–75 não validada visualmente.** A nova faixa foi escolhida por critério métrico
   (percentis de H, restrição de ≤3 pp nas faixas 320–640) em n=2 imagens. Nenhuma
   inspeção visual das sobreposições foi realizada pelo usuário até o momento.

10. **Cunha de carpete escuro fora da convexa (bitbots).** Na validação visual de 28/10/2026,
    uma cunha de carpete escuro (V<40, sombra) no centro-direita da imagem bitbots ficou fora
    da máscara convexa, até y≈570. Hipótese: V_min=40 elevado demais agora que o H separa
    as classes; não testada.

11. **Convexa sobe até a rede branca do gol (bitbots).** Na mesma validação, a convexa
    sobe até y≈95 na região da rede branca do gol, à esquerda da imagem bitbots. Pode
    introduzir FP brancos no topo da máscara nessa região.

---

## 8. Parâmetros não ajustados

Todos os parâmetros de `ConfigGrafo`, `ConfigLinhas`, `ConfigCruzamento` e
`ConfigSimplificacao` (em `features/`) permanecem com seus valores padrão.
A calibração `calibracao/real_provisoria_n2.json` foi escolhida entre 3 candidatos (A, B, C)
por recall centerline e precisão do avaliador (F1_mc), SEM validação visual do usuário,
e é provisória (n=2).
