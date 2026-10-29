# Fase 1 — Imagens reais da RoboCup: fatos medidos

Data da medição: 2026-10-27.
(atualizado em 2026-10-29: há medições e ajustes adicionais de 2026-10-28 — ver Seções 4, 5.4,
5.5, 5.5.1. A frase "sem ajuste de parâmetros do detector" vale para os dados de 27/10; a
Seção 5.5.1 documenta a decisão D39 que aplica angulo_stem_t_max_grau=150 no caminho real,
in-sample, a partir de 28/10/2026.)
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
(na data; 152 passed em 2026-10-28, ver 5.5.1, que cita 3 testes novos de config;
decomposição por commit git: 145→149 em 072fe5bc (4 testes adicionados em
`test_fluxo_reais.py`: test_casar_localizacao_basico, test_casar_distancia,
test_casar_localizacao_vazio, test_fluxo_gabarito_temporario); 149→152 em a88fb1c
(3 testes de config em `test_fluxo_reais.py`: test_default_config_stem_max_invariante,
test_cfg_cruzamento_from_calib_carrega_150, test_cfg_cruzamento_from_calib_sem_secao).)

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
(atualizado em 2026-10-29: gabarito criado em 28/10/2026 — `tests/gabarito_cruzamentos_reais.json`,
4 pontos; avaliador e análise na Seção 5.5.)

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

### 5.4 Branco e esqueleto acima do horizonte

Fonte: `scripts/medir_branco_acima_horizonte.py`; horizonte: `tests/gabarito_horizonte.json`.
Calibração: `calibracao/real_provisoria_n2.json`. Resolução original (640×640 px).

| Métrica | bitbots | eliza |
|---------|---------|-------|
| Horizonte (pontos anotados) | 4 pontos | 3 pontos |
| Horizonte y mín / y máx (px orig) | 127 / 233 | 185 / 285 |
| Branco total (px) | 54 859 | 43 557 |
| Acima do horizonte (px / % do total) | 21 104 (38,5%) | 25 299 (58,1%) |
| — dentro da convexa | 9 160 (43,4% do acima) | 2 073 (8,2% do acima) |
| — fora da convexa | 11 944 (56,6% do acima) | 23 226 (91,8% do acima) |
| Abaixo do horizonte (px / % do total) | 33 755 (61,5%) | 18 258 (41,9%) |
| — dentro da convexa | 31 213 (92,5% do abaixo) | 18 212 (99,7% do abaixo) |
| — fora da convexa | 2 542 (7,5% do abaixo) | 46 (0,3% do abaixo) |
| Esqueleto total (px orig, grade×8) | 24 832 | 33 984 |
| Esqueleto acima do horizonte | 643 (2,6%) | 4 677 (13,8%) |
| Esqueleto abaixo do horizonte | 24 189 (97,4%) | 29 307 (86,2%) |

**Avisos:**

1. **Horizonte aproximado.** A reta é definida por 3–4 pontos anotados manualmente com
   `scripts/anotar_horizonte.py` e interpolada linearmente por coluna. É uma aproximação
   grosseira do horizonte real; perspectiva e distorção da câmera não são corrigidas.

2. **Robôs e trave sobre o campo contam como "acima".** Objetos em pé sobre o campo que
   ultrapassam a reta do horizonte (robôs, trave) têm seus pixels brancos contabilizados
   como "acima do horizonte", mesmo que estejam projetados sobre o campo. A medição não
   distingue esses objetos de branco do fundo.

3. **Esqueleto em pixels originais, não comparável 1:1 com branco.** O esqueleto é contado
   após redimensionamento da grade (fator 8) de volta à resolução original via
   `cv2.INTER_NEAREST`; cada célula de grade vira um bloco 8×8 px. O número de "pixels de
   esqueleto" não é diretamente comparável ao número de pixels brancos.

**Convexa como filtro de fundo (eliza):** 91,8% do branco acima do horizonte fica
*fora* da convexa; 99,7% do branco abaixo fica *dentro*. A convexa funciona como filtro
eficaz de pixels de fundo no eliza. No bitbots esse filtro é menos efetivo (56,6% do
branco acima fica fora), provavelmente devido a objetos brancos sobre o campo (robôs,
trave) que estão dentro da convexa mas acima do horizonte.

**Branco de objetos sobre o campo (robôs, postes) permanece** e não é filtrado pela
convexa; o tratamento desses FP está previsto na Fase 3 (mascaramento com YOLO).

### 5.5 Avaliação de cruzamentos com gabarito real (n=4 pontos)

Fonte: `python3 avaliar_cruzamentos.py --reais` (2026-10-28).
Gabarito: `tests/gabarito_cruzamentos_reais.json` (4 pontos anotados manualmente).
Calibração: `calibracao/real_provisoria_n2.json`. Parâmetros: `ConfigCruzamento` default.
Diagnóstico detalhado: `scripts/diagnostico/diag_nos_gabarito_reais.py`.
Sobreposições: `debug_out/bitbots_diag_gabarito.png`, `debug_out/eliza_diag_gabarito.png`.

**Definição de coordenadas:** os pontos do gabarito e as detecções estão em
**células da grade subamostrada** (fator 8; 80×80 células para imagens 640×640).
Um ponto (x, y) na grade corresponde aproximadamente ao pixel (x×8, y×8) na imagem original.

#### Tabela do avaliador --reais (TOTAL)

| Métrica | Valor |
|---------|-------|
| GT total | 4 |
| Detecções total | 15 |
| TP (localização, tol=6 cél) | 2 |
| FP | 13 |
| FN | 2 |
| Precisão (loc) | 13% |
| Revocação (loc) | 50% |

Por tipo:

| Tipo | GT | Det | TP | FP | FN | Prec | Rev |
|------|----|----|----|----|----|----|-----|
| L | 2 | 14 | 2 | 12 | 0 | 14% | 100% |
| T | 2 | 0  | 0 | 0  | 2 | — | 0% |
| X | 0 | 1  | 0 | 1  | 0 | 0% | — |

Por imagem:

| Imagem | GT | Det | TP (loc) |
|--------|----|----|---------|
| bitbots | 2 | 10 | 1 |
| eliza   | 2 | 5  | 1 |

**Aviso: n=4 pontos do gabarito não têm validade estatística.** Os valores de precisão e
revocação são indicativos do comportamento nas duas imagens com a calibração provisória;
não generalizam para o conjunto real.

#### Análise por ponto do gabarito

**bitbots L(35,55) — OK.**
Nó mais próximo: (35,55), grau=2, dist=0 cél. Sobrevive a toda simplificação.
Detectado como L. TP(loc) ✓ TP(tipo) ✓.

**bitbots T(8,41) — FALHA na geometria.**
Nó mais próximo: (8,42), grau=3, dist=1.0 cél. Sobrevive a toda simplificação.
V1=1.00 (3 segmentos brancos em todos os 4 raios), V2=1.00, conf=1.000.
Borda D29: x−r_max = 0 ≥ 0 → não rejeitado por borda.
Rejeitado em `_verifica_geometria_t` (chamado em `classificar_candidatos`),
antes de entrar em `detectar_cruzamentos`. Não aparece em `debug_info`.

Braços e ângulos:

| Par | Ângulo | Critério barra (≥140°) | Critério stem vs outro braço |
|-----|--------|----------------------|------------------------------|
| braço0→(33,55) ↔ braço1→(6,42) | 152,5° | BARRA OK | stem=braço2: 146,3° vs braço1 > 130° **FALHA** |
| braço1→(6,42) ↔ braço2→(20,34) | 146,3° | BARRA OK | stem=braço0: 152,5° vs braço1 > 130° **FALHA** |
| braço0→(33,55) ↔ braço2→(20,34) | 61,2°  | insuf    | — |

Restrição vinculante medida: em ambos os pares de barra válidos, o stem faz ângulo
>130° com um dos braços da barra. Causa subjacente — curvatura da barra por perspectiva
e/ou geometria real do cruzamento — em teste (ver 5.5.1).

**eliza L(30,54) — OK.**
Nó mais próximo: detectado como L dentro da tolerância. TP(loc) ✓ TP(tipo) ✓.

**eliza T(48,43) — FALHA na geometria.**
Nó mais próximo: (47,44), grau=3, dist=1.4 cél. Sobrevive a toda simplificação.
V1=1.00 (3 branco, 2 verde em todos os raios), V2=1.00, conf=1.000.
Borda D29: não rejeitado.
Rejeitado em `_verifica_geometria_t`; não aparece em `debug_info`.

Braços e ângulos:

| Par | Ângulo | Critério barra (≥140°) | Critério stem vs outro braço |
|-----|--------|----------------------|------------------------------|
| braço0→(44,42) ↔ braço1→(49,44) | 146,3° | BARRA OK | stem=braço2: 135,0° vs braço1 > 130° **FALHA por 5°** |
| braço0→(44,42) ↔ braço2→(45,46) | 78,7°  | insuf    | — |
| braço1→(49,44) ↔ braço2→(45,46) | 135,0° | insuf    | — |

Restrição vinculante medida: único par de barra válido (0↔1 = 146,3°); stem faz 135,0°
com braço1, excedendo `angulo_stem_t_max_grau=130°` por 5°. Causa subjacente — curvatura
da barra por perspectiva e/ou geometria real do cruzamento — em teste (ver 5.5.1).

#### Nota sobre os overlays

Os overlays em `debug_out/*_diag_gabarito.png` mostram:
- Verde: pontos do gabarito.
- Vermelho: cruzamentos aceitos.
- Amarelo: candidato mais próximo em `debug_info` (pode não corresponder ao nó T
  rejeitado, pois nós rejeitados em `classificar_candidatos` não entram em `debug_info`).

#### 5.5.1 Sensibilidade do filtro de geometria T — medição (2026-10-28)

Fonte: `scripts/diagnostico/diag_sensibilidade_geom_t.py`.
Parâmetros alterados apenas localmente (ConfigCruzamento temporário); defaults não modificados.
n=4 GT reais, n=12 GT Webots — sem validade estatística; orientação apenas.

##### Tarefa 1: direção dos braços em escala de arco (`distancia_arco_geom_px`)

`_caminhar_ate_arco` avança a partir do vizinho imediato enquanto o nó corrente for
grau=2, acumulando distância. Para ambos os nós T reais, os vizinhos imediatos são
folhas (grau=1) ou junções (grau≥3), que param a caminhada imediatamente, ou
nós grau=2 geometricamente distantes (distância > d_arco) que também retornam o
vizinho imediato. Resultado: os ângulos são invariantes a d_arco para bitbots T;
para eliza T os ângulos mudam a partir de d_arco=5.

**bitbots T(8,42) — ângulos idênticos para todos d_arco ∈ {0,5,7,10}:**
braço0→(33,55) grau=2 a 28,2 cél (maior que qualquer d_arco testado → viz_start);
braço1→(6,42) grau=1 (folha); braço2→(20,34) grau=4 (junção).

| Par | Ângulo | Barra≥140° | stem vinculante | Necessita stem_max |
|-----|--------|------------|-----------------|-------------------|
| 0↔1 | 152,5° | OK | stem2↔1 = 146,3° | ≥ 147° |
| 1↔2 | 146,3° | OK | stem0↔1 = 152,5° | ≥ 153° |
| 0↔2 | 61,2°  | insuf | — | — |

`distancia_arco_geom_px` não altera nenhum ângulo (nenhum braço tem cadeia grau=2 percorrível no intervalo testado).

**eliza T(47,44) — ângulos mudam a partir de d_arco=5:**

| d_arco | braço0 usado | braço2 usado | Par barra | Ângulo barra | stem2↔1 | Necessita stem_max |
|--------|-------------|-------------|-----------|-------------|---------|-------------------|
| 0 | (44,42) grau=2 | (45,46) grau=2 | 0↔1 | 146,3° | 135,0° | ≥ 136° |
| ≥5 | (39,39) grau=4 | (31,54) grau=2 | 0↔1 | 148,0° | 148,0° | ≥ 149° |

Com d_arco≥5, braço0 atinge a junção (39,39) a 9,4 cél; o ângulo stem piora de 135° para 148°.
`distancia_arco_geom_px` piora a detecção do eliza T (ângulo stem aumenta).

**Conclusão:** a causa não é ruído de direção. Os ângulos são determinísticos no grafo
simplificado e violam o critério de haste independente de d_arco. A causa subjacente é
estrutural: a barra forma ângulo < 180° (bitbots: 152,5°; eliza: 146,3°), fazendo o stem
aparecer desalinhado com um dos braços da barra.

##### Tarefa 2: varredura `angulo_stem_t_max_grau` × `distancia_arco_geom_px`

W=Webots GT=12; R=Reais GT=4; tol=6 cél; W_Tdet/R_Tdet = T detectados; W_Ttt/R_Ttt = T TP tipo.

| stem | arco | W_Det | W_TP_l | W_TP_t | W_FP | W_FN | W_Tdet | W_Ttt | R_Det | R_TP_l | R_TP_t | R_FP | R_FN | R_Tdet | R_Ttt | FP_T_reais |
|------|------|-------|--------|--------|------|------|--------|-------|-------|--------|--------|------|------|--------|-------|------------|
| 130 | 0 | 27 | 6 | 3 | 21 | 6 | 8 | 0 | 15 | 2 | 2 | 13 | 2 | 0 | 0 | — |
| 130 | 5 | 29 | 7 | 3 | 22 | 5 | 8 | 0 | 15 | 2 | 2 | 13 | 2 | 0 | 0 | — |
| 130 | 7 | 26 | 5 | 3 | 21 | 7 | 5 | 0 | 15 | 2 | 2 | 13 | 2 | 1 | 0 | bitbots(46,30) c=0,83 |
| 130 | 10 | 25 | 5 | 2 | 20 | 7 | 5 | 0 | 15 | 2 | 2 | 13 | 2 | 1 | 0 | bitbots(46,30) c=0,83 |
| 135 | 0 | 28 | 6 | 3 | 22 | 6 | 9 | 0 | 16 | 3 | 3 | 13 | 1 | 1 | 1 | — |
| 135 | 5 | 29 | 7 | 3 | 22 | 5 | 8 | 0 | 15 | 2 | 2 | 13 | 2 | 0 | 0 | — |
| 135 | 7 | 26 | 5 | 3 | 21 | 7 | 5 | 0 | 15 | 2 | 2 | 13 | 2 | 1 | 0 | bitbots(46,30) c=0,83 |
| 135 | 10 | 25 | 5 | 2 | 20 | 7 | 5 | 0 | 15 | 2 | 2 | 13 | 2 | 1 | 0 | bitbots(46,30) c=0,83 |
| 140 | 0 | 28 | 6 | 3 | 22 | 6 | 9 | 0 | 16 | 3 | 3 | 13 | 1 | 1 | 1 | — |
| 140 | 5 | 29 | 7 | 3 | 22 | 5 | 8 | 0 | 15 | 2 | 2 | 13 | 2 | 0 | 0 | — |
| 140 | 7 | 26 | 5 | 3 | 21 | 7 | 5 | 0 | 15 | 2 | 2 | 13 | 2 | 1 | 0 | bitbots(46,30) c=0,83 |
| 140 | 10 | 25 | 5 | 2 | 20 | 7 | 5 | 0 | 15 | 2 | 2 | 13 | 2 | 1 | 0 | bitbots(46,30) c=0,83 |
| 150 | 0 | 28 | 6 | 3 | 22 | 6 | 9 | 0 | 17 | 4 | 4 | 13 | 0 | 2 | 2 | — |
| 150 | 5 | 29 | 7 | 3 | 22 | 5 | 8 | 0 | 17 | 4 | 4 | 13 | 0 | 2 | 2 | — |
| 150 | 7 | 26 | 5 | 3 | 21 | 7 | 5 | 0 | 17 | 4 | 4 | 13 | 0 | 3 | 2 | bitbots(46,30) c=0,83 |
| 150 | 10 | 25 | 5 | 2 | 20 | 7 | 5 | 0 | 17 | 4 | 4 | 13 | 0 | 3 | 2 | bitbots(46,30) c=0,83 |
| 170 | 0 | 28 | 6 | 3 | 22 | 6 | 9 | 0 | 17 | 4 | 4 | 13 | 0 | 2 | 2 | — |
| 170 | 5 | 29 | 7 | 3 | 22 | 5 | 8 | 0 | 17 | 4 | 4 | 13 | 0 | 3 | 2 | bitbots(46,30) c=0,83 |
| 170 | 7 | 26 | 5 | 3 | 21 | 7 | 5 | 0 | 17 | 4 | 4 | 13 | 0 | 3 | 2 | bitbots(46,30) c=0,83 |
| 170 | 10 | 25 | 5 | 2 | 20 | 7 | 5 | 0 | 17 | 4 | 4 | 13 | 0 | 3 | 2 | bitbots(46,30) c=0,83 |

**Observações:**

- **W_Ttt=0 para todos os stem_max** (Tarefa 4): os 2 T Webots são FN topológicos; nenhum
  ajuste de stem_max ou arco os recupera.
- Aumentar stem_max acrescenta T FP no Webots (W_Tdet sobe de 8 para 9) sem nenhum T TP.
- **Única combinação que detecta os 2 T reais sem T FP real: stem=150, arco=0.**
  (stem=150 arco=5 também detecta os 2 T, mas introduz T FP real para stem=170 arco=5.)
- **arco≥7 introduz T FP real** bitbots(46,30) conf=0,83 para qualquer stem_max, inclusive o default.
- **arco≥7 piora Webots**: W_TP_l cai de 6–7 para 5; W_FN aumenta de 5–6 para 7.
- **arco=5, stem=150**: detecta ambos T reais, sem T FP real, sem degradação Webots.
  R_FP permanece 13 (FP de tipo L e X); nenhum novo FP total é introduzido.
- **eliza T**: com arco=0, stem_max=135 já a detecta (135,0°≤135°); com arco≥5 o ângulo
  stem sobe para 148°, exigindo stem_max≥150 (arco≥5 piora a eliza).
- **bitbots T**: invariante a arco; exige stem_max≥150 (ângulo stem=146,3°) para qualquer arco.

**Custo de falso positivo no Webots:** com stem_max≥135 e arco=0, W_Tdet sobe de 8 para 9;
o nó extra é FP T (posição não verificada individualmente com n=12 Webots).

##### Tarefa 3: contagem de rejeições por `_verifica_geometria_t` (default)

| Imagem | Grau3 | Rej_geom | %Rej | Prox_GT_any(≤6 cél) | Prox_GT-T(≤6 cél) |
|--------|-------|----------|------|---------------------|-------------------|
| image_1 | 5 | 5 | 100% | 3 | 0 |
| image_2 | 10 | 5 | 50% | 1 | 0 |
| image | 4 | 1 | 25% | 0 | 0 |
| bitbots | 3 | 3 | 100% | 1 | 1 |
| eliza | 2 | 2 | 100% | 1 | 1 |

Rejeitados ≤6 cél de um GT-T:
- **bitbots:** rej(8,42) ↔ GT-T(8,41) dist=1,0 cél — é o nó T GT
- **eliza:** rej(47,44) ↔ GT-T(48,43) dist=1,4 cél — é o nó T GT
- **image_2:** nenhum rejeitado próximo de GT-T (os 2 GT T de image_2 são FN topológicos)

`_verifica_geometria_t` rejeita 100% dos nós grau=3 nas imagens reais. Os 2 GT-T reais
são os únicos rejeitados próximos de um GT-T. Em Webots, os 2 GT-T não aparecem como
rejeitados pelo filtro de geometria — pois nem chegam a ser nós grau=3 após simplificação.

##### Tarefa 4: T FN do Webots em image_2 (grade 163×86 cél)

**T(153,53) — FN topológico:**
Nó mais próximo antes da simplificação: (154,53) grau=**2** dist=1,0 cél.
`rem_curv` remove (154,53) (nó quase reto); substituto: (152,52) grau=2, ângulo=164,1°.
Após toda simplificação: grau=2, ângulo=164,1° → não é candidato T nem L.
D29: não rejeitado (dentro dos limites da grade).
**Causa do FN:** topológica — não existe nó de grau≥3 próximo à posição GT.
O ponto GT corresponde a uma região de linha simples no esqueleto simplificado.

**T(9,45) — FN topológico:**
Nó mais próximo antes da simplificação: (9,45) grau=**2** dist=0,0 cél.
`rem_curv` remove (9,45); substituto: (12,46) grau=**1** (folha).
Após toda simplificação: grau=1 → não é candidato.
D29: não rejeitado.
**Causa do FN:** topológica — o GT corresponde a uma folha (extremo de ramo) no esqueleto.

**Contraste com os T reais:** os 2 T FN reais têm grau=3 e sobrevivem à simplificação,
mas falham em `_verifica_geometria_t`. Os 2 T FN Webots nunca chegam a grau=3; nenhum
ajuste de `angulo_stem_t_max_grau` ou `distancia_arco_geom_px` os recupera.

**Aviso: n=4 reais, n=12 Webots. Os valores acima são indicativos, não generalizáveis.**

##### Decisão provisória aplicada — D39 (2026-10-28)

`angulo_stem_t_max_grau=150, distancia_arco_geom_px=0` gravados em
`calibracao/real_provisoria_n2.json` (seção `"cruzamentos"`).
`avaliar_cruzamentos.py --reais` lê essa seção via `_cfg_cruzamento_from_calib`
e aplica **apenas no caminho real**; Webots usa `ConfigCruzamento()` default (130) sem alteração.

**Verificação (2026-10-28):**
- pytest: 152 passed (incluindo 3 novos testes de regressão de config).
- Webots: Det=27, TP_loc=6, TP_tipo=3, FP_loc=21 — **idêntico ao commit 25fed10**. ✓

**Resultados reais com o override (stem=150, arco=0):**

TOTAL: GT=4, Det=17, TP_loc=4 (100%), TP_tipo=4 (100%), FP=13, FN=0.

| Imagem | GT | Det | TP_loc | FP | FN |
|--------|----|----|--------|----|----|
| bitbots | 2 | 11 | 2 | 9 | 0 |
| eliza | 2 | 6 | 2 | 4 | 0 |

Por tipo (reais, tol=6 cél):

| Tipo | GT | Det | TP | FP | FN | Prec | Rev |
|------|----|----|----|----|----|----|-----|
| L | 2 | 14 | 2 | 12 | 0 | 14% | 100% |
| T | 2 | 2  | 2 | 0  | 0 | 100% | 100% |
| X | 0 | 1  | 0 | 1  | 0 | 0% | — |

Lista completa de detecções aceitas (stem=150, arco=0):

**bitbots (11 detecções):**

| tipo | x | y | conf | px_orig |
|------|---|---|------|---------|
| L | 25 | 24 | 1,000 | (200,192) |
| L | 29 | 23 | 1,000 | (232,184) |
| **L** | **35** | **55** | **1,000** | **(280,440)** ← GT |
| L | 40 | 29 | 0,500 | (320,232) |
| L | 40 | 32 | 1,000 | (320,256) |
| L | 42 | 20 | 1,000 | (336,160) |
| L | 46 | 34 | 0,750 | (368,272) |
| L | 47 | 29 | 1,000 | (376,232) |
| L | 49 | 19 | 1,000 | (392,152) |
| L | 59 | 24 | 0,750 | (472,192) |
| **T** | **8** | **42** | **1,000** | **(64,336)** ← GT |

**eliza (6 detecções):**

| tipo | x | y | conf | px_orig |
|------|---|---|------|---------|
| L | 24 | 23 | 0,750 | (192,184) |
| **L** | **31** | **54** | **1,000** | **(248,432)** ← GT |
| L | 46 | 35 | 1,000 | (368,280) |
| L | 49 | 32 | 1,000 | (392,256) |
| **T** | **47** | **44** | **1,000** | **(376,352)** ← GT |
| X | 13 | 28 | 0,625 | (104,224) |

**Ângulos do stem nos 3 T detectados (arco=0):**

bitbots T(8,42): par barra 0↔1 = 152,5°; stem2↔braço0 = 61,2° ≤ 150° ✓; stem2↔braço1 = **146,3° ≤ 150° ✓**.
eliza T(47,44): par barra 0↔1 = 146,3°; stem2↔braço0 = 78,7° ≤ 150° ✓; stem2↔braço1 = **135,0° ≤ 150° ✓**.
Todos os ângulos de stem ficam abaixo de 150°. Sem discrepâncias.

**T FP potencial com arco≥7 — bitbots(46,30):**
Grade (46,30) → pixel original (368,240). Pixel no original: HSV=(21,17,196); classificado
como BRANCO (S≤60, V≥165). Vizinhança 20×20: ~49% branco, 0% verde. Posição no terço
superior-direito da imagem. **Não determinado** se é robô, trave ou linha (inspeção visual
não disponível programaticamente). Nota: esse FP T não aparece com arco=0 (configuração
aplicada); só emerge com arco≥7.

**Ressalvas obrigatórias (D39):**

1. **In-sample:** o limiar 150° foi escolhido nas mesmas 2 imagens em que é avaliado.
   Não há conjunto de validação independente. Recall T=100% é esperado por construção.
2. **n=2 imagens, 2 T:** sem validade estatística. Qualquer estimativa de precisão/recall
   é pontual e não generalizável.
3. **Filtro praticamente desligado:** com stem_max=150°, o filtro aceita hastes que fazem
   até 150° com um braço da barra (o bitbots T vinculante tem 146,3°). Junções Y
   ou outros artefatos com haste quase paralela a um braço da barra podem passar.
4. **Origem das imagens desconhecida:** os sufixos dos nomes de arquivo (`b0.8 s1.0 k0`,
   `b1.0 s1.2 k3`) não foram interpretados. A calibração pode não generalizar para
   imagens de outras condições de iluminação ou câmeras.

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
   **RESOLVIDO (2026-10-29):** horizonte anotado em `tests/gabarito_horizonte.json`; branco e
   esqueleto acima do campo medidos na Seção 5.4.

2. **eliza p2 precisa ser re-anotada como duas retas.** A polilinha p2 atual é curva;
   o maior fragmento detectado (36.3 cel) é classificado como ramo com 21.0° quando se usa
   a direção ponta-a-ponta. eliza poly_1 fica com cobertura de 78.9% pelas linhas extraídas.

3. **eliza p3 sob dúvida.** A polilinha p3 (4.3 cel / 34 px) não produziu nenhum fragmento
   detectado. Pode ser uma linha real muito curta ou um artefato de anotação.

4. **Gabarito de cruzamentos reais com n=4 pontos.** `tests/gabarito_cruzamentos_reais.json`
   foi criado em 28/10/2026 com 2 pontos por imagem (bitbots: L+T; eliza: L+T). Com n=4
   não é possível estimar precisão/revocação com validade estatística (ver Seção 5.5).

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
   **RESOLVIDO (2026-10-29):** validação visual realizada pelo usuário em 28/10/2026 (ver
   bloco em negrito na Seção 4); ressalvas documentadas nos itens 10 e 11.

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

(atualizado em 2026-10-29:)

**Estado atual da calibração e dos parâmetros:**
- Calibração verde: H 45–75 S≥40 V 40–180 (escolhida por percentis de H em n=2 imagens;
  validada visualmente pelo usuário em 28/10/2026 com ressalvas documentadas nos itens 10 e 11
  da Seção 7 — ver Seção 4).
- Calibração branco: V≥165 S≤60 H=0–179 (candidato B; escolhido por F1_mc; n=2).
- `ConfigGrafo`, `ConfigLinhas`, `ConfigSimplificacao`: todos os valores padrão.
- `ConfigCruzamento`: valores padrão **no caminho Webots**; no caminho real (`--reais`),
  `angulo_stem_t_max_grau=150` (default=130) — decisão D39, in-sample, 2026-10-28;
  `distancia_arco_geom_px=0` (padrão). Ver Seção 5.5.1.
