# Roadmap

Estado de partida (tag v0-features): detector de cruzamentos e linhas (Seções III-C e IV)
funcionando em imagens do Webots. Gabarito de 12 pontos: localização 6/12, tipo correto 3/12,
27 detecções (21 FP). Limitações em docs/decisoes.md.

Objetivo final: pipeline completo = segmentação + detecção com YOLO26n + features de linhas
(CrossingFeature / LineFeature) como entrada do filtro de partículas (Seção V, fora deste repo
por enquanto).

## Fase 1a — Imagens reais da RoboCup (alvo do algoritmo)
Hoje só calibramos e avaliamos no Webots. O alvo são imagens reais.
- Reunir um conjunto de imagens reais (bitbots, eliza e outras) e verificar quais geram esqueleto.
- Reprocessar a segmentação (III-A/III-B): calibração HSV (verde/branco), limiar do trow.
- Criar gabarito de cruzamentos E de linhas nas imagens reais (o gabarito atual é só Webots).
- Rodar o avaliador existente nesse conjunto. Os números do Webots deixam de ser a referência.
Critério: >= N imagens reais com esqueleto utilizável e gabarito anotado (N a definir).

**Status (2026-10-27; atualizado em 2026-10-29 com medições de 28/10: calibração verde
passou de H 35–100 para H 45–75; gabarito de cruzamentos criado; horizonte anotado;
D39 aplica angulo_stem_t_max_grau=150 no caminho real — ver Seções 4, 5.4, 5.5.1 de
`docs/fase1_imagens_reais.md`) — ver `docs/fase1_imagens_reais.md` para fatos completos:**
- ✅ 2 imagens reais disponíveis (bitbots, eliza, 640×640 px).
- ✅ Calibração provisória implementada (`calibracao/real_provisoria_n2.json`,
  `features/segmentacao_real.py`); F1_mc branco: bitbots=0.651, eliza=0.463.
  (atualizado em 2026-10-29: valores por método e calibração:
  — `avaliar_mascara_branco.py --com-convexa` com convexa default (H 35–100):
    bitbots=0.651, eliza=0.463.
  — `diag_h4575_pipeline.py` com PixelSegmentReal (H 45–75):
    bitbots=0.652, eliza=0.714 (Seção 4 de `docs/fase1_imagens_reais.md`).
  Os dois pares diferem em ferramenta E em faixa de H do verde (convexa), portanto a
  diferença não está isolada. Experimento controlado em /tmp (mesma lógica de
  `diag_h4575_pipeline.py`, apenas H variado, sem gravar no repo):
    H 45–75: bitbots=0.652, eliza=0.718; H 35–100: bitbots=0.651, eliza=0.466.
  A diferença eliza (0.466→0.718) é atribuível à mudança de H: com H 35–100 a convexa
  se estende ao topo (y_gauss_min=0.00); com H 45–75 recua (y_gauss_min=20.01).
  Para bitbots a diferença é desprezível (0.651→0.652).
  Nota: `diag_h4575_pipeline.py` cita 0.714 para eliza; o experimento controlado obteve
  0.718 com a mesma lógica — discrepância de 0.004 não investigada.)
- ✅ Gabarito de linhas anotado (3 polilinhas por imagem, `tests/gabarito_linhas.json`);
  cobertura pelo esqueleto: bitbots 93%, eliza 96%.
- ✅ Features extraídas e diagnosticadas (cruzamentos, linhas, fragmentação).
- ✅ Gabarito de cruzamentos reais: `tests/gabarito_cruzamentos_reais.json` (4 pontos, n=4 sem
  validade estatística; avaliador em Seção 5.5 de `docs/fase1_imagens_reais.md`).
  (atualizado em 2026-10-29: antes "inexistente")
- ✅ Horizonte anotado (`tests/gabarito_horizonte.json`); fração de branco acima do campo medida
  (ver Seção 5.4 de `docs/fase1_imagens_reais.md`).
  (atualizado em 2026-10-29: antes "horizonte não anotado; fração não medida")
- ❌ eliza p2 precisa ser re-anotada como duas retas; p3 sob dúvida.
- ❌ n=2 imagens (insuficiente); mais imagens necessárias.

## Fase 1b — Calibração de cor no local da competição

Motivação (medido, n=2 imagens reais):
- O limiar de branco fixo do Webots (V>=213) dá recall 0% no bitbots (escuro) e 94% no eliza.
- A mediana de H do carpete difere entre campos: eliza 58, bitbots 65 (Webots 57). A margem
  entre o carpete e o fundo é pequena (fundo do eliza começa em H=65).
- Uma calibração fixa não generaliza entre campos e luzes.

Passo 1 — Ferramenta manual assistida (prioridade):
- scripts/calibrar_cores.py: mostra 1–3 quadros do campo; o usuário clica em pontos de
  carpete e de linha; a ferramenta calcula faixas HSV do verde (H, S, V) e o critério de
  branco (limiar relativo ao V do campo); mostra as máscaras e a mascara_convexa sobre a
  imagem para confirmar visualmente; grava calibracao/<nome>.json no mesmo formato de
  real_provisoria_n2.json (com data, imagens usadas, status).
- Meta de uso: poucos minutos, com validação visual obrigatória antes de gravar.
- Integração: SegmentacaoReal lê o JSON escolhido.
  (atualizado em 2026-10-29: a classe se chama `PixelSegmentReal` — `features/segmentacao_real.py:44`)

Passo 2 — Estimação automática (depois, só com muitas imagens reais):
- Verde: pico dominante do histograma de H na região inferior da imagem (assumindo campo).
- Branco: relativo ao V do campo (V_linha / V_campo) ou top-hat, com tamanho do elemento
  estruturante ligado à largura esperada da linha por perspectiva.
- Checagens de sanidade: fração de verde dentro de um intervalo plausível, H do pico dentro
  de uma faixa de verde, quantidade mínima de branco; se falhar, cair para a calibração manual.
- Máscaras de robôs/bola do YOLO (Fase 3) excluídas do cálculo.

Critério de pronto:
- Passo 1: calibrar uma imagem real nova em < 5 min e obter máscaras com recall de linhas
  >= 0.9 e fração verde coerente na inspeção visual.
- Passo 2: avaliado em um conjunto de imagens reais de várias condições (a definir).

Dependências: conjunto de imagens reais maior (Fase 1). Não depende do YOLO para o Passo 1.

## Fase 2 — Qualidade do detector (sobre imagens reais)
- Curva precisão x confiança nos cruzamentos: com confiança acima de um limiar, quantos TP e FP sobram.
- Gabarito de linhas e métrica de ângulo/distância.
- Tratar as causas já medidas: X nas pontas da elipse com 3 ramos, faixa distante comprimida,
  parede e bola gerando esqueleto falso.
- Regra: mudanças em main.py (código do colega) só aditivas, com teste de regressão contra base-original.
Critério: a definir depois da curva da Fase 2, item 1.

## Fase 3 — YOLO26n e pipeline integrado
- Detecção de objetos (bola, robôs, outros a definir) com YOLO26n.
- Usar as caixas para mascarar a segmentação (bola e robôs não devem gerar esqueleto/FP).
- Interface única: imagem -> {detecções YOLO, CrossingFeature, LineFeature}.
- Avaliar o pipeline completo e o efeito do mascaramento nos FPs.
Dependências: Fase 1 (imagens reais) antes de treinar/avaliar.

## Fase 4 — Filtro de partículas (Seção V)
- Fora do escopo atual. Definir com o colega quem faz e como usa confiança e d_topo_celulas.

## Pendências de ambiente
- Fixar versões em requirements.txt (numpy, opencv, scipy, networkx).
- Ao fim de cada fase: tag no git e números medidos registrados em docs/.