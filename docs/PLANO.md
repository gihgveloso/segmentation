# PLANO — Seções III-C e IV (Line Structure-based Localization)

Entrada da minha parte: esqueleto das linhas brancas dentro do campo (saída do III-B, feita pelo colega).
Saída da minha parte: lista de `CrossingFeature` (tipo L/T/X, posição, orientação, confiança) e
`LineFeature` (extremos, distância e ângulo), prontas para o filtro de partículas (que fica para depois).

Fluxo:
`esqueleto -> nós (III-C) -> grafo conectado (III-C) -> grafo simplificado (IV-A) -> cruzamentos (IV-A)`
e, em paralelo a partir do grafo conectado, `-> linhas (IV-B)`.

> Observação: o `main.py` atual já extrai polilinhas do esqueleto com `approxPolyDP`
> (`_extrair_segmentos_conectados`). Isso é do colega e não muda. Minha parte constrói o grafo
> do artigo a partir do esqueleto, em paralelo, sem tocar nesse código.

## Etapa 0 — Descoberta (somente leitura, modo Plan)
Objetivo: descobrir o contrato real de entrada. Não escrever código.
- Quais chaves o dicionário de `Pixel_Segment.processar` devolve? Qual delas é o esqueleto, a máscara do campo, a máscara de branco/verde?
- Resolução da grade subamostrada e fator de escala para a imagem original.
- O esqueleto é binário ou tem níveis de cinza (densidade de branco após o passa-baixa)? Isso decide como calcular c(x,y):
  - cinza disponível: c(x,y) = nº de vizinhos-8 com valor >= ao do pixel central (exatamente como no artigo);
  - só binário: c(x,y) = nº de vizinhos-8 que também são esqueleto (adaptação, registrar em docs/decisoes.md).
- Existe calibração da câmera (matriz e distorção)? Se não, a etapa 8 fica atrás de flag.
Entregável: `docs/contrato_entrada.md` (arquivo novo) com o contrato e as dúvidas.

## Etapa 1 — Estruturas de dados e c(x,y)  [III-C]
- `features/graph.py`: `Node(id, x, y, gray)`, `StructureGraph` (adjacência, `degree(n)`, `add_edge`, `remove_node`, `neighbors`).
- `features/nodes.py::neighbor_count`: calcula c(x,y) para todos os pixels do esqueleto.
- Testes: esqueletos sintéticos pequenos com c conhecido (ponta = 1, meio de linha = 2, cruzamento = pico).

## Etapa 2 — Colocação de nós  [III-C]
- Nós nos picos (c = 0).
- Depois, nós em pixels de crista (c = 1 e c = 2) com distância mínima de 2 px entre nós (varredura gulosa; desempate pelo maior nível de cinza).
- Testes: linha reta horizontal de 20 px -> nós espaçados >= 2 px, extremos cobertos.

## Etapa 3 — Conexão por vizinhança 3x3  [III-C]
- Inserir aresta entre dois nós cujas regiões 3x3 se sobrepõem ou se tocam sobre o esqueleto.
- Implementação sugerida: rotular a região 3x3 de cada nó; conectar rótulos adjacentes (8-conexo) apenas por pixels do esqueleto.
- Teste: a linha reta vira uma cadeia de nós com grau 1 nas pontas e 2 no meio; um L e um T geram os graus esperados.

## Etapa 4 — Esticar extremos e dividir nós de grau 0  [III-C]
- Esticar: nó terminal (grau 1) é movido até o último pixel do esqueleto na direção da linha.
- Dividir: nó de grau 0 sobre um trecho de esqueleto vira dois nós nas extremidades do trecho, preenchendo lacunas.
- Ambíguo no artigo (remete a Behnke 1997): registrar a interpretação em docs/decisoes.md.
- Teste: segmentos curtos isolados passam a ter dois extremos conectados.

## Etapa 5 — Conexões candidatas (continuidade, fechamento, simplicidade)  [III-C]
- Gerar candidatas entre nós terminais (grau 0/1) próximos.
- Custo: distância entre nós pequena + cinza médio ao longo do segmento candidato parecido com o dos nós + continuidade angular com a direção da aresta existente.
- Restrições: grau máximo 4 (proibir cruzamento de grau > 4) e proibir cruzamentos que não formem linhas contínuas.
- Seleção gulosa pelo menor custo, abaixo de limiares em `ConfigConexao` (dataclass).
- Teste: linha com lacuna de 3-5 px é reconectada; duas linhas distantes não são ligadas.

## Etapa 6 — Simplificação do grafo  [IV-A]
- Suavizar as linhas, remover nós em trechos de baixa curvatura, eliminar linhas curtas que terminam em junção (espúrios), fundir junções próximas e conectadas em um único cruzamento.
- Nós movidos/criados sempre sobre o esqueleto (sem distorcer a visão do mundo).
- Teste: espúrio de 2-3 px em uma linha reta some; duas junções coladas viram um nó de grau 4.

## Etapa 7 — Candidatos e verificação de cruzamentos  [IV-A]
- Grau 2 com ângulo entre arestas relevante -> candidato L; grau 3 -> T; grau 4 -> X.
- Verificação 1: máquina de estados percorrendo um círculo em volta do cruzamento nas máscaras verde/branco subamostradas, checando a sequência verde-branco / branco-verde esperada para o tipo.
- Verificação 2: existe caminho branco do cruzamento até o vizinho em cada direção.
- Saída: `CrossingFeature(tipo, x, y, orientacao, confianca)`.
- Testes: L, T, X sintéticos viram os tipos corretos; um "T falso" (sem fundo verde) é rejeitado.

## Etapa 8 — Extração de linhas  [IV-B]
- Linha = conjunto conectado de nós de grau 2 + nós de grau diferente ligados diretamente a ele.
- Remover distorção da lente nas coordenadas dos nós (`cv2.undistortPoints`; identidade se não houver calibração).
- Split-and-merge: reta por mínimos quadrados, nó mais distante define o ponto de divisão, recursão; componentes com < 3 nós descartados; fusão de componentes com parâmetros parecidos.
- Extremos finais = projeção dos extremos do componente sobre a reta ajustada (não usar os extremos brutos).
- Opcional (flag): projeção no chão com altura/pitch/roll da câmera -> (distância, ângulo) egocêntricos.
- Saída: `LineFeature`.

## Etapa 9 — Integração, visualizador e revisão
- `run_features.py`: roda `Pixel_Segment` + minha parte e desenha nós, grafo, cruzamentos (L/T/X com cores) e linhas, no estilo do visualizador do colega (a/d navegam, q sai).
- Rodar nas imagens de `images/`; ajustar limiares só via dataclasses de config.
- Verificação final: `pytest`, `git diff --name-only` mostrando só arquivos novos, `docs/decisoes.md` completo, README da minha parte em `docs/features.md`.
- Pedir uma revisão de código ao Claude (`/review` ou "revise o diff desta branch contra o artigo").

---

# Prompts prontos para colar no Claude Code (um por etapa)

**Etapa 0** (em modo Plan, Shift+Tab até aparecer "plan mode"):
```
Leia CLAUDE.md e docs/paper.pdf (seções III-C e IV). Depois leia main.py inteiro, sem editar nada.
Responda: (1) quais chaves o dict de Pixel_Segment.processar devolve e qual é o esqueleto;
(2) resolução da grade subamostrada e escala para a imagem original; (3) o esqueleto é binário
ou tem cinza? (4) como eu calcularia c(x,y) a partir do que já existe. Grave o resultado em
docs/contrato_entrada.md e liste dúvidas que precisem de decisão minha. Não escreva código.
```

**Etapas 1 e 2:**
```
Seguindo PLANO.md etapas 1 e 2 e a seção III-C do artigo, use TDD: primeiro escreva os testes pytest
com esqueletos sintéticos, rode e veja falhar, depois implemente features/graph.py e features/nodes.py.
Não edite nenhum arquivo protegido. Ao final rode pytest e mostre o resultado. Registre decisões em
docs/decisoes.md. Não faça commit ainda, quero revisar o diff.
```

**Etapas 3 a 5 (uma por vez, mesmo formato):**
```
Etapa 3 do PLANO.md (conexão por 3x3). TDD, só arquivos novos, cite a seção III-C nos docstrings.
Mostre uma imagem de debug (salva em debug_out/, ignorada pelo git) do grafo sobre o esqueleto de
uma imagem real de images/ para eu conferir visualmente.
```
(repetir trocando para "Etapa 4 ... esticar extremos e dividir grau 0" e "Etapa 5 ... candidatas")

**Etapas 6 e 7:**
```
Etapas 6 e 7 do PLANO.md (seção IV-A). Use testes sintéticos para L, T, X e um T falso. A verificação
por círculo deve usar as máscaras verde/branco da grade subamostrada. Raio do círculo, limiar de
curvatura e tamanho mínimo de linha devem ficar em dataclass de config.
```

**Etapa 8:**
```
Etapa 8 do PLANO.md (seção IV-B). Implemente o split-and-merge com mínimos quadrados total, descarte
componentes com menos de 3 nós, e extremos por projeção na reta ajustada. Projeção no chão fica atrás
de uma flag e só roda se houver calibração; sem ela, devolva coordenadas na imagem sem distorção.
```

**Etapa 9:**
```
Crie run_features.py (arquivo novo) com visualizador de debug para rodar nas imagens de images/.
Depois faça a verificação final: pytest, git diff --name-only base-original (só arquivos novos),
e revise o diff inteiro contra as seções III-C e IV do artigo apontando qualquer divergência.
```
