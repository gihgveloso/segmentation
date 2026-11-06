# Projeto: Line Structure-based Localization (Schulz et al., 2009)

Implementação em Python do algoritmo do artigo em `docs/paper.pdf`, sobre o repositório
Segmentation_for_robocup. Este repositório é trabalhado por duas pessoas.

## Minha parte (a única que você deve implementar)
- Seção III-C: Placement and Connection of Nodes (colocação e conexão de nós no esqueleto)
- Seção IV: Feature Extraction (IV-A cruzamentos L/T/X e IV-B linhas)
- NÃO implementar: filtro de partículas (Seção V), nem nada de MCL/observation model.
  A saída da minha parte são apenas estruturas de dados `CrossingFeature` e `LineFeature`.

## Código que NÃO pode ser alterado (parte do colega, seções III-A e III-B)
`main.py`, `test.py`, `def_white_threshold.py`, `unique_table_values.py`, `*.csv`, `images/`, `README.md`.
- Nunca editar, renomear, formatar ou "limpar" esses arquivos.
- Para reutilizar: `from main import Pixel_Segment` (o bloco `__main__` do main.py não roda no import).
- Se precisar de algo que o código existente não expõe (ex.: um resultado intermediário),
  PARE e me pergunte; resolva com wrapper/adaptador em arquivo novo, nunca editando o original.

## Onde escrever (somente arquivos novos)
```
features/
  types.py        # CrossingFeature, LineFeature (interface para o futuro filtro de partículas)
  graph.py        # Node, StructureGraph (nós, arestas, grau)
  nodes.py        # III-C: c(x,y) e colocação de nós
  connections.py  # III-C: conexão por 3x3, esticar extremos, dividir grau 0, candidatas
  simplify.py     # IV-A: suavização / remoção de nós de baixa curvatura / fusão de junções
  crossings.py    # IV-A: candidatos L/T/X + verificação (máquina de estados no círculo)
  lines.py        # IV-B: componentes de grau 2, split-and-merge, parâmetros da linha
  geometry.py     # undistort e projeção no chão (opcional, via config)
run_features.py   # visualizador de debug (não substitui o main.py)
tests/            # pytest, imagens sintéticas
docs/decisoes.md  # toda interpretação de trecho ambíguo do artigo, com justificativa
```

## Convenções
- Seguir o estilo do código existente: nomes em português quando for lógica do domínio
  (ex.: `processar`), type hints, numpy/OpenCV, sem classes desnecessárias.
- Todo trabalho acontece na grade subamostrada (a mesma que o `Pixel_Segment` produz),
  conforme o artigo; usar precisão subpixel só quando o artigo indica.
- Cada função cita a seção do artigo no docstring (ex.: "III-C: nós em picos c=0").
- Trechos ambíguos do artigo (ex.: "degree-0 nodes are split") -> registrar a interpretação
  escolhida em `docs/decisoes.md` e deixar parâmetros em dataclass de config, nunca números mágicos.
- TDD: testes pytest com esqueletos sintéticos (linha reta, L, T, X, linha com lacuna) antes do código.
- NÃO fazer commit nem push. Quem commita sou eu, só no final.
- Antes de dizer que terminou: rodar `pytest` e `git diff --name-only base-original` e confirmar
  que só aparecem arquivos novos.


# Regras do projeto (segmentation / localização por linhas)

## Proibido
- NUNCA fazer git add / commit / push / reset / checkout / stash / rebase / clean. Quem commita é a Giovanna.
- NÃO alterar main.py, test.py, def_white_threshold.py, unique_table_values.py, README.md. Se precisar de algo desses arquivos, PARE e pergunte (ver regra acima).
- NÃO instalar pacotes sem registrar no relatório.
- cv2.resize: sempre interpolation= por keyword (o flag posicional vira dst e usa bilinear).

## Invariantes
- Baseline Webots deve permanecer idêntico: GT12 / Det27 / TP6 / TPtipo3 / FP21 / FN6. Comando: python3 avaliar_cruzamentos.py
- Suíte completa verde: python -m pytest -q -rs (reportar nº de testes e quais foram SKIPPED).

## Evidência
- Todo número em doc, relatório ou commit message cita o arquivo (e linha/tabela) de onde veio. Sem fonte, é "HIPÓTESE (não verificada)".
- Medições e scripts temporários ficam em ~/medicoes_calibracao (fora do repo). Jobs longos: setsid nohup python -u ... > log 2>&1 &, com resultados parciais em disco.

## Relatório padrão (obrigatório ao fim de TODA tarefa)
Escreva RELATORIO.md em ~/medicoes_calibracao/ (fora do repo) e imprima o conteúdo no terminal, com:
1. Arquivos criados/alterados (git status --short e git diff --stat colados).
2. Nº de testes antes/depois e SKIPPED (pytest -rs).
3. Baseline Webots (saída literal).
4. MEDIDO / LIMITAÇÕES / HIPÓTESES / PERGUNTAS EM ABERTO.
5. Decisões que você tomou por conta própria (marque como tal).
6. O que precisa de olho humano.
Não resolva por conta própria ambiguidade de produto: liste como pergunta.
