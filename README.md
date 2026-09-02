# Segmentação de Bancos de Areia no Rio Tapajós

Pipeline de segmentação semântica binária para identificar bancos de areia em imagens multiespectrais Sentinel-2, usando uma U-Net com encoder ResNet (transfer learning), busca de hiperparâmetros com Optuna e validação cruzada aninhada.

Projeto desenvolvido por **Gabriel Alves** e **Giann Kenyd** — Instituto Nacional de Pesquisas Espaciais (INPE), 2026.

---

## Sumário

- [Problema](#problema)
- [Dados](#dados)
- [Pipeline do método](#pipeline-do-método)
- [Arquitetura do modelo](#arquitetura-do-modelo)
- [Treinamento e validação](#treinamento-e-validação)
- [Métricas](#métricas)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Checkpoints](#checkpoints)
- [Limitações](#limitações)
- [Próximos passos](#próximos-passos)
- [Referências](#referências)

---

## Problema

Bancos de areia no Rio Tapajós mudam de posição e tamanho ao longo do ano, e monitorá-los manualmente é lento. O objetivo é identificar automaticamente bancos de areia em imagens multiespectrais, tratando o problema como uma **segmentação binária por pixel**: banco de areia ou fundo.

Principais desafios:
- Bancos de areia ocupam poucos pixels por cena (classe rara e pequena).
- Margens, ilhas, solo exposto e nuvens têm aparência parecida com areia.
- O objetivo prático é reduzir falsos positivos sem deixar de detectar bancos pequenos.

## Dados

- **Fonte:** recortes Sentinel-2 com 6 bandas — B2, B3, B4 (visível), B8 (NIR), B11 e B12 (SWIR).
- **Índices espectrais:** NDWI e MNDWI são calculados dentro do próprio `Dataset` (PyTorch) e concatenados como canais extras, totalizando **8 canais de entrada**.

  ```
  NDWI  = (B3 - B8)  / (B3 + B8)
  MNDWI = (B3 - B11) / (B3 + B11)
  ```

- **Ground truth:** máscaras binárias (0 = fundo, 255 = banco de areia), construídas por triagem manual → pré-máscara por índices espectrais → correção manual no QGIS → rasterização final.
- **Dataset final:** 200 cenas (de ~2.396 avaliadas na triagem) — 50 positivas e 150 negativas (proporção 1:3). As negativas incluem água sem banco, margem clara, ilha vegetada, nuvem e solo exposto, para ensinar o modelo a rejeitar alvos parecidos.

## Pipeline do método

```
Imagem Sentinel-2 → Bandas + NDWI/MNDWI (8 canais) → U-Net (encoder ResNet)
  → Mapa de probabilidade (sigmoid) → Threshold → Máscara binária final
```

Antes do treino, as cenas são convertidas em patches de tamanho fixo (**144×144**) via padding/crop. A separação treino/teste é sempre feita **por cena inteira, antes** da divisão em patches, para evitar vazamento de dados entre splits.

## Arquitetura do modelo

- **U-Net** para segmentação binária: encoder que resume padrões espectrais/espaciais, decoder que reconstrói a máscara pixel a pixel, e **skip connections** que preservam bordas e objetos pequenos, essenciais para um alvo tão pequeno e irregular quanto o banco de areia.
- **Encoder:** ResNet (`resnet18` ou `resnet34`, escolhido pela busca de hiperparâmetros) pré-treinada em ImageNet (*transfer learning*). A primeira camada convolucional foi adaptada para aceitar 8 canais em vez de 3. O encoder aprende com uma taxa de aprendizado menor que o restante da rede.
- **Saída:** 1 canal (logit), com `sigmoid` aplicado apenas na inferência.
- **Pré-processamento:** tratamento de NaN/Inf (nodata), normalização das bandas, padding/crop para 144×144.
- **Função de perda:** combinação de **Focal Loss** + **Dice Loss** (pesos buscados via Optuna).

## Treinamento e validação

- **Validação cruzada aninhada** com **5 folds externos**:
  - Cada fold externo estima o desempenho fora da amostra.
  - Em cada fold, um split interno é usado exclusivamente para a busca de hiperparâmetros e escolha do threshold.
  - O teste externo de cada fold nunca influencia a escolha de hiperparâmetros/threshold, só é usado para medir o desempenho daquele fold.
- **Busca de hiperparâmetros (Optuna):** amostrador TPE + pruning por mediana, 15 tentativas por fold externo, 12 épocas por tentativa. Espaço de busca:

  | Hiperparâmetro | Faixa | Escala |
  |---|---|---|
  | Taxa de aprendizado | 5e-5 – 2e-3 | log |
  | Fator de lr do encoder | 0,05 – 0,5 | log |
  | Weight decay | 1e-6 – 5e-4 | log |
  | Encoder | resnet18 / resnet34 | – |
  | Batch size | 4 / 8 / 16 | – |
  | Peso da Dice na loss | 0,3 – 1,5 | linear |
  | α (Focal Loss) | 0,25 – 0,9 | linear |
  | γ (Focal Loss) | 0,5 – 3,0 | linear |

- **Modelo final:** após a validação, um único modelo é treinado com **todas as 200 cenas**, usando os hiperparâmetros do fold externo com melhor IoU/Dice — esse é o modelo salvo para uso posterior (inferência), distinto dos 5 checkpoints por fold, que servem só para reproduzir a validação.

## Métricas

Avaliadas de forma **out-of-fold** (agregando TP/FP/FN de todos os pixels de todos os folds de teste, não a média simples entre folds):

- **Dice** = 2·TP / (2·TP + FP + FN) — sobreposição entre máscara prevista e real. Equivalente ao F1-score em segmentação binária por pixel.
- **Precision** = TP / (TP + FP) — dos pixels previstos como banco, quantos eram de fato.
- **Recall** = TP / (TP + FN) — dos bancos de areia reais, quantos foram detectados.

**Por que Dice em vez de IoU:** bancos de areia ocupam poucos pixels por cena, e o IoU penaliza fortemente qualquer erro pequeno de borda nesse cenário (denominador pequeno). O Dice é matematicamente mais tolerante (Dice = 2·IoU / (1+IoU)), por isso é a métrica mais usada para avaliar segmentação de objetos pequenos.


## Checkpoints

| Arquivo | Conteúdo | Uso |
|---|---|---|
| `unet_outer_fold{n}.pt` | pesos, época, IoU e hiperparâmetros daquele fold | reproduzir/auditar a validação cruzada |
| `unet_modelo_final.pt` | pesos, hiperparâmetros, threshold e desempenho estimado pela CV | inferência / uso em produção |

## Limitações

- Apenas 50 cenas positivas — dataset positivo ainda pequeno.
- Bancos de areia ocupam poucos pixels; algumas máscaras são muito pequenas.
- Thresholds de decisão variaram entre folds (sinal de instabilidade ligado ao tamanho do dataset).
- O tiling não aumentou o dataset quando a cena já era menor que 144×144.
- A busca do Optuna ainda é sensível ao conjunto pequeno de dados.

## Próximos passos

- Aumentar o dataset incorporando os rios Araguaia e Madeira (período de outubro).
- Comparar entradas RGB, 6 bandas e 8 canais (com índices) para avaliar se mais canais realmente ajudam.
- Investigar formas de reduzir falsos positivos (ex.: mais exemplos negativos difíceis, ajuste do α da Focal Loss).

## Referências

- Ronneberger, O.; Fischer, P.; Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015.
- Lin, T.-Y. et al. *Focal Loss for Dense Object Detection*. ICCV, 2017.
- Milletari, F.; Navab, N.; Ahmadi, S.-A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV, 2016.
- ESA. *Sentinel-2 User Handbook*. European Space Agency.

---

## Autores

- Gabriel Alves
- Giann Kenyd

Instituto Nacional de Pesquisas Espaciais (INPE) — 2026.