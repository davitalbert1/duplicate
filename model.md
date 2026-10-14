# Sistema de Detecção de Slides Fotográficos Duplicados

## 1. Pesquisa e Arquitetura Adotada

Para este tipo de imagem — slides de filmes fotográficos digitalizados por câmera fotográfica ou mesa de luz — foram identificados dois desafios principais.

### 1.1. Desafios Identificados

**Presença de molduras e suportes de digitalização**

As fotos contêm molduras de plástico ou papelão brancas ou bege, além de cantos da mesa de luz. Comparar as imagens brutas faria a IA focar nas molduras plásticas em vez do filme fotográfico.

**Variações ópticas e fotográficas**

Fotos tiradas em momentos diferentes do mesmo slide podem apresentar:

- Variações de alinhamento.
- Ligeiras rotações.
- Diferenças de temperatura de cor e balanço de branco.
- Ruído fotográfico.
- Pequenas variações de foco.

### 1.2. Modelo Escolhido

**Meta DINOv2 (Vision Transformer — ViT-Small/14)**

Modelo público de ponta, treinado por auto-supervisão, sem necessidade de anotações humanas. Ele aprendeu representações semânticas densas e robustas sobre imagens do mundo real.

**Cabeça de projeção contrastiva MLP (256-D) + NT-Xent Loss**

Uma camada de projeção MLP de 256 dimensões, combinada com a função de perda NT-Xent, inspirada nas abordagens SimCLR/DINO, especializa os vetores do DINOv2 no espaço de similaridade para slides fotográficos.

### 1.3. Pipeline de Pré-processamento

O módulo `slide_detector.py` implementa as seguintes funcionalidades:

- Detecta automaticamente a abertura interna da película de 35 mm.
- Suporta orientações horizontal e vertical.
- Descarta a moldura do slide.
- Utiliza subamostragem por meio de decodificação JPEG no modo `draft`.
- Otimiza a velocidade de processamento, com tempo estimado de aproximadamente 100 ms por foto de 24 MP.

### 1.4. Função de Probabilidade Sigmoide Calibrada

A similaridade cosseno entre os vetores neurais é convertida em probabilidades calibradas de 0% a 100%.

Essa probabilidade representa a estimativa do modelo sobre a possibilidade de duas imagens corresponderem ao mesmo slide fotográfico.

> **Observação:** a qualidade da calibração depende dos dados utilizados e do método de ajuste. Uma probabilidade de 90%, por exemplo, não deve ser interpretada automaticamente como uma garantia de 90% de chance de duplicação sem validação empírica.

---

## 2. Estrutura dos Arquivos Criados

O projeto está organizado nos seguintes arquivos:

| Arquivo | Descrição |
|---|---|
| `slide_detector.py` | Módulo de visão computacional que localiza o filme dentro do slide, descarta a moldura plástica e aplica subamostragem JPEG otimizada. |
| `model.py` | Define a rede neural profunda `SlideDuplicateModel`, utilizando DINOv2 e a função de perda contrastiva auto-supervisionada `NTXentLoss`. |
| `dataset.py` | Gera dados de treinamento com simulações realistas de novas digitalizações, incluindo pequenos deslocamentos, rotações e variações de exposição e contraste. |
| `train.py` | Executa o treinamento do modelo e salva o checkpoint `slide_duplicate_model.pt`. |
| `detect_duplicates.py` | Script de inferência que permite analisar pastas inteiras ou arquivos individuais. |
| `README.md` | Manual prático de utilização do projeto. |

---

## 3. Resumo

O sistema combina detecção da área útil do slide, extração de características visuais com DINOv2 e aprendizado contrastivo para identificar possíveis duplicatas de filmes fotográficos digitalizados.

A arquitetura foi projetada para lidar com diferenças de enquadramento, orientação, exposição, contraste e foco, reduzindo a influência das molduras externas.

O objetivo final é facilitar a organização de grandes coleções de slides fotográficos, automatizando a identificação de imagens potencialmente duplicadas e permitindo a revisão dos resultados pelo usuário.
