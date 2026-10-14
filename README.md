# Detector de Duplicatas em Slides de Filmes Digitalizados com IA

Solucao especializada desenvolvida com Deep Learning auto-supervisionado (**Meta DINOv2 / Vision Transformer**) para identificar automaticamente fotos que correspondem ao mesmo filme/slide fotografado mais de uma vez.

---

## 💡 Como Funciona a IA

1. **Extrator Inteligente de Janela de Filme (`slide_detector.py`)**:
   - Como as fotos de slides contem molduras plasticas brancas/bege, bordas de mesa de luz e cantos do equipamento de digitalizacao, a IA nao pode simplesmente comparar a foto inteira (senao as molduras causariam falsos positivos).
   - O detector identifica automaticamente a abertura interna do filme 35mm (suportando orientacoes horizontal e vertical) com remocao da moldura e subamostragem JPEG ultrarrapida (~100ms por foto de 24MP).

2. **Modelo de Representacao Visual (`model.py`)**:
   - Baseado no **Meta DINOv2 (ViT-Small/14)**, modelo de ponta treinado por auto-supervisao sem labels.
   - Adicionada camada projetora contrastiva MLP (256-D) calibrada com **NT-Xent Loss** (Normalized Temperature-scaled Cross Entropy Loss).
   - O modelo aprende representacoes invariantes a pequenas variacoes de exposicao, temperatura de cor, pequenas rotacoes e poeira, focando exclusivamente no conteudo filmico.

3. **Conversor de Similaridade em Probabilidade Calibrada**:
   - A similaridade cosseno e mapeada para uma probabilidade realistica (0% a 100%) atraves de uma funcao logistica sigmoide com limiar calibrado.

---

## 📦 Como Usar

### 1. Treinamento do Modelo (Fine-Tuning com seus Slides)

Para treinar ou refinar a IA com suas proprias fotos:

```powershell
py -3.14 train.py --data "C:\digitalização" --max_samples 500 --batch_size 8 --epochs 100
```

Parametros principais:
- `--data`: Caminho da pasta contendo as fotos dos slides.
- `--epochs`: Numero de epocas de treinamento (padrao: 5).
- `--batch_size`: Tamanho do lote (padrao: 8).
- `--max_samples`: Quantidade maxima de amostras para o treino (ou deixe sem para treinar com todas).
- `--output`: Nome do arquivo para salvar os pesos (padrao: `slide_duplicate_model.pt`).

---

### 2. Identificar Duplicatas em uma Pasta Inteira

Para analisar uma pasta de fotos e listar todos os pares duplicados ordenados do mais provavel para o menos provavel:

```powershell
python detect_duplicates.py "E:\backup\digitalização" --min_prob 0.50 --top_k 20
```

Parametros uteis:
- `--min_prob`: Probabilidade minima de corte (ex: `0.50` para 50%, `0.70` para 70%).
- `--max_images`: Limitar o numero de imagens analisadas (ideal para testes rapidos, ex: `--max_images 50`).
- `--top_k`: Numero maximo de pares a exibir (padrao: 30).
- `--recursive`: Adicione se quiser pesquisar em subpastas tambem.

---

### 3. Comparar um Arquivo Especifico contra uma Pasta

Se voce tem uma foto especifica e quer saber se ela ja foi tirada ou existe duplicada em uma pasta:

```powershell
python detect_duplicates.py "E:\backup\digitalização\IMG_0534.JPG" --against "E:\backup\digitalização" --top_k 10
```

O sistema retornara um ranking ordenado das fotos mais provaveis de serem o mesmo filme daquela foto especifica.
