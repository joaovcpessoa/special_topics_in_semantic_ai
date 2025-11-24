# Saliency Maps

## Visão Geral

Em redes profundas (como CNNs grandes, ResNets etc.), nem todas as amostras precisam passar por todas as camadas.
- As amostras “fáceis” (ex: imagem muito nítida de um gato) podem ser classificadas com segurança em camadas intermediárias.
- As “difíceis” (ex: gato deitado no escuro) continuam até o final.

Assim, a rede aprende a interromper o processamento cedo se a confiança for suficiente, economizando tempo e energia.
A partir daí temos um problema entre confiança ser diferente de certeza. A saída “softmax” nem sempre é um bom indicador de confiança, pois os modelos podem estar "overconfident".

Aí que entrariam os mapas de saliência, podemos usar eles como métrica de “atenção confiável”

## O que são Mapas de Saliência?

Um **mapa de saliência** é uma representação visual que destaca as regiões mais importantes ou relevantes de uma imagem para um modelo de aprendizado de máquina. Em essência, o objetivo de um mapa de saliência é refletir o grau de importância de um pixel para o modelo.
São ferramentas proeminentes em XAI (Inteligência Artificial Explicável), fornecendo explicações visuais do processo de tomada de decisão de modelos de aprendizado de máquina, especialmente redes neurais profundas. Eles destacam as regiões na entrada (imagens, texto, etc.) que são mais influentes na saída do modelo, indicando onde o modelo está "olhando" ao fazer uma previsão. Se a rede realmente “olha” para a região correta da imagem (segundo o mapa de saliência), então ela está pronta para sair cedo (early exit), ou seja, podemos definir um critério de ativação antecipada baseado em coerência espacial.

Pode fazer isso de algumas maneiras:

1. Medida de concentração espacial
- Você pode medir quão focado é o mapa de saliência:
- Calcule a entropia espacial (quanto mais difuso, menos confiança).
- Calcule o momento de inércia do mapa (se há uma única região dominante ou várias).

2. Medida de coerência intercamadas
- Gere o mapa de saliência em camadas intermediárias.
- Compare-o com o mapa final (usando cosine similarity ou SSIM).
- Se o mapa intermediário já está “olhando para o mesmo lugar” que o final, isso indica que a decisão já amadureceu — pode sair.

## Algoritmos

Existem diversas abordagens para a criação de mapas de saliência.

*   **Saliência Estática**: Baseia-se em características e estatísticas da imagem para localizar regiões de interesse.
*   **Saliência de Movimento**: Utiliza o movimento em um vídeo, detectado por fluxo óptico, onde objetos em movimento são considerados salientes.
*   **Objectness**: Reflete a probabilidade de uma janela de imagem cobrir um objeto, gerando caixas delimitadoras onde um objeto pode estar.
*   **TASED-Net**: Uma rede de codificador-decodificador que extrai características espaço-temporais de baixa resolução e as decodifica para gerar o mapa de saliência.
*   **STRA-Net**: Integra características espaço-temporais via acoplamento de aparência e fluxo óptico, aprendendo saliência multi-escala através de mecanismos de atenção.
*   **STAViS**: Combina informações visuais e auditivas espaço-temporais para localizar fontes sonoras e fundir as saliências.

As Redes Neurais Profundas (DNNs) convencionais geralmente possuem um único ponto de saída, realizando previsões após processar todas as camadas. No entanto, nem todas as entradas exigem a mesma quantidade de computação para alcançar uma previsão confiante. **Early Exit** (ou saída antecipada) é uma técnica que incorpora múltiplos "pontos de saída" em uma arquitetura de DNN, permitindo que a inferência seja interrompida precocemente em pontos intermediários.

Em uma DNN com Early Exit, ramificações laterais são adicionadas em diferentes profundidades da rede principal. Cada ramificação lateral possui um classificador que pode fazer uma previsão. Durante a inferência, a rede avalia a confiança da previsão em cada ponto de saída. Se a confiança atingir um determinado limiar, a inferência é interrompida e a previsão é aceita, sem a necessidade de processar as camadas subsequentes. Para entradas mais complexas, a inferência continua através de mais camadas da rede principal até que um ponto de saída atinja o limiar de confiança ou até que a saída final da rede seja alcançada.

### Extra

Isso era mentira em 1989 e continua sendo hoje em 2025.Os modelos que temos hoje como resnets, llms etc. Contém centenas de camadas, mas o mesmo tempo não temos nenhuma visão prática sobre o que realmente está sendo feito dentro dessas camadas de uma maneira semanticamente útil.

Já conhecemos o tipo de computação que acontece nessas camadas, mas não sabemos quais delas são necessárias e o que elas realmente estão fazendo com o problema.

Para tomadas de decisão, muitas pessoas gostariam de possuir algum grau de interpretabilidade, que podemos simplificar em “olhar por dentro da caixa-preta” do modelo para entender por que ele faz o que faz. Instrospeção dos modelos.

Isso faz com que dentro de processos de tomada de decisão ainda precisem de um capital humano (discutível).

A pergunta é: Esse tipo de instrospeção é possível em Redes neurais profundas?

Se é possível, quais propriedades elas precisam satisfazer
para que sejam interpretáveis?

É claro que se o modelo for arbitratiamente complicado, não há como o ser humano interpretá-lo, então somente se tivermos certas suposições estruturais sobre o modelo podemos falar sobre interpretá-lo.

Um modelo pode ser matematicamente preciso e estatisticamente ótimo, mas tão complexo e não linear que não conseguimos compreender o raciocínio interno dele de forma intuitiva.

Uma rede neural profunda com bilhões de parâmetros (como GPT-5) não é algo que um ser humano possa entender diretamente olhando para pesos e equações. Mesmo que tivéssemos todos os parâmetros do modelo, a estrutura é tão intrincada que é impossível traduzir “por que” ele tomou uma decisão específica em linguagem humana.

Isso significa que, para entender um modelo, precisamos impor restrições ou estruturas conhecidas que tornem sua lógica mais transparente. Essas suposições estruturais podem ser de vários tipos:

Tipo de Suposição
- Linearidade	-> Assume que a relação entre variáveis é linear
- Sparsidade (esparsidade) -> Assume que poucas variáveis são relevantes
- Hierarquia de decisão -> Assume decisões em etapas (if/else)
- Modularidade -> Divide o modelo em blocos compreensíveis	Redes neurais explicáveis por camadas/funções
- Atenção interpretável	-> Impõe que o modelo “explique” o que prioriza

A maneira mais comum de visualizar saliências para uma rede neural, em especial para CNNs, é usar a saliência de gradiente de entrada, onde a importância é essencialmente codificada pela sensibilidade dos pixels.

Se f é a rede neural de valor escalar, ela mapeia a entrada vetorial x para um valor escalar y. Então o gradiente da entrada em relação a saída é o que iremos definir como mapa de saliência.

Podemos visualizar a magnitude dos gradientes que destaca as partes importantes da imagem

Um dos principais problemas de utilizar gradientes para gerar mapas de saliência é que eles dependem das variações locais da função de saída em relação à entrada. Isso significa que regiões planas ou uniformes da imagem, onde a função do modelo muda pouco ou nada , recebem pouca ou nenhuma atribuição, mesmo que sejam semanticamente importantes para a decisão do modelo. Em outras palavras, os gradientes podem ignorar características relevantes simplesmente porque a função é localmente constante nessa região.

Atribuição de importância em métodos de gradiente mede quanto a saída do modelo muda se você alterar um pixel.
Se a saída muda muito ao mexer em um pixel, o modelo considera esse pixel importante.
Se a saída muda pouco ou nada, o pixel é considerado irrelevante, mesmo que faça parte de uma característica visual significativa.
Em outras palavras, a importância não é dada pelo conteúdo semântico da imagem, mas pelo grau de sensibilidade da função do modelo à entrada.

Isso é o que gradientes capturam

Recomendação futura: Fazer um modelo com várias saídas e ver se o mapa de saliência difere

## Resultado do primeiro experimento (CNN_binary.ipynb)

O modelo está aprendendo efetivamente, com redução consistente da perda e aumento da acurácia, finalizando com ~85-90%, o que indica um desempenho razoável. Não há sinais de "underfitting", pois o modelo consegue aprender os padrões dos dados.
O gap crescente entre as perdas de treinamento e validação (especialmente visível após a época 60) sugere que o modelo pode estar começando a memorizar os dados de treinamento em vez de generalizar. As oscilações na curva vermelha de acurácia sugerem que o conjunto de validação pode ser pequeno ou que o modelo é sensível a variações nos dados.

Nesse contexto, foi possível avaliar os diferentes algoritmos baseados em gradiente e esboçar o que de fato vem a ser um mapa de saliência.

## Resultado do segundo experimento (CNN_early_exit.ipynb)

Aqui o modelo performou de maneira mais limitada, mas no final, nosso objetivo era ver como o mapa de saliência mudava conforme avançava nas camadas e isso foi um sucesso.

## Resultado do terceiro experimento (CNN_early_exit.ipynb)

### U-Net

Enquanto CNNs são tipicamente usadas para tarefas de classificação (onde a saída é um único rótulo de classe para uma imagem inteira), em problemas de segmentação biomédica o objetivo é o que chamam de **localização**, que consiste em atribuir um rótulo de classe a cada pixel.

Antes do U-Net, uma abordagem comum era usar uma rede de janela deslizante (sliding-window setup) para prever o rótulo de cada pixel fornecendo uma região local (patch) ao redor dele. Embora eficaz, essa estratégia tinha duas desvantagens principais:
- Lentidão: A rede precisava ser executada separadamente para cada patch, com muita redundância.
- Trade-off: Havia um compromisso entre a precisão da localização e o uso do contexto. Patches maiores exigiam mais camadas de *max-pooling*, reduzindo a precisão da localização.

O U-Net se baseia na arquitetura "fully convolutional network", modificada e estendida para funcionar com muito poucas imagens de treinamento e produzir segmentações mais precisas. (Talvez sirva para nosso artigo?)

### Arquitetura da Rede

O nome "U-Net" deriva de sua forma em U, simétrica. A arquitetura, ilustrada na Figura 1 do artigo, é composta por dois caminhos principais:

1.  **Caminho Contrativo (Contracting Path):** Localizado no lado esquerdo da arquitetura, este caminho tem como objetivo **capturar o contexto**.
    *   Segue a arquitetura típica de uma CNN: aplicação repetida de duas convoluções $3\times3$ (não acolchoadas, ou *unpadded*), cada uma seguida por uma unidade linear retificada (**ReLU**), e uma operação de *max pooling* $2\times2$ (com passo 2) para subamostragem (*downsampling*).
    *   A cada etapa de subamostragem, o número de canais de *feature maps* é dobrado.

2.  **Caminho Expansivo (Expansive Path):** Localizado no lado direito, este caminho é simétrico ao contratante e permite a **localização precisa**.
    *   Cada etapa consiste em uma superamostragem (*upsampling*) do *feature map*, seguida por uma convolução $2\times2$ ("up-convolution") que reduz pela metade o número de canais de *features*.
    *   Em seguida, ocorre uma **concatenação** com o *feature map* correspondente (recortado) do caminho contrativo. Essa etapa combina as informações de alta resolução do caminho contrativo com as informações de contexto propagadas (já que o caminho expansivo também possui um grande número de canais de *features*).
    *   Finaliza com duas convoluções $3\times3$, cada uma seguida por uma ReLU.
    *   O recorte (*cropping*) é necessário devido à perda de pixels de borda em cada convolução.

A rede U-Net não possui camadas totalmente conectadas (*fully connected layers*). Na camada final, uma convolução $1\times1$ é usada para mapear cada vetor de *features* de 64 componentes para o número desejado de classes. A rede possui um total de **23 camadas convolucionais**.

### Estratégia de Treinamento

A rede foi treinada usando a implementação de gradiente descendente estocástico do Caffe.

#### 1. Aumento de Dados (Data Augmentation)

O **aumento excessivo de dados** é crucial para o sucesso da U-Net, pois permite que a rede use as poucas amostras anotadas de forma mais eficiente.
*   É essencial para ensinar a rede a **invariância à rotação e ao deslocamento** e robustez a variações de valor de cinza e, principalmente, a **deformações elásticas**.
*   As deformações elásticas aleatórias dos dados de treinamento são o conceito-chave para treinar a rede com pouquíssimas imagens anotadas, pois simulam as variações mais comuns em tecidos biológicos.

#### 2. Função de Perda Ponderada (Weighted Loss)

Um desafio em muitas tarefas de segmentação celular é a **separação de objetos tocantes** da mesma classe. Para resolver isso, os autores propuseram o uso de uma função de perda ponderada:
*   A função de energia (loss function) é calculada usando um *soft-max* pixel a pixel sobre o *feature map* final combinado com a função de perda de entropia cruzada.
*   É introduzido um **mapa de peso** $w(x)$ para dar mais importância a certos pixels durante o treinamento.
*   O mapa de peso é pré-calculado para:
    *   Compensar a diferente frequência de pixels de uma determinada classe.
    *   Forçar a rede a aprender as pequenas **bordas de separação** introduzidas entre as células tocantes. As labels de fundo (background) que separam células tocantes recebem um grande peso na função de perda.

#### 3. Estratégia de Sobreposição de Tiles (Overlap-tile strategy)

O U-Net usa apenas a parte válida de cada convolução, de modo que o mapa de segmentação de saída contém apenas pixels para os quais o contexto completo estava disponível na imagem de entrada. Essa estratégia permite a **segmentação contínua de imagens arbitrariamente grandes**.
*   Para prever os pixels na região da borda da imagem, o contexto ausente é extrapolado pelo **espelhamento da imagem de entrada**.
*   Esta estratégia de ladrilhamento (*tiling*) é importante para aplicar a rede a imagens grandes, pois o contrário limitaria a resolução pela memória da GPU.

### Bases de Dados (Datasets) e Resultados

Os experimentos demonstraram a aplicação do U-Net em três tarefas distintas de segmentação biomédica.

#### 1. Segmentação de Estruturas Neurais (EM Stacks)

*   **Desafio:** O Desafio de Segmentação EM (EM segmentation challenge), iniciado no ISBI 2012.
*   **Base de Dados:** Consiste em 30 imagens ($512\times512$ pixels) de microscopia eletrônica de transmissão por seção serial (EM) do cordão nervoso ventral (VNC) da larva de primeiro instar de *Drosophila*.
*   **Anotações:** Cada imagem possui um mapa de segmentação de verdade fundamental (*ground truth*) totalmente anotado para **células (branco) e membranas (preto)**.
*   **Avaliação:** Usa o "erro de *warping*", o "erro Rand" e o "erro de pixel".
*   **Resultado do U-Net:** O U-Net superou o método anterior (rede de janela deslizante de Ciresan et al.). Alcançou um erro de *warping* de 0.0003529 (o novo melhor resultado) e um erro Rand de 0.0382.

#### 2. Segmentação Celular em Imagens de Microscopia de Luz (ISBI Cell Tracking Challenge 2015)

O U-Net também foi aplicado a esta tarefa de segmentação celular, que faz parte do desafio ISBI Cell Tracking Challenge.

**Dataset 1: "PhC-U373"**
*   **Descrição:** Células de Glioblastoma-astrocitoma U373 registradas por microscopia de contraste de fase (Phase Contrast Microscopy).
*   **Anotações:** 35 imagens de treinamento parcialmente anotadas.
*   **Resultado do U-Net:** Atingiu uma IOU (*Intersection over Union*) média de **92%**, sendo significativamente melhor do que o segundo melhor algoritmo (83%).

**Dataset 2: "DIC-HeLa"**
*   **Descrição:** Células HeLa em vidro plano registradas por microscopia de contraste de interferência diferencial (DIC - Differential Interference Contrast).
*   **Anotações:** 20 imagens de treinamento parcialmente anotadas.
*   **Resultado do U-Net:** Atingiu uma IOU média de **77.5%**, sendo significativamente melhor do que o segundo melhor algoritmo (46%).

Em essência, a U-Net funciona como um filtro complexo e inteligente que aprende a "desenhar" o contorno das estruturas biológicas. Se o treinamento padrão é como aprender a reconhecer um objeto em uma foto (classificação), a U-Net é como aprender a traçar o contorno exato de cada objeto nessa foto (segmentação), mesmo quando os objetos estão encostados, usando a estratégia de aumento de dados como um "treinamento de elasticidade" para garantir que a rede reconheça as células, independentemente de estarem espremidas ou deformadas.



### Dataset Oxford-IIIT Pet

The dataset contains photographs of cats and dogs across 37 categories, with approximately 200 images per class. The images exhibit significant variation in scale, pose, and lighting conditions. Each image is accompanied by ground-truth annotations, including the breed, head region of interest (ROI), and pixel-level trimap segmentation. This dataset is widely used for <b>image classification</b>, as well as for <b>segmentation</b> and <b>object detection</b> tasks.

| **Task**                 | **Question it answers**                       | **Output**              | **Example use case**                    |
| ------------------------ | --------------------------------------------- | ----------------------- | --------------------------------------- |
| **Image Classification** | What is in the image?                         | One or more labels      | “It’s a Siamese cat.”                   |
| **Object Detection**     | What is in the image and where is it located? | Bounding boxes + labels | Detecting multiple animals in one photo |
| **Image Segmentation**   | What is in each pixel?                        | Pixel-wise mask         | Separating the cat from the background  |

Link: https://www.robots.ox.ac.uk/~vgg/data/pets/

![image.png](attachment:image.png)

##### Pixel-wise mask

A pixel mask is an auxiliary image used to represent which parts of the original image belong to a specific object or class. It is fundamental in image segmentation tasks.
A pixel mask has the same size as the original image (same width and height), but instead of containing real colors (RGB), each pixel stores a numerical value indicating the class that pixel belongs to.

| Pixel | Mask Value       | Meaning                                      |
| ----- | ---------------- | -------------------------------------------- |
| 0     | Background       | Irrelevant area                              |
| 1     | Cat              | Part of the animal                           |
| 2     | Uncertain border | Transition region between background and cat |

Practical Example -> Imagine the following image:
- Original image: A white cat sitting on a sofa.
- Pixel mask: Pixels forming the cat → value 1

Pixels from the background (sofa, wall) → value 0
When visualized, this mask appears as a grayscale or artificially colored image, highlighting only the shape of the object.

Masks allow the model to learn the exact shape and precise boundaries of an object.
Unlike detection (which uses bounding boxes), segmentation shows which pixels belong to the object.

| Type           | Description                                         | Example of use                         |
| -------------- | --------------------------------------------------- | -------------------------------------- |
| **Binary**     | Contains only 0 (background) and 1 (object)         | Segmenting the cat from the background |
| **Multiclass** | Each number represents a different class            | Separating cat, dog, and background    |
| **Trimap**     | Three levels (background, object, uncertain border) | Used in the *Oxford-IIIT Pet* dataset  |

---

# Experimento U-NET

Como acabei não conseguindo explicar as coisas corretamente durante a aula, tentei me redimir escrevendo este texto com o máximo de profundidade e detalhamento que consegui. Peço desculpas pelo uso do português.

## Artigo Original

O artigo apresenta uma nova arquitetura de rede neural e uma estratégia de treinamento desenvolvidas especificamente para a segmentação de imagens biomédicas. A motivação para esse trabalho surgiu da seguinte observação: o treinamento bem-sucedido de redes neurais profundas geralmente requer milhares de amostras de treinamento anotadas. Contudo, em tarefas de processamento de imagens biomédicas, a obtenção de um grande número de imagens devidamente anotadas é tipicamente difícil. Além disso, essas tarefas demandam localização precisa, ou seja, a atribuição de um rótulo de classe a cada pixel, em vez de uma única classificação para toda a imagem.

Entre os trabalhos de referência que inspiraram o estudo, destaca-se o de Ciresan et al., que utilizou uma abordagem baseada em janelas deslizantes (sliding windows) para prever o rótulo de cada pixel a partir de patches locais. Embora eficaz, esse método apresentava limitações, sobretudo em termos de velocidade e na conciliação entre a precisão da localização e o uso de contexto.

Com base nessas limitações, os autores desenvolveram uma nova arquitetura, denominada U-Net, que se baseia e expande o conceito de redes totalmente convolucionais (fully convolutional networks). Seu nome deriva da característica forma de U da arquitetura, que reflete a simetria entre as fases de contração e expansão do modelo.

![image](./image.png)

Vou tentar explicar a estrutura geral da rede

Como mencionei antes, a rede tem esse nome por causa do formato em “U” do seu fluxo de dados:
- Lado esquerdo (descendente): caminho de contração (encoder).
- Lado direito (ascendente): caminho de expansão (decoder).
- Centro (parte inferior): o gargalo (bottleneck), que conecta as duas fases.

##### Encoder

- Cada bloco no lado esquerdo aplica duas convoluções 3×3 seguidas de função de ativação ReLU (parte azul escuro).
- Depois, aplica-se uma operação de max pooling 2×2 (setas vermelhas), que reduz as dimensões espaciais da imagem pela metade, mas duplica o número de canais (ou filtros).
- Isso permite que a rede aprenda características mais abstratas e de maior contexto, mas com menor resolução espacial.

Na imagem temos um exemplo (no topo à esquerda):

- Entrada: 572×572×1
- Após convoluções: 570×570×64, depois 568×568×64
- Após pooling: 284×284×128 (metade da resolução, dobro de filtros)

##### Bottleneck

No fundo do “U”, a imagem tem dimensões muito reduzidas (28×28), mas muitos canais (1024). Aqui a rede aprende representações altamente abstratas, capturando características globais da imagem.

##### Decoder

- Cada etapa realiza uma up-convolução (ou transposed convolution) 2×2 (setas verdes), que aumenta a resolução da imagem pela metade e reduz o número de canais.
- Em seguida, há uma concatenação (copy and crop) com a saída correspondente do encoder (setas cinza). Isso recupera informações espaciais finas perdidas na compressão.
- Depois, duas convoluções 3×3 + ReLU novamente refinam o mapa de características.

##### Camada de Saída

Por fim, há uma convolução 1×1 (seta verde-clara) que reduz os canais para o número de classes desejado (neste caso, 2 → fundo e objeto). O resultado é o mapa de segmentação (output segmentation map), onde cada pixel recebe uma probabilidade de pertencer a cada classe.

##### Resumo

O encoder extrai recursos (o “o que” da imagem), o decoder reconstrói a forma espacial (o “onde”) e as conexões em espelho preservam detalhes finos e bordas.

### Treinamento

A estratégia de treinamento foi concebida para usar poucas amostras anotadas de forma eficiente, provavelmente pela falta de dados para o problema, assim como no nosso desafio do ICASSP. Então no projeto do artigo, foram utilizadas as seguintes considerações:

- Data Augmentation: Usado excessivamente e considerado crucial devido à escassez de dados de treinamento. O método principal foi aplicar deformações elásticas aleatórias às imagens. Isso permite que a rede aprenda a invariança a tais deformações, que são a variação mais comum em tecidos biomédicos.
- Função de Perda Ponderada: Para resolver o desafio de separar objetos da mesma classe que se tocam (como células), os autores introduziram uma função de perda ponderada por pixel. Os rótulos de fundo que separam células em contato recebem um grande peso na função de perda para forçar a rede a aprender as pequenas bordas de separação. O peso ($w(x)$) é calculado com base nas distâncias à borda da célula mais próxima ($d_1(x)$) e da segunda célula mais próxima ($d_2(x)$).
- Otimização: O treinamento utiliza o gradiente descendente estocástico (SGD), implementado no Caffe. É utilizada uma alta taxa de momentum (0.99) e se favorece o uso de grandes blocos de entrada (*input tiles*) em vez de um grande tamanho de lote (*batch size*), que é reduzido a uma única imagem.

**Observação:** Caffe (Convolutional Architecture for Fast Feature Embedding) é um framework de aprendizado profundo desenvolvido por Yangqing Jia no Berkeley Vision and Learning Center (BVLC), da Universidade da Califórnia, em Berkeley. Foi um dos primeiros frameworks populares de deep learning (lançado em 2014), antes do TensorFlow ou PyTorch, e ficou conhecido pela eficiência e velocidade no treinamento de redes convolucionais (CNNs), especialmente para visão computacional. O Caffe praticamente caiu em desuso por frameworks modernos como TensorFlow (Google), PyTorch (Meta/Facebook) e Keras, mas pelo que vi, na época era o padrão ouro para redes convolucionais.

### Resultados e Desempenho

A U-Net demonstrou desempenho superior em várias tarefas de segmentação biomédica:

*   **Segmentação de Estruturas Neuronais (EM Stacks):** A U-Net superou o método anterior de janela deslizante. Ela alcançou um erro de *warping* de 0.0003529 (o novo melhor resultado na época, em março de 2015) e um erro *Rand* de 0.0382 no desafio ISBI para segmentação EM.
*   **Desafio de Rastreamento de Células ISBI 2015 (Microscopia de Luz):** A rede venceu por uma grande margem em datasets de luz transmitida (contraste de fase e DIC).
    *   No dataset **"PhC-U373"**, alcançou IOU (Interseção sobre União) de 92%, superando significativamente o segundo melhor algoritmo (83%).
    *   No dataset **"DIC-HeLa"**, alcançou IOU de 77.5%, também superando significativamente o segundo melhor (46%).
*   **Velocidade:** A rede é rápida; a segmentação de uma imagem 512x512 leva **menos de um segundo** em uma GPU recente.

Como conclusão, a arquitetura U-Net ofereceu excelente desempenho em diversas aplicações biomédicas, exigindo **muito poucas imagens anotadas** graças à robustez da sua estratégia de aumento de dados com deformações elásticas. O tempo de treinamento é razoável (apenas 10 horas em uma NVidia Titan GPU).

### O que diabos o meu código está fazendo então?

Dito tudo isso, vamos entender o que foi que eu fiz.

Estou implementando um experimento para otimizar o treinamento de uma rede U-Net, descrita anteriormente, através do uso de Redução Espacial (Spatial Reduction) a partir de Saliência. Realizo o treinamento da rede em múltiplas iterações e em cada iteração, o modelo é forçado a concentrar-se em uma região progressivamente menor e mais relevante da imagem. Então o fluxo atual é:

1. Treinar o modelo.
2. Usar a previsão do modelo (saliência) para identificar quais pixels são mais importantes para a segmentação (e.g., Top 90%).
3. Zerar os pixels menos importantes, criando um novo conjunto de treino "reduzido".
4. Retreinar o modelo na próxima iteração apenas com os pixels considerados relevantes.

A hipótese proposta foi, ao focar o treinamento nas características essenciais e ignorar o fundo irrelevante de forma progressiva, o modelo pode melhorar a robustez, aumentar a velocidade de convergência, já que estamos treinando em menos dados relevantes e evitar que o modelo aprenda características desnecessárias. É um ciclo de treinamento iterativo e adaptativo.

O primeiro questionamento poderia ser: "Você está usando essa tal de U-net, mas não seria possível usar uma CNN simples para essa tarefa de segmentação?"

Essa é uma ótima pergunta que toca na diferença fundamental entre tarefas de Classificação e Segmentação em Deep Learning, que eu não sabia e por isso fiz até uma tabelinha para lembrar

| **Tarefa**                   | **Pergunta que responde**                  | **Saída (Output)**             | **Exemplo de uso**                        |
| ---------------------------- | ------------------------------------------ | ------------------------------ | ----------------------------------------- |
| **Classificação de Imagens** | O que há na imagem?                        | Um ou mais rótulos             | “É um gato siamês.”                       |
| **Detecção de Objetos**      | O que há na imagem e onde está localizado? | Caixas delimitadoras + rótulos | Detectar vários animais em uma única foto |
| **Segmentação de Imagens**   | O que há em cada pixel?                    | Máscara por pixel              | Separar o gato do fundo                   |

Aprendido isso, a resposta pode ser direta: Não! Um modelo CNN padrão não seria adequado para a tarefa de segmentação, pois ele não consegue gerar a saída esperada, que é um mapa de pixels.

A segmentação semântica é uma tarefa de predição densa (pixel-a-pixel).

| Característica | CNN Típica (Ex: VGG, ResNet para Classificação) | U-Net / CNN para Segmentação |
| --- | --- | --- |
| Objetivo | Classificar a imagem inteira (Ex: ""É um gato"") | Classificar cada pixel (Ex: ""Este pixel é 'gato', este é 'fundo'"") |
| Output | Um vetor de classes ([0.9, 0.1]). | Um tensor de imagem com a mesma resolução de entrada (Ex: 128x128x3) |
| Downsampling | Usa Max Pooling extensivamente, perdendo informações de localização espacial para obter um alto nível de abstração. | Usa Max Pooling, mas compensa essa perda na fase de Decoder (Expansão). |
| Camada Final | Camada Totalmente Conectada (Fully Connected) que colapsa toda a informação espacial em um vetor. | Camada de Convolução 1x1 que projeta a informação espacial de volta ao número de classes. |

Usando uma CNN típica, como eu fiz nos exemplos anteriores, o resultado final seria um único vetor (o rótulo da imagem) e não teria a saída no formato de imagem necessária para zerar os pixels menos importantes.

A U-Net resolve esse problema usando uma arquitetura de Encoder-Decoder com Skip Connections, como vou explicar daqui a pouco.

## Explicação do código

### Dataset

Oxford-IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/

Motivo? Nenhum em especial. Achei buscando na internet datasets para tarefas de segmentação. Por algum motivo não havia conseguido acessar o dataset usado no artigo.

#### Sobre o dataset
O conjunto de dados contém fotografias de cães e gatos com 37 categorias, contendo aproximadamente 200 imagens para cada classe. As imagens apresentam grande variação em escala, pose e iluminação. Todas as imagens possuem anotações de referência (ground truth) associadas, incluindo raça, região de interesse (ROI) da cabeça e segmentação trimap em nível de pixel.

O código foi dividido em partes que irei explicar a seguir:

### Preparação dos dados

- `class OxfordPetSegmentation(Dataset)`: É o Dataset que encapsula o conjunto de dados (imagens de pets e suas máscaras de anotação). Responsável por: Extrair os arquivos (.tar.gz), parear imagens com suas máscaras correspondentes, redimensionar as imagens (para 128x128), mapear os valores da máscara (por exemplo, 3 -> 0) e converter a máscara para o formato One-Hot Encoding (cada classe em um canal diferente), que é um formato comum para treinar modelos de segmentação.
- `create_dataloaders`: Divide o dataset total em conjuntos de Treino (80%), Validação (10%) e Teste (10%). Cria os DataLoaders que gerenciam o carregamento dos dados em batches (lotes) e o embaralhamento."
- `get_full_train_data`: Função auxiliar que agrupa todas as imagens e máscaras do dataloader de treino em dois grandes tensores. Isso é feito para que a etapa de redução espacial possa ser aplicada sobre todo o conjunto de treino de uma vez, sem depender dos batches.

### Modelo

- `conv_block`: Define o bloco básico de processamento. Aplica duas camadas de Convolução 2D, seguidas pela função de ativação ReLU (Rectified Linear Unit), que introduz não-linearidade no modelo.
- `class UNet(nn.Module)`: Monta a arquitetura completa U-Net. Encoder reduz a dimensão espacial da imagem (usando MaxPool2d) e aumenta a profundidade (número de canais, e.g., de 32 para 64) para capturar características de alto nível (o ""o quê"" é o objeto)." e Decoder aumenta a dimensão espacial de volta ao tamanho original (usando ConvTranspose2d) e diminui a profundidade, reconstruindo a imagem segmentada.
- `Skip Connections`: Conexões essenciais que concatenam características do Encoder com as do Decoder na mesma resolução. Isso permite que o Decoder recupere detalhes espaciais finos que foram perdidos durante a redução do Encoder, melhorando a precisão da segmentação de bordas.

### Funções de Saliência e Métricas

- `create_saliency_mask`: Calcula a probabilidade (softmax) e soma as probabilidades das classes de interesse (o que não faz parte do fundo) para obter um mapa de relevância. Define um limiar (e.g., Top 90% dos pixels mais relevantes) e cria uma máscara binária (1 para relevante, 0 para irrelevante)

- `apply_mask`: Aplica a máscara de saliência à imagem original, zerando os pixels não-relevantes (transformando-os em preto/escuro) para a próxima iteração.

- calculate_metrics,iou = intersection / (union + 1e-6),"Calcula as métricas de desempenho para segmentação: mIoU (mean Intersection over Union) e mDice (mean Dice Coefficient). Ambos medem o quanto a previsão do modelo se sobrepõe à verdade fundamental (target), variando de 0 a 1 (quanto mais perto de 1, melhor)."

#### Sobre o Algoritmo de Saliência

Uma questão que discutimos anteriormente é o motivo do código não utilizar um dos algoritmos de saliência mais conhecidos e complexos, como Grad-CAM ou Integrated Gradients, mas sim um método direto e personalizado, baseado na saída do próprio modelo. O algoritmo de saliência utilizado neste caso é um método de Saliência Baseada na Previsão com Limiar de Percentil.

**Cálculo do Score de Relevância (Agregação de Probabilidade)**

Esta é a parte que determina o quão "saliência" é um pixel.

- Softmax: O código aplica a função `F.softmax` nas previsões (`predictions`) da rede U-Net. Isso transforma os valores brutos de saída em probabilidades (a soma das probabilidades de todas as classes para um pixel é 1).

- Agregação: Em seguida, ele calcula o `relevance_scores` pela seguinte linha:
```python
relevance_scores = torch.sum(probabilities[:, 1:, :, :], dim=1) 
```
O `probabilities[:, 1:, :, :]` seleciona todos os canais de classe, exceto o canal de índice 0 (que geralmente representa o "fundo" ou *background* no problema de segmentação). Ao usar o `torch.sum` ao longo do eixo das classes (`dim=1`), o *score* final de relevância de um pixel é a soma das probabilidades de ele pertencer a qualquer classe de objeto de interesse (no caso, o animal ou seu contorno).

**Criação da Máscara Binária por Percentil**

Em vez de usar um valor fixo, o código utiliza um método adaptativo para definir o limiar.

- **Percentil:** Para cada imagem, o código determina o valor de *score* de relevância que corresponde ao `threshold_percentile` (e.g., 90%).
- **`torch.kthvalue`:** A função `torch.kthvalue` é usada para encontrar o score no percentil desejado (por exemplo, o score do pixel que está no Top 90% dos mais relevantes).
- **Binarização:** Todos os pixels cujo score de relevância **exceda esse valor de limiar** são marcados com **1** (relevante), e os demais são marcados com **0** (irrelevante), criando a máscara binária de saliência.

Parecia ser um método específico e eficaz para o propósito de Redução Espacial Iterativa, ao somar as probabilidades das classes de interesse e ignorar a probabilidade do fundo (classe 0), o algoritmo garante que o mapa de saliência destaque apenas a presença do objeto. O uso do limiar de percentil garante que, independentemente da imagem, o modelo sempre manterá um percentual fixo para a próxima iteração de treinamento, implementando a estratégia de redução espacial de forma controlada.

### Treino iterativo

- `train_iteration`: Implementa o ciclo de treino padrão (cálculo de perda, retropropagação, atualização de pesos). Inclui um ReduceLROnPlateau Scheduler que reduz a taxa de aprendizado se a perda de validação parar de melhorar por um certo tempo, ajudando a convergir melhor. Após o treino, executa a avaliação no conjunto de validação para calcular o mIoU e mDice.
- `Loop,"for i in range(1, MAX_ITERATIONS + 1)`: O ciclo que coordena todo o processo: treina o modelo, usa o modelo para identificar os pixels mais relevantes no conjunto de treino, aplica a redução espacial (o apply_mask) e, em seguida, cria um novo dataloader para o próximo ciclo de treinamento, usando as imagens reduzidas.

### Resumo

Caso não tenha ficado claro até aqui. O artigo que descreve a arquitetura base que estou usando, no entanto o processo realizado no código não é idêntico ao descrito no artigo, mas sim uma extensão focada na estratégia de treinamento. A principal diferença reside no Ciclo de Treinamento Iterativo com Redução Espacial baseada em Saliência que implementei, o qual não faz parte da metodologia central do artigo original da U-Net. Vou até fazer uma tabela para não perder o costume.

Artigo

|Aspecto | Metodologia U-Net Original |
| ------ | -------------------------- |
| Arquitetura	| Um caminho contrativo (Encoder) para captura de contexto, um caminho expansivo (Decoder) para localização precisa, e Skip Connections para unir ambos. |
| Treinamento	| End-to-End Único: O modelo é treinado em um único ciclo (algumas épocas), usando o conjunto de dados completo (imagens e máscaras) em cada iteração. |
| Perda (Loss) | Utiliza uma função de perda com mapa de peso de pixels (pixel-wise weight map). Isso é usado para forçar a rede a dar mais importância aos pixels das bordas, ajudando a separar objetos adjacentes, o que é crucial em aplicações biomédicas como a separação de células. |
| Aumento de Dados | Uso massivo de Data Augmentation (aumento de dados) para simular elasticidade e deformações, crucial quando há poucas amostras de treinamento. 

Código do João

| Aspecto | Metodologia adotada |
| ------- | ------------------- |
| Arquitetura | Mesma do artigo |
| Treinamento | Iterativo com Redução Espacial: O modelo é treinado em vários ciclos (7 iterações), onde o conjunto de dados de treino é modificado a cada ciclo. A rede não é apenas treinada, mas forçada a focar nas regiões mais relevantes identificadas pelo seu próprio mapa de saliência na iteração anterior. |
| Foco na Saliência (LÁ ELE) | Em vez de um peso estático para as bordas (como no original), seu código usa o resultado da previsão para criar uma máscara de saliência (Baseada na Previsão) e ignorar pixels com baixa probabilidade de serem objetos. |

O Marcelo pediu para dar uma olhada nas referências para garantir que não estamos tentando *reinventar a roda*. O que descobri em minhas peregrinações pelo IEEE é que esse processo exato de Redução Espacial Iterativa baseada em Saliência com a metodologia de limiar de percentil de probabilidade (nome grande...) é uma variação específica. O conceito geral de refinar o foco de treinamento de forma iterativa usando mapas de relevância é algo ativamente explorado na pesquisa. Há muitos artigos que buscam o  **Refinamento Iterativo e Melhoria da Saliência** em tarefas de segmentação, especialmente em contextos de aprendizado de poucas amostras (Zero-Shot ou Few-Shot) ou detecção de objetos.

1. IteRPrimE: Refinamento Grad-CAM Iterativo
Título: [2503.00936] IteRPrimE: Zero-shot Referring Image Segmentation with Iterative Grad-CAM Refinement and Primary Word Emphasis.

Técnica: Este trabalho propõe uma estratégia de refinamento iterativo para segmentação de imagens baseada em texto (Referring Image Segmentation).

Mecanismo: Em vez de usar a probabilidade de softmax diretamente  ele usa mapas de Grad-CAM gerados por um modelo Vision-Language. A ideia é a mesma: melhorar progressivamente o foco do modelo na região alvo para superar a imprecisão posicional.

2. Aperfeiçoamento Iterativo de Saliência
Título: [2112.00665] Iterative Saliency Enhancement using Superpixel Similarity.

Técnica: O estudo introduz uma abordagem híbrida que iterativamente gera mapas de saliência aprimorados alternando entre a segmentação de superpixels e a estimação de saliência baseada em superpixels.

Mecanismo: O foco aqui é aprimorar o mapa de saliência em si, para que ele se torne mais nítido e preciso, repetindo o processo em ciclos até que os mapas gerados sejam combinados.

3. Redução de Custo Computacional com Saliência Semântica
Título: Saliency Prediction with Active Semantic Segmentation.

Técnica: Embora o objetivo principal seja a predição de saliência, o trabalho introduz um modelo de predição de saliência baseado em segmentação semântica ativa.

Mecanismo: O uso de informações de segmentação semântica para orientar e reduzir o subconjunto de regiões a serem processadas para estimar a saliência compartilha a lógica de usar o conhecimento do modelo (segmentação/semântica) para refinar o processamento espacial.

A partir disso, que podemos fazer?