## Descrição

Este projeto é uma implementação em Python do algoritmo Affinity2Vec, conforme descrito no artigo "Affinity2Vec: drug-target binding affinity prediction through representation learning, graph mining, and machine learning". O objetivo é prever a afinidade de ligação (um valor contínuo) entre compostos (fármacos) e alvos proteicos.

O método trata o problema como uma rede heterogênea, combinando similaridade entre fármacos, similaridade entre alvos e interações conhecidas. A partir dessa rede, extrai características baseadas em meta-caminhos e embeddings para treinar um modelo de regressão (XGBoost).

## Como Usar

Siga os passos abaixo para configurar o ambiente e executar um experimento.
1. Pré-requisitos
    Python 3.8 ou superior
    Gerenciador de pacotes pip

2. Instalação
Clone este repositório e instale as dependências listadas no arquivo requirements.txt:

`git clone https://github.com/arthruur/Affinity2Vec_ChEMBL.git`

`cd Affinity2Vec_ChEMBL`

`pip install -r requirements.txt`

3. Preparação dos Dados

Antes de executar os scripts, certifique-se de que os seguintes arquivos estão nos diretórios corretos:

Afinidades: Coloque o arquivo de afinidades (ex: affinitiesChEMBL34-filtered.tsv) no diretório Input/ChEMBL/.

Embeddings: Coloque os arquivos de embedding para fármacos e alvos (ex: Dr_ChemBERTa_EMBED.tsv, Pr_ESM2_EMBED.tsv ou Dr_seq2seq_EMBED.tsv, Pr_ProtVec_EMBED.tsv) no diretório EMBED/ChEMBL/.

Folds (Divisões de Dados): Se estiver usando uma divisão fixa, coloque os arquivos de índice (ex: train_val_idx.txt, test_idx.txt) no diretório Input/ChEMBL/folds/.

4. Executando um Experimento

Para treinar e avaliar o modelo, execute um dos scripts principais. Por exemplo, para rodar o experimento com o dataset ChEMBL usando os folds fixos:

`python Affinity2Vec_ChEMBL_FixedFolds.py`

O script irá executar todas as etapas do pipeline:

- Carregar e pré-processar os dados.

- Dividir os dados em treino e teste conforme os arquivos de índice.

- Gerar as características de meta-caminho e de embedding.
- Treinar o modelo XGBoost.

- Avaliar o modelo no conjunto de teste.

Dentro do script é possível alternar entre os modelos Affinity2Vec_Pscores e Affinity2Vec_Hybrid pela variável "USE_EMBEDDING_FEATURES" no começo do código, além disso, é possível alternar entre os embeddings disponibilizados no artigo original, ou os gerados para esse experimento modificando as variáveis "drug_embedding_file" e "target_embedding_file". Em alternativa ao `Affinity2Vec_ChEMBL_FixedFolds` pode ser executado o `Affinity2Vec_ChEMBL` que faz outra divisão dos dados e utiliza K-FOLD para validação cruzada. 


Após a execução, os seguintes arquivos serão salvos no diretório definido pela variável MODEL_OUTPUT_FOLDER, results/.

  final_metrics.txt: Um arquivo de texto com as métricas de performance finais (MSE, CI, r_m2).5. Verificando os Resultados

  real_vs_predicted.png: Um gráfico de dispersão comparando os valores de afinidade reais com os valores previstos pelo modelo no conjunto de teste.