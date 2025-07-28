#!/usr/bin/env python
# coding: utf-8

"""
Affinity2Vec_ChEMBL_FixedFolds.py

Esta versão foi modificada para usar conjuntos de treino e teste pré-definidos
a partir de arquivos de índice, em vez de usar validação cruzada K-Fold.
"""

# Pacotes Gerais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
import os
import json

# Pacotes de ML
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index

# Funções Proprietárias e Adaptadas
from evaluation import *
from OptimizedpathScores_functions import *

# =============================================================================
# --- CONFIGURAÇÃO PARA EXPERIMENTOS ---
# =============================================================================
USE_EMBEDDING_FEATURES = True
MODEL_OUTPUT_FOLDER = "results/exp_fixed_folds_hybrid_transformer/"

# =============================================================================
# --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO INICIAL ---
# =============================================================================
print("--- Etapa 1: Carregando e Pré-processando os Dados ---")

# --- Caminhos para os arquivos de entrada ---
input_folder = "Input/ChEMBL/"
folds_folder = os.path.join(input_folder, "folds/")
drug_embed_file = "EMBED/ChEMBL/Dr_ChemBERTa_EMBED.tsv"
target_embed_file = "EMBED/ChEMBL/Pr_ESM2_EMBED.tsv"
affinityFile = input_folder + "affinitiesChEMBL34-filtered.tsv"
train_indices_file = os.path.join(folds_folder, "train_val_idx.txt")
test_indices_file = os.path.join(folds_folder, "test_idx.txt")


if not os.path.exists(MODEL_OUTPUT_FOLDER):
    os.makedirs(MODEL_OUTPUT_FOLDER)

# Carrega embeddings e afinidades
drEMBED = pd.read_csv(drug_embed_file, delimiter='\t', header=None).set_index(0)
tgEMBED = pd.read_csv(target_embed_file, delimiter='\t', header=None).set_index(0)
Affinities = pd.read_csv(affinityFile, delimiter='\t')

# Mapeia IDs para índices numéricos
drug_ids = drEMBED.index.tolist()
target_ids = tgEMBED.index.tolist()
drug_to_idx = {drug: i for i, drug in enumerate(drug_ids)}
target_to_idx = {target: i for i, target in enumerate(target_ids)}

# Calcula matrizes de similaridade a partir dos embeddings
print("Calculando matrizes de similaridade...")
DD_Sim_matrix = MinMaxScaler().fit_transform(cosine_similarity(drEMBED.values).astype(np.float32))
TT_Sim_matrix = MinMaxScaler().fit_transform(cosine_similarity(tgEMBED.values).astype(np.float32))

# Cria a matriz de afinidade completa com NaNs para pares desconhecidos
affinity_matrix = np.full((len(drug_ids), len(target_ids)), np.nan, dtype=np.float32)
for _, row in Affinities.iterrows():
    if row['compound'] in drug_to_idx and row['target'] in target_to_idx:
        affinity_matrix[drug_to_idx[row['compound']], target_to_idx[row['target']]] = row['pchembl_value']

print("Dados carregados e pré-processados.")

# =============================================================================
# --- 2. DIVISÃO DE DADOS USANDO OS ÍNDICES FORNECIDOS ---
# =============================================================================
print("\n--- Etapa 2: Dividindo os dados com base nos folds fornecidos ---")

# Encontra os índices de todos os pares com afinidade conhecida
positive_row_inds, positive_col_inds = np.where(np.isnan(affinity_matrix) == False)
positive_labels = affinity_matrix[positive_row_inds, positive_col_inds]

# Carrega os índices de treino e teste dos arquivos
try:
    train_indices_folds = np.loadtxt(train_indices_file, dtype=int)
    train_indices = train_indices_folds.flatten() # Achata para um único array de treino
    test_indices = np.loadtxt(test_indices_file, dtype=int)
    print(f"Carregados {len(train_indices)} índices de treino e {len(test_indices)} de teste.")
except FileNotFoundError as e:
    print(f"Erro: Arquivo de índice não encontrado - {e}. Verifique os caminhos.")
    exit()


# Divide os dados de acordo com os índices carregados
train_pos_rows, train_pos_cols = positive_row_inds[train_indices], positive_col_inds[train_indices]
test_pos_rows, test_pos_cols = positive_row_inds[test_indices], positive_col_inds[test_indices]
Y_train = positive_labels[train_indices]
Y_test = positive_labels[test_indices]
print(f"Divisão final: {len(Y_train)} amostras de treino, {len(Y_test)} amostras de teste.")

# =============================================================================
# --- 3. GERAÇÃO DE CARACTERÍSTICAS ---
# =============================================================================
print("\n--- Etapa 3: Gerando características de Meta-Caminho e Embedding ---")

# --- 3.1 Criação do Grafo Mascarado para Treino ---
# Normaliza os rótulos de treino e cria o grafo de afinidade apenas com dados de treino
scaler_aff = MinMaxScaler()
Y_train_scaled = scaler_aff.fit_transform(Y_train.reshape(-1, 1)).flatten()

affinity_matrix_norm_train = np.full_like(affinity_matrix, np.nan)
affinity_matrix_norm_train[train_pos_rows, train_pos_cols] = Y_train_scaled

affinity_graph_weights_train = np.nan_to_num(affinity_matrix_norm_train, nan=0.0)
print("Grafo de treino mascarado criado.")

# --- 3.2 Cálculo dos Meta-Path Scores para o conjunto de TREINO ---
print("Calculando scores de meta-path para o conjunto de treino...")
sumDDD, maxDDD = DDD_TTT_sim(DD_Sim_matrix)
sumTTT, maxTTT = DDD_TTT_sim(TT_Sim_matrix)

sumDDT_train, maxDDT_train = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_train, 2)
sumDTT_train, maxDTT_train = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_train, 2)
sumDDDT_train, maxDDDT_train = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_train, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_train, 3)[1]
sumDTTT_train, maxDTTT_train = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_train, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_train, 3)[1]
sumDTDT_train, maxDTDT_train = metaPath_DTDT(affinity_graph_weights_train)
sumDDTT_train, maxDDTT_train = metaPath_DDTT(affinity_graph_weights_train, DD_Sim_matrix, TT_Sim_matrix)

# --- 3.3 Cálculo dos Meta-Path Scores para o conjunto de TESTE ---
# Para o teste, simula-se um cenário real onde o grafo completo (com todas as afinidades conhecidas) é usado para extrair caminhos.
print("Calculando scores de meta-path para o conjunto de teste...")
all_pos_labels_scaled = MinMaxScaler().fit_transform(positive_labels.reshape(-1, 1)).flatten()
affinity_matrix_norm_full = np.full_like(affinity_matrix, np.nan)
affinity_matrix_norm_full[positive_row_inds, positive_col_inds] = all_pos_labels_scaled
affinity_graph_weights_full = np.nan_to_num(affinity_matrix_norm_full, nan=0.0)

sumDDT_full, maxDDT_full = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_full, 2)
sumDTT_full, maxDTT_full = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_full, 2)
sumDDDT_full, maxDDDT_full = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_full, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_full, 3)[1]
sumDTTT_full, maxDTTT_full = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_full, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_full, 3)[1]
sumDTDT_full, maxDTDT_full = metaPath_DTDT(affinity_graph_weights_full)
sumDDTT_full, maxDDTT_full = metaPath_DDTT(affinity_graph_weights_full, DD_Sim_matrix, TT_Sim_matrix)

# --- 3.4 Montagem dos Vetores de Features ---
print("Montando vetores de características...")
X_path_train = np.vstack([
    sumDDT_train[train_pos_rows, train_pos_cols], maxDDT_train[train_pos_rows, train_pos_cols],
    sumDTT_train[train_pos_rows, train_pos_cols], maxDTT_train[train_pos_rows, train_pos_cols],
    sumDDDT_train[train_pos_rows, train_pos_cols], maxDDDT_train[train_pos_rows, train_pos_cols],
    sumDTTT_train[train_pos_rows, train_pos_cols], maxDTTT_train[train_pos_rows, train_pos_cols],
    sumDTDT_train[train_pos_rows, train_pos_cols], maxDTDT_train[train_pos_rows, train_pos_cols],
    sumDDTT_train[train_pos_rows, train_pos_cols], maxDDTT_train[train_pos_rows, train_pos_cols]
]).T

X_path_test = np.vstack([
    sumDDT_full[test_pos_rows, test_pos_cols], maxDDT_full[test_pos_rows, test_pos_cols],
    sumDTT_full[test_pos_rows, test_pos_cols], maxDTT_full[test_pos_rows, test_pos_cols],
    sumDDDT_full[test_pos_rows, test_pos_cols], maxDDDT_full[test_pos_rows, test_pos_cols],
    sumDTTT_full[test_pos_rows, test_pos_cols], maxDTTT_full[test_pos_rows, test_pos_cols],
    sumDTDT_full[test_pos_rows, test_pos_cols], maxDTDT_full[test_pos_rows, test_pos_cols],
    sumDDTT_full[test_pos_rows, test_pos_cols], maxDDTT_full[test_pos_rows, test_pos_cols]
]).T

# Adiciona características de embedding se a flag estiver ativa (modelo híbrido)
if USE_EMBEDDING_FEATURES:
    X_embed_train = np.hstack([drEMBED.values[train_pos_rows], tgEMBED.values[train_pos_cols]])
    X_embed_test = np.hstack([drEMBED.values[test_pos_rows], tgEMBED.values[test_pos_cols]])
    X_train = np.hstack([X_path_train, X_embed_train])
    X_test = np.hstack([X_path_test, X_embed_test])
else:
    X_train = X_path_train
    X_test = X_path_test

print(f"Dimensões finais das características: Treino={X_train.shape}, Teste={X_test.shape}")

# =============================================================================
# --- 4. TREINAMENTO E AVALIAÇÃO DO MODELO ---
# =============================================================================
print("\n--- Etapa 4: Treinando e Avaliando o Modelo ---")

# Normaliza as características
scaler_feat = MinMaxScaler()
X_train_scaled = scaler_feat.fit_transform(X_train)
X_test_scaled = scaler_feat.transform(X_test)

# Define o regressor XGBoost
xg_reg = xgb.XGBRegressor(booster='gbtree', tree_method='hist', device='cuda', objective='reg:squarederror', eval_metric='rmse', colsample_bytree=0.8, learning_rate=0.03, max_depth=19, n_estimators=855, seed=42, n_jobs=-1)

print("Treinando o modelo XGBoost...")
xg_reg.fit(X_train_scaled, Y_train)

print("Realizando predições no conjunto de teste...")
Y_hat_test = xg_reg.predict(X_test_scaled)

# Calcula as métricas de avaliação
mse = mean_squared_error(Y_test, Y_hat_test)
ci = concordance_index(Y_test, Y_hat_test)
rm2 = get_rm2(Y_test, Y_hat_test)

# =============================================================================
# --- 5. RESULTADOS FINAIS ---
# =============================================================================
print(f"\n{'='*20} RESULTADOS FINAIS DA AVALIAÇÃO {'='*20}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Concordance Index (CI): {ci:.4f}")
print(f"R_m^2: {rm2:.4f}")

# Salva os resultados em um arquivo
metrics_path = os.path.join(MODEL_OUTPUT_FOLDER, "final_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Resultados da Avaliação com Folds Fixos\n")
    f.write("-" * 30 + "\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Concordance Index (CI): {ci:.4f}\n")
    f.write(f"R_m^2: {rm2:.4f}\n")

print(f"\nMétricas detalhadas salvas em: {metrics_path}")

# Opcional: Plotar valores reais vs. previstos
plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_hat_test, alpha=0.5)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.title("Valores Reais vs. Previstos no Conjunto de Teste")
plt.xlabel("Valores Reais (pchembl_value)")
plt.ylabel("Valores Previstos (pchembl_value)")
plt.grid(True)
plot_path = os.path.join(MODEL_OUTPUT_FOLDER, "real_vs_predicted.png")
plt.savefig(plot_path)
print(f"Gráfico de dispersão salvo em: {plot_path}")
# plt.show()
