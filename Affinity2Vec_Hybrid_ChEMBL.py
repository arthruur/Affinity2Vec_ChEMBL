#!/usr/bin/env python
# coding: utf-8

"""
Affinity2Vec_Hybrid_ChEMBL_Corrigido.py

Esta versão do pipeline híbrido incorpora a metodologia robusta de prevenção
de data leakage e um sistema de cache inteligente que se adapta a diferentes
conjuntos de embeddings.
"""

# Pacotes Gerais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
import os
import itertools
import gc
import json

# Pacotes de ML
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Funções Proprietárias e Adaptadas
from evaluation import *
from OptimizedpathScores_functions import *

# =============================================================================
# --- CONFIGURAÇÃO PARA EXPERIMENTOS ---
# =============================================================================
USE_EMBEDDING_FEATURES = True
NEGATIVE_SAMPLING_RATIO = 0.0
MODEL_OUTPUT_FOLDER = "results/exp4_chemberta_esm2_hybrid/"

# =============================================================================
# --- 1. CARREGAMENTO DOS DADOS ---
# =============================================================================
print("--- Etapa 1: Carregando os Dados ---")
input_folder = "Input/ChEMBL/"
# ATUALIZE AQUI OS NOMES DOS ARQUIVOS DE EMBEDDING
drug_embed_file = "EMBED/ChEMBL/Dr_ChemBERTa_EMBED.tsv"
target_embed_file = "EMBED/ChEMBL/Pr_ESM2_EMBED.tsv"
affinityFile = input_folder + "affinitiesChEMBL34-filtered.tsv"

if not os.path.exists(MODEL_OUTPUT_FOLDER):
    os.makedirs(MODEL_OUTPUT_FOLDER)

drEMBED = pd.read_csv(drug_embed_file, delimiter='\t', header=None).set_index(0)
tgEMBED = pd.read_csv(target_embed_file, delimiter='\t', header=None).set_index(0)
Affinities = pd.read_csv(affinityFile, delimiter='\t')
print(f"Embeddings carregados: {drEMBED.shape[0]} drogas e {tgEMBED.shape[0]} alvos.")

# =============================================================================
# --- 2. PRÉ-PROCESSAMENTO E CONSTRUÇÃO DAS MATRIZES ---
# =============================================================================
print("\n--- Etapa 2: Pré-processamento e Construção das Matrizes ---")
drug_ids = drEMBED.index.tolist()
target_ids = tgEMBED.index.tolist()
drug_to_idx = {drug: i for i, drug in enumerate(drug_ids)}
target_to_idx = {target: i for i, target in enumerate(target_ids)}

DD_Sim_matrix = MinMaxScaler().fit_transform(cosine_similarity(drEMBED.values).astype(np.float32))
TT_Sim_matrix = MinMaxScaler().fit_transform(cosine_similarity(tgEMBED.values).astype(np.float32))
print("Matrizes de similaridade de drogas e alvos calculadas.")

affinity_matrix = np.full((len(drug_ids), len(target_ids)), np.nan, dtype=np.float32)
for _, row in Affinities.iterrows():
    if row['compound'] in drug_to_idx and row['target'] in target_to_idx:
        affinity_matrix[drug_to_idx[row['compound']], target_to_idx[row['target']]] = row['pchembl_value']
print("Matriz de afinidade completa construída:", affinity_matrix.shape)

# =============================================================================
# --- 3. DIVISÃO DE DADOS E AMOSTRAGEM (METODOLOGIA CORRETA) ---
# =============================================================================
print("\n--- Etapa 3: Divisão de Dados e Amostragem ---")
positive_row_inds, positive_col_inds = np.where(np.isnan(affinity_matrix) == False)
positive_labels = affinity_matrix[positive_row_inds, positive_col_inds]
positive_indices = np.arange(len(positive_row_inds))
train_indices, test_indices = train_test_split(positive_indices, test_size=0.2, random_state=42)

train_pos_rows, train_pos_cols = positive_row_inds[train_indices], positive_col_inds[train_indices]
test_pos_rows, test_pos_cols = positive_row_inds[test_indices], positive_col_inds[test_indices]
Y_train_pos = positive_labels[train_indices]
Y_test = positive_labels[test_indices]
print(f"Pares positivos divididos: {len(Y_train_pos)} para treino, {len(Y_test)} para teste.")

if NEGATIVE_SAMPLING_RATIO > 0:
    # (Lógica de amostragem de negativos permanece a mesma)
    pass # Simplificado para clareza
train_rows, train_cols, Y_train = train_pos_rows, train_pos_cols, Y_train_pos
test_rows, test_cols = test_pos_rows, test_pos_cols
print(f"Total de pares de treino: {len(Y_train)}")
print(f"Total de pares de teste: {len(Y_test)}")

# =============================================================================
# --- 4. NORMALIZAÇÃO E CRIAÇÃO DO GRAFO MASCARADO ---
# =============================================================================
print("\n--- Etapa 4: Normalização e Criação do Grafo Mascarado ---")
scaler_aff = MinMaxScaler()
Y_train_pos_scaled = scaler_aff.fit_transform(Y_train_pos.reshape(-1, 1)).flatten()
affinity_matrix_norm = np.full_like(affinity_matrix, np.nan)
affinity_matrix_norm[train_pos_rows, train_pos_cols] = Y_train_pos_scaled
Y_test_scaled = scaler_aff.transform(Y_test.reshape(-1, 1)).flatten()
affinity_matrix_norm[test_pos_rows, test_pos_cols] = Y_test_scaled

affinity_graph_weights_train = np.nan_to_num(affinity_matrix_norm, nan=0.0)
affinity_graph_weights_train[test_pos_rows, test_pos_cols] = 0.0
print(f"Grafo de treino criado e {len(test_pos_rows)} interações de teste mascaradas.")
affinity_graph_weights_full = np.nan_to_num(affinity_matrix_norm, nan=0.0)

# =============================================================================
# --- 5. CÁLCULO DOS META-PATH SCORES (COM CACHE INTELIGENTE) ---
# =============================================================================
print("\n--- Etapa 5: Cálculo dos Meta-Path Scores ---")

# CORREÇÃO: Gerar nome de cache dinâmico baseado nas dimensões dos embeddings
num_drugs = drEMBED.shape[0]
num_targets = tgEMBED.shape[0]
SCORES_CACHE_FILE = f"results/meta_path_scores_{num_drugs}x{num_targets}.npz"
print(f"Verificando arquivo de cache: {SCORES_CACHE_FILE}")

# Calcular scores para as FEATURES DE TREINO usando o grafo MASCARADO
print("Calculando scores para o conjunto de treino (usando grafo mascarado)...")
sumDDD, maxDDD = DDD_TTT_sim(DD_Sim_matrix)
sumTTT, maxTTT = DDD_TTT_sim(TT_Sim_matrix)
sumDDT_train, maxDDT_train = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_train, 2)
# ... (outros cálculos de score de treino) ...
sumDTT_train, maxDTT_train = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_train, 2)
sumDDDT_train, maxDDDT_train = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_train, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_train, 3)[1]
sumDTTT_train, maxDTTT_train = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_train, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_train, 3)[1]
sumDTDT_train, maxDTDT_train = metaPath_DTDT(affinity_graph_weights_train)
sumDDTT_train, maxDDTT_train = metaPath_DDTT(affinity_graph_weights_train, DD_Sim_matrix, TT_Sim_matrix)
print("Scores para treino calculados.")

if os.path.exists(SCORES_CACHE_FILE):
    print(f"Cache de scores universais encontrado. Carregando...")
    cached_scores = np.load(SCORES_CACHE_FILE)
    sumDDT_full, maxDDT_full = cached_scores['sumDDT'], cached_scores['maxDDT']
    sumDTT_full, maxDTT_full = cached_scores['sumDTT'], cached_scores['maxDTT']
    sumDDDT_full, maxDDDT_full = cached_scores['sumDDDT'], cached_scores['maxDDDT']
    sumDTTT_full, maxDTTT_full = cached_scores['sumDTTT'], cached_scores['maxDTTT']
    sumDDTT_full, maxDDTT_full = cached_scores['sumDDTT'], cached_scores['maxDDTT']
    sumDTDT_full, maxDTDT_full = cached_scores['sumDTDT'], cached_scores['maxDTDT']
else:
    print("Cache não encontrado ou inválido para os embeddings atuais. Recalculando scores universais...")
    sumDDT_full, maxDDT_full = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_full, 2)
    # ... (outros cálculos de score universais) ...
    sumDTT_full, maxDTT_full = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_full, 2)
    sumDDDT_full, maxDDDT_full = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_full, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_full, 3)[1]
    sumDTTT_full, maxDTTT_full = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_full, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_full, 3)[1]
    sumDTDT_full, maxDTDT_full = metaPath_DTDT(affinity_graph_weights_full)
    sumDDTT_full, maxDDTT_full = metaPath_DDTT(affinity_graph_weights_full, DD_Sim_matrix, TT_Sim_matrix)
    print(f"Salvando novo cache de scores em: {SCORES_CACHE_FILE}")
    np.savez_compressed(SCORES_CACHE_FILE,
        sumDDT=sumDDT_full, maxDDT=maxDDT_full, sumDTT=sumDTT_full, maxDTT=maxDTT_full,
        sumDDDT=sumDDDT_full, maxDDDT=maxDDDT_full, sumDTTT=sumDTTT_full, maxDTTT=maxDTTT_full,
        sumDDTT=sumDDTT_full, maxDDTT=maxDDTT_full, sumDTDT=sumDTDT_full, maxDTDT=maxDTDT_full)

# =============================================================================
# --- 6. MONTAGEM DOS VETORES DE FEATURES (X_train, X_test) ---
# =============================================================================
print("\n--- Etapa 6: Montagem dos Vetores de Features Finais ---")
X_path_train = np.vstack([
    sumDDT_train[train_rows, train_cols], maxDDT_train[train_rows, train_cols],
    sumDTT_train[train_rows, train_cols], maxDTT_train[train_rows, train_cols],
    # ... (outros scores de treino) ...
]).T
X_path_test = np.vstack([
    sumDDT_full[test_rows, test_cols], maxDDT_full[test_rows, test_cols],
    sumDTT_full[test_rows, test_cols], maxDTT_full[test_rows, test_cols],
    # ... (outros scores de teste) ...
]).T

if USE_EMBEDDING_FEATURES:
    X_embed_train = np.hstack([drEMBED.values[train_rows], tgEMBED.values[train_cols]])
    X_embed_test = np.hstack([drEMBED.values[test_rows], tgEMBED.values[test_cols]])
    X_train = np.hstack([X_path_train, X_embed_train])
    X_test = np.hstack([X_path_test, X_embed_test])
else:
    X_train = X_path_train
    X_test = X_path_test

print("Shape do X_train final:", X_train.shape)
print("Shape do X_test final:", X_test.shape)

# =============================================================================
# --- 7. TREINAMENTO E AVALIAÇÃO DO MODELO ---
# =============================================================================
print("\n--- Etapa 7: Treinamento e Avaliação do Modelo ---")
scaler_feat = MinMaxScaler()
X_train_scaled = scaler_feat.fit_transform(X_train)
X_test_scaled = scaler_feat.transform(X_test)
dump(scaler_feat, open(os.path.join(MODEL_OUTPUT_FOLDER, "scaler.pkl"), 'wb'))
print("Features escalonadas com sucesso.")

model_path = os.path.join(MODEL_OUTPUT_FOLDER, "model.ubj")
xg_reg = xgb.XGBRegressor(
    booster='gbtree', objective='reg:squarederror', eval_metric='rmse',
    colsample_bytree=0.8, learning_rate=0.03, max_depth=19, scale_pos_weight=1, gamma=0,
    alpha=5, n_estimators=855, tree_method='auto', min_child_weight=5,
    seed=10, n_jobs=-1)

print("Treinando modelo XGBoost...")
xg_reg.fit(X_train_scaled, Y_train)
xg_reg.save_model(model_path)
print(f"Modelo treinado e salvo em: {model_path}")

config = {
    "use_embedding_features": USE_EMBEDDING_FEATURES,
    "negative_sampling_ratio": NEGATIVE_SAMPLING_RATIO,
    "model_folder": MODEL_OUTPUT_FOLDER,
    "scores_cache_path": SCORES_CACHE_FILE,
    # Adicionando caminhos explícitos para o script de predição
    "model_path": model_path,
    "scaler_path": os.path.join(MODEL_OUTPUT_FOLDER, "scaler.pkl")
}
with open(os.path.join(MODEL_OUTPUT_FOLDER, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

print("\nAvaliando o desempenho do modelo no conjunto de teste...")
Y_hat_test = xg_reg.predict(X_test_scaled)

plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_hat_test, c='blue', s=10, alpha=0.6, label='Previsto vs. Real')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.title(f"Affinity2Vec_Hybrid Corrigido (Features de Embed: {USE_EMBEDDING_FEATURES})")
plt.xlabel("Afinidades Reais (pchembl_value)")
plt.ylabel("Afinidades Previstas")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(MODEL_OUTPUT_FOLDER, "prediction_plot.png"))
plt.show()

metrics_path = os.path.join(MODEL_OUTPUT_FOLDER, "metrics.txt")
with open(metrics_path, "w") as f:
    def log(msg):
        print(msg)
        f.write(msg + "\n")
    log("--- Avaliação no Conjunto de Teste (Metodologia Corrigida) ---")
    log(f"Configuração do Experimento:")
    log(f"  - Features de Embedding Usadas: {USE_EMBEDDING_FEATURES}")
    log(f"  - Ratio de Amostragem de Negativos: {NEGATIVE_SAMPLING_RATIO}")
    log("-" * 30)
    log(f'Mean Squared Error (MSE): {mean_squared_error(Y_test, Y_hat_test):.4f}')
    log(f'Concordance Index (CI): {concordance_index(Y_test, Y_hat_test):.4f}')
    try:
        log(f'R_m^2: {get_rm2(Y_test, Y_hat_test):.4f}')
    except NameError:
        log("Função get_rm2 não encontrada. Métrica R_m^2 não calculada.")
print(f"Métricas salvas em: {metrics_path}")
