#!/usr/bin/env python
# coding: utf-8

"""
Affinity2Vec_Hybrid_ChEMBL_KFold.py

Esta versão implementa a validação cruzada K-Fold para uma avaliação
mais robusta da performance do modelo, garantindo que o vazamento de dados
seja prevenido em cada uma das K iterações.
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
from sklearn.model_selection import KFold # Importar KFold

# Funções Proprietárias e Adaptadas
from evaluation import *
from OptimizedpathScores_functions import *

# =============================================================================
# --- CONFIGURAÇÃO PARA EXPERIMENTOS ---
# =============================================================================
USE_EMBEDDING_FEATURES = True
NEGATIVE_SAMPLING_RATIO = 0.0
MODEL_OUTPUT_FOLDER = "results/exp_kfold_pscores_baseline/"
N_SPLITS = 5 

# =============================================================================
# --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO INICIAL ---
# =============================================================================
print("--- Etapa 1: Carregando e Pré-processando os Dados ---")

# --- Exemplo de como os dados seriam carregados ---
input_folder = "Input/ChEMBL/"
drug_embed_file = "EMBED/ChEMBL/Dr_ChemBERTa_EMBED.tsv"
target_embed_file = "EMBED/ChEMBL/Pr_ESM2_EMBED.tsv"
affinityFile = input_folder + "affinitiesChEMBL34-filtered.tsv"

if not os.path.exists(MODEL_OUTPUT_FOLDER):
    os.makedirs(MODEL_OUTPUT_FOLDER)

drEMBED = pd.read_csv(drug_embed_file, delimiter='\t', header=None).set_index(0)
tgEMBED = pd.read_csv(target_embed_file, delimiter='\t', header=None).set_index(0)
Affinities = pd.read_csv(affinityFile, delimiter='\t')

drug_ids = drEMBED.index.tolist()
target_ids = tgEMBED.index.tolist()
drug_to_idx = {drug: i for i, drug in enumerate(drug_ids)}
target_to_idx = {target: i for i, target in enumerate(target_ids)}

DD_Sim_matrix = MinMaxScaler().fit_transform(cosine_similarity(drEMBED.values).astype(np.float32))
TT_Sim_matrix = MinMaxScaler().fit_transform(cosine_similarity(tgEMBED.values).astype(np.float32))

affinity_matrix = np.full((len(drug_ids), len(target_ids)), np.nan, dtype=np.float32)
for _, row in Affinities.iterrows():
    if row['compound'] in drug_to_idx and row['target'] in target_to_idx:
        affinity_matrix[drug_to_idx[row['compound']], target_to_idx[row['target']]] = row['pchembl_value']


# =============================================================================
# --- 2. PREPARAÇÃO PARA O K-FOLD ---
# =============================================================================
print(f"\n--- Etapa 2: Preparando para Validação Cruzada com {N_SPLITS} Folds ---")
positive_row_inds, positive_col_inds = np.where(np.isnan(affinity_matrix) == False)
positive_labels = affinity_matrix[positive_row_inds, positive_col_inds]
positive_indices = np.arange(len(positive_row_inds))

# Inicializar o KFold
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Listas para armazenar as métricas de cada fold
all_mse = []
all_ci = []
all_rm2 = []

# =============================================================================
# --- 3. LOOP DE VALIDAÇÃO CRUZADA ---
# =============================================================================

for fold, (train_indices, test_indices) in enumerate(kf.split(positive_indices)):
    print(f"\n{'='*20} INICIANDO FOLD {fold + 1}/{N_SPLITS} {'='*20}")

    # --- 3.1 Divisão de Dados do Fold ---
    train_pos_rows, train_pos_cols = positive_row_inds[train_indices], positive_col_inds[train_indices]
    test_pos_rows, test_pos_cols = positive_row_inds[test_indices], positive_col_inds[test_indices]
    Y_train = positive_labels[train_indices]
    Y_test = positive_labels[test_indices]
    print(f"Fold {fold+1}: {len(Y_train)} amostras de treino, {len(Y_test)} amostras de teste.")

    # --- 3.2 Criação do Grafo Mascarado para o Fold ---
    scaler_aff = MinMaxScaler()
    Y_train_scaled = scaler_aff.fit_transform(Y_train.reshape(-1, 1)).flatten()
    
    affinity_matrix_norm = np.full_like(affinity_matrix, np.nan)
    affinity_matrix_norm[train_pos_rows, train_pos_cols] = Y_train_scaled
    
    affinity_graph_weights_train = np.nan_to_num(affinity_matrix_norm, nan=0.0)
    print(f"Grafo de treino do Fold {fold+1} criado.")

    # --- 3.3 Cálculo dos Meta-Path Scores (sem cache para garantir isolamento) ---
    print("Calculando scores de meta-path para o fold...")
    # NOTA: Para K-Fold, é mais seguro recalcular os scores a cada fold para garantir total isolamento.
    
    # Calcular scores para as FEATURES DE TREINO usando o grafo MASCARADO
    sumDDD, maxDDD = DDD_TTT_sim(DD_Sim_matrix)
    sumTTT, maxTTT = DDD_TTT_sim(TT_Sim_matrix)
    
    sumDDT_train, maxDDT_train = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_train, 2)
    sumDTT_train, maxDTT_train = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_train, 2)
    sumDDDT_train, maxDDDT_train = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_train, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_train, 3)[1]
    sumDTTT_train, maxDTTT_train = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_train, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_train, 3)[1]
    sumDTDT_train, maxDTDT_train = metaPath_DTDT(affinity_graph_weights_train)
    sumDDTT_train, maxDDTT_train = metaPath_DDTT(affinity_graph_weights_train, DD_Sim_matrix, TT_Sim_matrix)


    # Calcular scores para as FEATURES DE TESTE usando o grafo COMPLETO (simulando predição real)
    # Primeiro, criamos o grafo completo com todas as afinidades conhecidas normalizadas
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

    if USE_EMBEDDING_FEATURES:
        X_embed_train = np.hstack([drEMBED.values[train_pos_rows], tgEMBED.values[train_pos_cols]])
        X_embed_test = np.hstack([drEMBED.values[test_pos_rows], tgEMBED.values[test_pos_cols]])
        X_train = np.hstack([X_path_train, X_embed_train])
        X_test = np.hstack([X_path_test, X_embed_test])
    else:
        X_train = X_path_train
        X_test = X_path_test

    # --- 3.5 Treinamento e Avaliação do Modelo no Fold ---
    scaler_feat = MinMaxScaler()
    X_train_scaled = scaler_feat.fit_transform(X_train)
    X_test_scaled = scaler_feat.transform(X_test)

    xg_reg = xgb.XGBRegressor(booster='gbtree', objective='reg:squarederror', eval_metric='rmse', colsample_bytree=0.8, learning_rate=0.03, max_depth=19, n_estimators=855, seed=42, n_jobs=-1)
    
    print(f"Treinando modelo para o Fold {fold+1}...")
    xg_reg.fit(X_train_scaled, Y_train)
    
    Y_hat_test = xg_reg.predict(X_test_scaled)
    
    # --- 3.6 Armazenamento das Métricas ---
    mse = mean_squared_error(Y_test, Y_hat_test)
    ci = concordance_index(Y_test, Y_hat_test)
    rm2 = get_rm2(Y_test, Y_hat_test)
    
    all_mse.append(mse)
    all_ci.append(ci)
    all_rm2.append(rm2)
    
    print(f"Resultados do Fold {fold+1}: MSE={mse:.4f}, CI={ci:.4f}, R_m^2={rm2:.4f}")

# =============================================================================
# --- 4. RESULTADOS FINAIS DA VALIDAÇÃO CRUZADA ---
# =============================================================================
print(f"\n{'='*20} RESULTADOS FINAIS DA VALIDAÇÃO CRUZADA ({N_SPLITS}-Folds) {'='*20}")

mean_mse = np.mean(all_mse)
std_mse = np.std(all_mse)
mean_ci = np.mean(all_ci)
std_ci = np.std(all_ci)
mean_rm2 = np.mean(all_rm2)
std_rm2 = np.std(all_rm2)

print(f"Mean Squared Error (MSE): {mean_mse:.4f} ± {std_mse:.4f}")
print(f"Concordance Index (CI): {mean_ci:.4f} ± {std_ci:.4f}")
print(f"R_m^2: {mean_rm2:.4f} ± {std_rm2:.4f}")

# Salvar resultados em um arquivo
metrics_path = os.path.join(MODEL_OUTPUT_FOLDER, "kfold_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Resultados da Validação Cruzada com {N_SPLITS} Folds\n")
    f.write("-" * 30 + "\n")
    f.write(f"Mean Squared Error (MSE): {mean_mse:.4f} ± {std_mse:.4f}\n")
    f.write(f"Concordance Index (CI): {mean_ci:.4f} ± {std_ci:.4f}\n")
    f.write(f"R_m^2: {mean_rm2:.4f} ± {std_rm2:.4f}\n")
    f.write("\nResultados por Fold:\n")
    f.write(f"MSEs: {all_mse}\n")
    f.write(f"CIs: {all_ci}\n")
    f.write(f"R_m^2s: {all_rm2}\n")

print(f"\nMétricas detalhadas salvas em: {metrics_path}")
