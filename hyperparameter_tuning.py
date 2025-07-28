#!/usr/bin/env python
# coding: utf-8

"""
hyperparameter_tuning_gpu.py

Este script realiza a otimização de hiperparâmetros para o modelo XGBoost
utilizando a biblioteca Optuna. Ele aproveita a aceleração de GPU com CuPy
e a metodologia robusta de validação cruzada K-Fold para encontrar a
melhor combinação de parâmetros de forma eficiente.
"""

# Pacotes Gerais
import pandas as pd
import os
import json
import time

# Pacotes de Otimização
import optuna

# MUDANÇA PARA GPU: Importar NumPy e CuPy separadamente para clareza
import numpy as np
import cupy

# Pacotes de ML
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

# Funções Proprietárias
from evaluation import *
from OptimizedpathScores_functions import *

# =============================================================================
# --- CONFIGURAÇÃO PARA O EXPERIMENTO DE OTIMIZAÇÃO ---
# =============================================================================
# Número de combinações de hiperparâmetros a serem testadas pelo Optuna.
N_TRIALS = 50
# Número de folds para a validação cruzada dentro de cada trial.
N_SPLITS = 5
# Pasta para salvar os resultados da otimização.
MODEL_OUTPUT_FOLDER = "results/hyperparam_tuning_gpu/"
# Usar features de embedding?
USE_EMBEDDING_FEATURES = True

# =============================================================================
# --- FUNÇÃO OBJECTIVE (O CORAÇÃO DA OTIMIZAÇÃO) ---
# =============================================================================

def objective(trial, preloaded_data):
    """
    Esta função é chamada pelo Optuna em cada trial.
    Ela define os hiperparâmetros, treina o modelo com K-Fold e retorna
    a métrica de performance (MSE) que o Optuna tentará minimizar.
    """
    # --- 1. Desempacotar os dados pré-carregados ---
    (drEMBED_gpu, tgEMBED_gpu, DD_Sim_matrix, TT_Sim_matrix, affinity_matrix,
     positive_row_inds, positive_col_inds, positive_labels, positive_indices) = preloaded_data

    # --- 2. Definir o espaço de busca de hiperparâmetros ---
    # O Optuna irá sugerir valores para cada parâmetro a cada trial.
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',  # Usar GPU para o treinamento
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'seed': 42,
        'n_jobs': -1
    }

    all_mse = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # --- 3. Loop de Validação Cruzada (igual ao script anterior) ---
    for fold, (train_indices_cpu, test_indices_cpu) in enumerate(kf.split(cupy.asnumpy(positive_indices))):
        train_indices, test_indices = cupy.asarray(train_indices_cpu), cupy.asarray(test_indices_cpu)
        
        train_pos_rows, train_pos_cols = positive_row_inds[train_indices], positive_col_inds[train_indices]
        test_pos_rows, test_pos_cols = positive_row_inds[test_indices], positive_col_inds[test_indices]
        Y_train = positive_labels[train_indices]
        Y_test = positive_labels[test_indices]

        # Criação do grafo mascarado e cálculo dos scores...
        scaler_aff = MinMaxScaler()
        Y_train_scaled = cupy.asarray(scaler_aff.fit_transform(cupy.asnumpy(Y_train.reshape(-1, 1)))).flatten()
        affinity_matrix_norm = cupy.full_like(affinity_matrix, np.nan)
        affinity_matrix_norm[train_pos_rows, train_pos_cols] = Y_train_scaled
        affinity_graph_weights_train = cupy.nan_to_num(affinity_matrix_norm, nan=0.0)

        # Cálculo dos scores de meta-path para o treino
        sumDDD, maxDDD = DDD_TTT_sim(DD_Sim_matrix)
        sumTTT, maxTTT = DDD_TTT_sim(TT_Sim_matrix)
        sumDDT_train, maxDDT_train = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_train, 2)
        sumDTT_train, maxDTT_train = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_train, 2)
        sumDDDT_train, maxDDDT_train = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_train, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_train, 3)[1]
        sumDTTT_train, maxDTTT_train = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_train, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_train, 3)[1]
        sumDTDT_train, maxDTDT_train = metaPath_DTDT(affinity_graph_weights_train)
        sumDDTT_train, maxDDTT_train = metaPath_DDTT(affinity_graph_weights_train, DD_Sim_matrix, TT_Sim_matrix)

        # Cálculo dos scores de meta-path para o teste (usando grafo completo)
        all_pos_labels_scaled = cupy.asarray(MinMaxScaler().fit_transform(cupy.asnumpy(positive_labels.reshape(-1, 1)))).flatten()
        affinity_matrix_norm_full = cupy.full_like(affinity_matrix, np.nan)
        affinity_matrix_norm_full[positive_row_inds, positive_col_inds] = all_pos_labels_scaled
        affinity_graph_weights_full = cupy.nan_to_num(affinity_matrix_norm_full, nan=0.0)
        sumDDT_full, maxDDT_full = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_full, 2)
        sumDTT_full, maxDTT_full = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_full, 2)
        sumDDDT_full, maxDDDT_full = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_full, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_full, 3)[1]
        sumDTTT_full, maxDTTT_full = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_full, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_full, 3)[1]
        sumDTDT_full, maxDTDT_full = metaPath_DTDT(affinity_graph_weights_full)
        sumDDTT_full, maxDDTT_full = metaPath_DDTT(affinity_graph_weights_full, DD_Sim_matrix, TT_Sim_matrix)

        # Montagem dos vetores de features
        X_path_train = cupy.vstack([
            sumDDT_train[train_pos_rows, train_pos_cols], maxDDT_train[train_pos_rows, train_pos_cols],
            sumDTT_train[train_pos_rows, train_pos_cols], maxDTT_train[train_pos_rows, train_pos_cols],
            sumDDDT_train[train_pos_rows, train_pos_cols], maxDDDT_train[train_pos_rows, train_pos_cols],
            sumDTTT_train[train_pos_rows, train_pos_cols], maxDTTT_train[train_pos_rows, train_pos_cols],
            sumDTDT_train[train_pos_rows, train_pos_cols], maxDTDT_train[train_pos_rows, train_pos_cols],
            sumDDTT_train[train_pos_rows, train_pos_cols], maxDDTT_train[train_pos_rows, train_pos_cols]
        ]).T
        
        X_path_test = cupy.vstack([
            sumDDT_full[test_pos_rows, test_pos_cols], maxDDT_full[test_pos_rows, test_pos_cols],
            sumDTT_full[test_pos_rows, test_pos_cols], maxDTT_full[test_pos_rows, test_pos_cols],
            sumDDDT_full[test_pos_rows, test_pos_cols], maxDDDT_full[test_pos_rows, test_pos_cols],
            sumDTTT_full[test_pos_rows, test_pos_cols], maxDTTT_full[test_pos_rows, test_pos_cols],
            sumDTDT_full[test_pos_rows, test_pos_cols], maxDTDT_full[test_pos_rows, test_pos_cols],
            sumDDTT_full[test_pos_rows, test_pos_cols], maxDDTT_full[test_pos_rows, test_pos_cols]
        ]).T

        if USE_EMBEDDING_FEATURES:
            X_embed_train = cupy.hstack([drEMBED_gpu[train_pos_rows], tgEMBED_gpu[train_pos_cols]])
            X_embed_test = cupy.hstack([drEMBED_gpu[test_pos_rows], tgEMBED_gpu[test_pos_cols]])
            X_train = cupy.hstack([X_path_train, X_embed_train])
            X_test = cupy.hstack([X_path_test, X_embed_test])
        else:
            X_train = X_path_train
            X_test = X_path_test

        # Treinamento e Avaliação
        X_train_cpu, X_test_cpu = cupy.asnumpy(X_train), cupy.asnumpy(X_test)
        Y_train_cpu, Y_test_cpu = cupy.asnumpy(Y_train), cupy.asnumpy(Y_test)

        scaler_feat = MinMaxScaler()
        X_train_scaled = scaler_feat.fit_transform(X_train_cpu)
        X_test_scaled = scaler_feat.transform(X_test_cpu)

        # Instanciar o modelo com os parâmetros sugeridos pelo trial
        xg_reg = xgb.XGBRegressor(**param)
        xg_reg.fit(X_train_scaled, Y_train_cpu)
        Y_hat_test = xg_reg.predict(X_test_scaled)
        
        mse = mean_squared_error(Y_test_cpu, Y_hat_test)
        all_mse.append(mse)

    # --- 4. Retornar a métrica a ser otimizada ---
    # A média do MSE através dos K folds
    average_mse = np.mean(np.array(all_mse))
    print(f"Trial {trial.number} finalizado. MSE médio: {average_mse:.4f}")
    
    return average_mse

# =============================================================================
# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_OUTPUT_FOLDER):
        os.makedirs(MODEL_OUTPUT_FOLDER)

    # --- 1. Carregar e pré-processar os dados UMA VEZ ---
    print("--- Pré-carregando e processando todos os dados necessários... ---")
    input_folder = "Input/ChEMBL/"
    drug_embed_file = "EMBED/ChEMBL/Dr_ChemBERTa_EMBED.tsv"
    target_embed_file = "EMBED/ChEMBL/Pr_ESM2_EMBED.tsv"
    affinityFile = input_folder + "affinitiesChEMBL34-filtered.tsv"

    df_drEMBED = pd.read_csv(drug_embed_file, delimiter='\t', header=None).set_index(0)
    df_tgEMBED = pd.read_csv(target_embed_file, delimiter='\t', header=None).set_index(0)
    Affinities = pd.read_csv(affinityFile, delimiter='\t')

    drEMBED_gpu = cupy.array(df_drEMBED.values, dtype=cupy.float32)
    tgEMBED_gpu = cupy.array(df_tgEMBED.values, dtype=cupy.float32)

    drug_ids = df_drEMBED.index.tolist()
    target_ids = df_tgEMBED.index.tolist()
    drug_to_idx = {drug: i for i, drug in enumerate(drug_ids)}
    target_to_idx = {target: i for i, target in enumerate(target_ids)}

    def cosine_similarity_gpu(matrix):
        norm = cupy.sqrt(cupy.sum(matrix**2, axis=1, keepdims=True))
        normalized_matrix = matrix / norm
        return normalized_matrix @ normalized_matrix.T

    DD_Sim_matrix = cosine_similarity_gpu(drEMBED_gpu)
    TT_Sim_matrix = cosine_similarity_gpu(tgEMBED_gpu)

    scaler_sim = MinMaxScaler()
    DD_Sim_matrix = cupy.asarray(scaler_sim.fit_transform(cupy.asnumpy(DD_Sim_matrix)))
    TT_Sim_matrix = cupy.asarray(scaler_sim.fit_transform(cupy.asnumpy(TT_Sim_matrix)))

    affinity_matrix_cpu = np.full((len(drug_ids), len(target_ids)), float('nan'), dtype=np.float32)
    for _, row in Affinities.iterrows():
        if row['compound'] in drug_to_idx and row['target'] in target_to_idx:
            # CORREÇÃO: Usar target_to_idx para o alvo
            affinity_matrix_cpu[drug_to_idx[row['compound']], target_to_idx[row['target']]] = row['pchembl_value']
    affinity_matrix = cupy.asarray(affinity_matrix_cpu)

    positive_row_inds, positive_col_inds = cupy.where(cupy.isnan(affinity_matrix) == False)
    positive_labels = affinity_matrix[positive_row_inds, positive_col_inds]
    positive_indices = cupy.arange(len(positive_row_inds))
    
    # Empacotar todos os dados necessários para passar para a função objective
    preloaded_data = (
        drEMBED_gpu, tgEMBED_gpu, DD_Sim_matrix, TT_Sim_matrix, affinity_matrix,
        positive_row_inds, positive_col_inds, positive_labels, positive_indices
    )
    print("--- Dados pré-carregados com sucesso. Iniciando otimização... ---")

    # --- 2. Criar e rodar o estudo do Optuna ---
    study = optuna.create_study(direction='minimize')
    
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, preloaded_data), n_trials=N_TRIALS)
    end_time = time.time()

    print(f"\nOtimização concluída em {(end_time - start_time) / 60:.2f} minutos.")

    # --- 3. Exibir e salvar os melhores resultados ---
    best_trial = study.best_trial
    print("\n================ MELHOR TRIAL ENCONTRADO ================")
    print(f"  Valor (MSE Mínimo): {best_trial.value:.4f}")
    print("  Melhores Hiperparâmetros:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Salvar os resultados em um arquivo JSON
    results_path = os.path.join(MODEL_OUTPUT_FOLDER, "best_params.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_mse": best_trial.value,
            "best_params": best_trial.params
        }, f, indent=4)
        
    print(f"\nResultados da otimização salvos em: {results_path}")

