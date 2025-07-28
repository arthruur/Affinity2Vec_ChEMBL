#!/usr/bin/env python
# coding: utf-8

"""
evaluate_model.py

Script para carregar um modelo Affinity2Vec pré-treinado e seus artefatos
para reavaliar seu desempenho no conjunto de teste original.

Uso:
    python evaluate_model.py /caminho/para/pasta/do/modelo/

A pasta do modelo deve conter:
- O modelo XGBoost treinado (*.ubj)
- O objeto scaler (*.pkl)
- O cache de scores (*.npz)
- Um arquivo 'config.json' com os parâmetros do treinamento.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from pickle import load
import matplotlib.pyplot as plt
import itertools

# Importar funções necessárias do seu projeto
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from evaluation import get_rm2  # Supondo que get_rm2 esteja em evaluation.py

def evaluate(model_dir):
    """
    Carrega um modelo pré-treinado e seus artefatos para reavaliar seu
    desempenho no conjunto de teste original.
    """
    # --- 1. Carregar Configuração ---
    print(f"--- 1. Carregando configuração de '{model_dir}' ---")
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Erro: 'config.json' não encontrado em '{model_dir}'.")
        print("Por favor, garanta que seu script de treinamento salve a configuração do modelo.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extrair parâmetros de configuração
    USE_EMBEDDING_FEATURES = config.get('use_embedding_features', True)
    NEGATIVE_SAMPLING_RATIO = config.get('negative_sampling_ratio', 0.0)
    SCORES_CACHE_FILE = config.get('scores_cache_path')
    MODEL_PATH = config.get('model_path')
    SCALER_PATH = config.get('scaler_path')

    # --- 2. Carregar Artefatos Principais ---
    print("--- 2. Carregando modelo, scaler e scores em cache ---")
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, SCORES_CACHE_FILE]):
        print("Erro: Um ou mais artefatos necessários (modelo, scaler ou cache de scores) estão faltando.")
        return

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    scaler = load(open(SCALER_PATH, 'rb'))
    cached_scores = np.load(SCORES_CACHE_FILE)
    print("Artefatos carregados com sucesso.")

    # --- 3. Reconstruir o Dataset (X e Y) ---
    # Esta seção espelha a preparação de dados do script de treinamento para
    # garantir que as matrizes X e Y geradas sejam idênticas.
    print("--- 3. Reconstruindo o dataset para isolar o conjunto de teste original ---")
    # Carregar dados base
    affinityFile = "Input/ChEMBL/affinitiesChEMBL34-filtered.tsv"
    drEMBED = pd.read_csv("EMBED/ChEMBL/Dr_seq2seq_EMBED.tsv", delimiter='\t', header=None).set_index(0)
    tgEMBED = pd.read_csv("EMBED/ChEMBL/Pr_ProtVec_EMBED.tsv", delimiter='\t', header=None).set_index(0)
    Affinities = pd.read_csv(affinityFile, delimiter='\t')

    drug_ids = drEMBED.index.tolist()
    target_ids = tgEMBED.index.tolist()
    drug_to_idx = {drug: i for i, drug in enumerate(drug_ids)}
    target_to_idx = {target: i for i, target in enumerate(target_ids)}

    affinity_matrix = np.full((len(drug_ids), len(target_ids)), np.nan, dtype=np.float32)
    for _, row in Affinities.iterrows():
        if row['compound'] in drug_to_idx and row['target'] in target_to_idx:
            affinity_matrix[drug_to_idx[row['compound']], target_to_idx[row['target']]] = row['pchembl_value']

    # Realizar a mesma amostragem do treinamento
    positive_row_inds, positive_col_inds = np.where(np.isnan(affinity_matrix) == False)
    positive_pairs = set(zip(positive_row_inds, positive_col_inds))

    if NEGATIVE_SAMPLING_RATIO > 0:
        # Usar uma semente fixa para reprodutibilidade da seleção de amostras negativas
        np.random.seed(42)
        all_possible_pairs = set(itertools.product(range(len(drug_ids)), range(len(target_ids))))
        negative_pairs_set = all_possible_pairs - positive_pairs
        num_negatives_to_sample = min(int(len(positive_pairs) * NEGATIVE_SAMPLING_RATIO), len(negative_pairs_set))
        sampled_negative_pairs_list = np.random.permutation(list(negative_pairs_set))[:num_negatives_to_sample]
        if len(sampled_negative_pairs_list) > 0:
            negative_row_inds, negative_col_inds = zip(*sampled_negative_pairs_list)
        else:
            negative_row_inds, negative_col_inds = [], []
        final_row_inds = np.concatenate([positive_row_inds, negative_row_inds])
        final_col_inds = np.concatenate([positive_col_inds, negative_col_inds])
        positive_labels = affinity_matrix[positive_row_inds, positive_col_inds]
        negative_labels = np.zeros(len(negative_row_inds), dtype=np.float32)
        Y = np.concatenate([positive_labels, negative_labels])
    else:
        final_row_inds, final_col_inds = positive_row_inds, positive_col_inds
        Y = affinity_matrix[final_row_inds, final_col_inds]

    # --- 4. Montar o Vetor de Features X ---
    print("--- 4. Montando o vetor de features completo X ---")
    path_scores_list = []
    for i in range(len(final_row_inds)):
        row, col = final_row_inds[i], final_col_inds[i]
        scores = (
            cached_scores['sumDDT'][row, col], cached_scores['maxDDT'][row, col],
            cached_scores['sumDTT'][row, col], cached_scores['maxDTT'][row, col],
            cached_scores['sumDDDT'][row, col], cached_scores['maxDDDT'][row, col],
            cached_scores['sumDTTT'][row, col], cached_scores['maxDTTT'][row, col],
            cached_scores['sumDDTT'][row, col], cached_scores['maxDDTT'][row, col],
            cached_scores['sumDTDT'][row, col], cached_scores['maxDTDT'][row, col]
        )
        path_scores_list.append(scores)
    X_path_scores = np.array(path_scores_list, dtype=np.float32)

    if USE_EMBEDDING_FEATURES:
        drug_embed_features = drEMBED.values[final_row_inds]
        target_embed_features = tgEMBED.values[final_col_inds]
        X = np.concatenate([X_path_scores, drug_embed_features, target_embed_features], axis=1)
    else:
        X = X_path_scores

    print(f"Reconstruído X com shape {X.shape} e Y com shape {Y.shape}")

    # --- 5. Reproduzir a Divisão do Conjunto de Teste ---
    print("--- 5. Reproduzindo a divisão treino/teste ---")
    # Usar o mesmo random_state garante que a divisão seja idêntica
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Conjunto de teste isolado com {len(X_test)} amostras.")

    # --- 6. Avaliar o Modelo ---
    print("--- 6. Avaliando o desempenho do modelo no conjunto de teste ---")
    X_test_scaled = scaler.transform(X_test)
    Y_hat_test = model.predict(X_test_scaled)

    # --- 7. Exibir Resultados ---
    print("\n" + "="*50)
    print("          RESULTADOS DA AVALIAÇÃO DO MODELO")
    print("="*50)
    print(f"Carregado de: '{model_dir}'")
    print("-" * 50)
    print(f"Configuração:")
    print(f"  - Features de Embedding Usadas: {USE_EMBEDDING_FEATURES}")
    print(f"  - Ratio de Amostragem de Negativos: {NEGATIVE_SAMPLING_RATIO}")
    print("-" * 50)

    mse = mean_squared_error(Y_test, Y_hat_test)
    ci = concordance_index(Y_test, Y_hat_test)
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Concordance Index (CI): {ci:.4f}')
    try:
        rm2 = get_rm2(Y_test, Y_hat_test)
        print(f'R_m^2: {rm2:.4f}')
    except (NameError, ImportError):
        print("R_m^2: Não calculado (função get_rm2 não encontrada).")
    print("="*50 + "\n")

    # Plotar resultados
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_test, Y_hat_test, c=Y_test, cmap='viridis', s=10, alpha=0.6)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
    plt.title(f"Reavaliação: Previsto vs. Real\n(Modelo: {os.path.basename(model_dir.strip('/'))})")
    plt.xlabel("Afinidades Reais (pchembl_value)")
    plt.ylabel("Afinidades Previstas")
    plt.colorbar(label='Valor Real de pchembl')
    plt.grid(True)
    plot_path = os.path.join(model_dir, "reevaluation_plot.png")
    plt.savefig(plot_path)
    print(f"Gráfico de predição salvo em: '{plot_path}'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Avaliar um modelo Affinity2Vec pré-treinado.")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Caminho para o diretório contendo o modelo treinado e seus artefatos (modelo, scaler, scores, config)."
    )
    args = parser.parse_args()
    evaluate(args.model_dir)
