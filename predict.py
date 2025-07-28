# -*- coding: utf-8 -*-
"""
Script de Predição Final para o Modelo Affinity2Vec

Este script carrega um modelo Affinity2Vec treinado e seus artefatos para
prever a afinidade de ligação para novos pares de composto-alvo.

Para usar, configure as variáveis na seção 'CONFIGURAÇÃO DO USUÁRIO' abaixo.
"""

# Pacotes gerais
import pandas as pd
import numpy as np
from pickle import load
import os
import json

# Pacotes de ML
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# --- CONFIGURAÇÃO DO USUÁRIO ---
# Altere as variáveis nesta seção para apontar para o seu experimento e dados.
# =============================================================================

# 1. Especifique a pasta de resultados do modelo que você quer usar para a predição.
#    Esta pasta deve conter os arquivos 'model.ubj', 'scaler.pkl' e 'config.json'.
PASTA_DO_EXPERIMENTO = "results/exp1_baseline_hybrid/"

# 2. (Opcional) Forneça um caminho para um arquivo CSV com os pares a serem previstos.
#    O arquivo deve ter duas colunas: 'compound_id' e 'target_id'.
#    Se for None, o script usará uma lista de exemplos padrão.
ARQUIVO_DE_ENTRADA_CSV = None  # Ex: "dados/meus_compostos.csv"

# 3. (Opcional) Especifique um caminho para salvar os resultados em um arquivo CSV.
#    Se for None, os resultados serão apenas impressos no console.
ARQUIVO_DE_SAIDA_CSV = "resultados_predicao.csv"

# =============================================================================
# --- LÓGICA DO SCRIPT DE PREDIÇÃO ---
# (Não é necessário alterar abaixo desta linha)
# =============================================================================

def predict_affinity(model_dir, pairs_to_predict):
    """
    Carrega os artefatos de um modelo e prevê a afinidade para uma lista de pares.

    Args:
        model_dir (str): Caminho para a pasta do experimento do modelo.
        pairs_to_predict (list of tuples): Lista de tuplas (compound_id, target_id).

    Returns:
        pd.DataFrame: DataFrame com os pares e suas afinidades previstas.
    """
    print(f"--- Iniciando predição usando o modelo de '{model_dir}' ---")

    # --- 1. Validação e Construção de Caminhos ---
    # Assume uma estrutura de pastas padrão
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "modelChEMBL-xgboost.ubj")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scores_path = "results/meta_path_scores.npz" 
    embed_path = "EMBED/ChEMBL/"

    required_files = [config_path, model_path, scaler_path, scores_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"\nERRO: Arquivo essencial não encontrado: {f_path}")
            print("Verifique se os caminhos e a PASTA_DO_EXPERIMENTO estão corretos.")
            return None
            
    # --- 2. Carregamento da Configuração e Artefatos ---
    print("\nCarregando configuração e artefatos do modelo...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    USE_EMBEDDING_FEATURES = config.get("use_embedding_features", False)
    print(f"Configuração do modelo: Usar Embeddings = {USE_EMBEDDING_FEATURES}")

    xg_reg = xgb.XGBRegressor()
    xg_reg.load_model(model_path)
    scaler = load(open(scaler_path, 'rb'))
    print("Modelo e scaler carregados.")

    # --- 3. Carregamento dos Dados de Mapeamento e Scores ---
    print("\nCarregando dados de mapeamento e scores...")
    drEMBED = pd.read_csv(os.path.join(embed_path, 'Dr_seq2seq_EMBED.tsv'), delimiter='\t', header=None).set_index(0)
    tgEMBED = pd.read_csv(os.path.join(embed_path, 'Pr_ProtVec_EMBED.tsv'), delimiter='\t', header=None).set_index(0)
    drug_to_idx = {drug: i for i, drug in enumerate(drEMBED.index)}
    target_to_idx = {target: i for i, target in enumerate(tgEMBED.index)}

    loaded_scores = np.load(scores_path)
    score_matrices = {key: loaded_scores[key] for key in loaded_scores}
    print("Mapeamentos e scores carregados com sucesso.")

    # --- 4. Predição para os Pares Fornecidos ---
    print("\n--- Iniciando Predições ---")
    predictions = []
    for dr_id, tg_id in pairs_to_predict:
        if dr_id not in drug_to_idx or tg_id not in target_to_idx:
            status = "Composto ou Alvo não encontrado no banco de dados."
            print(f"AVISO: Par ({dr_id}, {tg_id}) pulado. {status}")
            predictions.append({'compound_id': dr_id, 'target_id': tg_id, 'predicted_pchembl': np.nan, 'status': status})
            continue

        dr_idx, tg_idx = drug_to_idx[dr_id], target_to_idx[tg_id]

        path_scores_vector = np.array([[
            score_matrices['sumDDT'][dr_idx, tg_idx], score_matrices['maxDDT'][dr_idx, tg_idx],
            score_matrices['sumDTT'][dr_idx, tg_idx], score_matrices['maxDTT'][dr_idx, tg_idx],
            score_matrices['sumDDDT'][dr_idx, tg_idx], score_matrices['maxDDDT'][dr_idx, tg_idx],
            score_matrices['sumDTTT'][dr_idx, tg_idx], score_matrices['maxDTTT'][dr_idx, tg_idx],
            score_matrices['sumDDTT'][dr_idx, tg_idx], score_matrices['maxDDTT'][dr_idx, tg_idx],
            score_matrices['sumDTDT'][dr_idx, tg_idx], score_matrices['maxDTDT'][dr_idx, tg_idx]
        ]], dtype=np.float32)

        if USE_EMBEDDING_FEATURES:
            drug_embed = drEMBED.values[dr_idx].reshape(1, -1)
            target_embed = tgEMBED.values[tg_idx].reshape(1, -1)
            feature_vector = np.concatenate([path_scores_vector, drug_embed, target_embed], axis=1)
        else:
            feature_vector = path_scores_vector
        
        feature_vector_scaled = scaler.transform(feature_vector)
        predicted_pchembl = xg_reg.predict(feature_vector_scaled)[0]
        
        status = "Sucesso"
        predictions.append({'compound_id': dr_id, 'target_id': tg_id, 'predicted_pchembl': float(predicted_pchembl), 'status': status})
        print(f"  - Par ({dr_id}, {tg_id}) -> Predição pChEMBL: {predicted_pchembl:.4f}")

    return pd.DataFrame(predictions)

# -- BLOCO PRINCIPAL DE EXECUÇÃO -- #
if __name__ == '__main__':
    # Definir os pares para teste
    if ARQUIVO_DE_ENTRADA_CSV and os.path.exists(ARQUIVO_DE_ENTRADA_CSV):
        try:
            print(f"\nLendo pares do arquivo: {ARQUIVO_DE_ENTRADA_CSV}")
            input_df = pd.read_csv(ARQUIVO_DE_ENTRADA_CSV)
            pares_para_teste = list(zip(input_df['compound_id'], input_df['target_id']))
        except Exception as e:
            print(f"ERRO ao ler o arquivo CSV de entrada: {e}")
            pares_para_teste = [] # Evita que o script continue com dados vazios
    else:
        print("\nNenhum arquivo de entrada válido fornecido. Usando lista de pares de exemplo...")
        pares_para_teste = [
            ('CHEMBL42', 'CHEMBL231'),   # Clozapina vs. H1 Receptor
            ('CHEMBL192', 'CHEMBL1801'),  # Sildenafil vs. PDE5
            ('CHEMBL118', 'CHEMBL230'),   # Celecoxib vs. COX-2
            ('CHEMBL113', 'CHEMBL231'),   # Cafeína vs. H1 Receptor
            ('COMPOSTO_INEXISTENTE', 'CHEMBL232'),
            ('CHEMBL42', 'ALVO_INEXISTENTE')
        ]
    
    if pares_para_teste:
        # Executar a predição
        results_df = predict_affinity(PASTA_DO_EXPERIMENTO, pares_para_teste)

        # Exibir e salvar os resultados
        if results_df is not None:
            print("\n--- Resumo Final das Predições ---")
            print(results_df.to_string())
            
            if ARQUIVO_DE_SAIDA_CSV:
                try:
                    output_dir = os.path.dirname(ARQUIVO_DE_SAIDA_CSV)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    results_df.to_csv(ARQUIVO_DE_SAIDA_CSV, index=False, float_format='%.4f')
                    print(f"\nResultados salvos com sucesso em: {ARQUIVO_DE_SAIDA_CSV}")
                except Exception as e:
                    print(f"\nERRO ao salvar o arquivo CSV de saída: {e}")
    else:
        print("\nNenhum par para predição foi fornecido. Encerrando o script.")

