# Pacotes gerais
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
import os
import itertools

# Pacotes de ML
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Arquivos proprietários e adaptado por Duarte
from evaluation import *
from OptimizedpathScores_functions import *

# --- CONFIGURAÇÃO ---
MODEL_FOLDER = "ChEMBL_trained_model_Pscore_Corrigido/"
SCORES_CACHE_FILE = os.path.join(MODEL_FOLDER, "meta_path_scores_universal.npz")

# --- 1. CARREGAMENTO DOS DADOS ---
print("--- Etapa 1: Carregando os Dados ---")
input_folder = "Input/ChEMBL/" 
drugFile = input_folder+"compoundsChEMBL34_SMILESgreaterThan5.tsv"
targetFile= input_folder+"targetsChEMBL34_noRepo3D.tsv"
affinityFile= input_folder+"affinitiesChEMBL34-filtered.tsv"

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

drEMBED = pd.read_csv("EMBED/ChEMBL/Dr_seq2seq_EMBED.tsv", delimiter='\t', header=None).set_index(0)
tgEMBED = pd.read_csv("EMBED/ChEMBL/Pr_ProtVec_EMBED.tsv", delimiter='\t', header=None).set_index(0)
Affinities = pd.read_csv(affinityFile, delimiter='\t')
print(f"Embeddings carregados: {drEMBED.shape[0]} drogas e {tgEMBED.shape[0]} alvos.")


# --- 2. PRÉ-PROCESSAMENTO E CONSTRUÇÃO DAS MATRIZES ---
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


# --- 3. DIVISÃO DE DADOS (METODOLOGIA CORRETA E CONSISTENTE) ---
print("\n--- Etapa 3: Divisão de Dados ---")
# Identificar todos os pares com afinidade conhecida
known_row_inds, known_col_inds = np.where(np.isnan(affinity_matrix) == False)
known_labels = affinity_matrix[known_row_inds, known_col_inds]

# Criar um array de índices para facilitar a divisão
indices = np.arange(len(known_row_inds))

# Dividir os ÍNDICES em treino e teste para evitar data leakage. Esta é a ÚNICA divisão.
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Obter as coordenadas e labels para treino e teste
train_rows, train_cols = known_row_inds[train_indices], known_col_inds[train_indices]
test_rows, test_cols = known_row_inds[test_indices], known_col_inds[test_indices]
Y_train = known_labels[train_indices]
Y_test = known_labels[test_indices]

print(f"Dados divididos consistentemente:")
print(f" - Amostras de Treino: {len(Y_train)}")
print(f" - Amostras de Teste: {len(Y_test)}")


# --- 4. NORMALIZAÇÃO E CRIAÇÃO DO GRAFO MASCARADO ---
print("\n--- Etapa 4: Normalização e Criação do Grafo Mascarado ---")
scaler_aff = MinMaxScaler()
Y_train_scaled = scaler_aff.fit_transform(Y_train.reshape(-1, 1)).flatten()

affinity_matrix_norm = np.full_like(affinity_matrix, np.nan)
affinity_matrix_norm[train_rows, train_cols] = Y_train_scaled

# Criar o grafo de afinidade MASCARADO para o cálculo das features de TREINO
affinity_graph_weights_train = np.nan_to_num(affinity_matrix_norm, nan=0.0)
print(f"Grafo de treino criado. As {len(test_rows)} interações de teste não estão presentes.")


# --- 5. CÁLCULO DOS META-PATH SCORES ---
print("\n--- Etapa 5: Cálculo dos Meta-Path Scores ---")
# Calcular scores para as FEATURES DE TREINO usando o grafo MASCARADO
print("Calculando scores para o conjunto de treino (usando grafo mascarado)...")
sumDDD, maxDDD = DDD_TTT_sim(DD_Sim_matrix)
sumTTT, maxTTT = DDD_TTT_sim(TT_Sim_matrix)
sumDDT_train, maxDDT_train = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_train, 2)
sumDTT_train, maxDTT_train = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_train, 2)
sumDDDT_train, maxDDDT_train = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_train, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_train, 3)[1]
sumDTTT_train, maxDTTT_train = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_train, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_train, 3)[1]
sumDTDT_train, maxDTDT_train = metaPath_DTDT(affinity_graph_weights_train)
sumDDTT_train, maxDDTT_train = metaPath_DDTT(affinity_graph_weights_train, DD_Sim_matrix, TT_Sim_matrix)
print("Scores para treino calculados.")

# Calcular ou carregar scores UNIVERSAIS (para teste e predição futura)
if os.path.exists(SCORES_CACHE_FILE):
    print(f"Cache de scores universais encontrado. Carregando...")
    cached_scores = np.load(SCORES_CACHE_FILE)
    # CORREÇÃO: Carregar TODAS as matrizes de score do cache
    sumDDT_full, maxDDT_full = cached_scores['sumDDT'], cached_scores['maxDDT']
    sumDTT_full, maxDTT_full = cached_scores['sumDTT'], cached_scores['maxDTT']
    sumDDDT_full, maxDDDT_full = cached_scores['sumDDDT'], cached_scores['maxDDDT']
    sumDTTT_full, maxDTTT_full = cached_scores['sumDTTT'], cached_scores['maxDTTT']
    sumDDTT_full, maxDDTT_full = cached_scores['sumDDTT'], cached_scores['maxDDTT']
    sumDTDT_full, maxDTDT_full = cached_scores['sumDTDT'], cached_scores['maxDTDT']
    print("Todos os scores universais carregados do cache.")
else:
    print("Cache não encontrado. Calculando scores universais...")
    # Normalizar a matriz de afinidade completa para o cache
    affinity_matrix_norm_full = np.copy(affinity_matrix)
    affinity_matrix_norm_full[known_row_inds, known_col_inds] = scaler_aff.transform(known_labels.reshape(-1, 1)).flatten()
    affinity_graph_weights_full = np.nan_to_num(affinity_matrix_norm_full, nan=0.0)
    
    sumDDT_full, maxDDT_full = metaPath_Dsim_DT(DD_Sim_matrix, affinity_graph_weights_full, 2)
    sumDTT_full, maxDTT_full = metaPath_DT_Tsim(TT_Sim_matrix, affinity_graph_weights_full, 2)
    sumDDDT_full, maxDDDT_full = metaPath_Dsim_DT(sumDDD, affinity_graph_weights_full, 3)[0], metaPath_Dsim_DT(maxDDD, affinity_graph_weights_full, 3)[1]
    sumDTTT_full, maxDTTT_full = metaPath_DT_Tsim(sumTTT, affinity_graph_weights_full, 3)[0], metaPath_DT_Tsim(maxTTT, affinity_graph_weights_full, 3)[1]
    sumDTDT_full, maxDTDT_full = metaPath_DTDT(affinity_graph_weights_full)
    sumDDTT_full, maxDDTT_full = metaPath_DDTT(affinity_graph_weights_full, DD_Sim_matrix, TT_Sim_matrix)
    
    print(f"Salvando scores universais em cache: {SCORES_CACHE_FILE}")
    np.savez_compressed(SCORES_CACHE_FILE,
        sumDDT=sumDDT_full, maxDDT=maxDDT_full, sumDTT=sumDTT_full, maxDTT=maxDTT_full,
        sumDDDT=sumDDDT_full, maxDDDT=maxDDDT_full, sumDTTT=sumDTTT_full, maxDTTT=maxDTTT_full,
        sumDDTT=sumDDTT_full, maxDDTT=maxDDTT_full, sumDTDT=sumDTDT_full, maxDTDT=maxDTDT_full)


# --- 6. MONTAGEM DOS VETORES DE FEATURES (X_train, X_test) ---
print("\n--- Etapa 6: Montagem dos Vetores de Features Finais ---")
# Montar X_train com scores do grafo MASCARADO
X_train = np.vstack([
    sumDDT_train[train_rows, train_cols], maxDDT_train[train_rows, train_cols],
    sumDTT_train[train_rows, train_cols], maxDTT_train[train_rows, train_cols],
    sumDDDT_train[train_rows, train_cols], maxDDDT_train[train_rows, train_cols],
    sumDTTT_train[train_rows, train_cols], maxDTTT_train[train_rows, train_cols],
    sumDDTT_train[train_rows, train_cols], maxDDTT_train[train_rows, train_cols],
    sumDTDT_train[train_rows, train_cols], maxDTDT_train[train_rows, train_cols]
]).T

# Montar X_test com scores do grafo COMPLETO/UNIVERSAL
X_test = np.vstack([
    sumDDT_full[test_rows, test_cols], maxDDT_full[test_rows, test_cols],
    sumDTT_full[test_rows, test_cols], maxDTT_full[test_rows, test_cols],
    sumDDDT_full[test_rows, test_cols], maxDDDT_full[test_rows, test_cols],
    sumDTTT_full[test_rows, test_cols], maxDTTT_full[test_rows, test_cols],
    sumDDTT_full[test_rows, test_cols], maxDDTT_full[test_rows, test_cols],
    sumDTDT_full[test_rows, test_cols], maxDTDT_full[test_rows, test_cols]
]).T

print("Shape do X_train final:", X_train.shape)
print("Shape do X_test final:", X_test.shape)


# --- 7. TREINAMENTO E AVALIAÇÃO DO MODELO ---
print("\n--- Etapa 7: Treinamento e Avaliação do Modelo ---")
scaler_feat = MinMaxScaler()
X_train_scaled = scaler_feat.fit_transform(X_train)
X_test_scaled = scaler_feat.transform(X_test)
dump(scaler_feat, open(os.path.join(MODEL_FOLDER, "scaler_pscore_corrected.pkl"), 'wb'))
print("Features escalonadas com sucesso.")

model_path = os.path.join(MODEL_FOLDER, "modelChEMBL-xgboost_pscore_corrected.ubj")
xg_reg = xgb.XGBRegressor(
    booster='gbtree', objective='reg:squarederror', eval_metric='rmse',
    colsample_bytree=0.8, learning_rate=0.03, max_depth=19, scale_pos_weight=1, gamma=0,
    alpha=5, n_estimators=855, tree_method='auto', min_child_weight=5,
    seed=10, n_jobs=-1)

if os.path.exists(model_path):
    print(f"\nModelo encontrado em '{model_path}'. Carregando modelo, pulando treinamento.")
    xg_reg.load_model(model_path)
else:
    print("\nIniciando o treinamento do modelo XGBoost...")
    xg_reg.fit(X_train_scaled, Y_train)
    print("Treinamento concluído.")
    xg_reg.save_model(model_path)
    print(f"Modelo salvo em: {model_path}")

print("\nAvaliando o desempenho do modelo no conjunto de teste...")
Y_hat_test = xg_reg.predict(X_test_scaled)

plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_hat_test, color="blue", s=10, alpha=0.5, label='Previsto vs. Real')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Linha Ideal (y=x)')
plt.title("Affinity2Vec_Pscore Corrigido: Previsto vs. Real (ChEMBL Test Set)")
plt.xlabel("Afinidades Reais (pchembl_value)")
plt.ylabel("Afinidades Previstas")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_FOLDER, "prediction_plot_pscore_corrected.png"))
plt.show()

metrics_path = os.path.join(MODEL_FOLDER, "metrics_pscore_corrected.txt")
with open(metrics_path, "w") as f:
    def log(msg):
        print(msg)
        f.write(msg + "\n")
    log("\n--- Avaliação no Conjunto de Teste (Metodologia Corrigida) ---")
    log(f'Mean Squared Error (MSE): {mean_squared_error(Y_test, Y_hat_test):.4f}')
    log(f'Concordance Index (CI): {concordance_index(Y_test, Y_hat_test):.4f}')
    try:
        log(f'rm2: {get_rm2(Y_test, Y_hat_test):.4f}')
    except NameError:
        log("Função get_rm2 não encontrada. Métrica rm2 não calculada.")
print(f"Métricas de avaliação salvas em: {metrics_path}")
