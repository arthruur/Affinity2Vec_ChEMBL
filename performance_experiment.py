import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import importlib
import os

def check_modules():
    """Verifica se os módulos necessários podem ser importados."""
    try:
        importlib.import_module("pathScores_functions")
        importlib.import_module("OptimizedpathScores_functions")
        importlib.import_module("seaborn")
        return True
    except ImportError as e:
        print(f"Erro de importação: {e}")
        print("Certifique-se de que os pacotes necessários (pandas, matplotlib, seaborn) estão instalados e que os arquivos .py estão no diretório correto.")
        return False

def load_data():
    """
    Carrega os dados necessários para o experimento.
    """
    try:
        fpath = 'Input/Davis/'
        dsim_file = "Input/Davis/drug-drug_similarities_2D.txt"
        psim_file = "Input/Davis/target-target_similarities_WS.txt"

        Dr_SimM = np.loadtxt(dsim_file, dtype=float, skiprows=0)
        Pr_SimM = np.loadtxt(psim_file, delimiter=" ", dtype=float, skiprows=0)

        affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
        aff = -np.log10(np.array(affinity, dtype=np.float64) / (10**9))
        train_aff_M = np.exp((-1) * aff)
        
        return Dr_SimM, Pr_SimM, train_aff_M

    except FileNotFoundError as e:
        print(f"Erro: Arquivo de dados não encontrado. Verifique a estrutura de pastas 'Input/Davis/'.")
        print(f"Detalhe do erro: {e}")
        return None, None, None

def run_single_test(test_function, *args):
    """
    Executa uma função de teste específica, medindo tempo e memória.
    """
    tracemalloc.start()
    start_time = time.time()
    
    result = test_function(*args)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time = end_time - start_time
    peak_memory_mb = peak / 10**6
    
    return execution_time, peak_memory_mb, result

def run_all_paths(path_module, Dr_SimM, Pr_SimM, train_aff_M):
    """Executa todas as funções de cálculo de caminho em sequência."""
    sumDDD, maxDDD = path_module.DDD_TTT_sim(Dr_SimM)
    sumTTT, maxTTT = path_module.DDD_TTT_sim(Pr_SimM)
    sumDDT, maxDDT = path_module.metaPath_Dsim_DT(Dr_SimM, train_aff_M, 2)
    sumDTT, maxDTT = path_module.metaPath_DT_Tsim(Pr_SimM, train_aff_M, 2)
    sumDDDT, _ = path_module.metaPath_Dsim_DT(sumDDD, train_aff_M, 3)
    _, maxDDDT = path_module.metaPath_Dsim_DT(maxDDD, train_aff_M, 3)
    sumDTTT, _ = path_module.metaPath_DT_Tsim(sumTTT, train_aff_M, 3)
    _, maxDTTT = path_module.metaPath_DT_Tsim(maxTTT, train_aff_M, 3)
    sumDTDT, maxDTDT = path_module.metaPath_DTDT(train_aff_M)
    sumDDTT, maxDDTT = path_module.metaPath_DDTT(train_aff_M, Dr_SimM, Pr_SimM)
    return None # Não precisamos dos resultados aqui

def main():
    """
    Função principal que executa os experimentos de comparação.
    """
    if not check_modules():
        return
        
    import pathScores_functions as original_ps
    import OptimizedpathScores_functions as optimized_ps

    print("Carregando dados para o experimento...")
    Dr_SimM, Pr_SimM, train_aff_M = load_data()
    
    if Dr_SimM is None: return

    N_ITERATIONS = 5
    modules_to_test = {"Original": original_ps, "Otimizada": optimized_ps}
    
    # --- PARTE 1: TESTE GERAL ---
    print("\n" + "="*50)
    print(" PARTE 1: Teste de Desempenho Geral (Todos os Caminhos)")
    print("="*50)
    
    general_results = []
    for version, module in modules_to_test.items():
        print(f"\n--- Testando versão: {version} ---")
        for i in range(N_ITERATIONS):
            exec_time, mem_peak, _ = run_single_test(run_all_paths, module, Dr_SimM.copy(), Pr_SimM.copy(), train_aff_M.copy())
            general_results.append({'version': version, 'execution_time': exec_time, 'peak_memory_mb': mem_peak})
            print(f"  Iteração {i+1}/{N_ITERATIONS} concluída.")

    df_general = pd.DataFrame(general_results)
    summary_general = df_general.groupby('version').agg(avg_time=('execution_time', 'mean'), avg_mem=('peak_memory_mb', 'mean')).reset_index()
    print("\n--- Resumo do Desempenho Geral ---")
    print(summary_general.to_string())
    
    # Salvar resultados gerais em TXT
    summary_general.to_csv('general_performance_summary.txt', sep='\t', index=False)
    print("\nResumo geral salvo em 'general_performance_summary.txt'")


    # --- PARTE 2: TESTE DETALHADO ---
    print("\n" + "="*50)
    print(" PARTE 2: Teste de Desempenho Detalhado (por Metacaminho)")
    print("="*50)
    
    detailed_results = []
    # Usamos os resultados do teste geral para os caminhos mais complexos
    sumDDD_opt, _ = optimized_ps.DDD_TTT_sim(Dr_SimM)
    sumTTT_opt, _ = optimized_ps.DDD_TTT_sim(Pr_SimM)

    metapaths_to_test = {
        "D-D-D / T-T-T": (lambda mod: mod.DDD_TTT_sim(Dr_SimM)),
        "D-T-T": (lambda mod: mod.metaPath_DT_Tsim(Pr_SimM, train_aff_M, 2)),
        "D-D-T": (lambda mod: mod.metaPath_Dsim_DT(Dr_SimM, train_aff_M, 2)),
        "D-T-D-T": (lambda mod: mod.metaPath_DTDT(train_aff_M)),
        "D-D-T-T": (lambda mod: mod.metaPath_DDTT(train_aff_M, Dr_SimM, Pr_SimM)),
        "D-D-D-T": (lambda mod: mod.metaPath_Dsim_DT(sumDDD_opt, train_aff_M, 3)),
        "D-T-T-T": (lambda mod: mod.metaPath_DT_Tsim(sumTTT_opt, train_aff_M, 3)),
    }

    for path_name, test_lambda in metapaths_to_test.items():
        print(f"\n--- Testando Metacaminho: {path_name} ---")
        for version, module in modules_to_test.items():
            for i in range(N_ITERATIONS):
                exec_time, mem_peak, _ = run_single_test(test_lambda, module)
                detailed_results.append({'metapath': path_name, 'version': version, 'execution_time': exec_time, 'peak_memory_mb': mem_peak})
            print(f"  Versão {version} concluída.")

    df_detailed = pd.DataFrame(detailed_results)
    summary_detailed = df_detailed.groupby(['metapath', 'version']).agg(avg_time=('execution_time', 'mean'), avg_mem=('peak_memory_mb', 'mean')).reset_index()
    print("\n--- Resumo do Desempenho por Metacaminho ---")
    print(summary_detailed.to_string())

    # Salvar resultados detalhados em TXT
    summary_detailed.to_csv('detailed_performance_summary.txt', sep='\t', index=False)
    print("\nResumo detalhado salvo em 'detailed_performance_summary.txt'")


    # --- PARTE 3: VISUALIZAÇÃO ---
    print("\nGerando gráficos...")
    sns.set_theme(style="whitegrid")

    # Gráfico Geral - Tempo
    plt.figure(figsize=(8, 6))
    ax_general_time = sns.barplot(data=summary_general, x='version', y='avg_time', palette=['skyblue', 'lightgreen'])
    ax_general_time.set_title('Desempenho Geral - Tempo de Execução', fontsize=14)
    ax_general_time.set_xlabel('Versão')
    ax_general_time.set_ylabel('Tempo Médio (segundos)')
    plt.tight_layout()
    plt.savefig('general_performance_time.png')
    print("- Gráfico 'general_performance_time.png' salvo.")
    plt.close()

    # Gráfico Geral - Memória
    plt.figure(figsize=(8, 6))
    ax_general_mem = sns.barplot(data=summary_general, x='version', y='avg_mem', palette=['skyblue', 'lightgreen'])
    ax_general_mem.set_title('Desempenho Geral - Uso de Memória', fontsize=14)
    ax_general_mem.set_xlabel('Versão')
    ax_general_mem.set_ylabel('Pico Médio de Memória (MB)')
    plt.tight_layout()
    plt.savefig('general_performance_memory.png')
    print("- Gráfico 'general_performance_memory.png' salvo.")
    plt.close()

    # Gráfico Detalhado - Tempo
    plt.figure(figsize=(12, 7))
    ax_detailed_time = sns.barplot(data=summary_detailed, x='metapath', y='avg_time', hue='version', palette=['skyblue', 'lightgreen'])
    ax_detailed_time.set_title('Desempenho Detalhado - Tempo de Execução por Metacaminho', fontsize=14)
    ax_detailed_time.set_xlabel('Metacaminho')
    ax_detailed_time.set_ylabel('Tempo Médio (segundos)')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig('detailed_performance_time.png')
    print("- Gráfico 'detailed_performance_time.png' salvo.")
    plt.close()

    # Gráfico Detalhado - Memória
    plt.figure(figsize=(12, 7))
    ax_detailed_mem = sns.barplot(data=summary_detailed, x='metapath', y='avg_mem', hue='version', palette=['skyblue', 'lightgreen'])
    ax_detailed_mem.set_title('Desempenho Detalhado - Uso de Memória por Metacaminho', fontsize=14)
    ax_detailed_mem.set_xlabel('Metacaminho')
    ax_detailed_mem.set_ylabel('Pico Médio de Memória (MB)')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig('detailed_performance_memory.png')
    print("- Gráfico 'detailed_performance_memory.png' salvo.")
    plt.close()

    print("\nTodos os arquivos foram gerados com sucesso!")


if __name__ == "__main__":
    main()
