# pathScores_functions.py (VERSÃO OTIMIZADA)
import numpy as np

def DDD_TTT_sim(simM):
    np.fill_diagonal(simM, 0)
    n = simM.shape[0]

    # Usar np.dot para a soma é mais eficiente
    sumM = np.dot(simM, simM)
    
    # O loop para o máximo já é razoavelmente otimizado em memória
    maxM = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # Multiplicação via broadcasting, a matriz temporária é apenas 2D
        temp_matrix = simM[i, :][:, np.newaxis] * simM
        maxM[i, :] = np.max(temp_matrix, axis=0)
         
    return sumM, maxM

def metaPath_Dsim_DT(Dsim, DT, length):
    np.fill_diagonal(Dsim, 0)
    num_drugs, num_targets = DT.shape

    # --- Otimização para sumM ---
    # Substitui o einsum ineficiente pela multiplicação de matriz padrão.
    sumM = np.dot(Dsim, DT)

    # --- Otimização para maxM ---
    # Usa um loop para evitar a criação da matriz 3D gigante.
    maxM = np.zeros((num_drugs, num_targets), dtype=np.float32)
    for i in range(num_drugs):
        # Broadcasting cria uma matriz 2D (3340, 1526), que é gerenciável.
        temp_matrix = Dsim[i, :][:, np.newaxis] * DT
        maxM[i, :] = np.max(temp_matrix, axis=0)
        
    return sumM, maxM

def metaPath_DT_Tsim(Tsim, DT, length):
    np.fill_diagonal(Tsim, 0)
    num_drugs, num_targets = DT.shape

    # --- Otimização para sumM ---
    # A ordem aqui é DT . Tsim.T, mas como Tsim é simétrica, Tsim.T = Tsim
    sumM = np.dot(DT, Tsim)

    # --- Otimização para maxM ---
    # Usa um loop para evitar a criação da matriz 3D gigante.
    maxM = np.zeros((num_drugs, num_targets), dtype=np.float32)
    for i in range(num_targets):
        # Broadcasting para o caminho D-T-T
        temp_matrix = DT[:, i][:, np.newaxis] * Tsim[i, :]
        # Precisamos adicionar isso ao resultado parcial
        if i == 0:
            maxM = temp_matrix
        else:
            # Acumula o máximo elemento a elemento
            maxM = np.maximum(maxM, temp_matrix)

    return sumM, maxM

def metaPath_DDTT(DT, Dsim, Tsim):
    sumDDT, maxDDT = metaPath_Dsim_DT(Dsim, DT, 3)
    sumDDTT, _ = metaPath_DT_Tsim(Tsim, sumDDT, 3)
    _, maxDDTT = metaPath_DT_Tsim(Tsim, maxDDT, 3)
    
    return sumDDTT, maxDDTT

def metaPath_DTDT(DT):
    TD = np.transpose(DT)
    # DD aqui é (drogas, drogas)
    DD = np.dot(DT, TD)
    sumM, maxM = metaPath_Dsim_DT(DD, DT, 3)
    
    return sumM, maxM