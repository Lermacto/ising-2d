import numpy as np
from numba import njit, prange


@njit  # Para que numba compile la función
def h(S):  # Calcula la energía de la red en   el estado S
    H = 0
    for i in np.arange(S.shape[0]):
        for j in np.arange(S.shape[1]):
            H += -S[i, j] * (S[i - 1, j] + S[i, j - 1])
    return H / S.size  # Aca S.size ya nos dá la normalización por L^2


@njit
def calculate_dE(sij, i, j, S):
    # xx s3 xx
    # s4 s0 s2
    # xx s1 xx
    L = S.shape[0]
    upper_i, upper_j = (i + 1) % L, (
        j + 1
    ) % L  # Si i+1 = L -> (i+1)%L=0, esto fuerza cc periódicas
    s1, s2, s3, s4 = S[upper_i, j], S[i, upper_j], S[i - 1, j], S[i, j - 1]
    return 2 * sij * (s1 + s2 + s3 + s4)


@njit
def metropolis(S, prob):  # Aplica el algoritmo de Metropolis al estado S
    dm = 0
    de = 0
    # La consigna dice que hay que aplicar el algoritmo de metrópolis por cada
    # sitio en la red, es decir, L cuadrado veces.
    for n in range(S.size):
        i, j = np.random.choice(
            S.shape[0], 2
        )  # Elegimos 2 posiciones al azar en el rango [0,L)
        sij = S[i, j]  # Obtenemos el spin de esa posición
        opp_sij = -sij  # Obtenemos el spin opuesto al original
        dE_sij = calculate_dE(sij, i, j, S)
        # La diferencia de energía al transicionar de sij a opp_sij
        p = np.random.random(1)
        if (
            dE_sij <= 0
            or (dE_sij == 4 and p < prob[0])
            or (dE_sij == 8 and p < prob[1])
        ):
            S[i, j] = opp_sij
            de += dE_sij
            dm += opp_sij
    return S, dm / S.size, de / S.size


@njit
def cor(S: np.ndarray) -> np.ndarray:
    L = S.shape[0]
    cor_vec = np.zeros(L // 2)
    for r, _ in enumerate(cor_vec):
        cor_filas = np.zeros(L)
        for i in prange(L):
            fila = S[i]
            fila_shift = np.roll(fila, r)
            cor_filas[i] = np.sum(fila * fila_shift)
        cor_vec[r] = np.sum(cor_filas)
    return cor_vec / (L**2)


@njit
def metropolis2(S: np.ndarray, prob: np.ndarray):
    # Aplica el algoritmo de Metropolis al estado S
    # if N == S.size:
    #    return S, h(S), np.mean(S)
    dm = 0
    dc = np.zeros(S.shape[0] // 2)
    # La consigna dice que hay que aplicar el algoritmo de metrópolis por cada
    # sitio en la red, es decir, L cuadrado veces.
    for _ in range(S.size):
        i, j = np.random.choice(
            S.shape[0], 2
        )  # Elegimos 2 posiciones al azar en el rango [0,L)
        sij = S[i, j]  # Obtenemos el spin de esa posición
        opp_sij = -sij  # Obtenemos el spin opuesto al original
        # dE_sij = calculate_dE(opp_sij, i, j, S) - calculate_dE(
        #     sij, i, j, S
        # )
        dE_sij = calculate_dE(sij, i, j, S)
        # La diferencia de energía al transicionar de sij a opp_sij
        p = np.random.random(1)
        c_viejo = cor(S)  # [Optim] Levantar de cuenta anterior?
        if (
            dE_sij <= 0
            or (dE_sij == 4 and p < prob[0])
            or (dE_sij == 8 and p < prob[1])
        ):
            S[i, j] = opp_sij
            dm += opp_sij
            dc += (
                cor(S) - c_viejo
            )  # [Optim] Se podría calcular la contribución especifica de este spin
    return S, dm / S.size, dc / S.size
