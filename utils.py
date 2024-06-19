import numpy as np
from numba import njit


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
    for _ in range(S.size):
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


def cor(S: np.ndarray) -> np.ndarray:  # Usando FFT es más rápido que con @njit
    L = S.shape[0]
    S_hat = np.fft.fft(S, axis=1)
    cor = np.sum(np.fft.ifft(S_hat * S_hat.conj(), axis=1), axis=0)
    return np.real(cor[: L // 2]) / S.size


def metropolis2_fft(S: np.ndarray, prob: np.ndarray):
    # Aplica el algoritmo de Metropolis al estado S
    c_original = cor(S)
    # La consigna dice que hay que aplicar el algoritmo de metrópolis por cada
    # sitio en la red, es decir, L cuadrado veces.
    S, dm, _ = metropolis(S, prob)
    dc = cor(S) - c_original
    return S, dm, dc / S.size
