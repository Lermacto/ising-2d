import numpy as np
from numba import njit


@njit  # Para que numba compile la función
def h(S: np.ndarray) -> float:  # Calcula la energía de la red en   el estado S
    H = 0
    for i in np.arange(S.shape[0]):
        for j in np.arange(S.shape[1]):
            H += -S[i, j] * (S[i - 1, j] + S[i, j - 1])
    return H / S.size  # Aca S.size ya nos dá la normalización por L^2


@njit
def calculate_dE(s0: int, i: int, j: int, S: np.ndarray) -> int:
    # xx s3 xx
    # s4 s0 s2
    # xx s1 xx
    L = S.shape[0]
    upper_i = (i + 1) % L  # Condición de periodicidad
    upper_j = (j + 1) % L  # Condición de periodicidad
    s1, s2, s3, s4 = S[upper_i, j], S[i, upper_j], S[i - 1, j], S[i, j - 1]
    return 2 * s0 * (s1 + s2 + s3 + s4)


@njit
def metropolis(S: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, float, float]:
    # Aplica el algoritmo de Metropolis al estado S
    dm = 0
    de = 0
    # La consigna dice que hay que aplicar el algoritmo de metrópolis por cada
    # sitio en la red, es decir, L cuadrado veces.
    for _ in range(S.size):
        i, j = np.random.choice(S.shape[0], 2)  # Las coordenadas al azar en [0, L)²
        sij = S[i, j]  # El spin de esa posición
        opp_sij = -sij  # El spin al que podría transicionar
        dE_sij = calculate_dE(sij, i, j, S)
        # La diferencia de energía al transicionar de sij a opp_sij
        p = np.random.random(1)
        if (  # El spin cambia su estado si:
            dE_sij <= 0  # La diferencia de energía es negativa o cero
            or (dE_sij == 4 and p < prob[0])  # Con prob[0] si dE = 4
            or (dE_sij == 8 and p < prob[1])  # Con prob[1] si dE = 8
        ):
            S[i, j] = opp_sij  # El spin cambia de estado
            de += dE_sij
            dm += opp_sij
    return S, dm / S.size, de / S.size


@njit
def termalizar_S(S: np.ndarray, prob: np.ndarray) -> np.ndarray:
    n = 0
    n_check = 100
    historic_de = np.zeros(n_check)
    while True:
        S, _, de = metropolis(S, prob)
        historic_de[n % n_check] = de
        n += 1
        sum_de = np.sum(historic_de)
        if n // n_check > 0:
            if np.abs(sum_de / (n_check)) < 1e-6:
                break
    return S


@njit
def metropolis_N_veces(
    S: np.ndarray, N: int, prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    M = np.zeros(N)  # Magnetización en función del paso
    E = np.zeros(N)  # Energía por particula en funcion del paso
    M[0] = np.sum(S) / S.size  # Promedio de spines
    E[0] = h(S)  # Energía por partícula
    for n in range(1, N):
        S, dm, de = metropolis(S, prob)
        M[n] = M[n - 1] + dm
        E[n] = E[n - 1] + de
    return E, M, S


@njit
def transicion_de_fase(
    T, N, S0
) -> tuple[float, float, float, float, np.ndarray]:

    beta = 1 / T
    dE = np.array([4, 8])  # únicos Delta_E positivos
    prob = np.exp(-beta * dE)

    S = termalizar_S(S0, prob)
    E, M = metropolis_N_veces(S, N, prob)

    avg_E = np.sum(E) / N
    avg_M = np.sum(np.abs(M)) / N

    std_E = np.sqrt(np.sum((E - avg_E) ** 2) / (N - 1))
    std_M = np.sqrt(np.sum((np.abs(M) - avg_M) ** 2) / (N - 1))

    return avg_E, avg_M, std_E, std_M, S


def cor(S: np.ndarray) -> np.ndarray:  # Usando FFT es más rápido que con @njit
    L = S.shape[0]
    S_hat = np.fft.fft(S, axis=1)
    cor = np.sum(np.fft.ifft(S_hat * S_hat.conj(), axis=1), axis=0)
    return np.real(cor[: L // 2]) / S.size


def metropolis2(
    S: np.ndarray,
    prob: np.ndarray,
    cor_S: np.ndarray = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    # Aplica el algoritmo de Metropolis al estado S
    if cor_S is None:  # Calculamos la correlación antes de evolucionar el sistema
        c_original = cor(S)
    else:
        c_original = cor_S
    S, dm, _ = metropolis(S, prob)
    dc = cor(S) - c_original  # Calculamos la correlación al finalizar la evolución
    return S, dm, dc / S.size
