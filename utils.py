import numpy as np
from numba import njit, prange
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm


@njit
def h(S: np.ndarray) -> float:
    """Calcula la energía (por partícula) de la red en el estado S."""
    H = 0
    for i in prange(S.shape[0]):
        for j in prange(S.shape[1]):
            H += -S[i, j] * (S[i - 1, j] + S[i, j - 1])
    return H / S.size  # Aca S.size ya nos dá la normalización por L^2


@njit
def calculate_dE(s0: int, i: int, j: int, S: np.ndarray) -> int:
    """Calcula la variación en la energía de la red al cambiar la proyección del spin \
en la posición (i, j).
    """
    L = S.shape[0]
    upper_i = (i + 1) % L  # Condición de periodicidad
    upper_j = (j + 1) % L  # Condición de periodicidad
    #    s3
    # s4 s0 s2
    #    s1
    s1, s2, s3, s4 = S[upper_i, j], S[i, upper_j], S[i - 1, j], S[i, j - 1]
    return 2 * s0 * (s1 + s2 + s3 + s4)


@njit
def metropolis(S: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Aplica L² pasos del algoritmo de Metropolis al estado S de la red de L×L spines \
usando las probabilidades provistas.
    """
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
            dm += 2 * opp_sij
    return S, dm / S.size, de / S.size


@njit
def termalize_S(S: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Aplica el algoritmo metropolis a una red S hasta que se alcanza el equilibrio \
térmico.
    """
    n = 0
    n_check = 100  # Tamaño de la ventana en la que miramos la variación de la energía
    # Nota: n_check es suficientemente chico como para que las variaciones espontaneas
    # del valor medio de M no afecten la condición de equilibrio.
    historic_de = np.zeros(n_check)
    historic_dm = np.zeros(n_check)
    while (  # Iterar mientras:
        (n < n_check)  # Hayamos hecho menos de n_check pasos
        # O si en los últimos n_check pasos:
        # las variaciones de E no se compensan
        or (np.abs(np.sum(historic_de) / (n_check)) > 1e-6)
        # las variaciones de M no se compensan
        or (np.abs(np.sum(historic_dm) / n_check) > 1e-6)
    ):
        S, dm, de = metropolis(S, prob)
        historic_de[n % n_check] = de
        historic_dm[n % n_check] = dm
        n += 1
    return S


def create_dense_domain_around_tc(
    temperatura_critica: float,
    n_puntos: int,
    dispersion: float = 0.5,
) -> np.ndarray:
    """Genera un array de temperaturas alrededor de la temperatura crítica distribuidos\
 de manera proporcional a una distribución gaussiana.
    """
    distribucion = norm(loc=temperatura_critica, scale=dispersion)  # Gaussiana
    # Genero un array de valores equiespaciados entre 0 y 1 para samplear la ICDF
    dominio_equiespaciado = np.linspace(1, 0, n_puntos + 2)[1:-1]
    # Evaluo la ICDF en los valores generados
    dominio_denso_en_tc = distribucion.isf(dominio_equiespaciado)
    return dominio_denso_en_tc


@njit
def do_sim_until_convergence(S: np.ndarray, prob: np.ndarray) -> int:
    """Devuelve el número de pasos montecarlo del algoritmo de Metropolis que \
fueron necesarios para que los promedios de la energía y magnetización de S converjan.
    """
    N_max = 50_000
    M = np.zeros(N_max)  # Magnetización en función del paso
    E = np.zeros(N_max)  # Energía por particula en funcion del paso
    M_avg = np.zeros_like(M)  # M_avg[i] = mean(|M[:i+1]|)
    E_avg = np.zeros_like(E)  # E_avg[i] = mean(E[:i+1])

    M[0] = np.sum(S) / S.size  # Promedio de spines
    E[0] = h(S)  # Energía por partícula
    M_avg[0] = np.abs(M[0])
    E_avg[0] = E[0]

    n = 1
    average_counter = 0
    while (n < N_max) and (average_counter < 500):
        S, dm, de = metropolis(S, prob)
        M[n] = M[n - 1] + dm
        E[n] = E[n - 1] + de
        M_avg[n] = np.mean(np.abs(M[: n + 1]))
        E_avg[n] = np.mean(E[: n + 1])
        # Si la diferencia absoluta entre este paso y el siguiente es menor a 10e-6
        # añado uno al contador de promedio
        # Si esto sucede 500 veces decimos que el sistema tiene un promedio estable
        if (abs(E_avg[n - 1] - E_avg[n]) < 1e-6) and (
            abs(M_avg[n - 1] - M_avg[n]) < 1e-6
        ):
            # Podría mejorarse pidiendo algo con respecto al error relativo
            average_counter += 1
        n += 1
    return n


def get_stable_ns_for_measurement(
    T_arr: np.ndarray,
    S0: list[np.ndarray],
) -> np.ndarray:
    n_promedios = np.zeros_like(T_arr)
    for ti in tqdm(range(T_arr.size), desc="T", total=T_arr.size):
        T = T_arr[ti]
        beta = 1 / T
        prob = np.exp(-beta * np.array([4, 8]))
        if T < 2.27:
            S = S0[0]  # T bajas
        else:
            S = S0[1]  # T altas
        S = termalize_S(S, prob)
        n_promedios[ti] = do_sim_until_convergence(S, prob)
    return n_promedios


def get_n_from_matrix(L: int, T: float) -> int:
    # Cargamos los N precalculados
    df_n = pd.read_csv("LT_matrix.csv", index_col=0, dtype=float)
    df_n.columns = [int(L_str) for L_str in df_n.columns]
    df_n = df_n.applymap(int)
    # Extraemos los rangos de T y L y los valores de N para cada par (T, L)
    L_vals = df_n.columns.to_numpy(dtype=float)
    T_vals = df_n.index.to_numpy(dtype=float)
    N_grid = df_n.to_numpy(dtype=int)
    # Buscamos los valores más cercanos:
    row_num = np.argmin(abs(T_vals - T))
    col_num = np.argmin(abs(L_vals - L))
    n_vecinos = 6
    # Promedio a n vecinos en T
    N_medio = np.mean(
        N_grid[max(row_num - n_vecinos // 2, 0) : row_num + n_vecinos // 2, col_num]
    )
    return int(np.round(N_medio))


@njit
def do_n_steps_of_simulation(S, N, prob):
    M = np.zeros(N)  # Magnetización en función del paso
    E = np.zeros(N)  # Energía por particula en funcion del paso
    M[0] = np.sum(S) / S.size  # Promedio de spines
    E[0] = h(S)  # Energía por partícula

    for n in range(1, N):
        S, dm, de = metropolis(S, prob)
        M[n] = M[n - 1] + dm
        E[n] = E[n - 1] + de

    return E, M


@njit
def phase_transition(T, n_stable, S0):
    beta = 1 / T
    dE = np.array([4, 8])  # únicos Delta_E positivos
    prob = np.exp(-beta * dE)

    # Termalizamos el sistema
    S = termalize_S(S0, prob)
    # Lo estabilizamos
    E, M = do_n_steps_of_simulation(S, n_stable, prob)

    avg_E = np.sum(E) / n_stable
    avg_M = np.sum(np.abs(M)) / n_stable

    var_E = np.sum((E - avg_E) ** 2) / (n_stable - 1)
    var_M = np.sum((np.abs(M) - avg_M) ** 2) / (n_stable - 1)

    return avg_E, avg_M, var_E, var_M, S


@njit(parallel=True)
def cor_paralela(S: np.ndarray, axis: int = 1) -> np.ndarray:
    L = S.shape[axis]
    cor_vec = np.ones(L // 2)
    for r in prange(1, L // 2):
        cor_tira = np.zeros(L)
        for i in prange(L):
            tira = S[i] if axis else S[:, i]
            tira_shift = np.roll(tira, r)
            cor_tira[i] = np.sum(tira * tira_shift)
        cor_vec[r] = np.sum(cor_tira)
    cor_vec[1:] /= L**2
    return cor_vec


def cor_fft(S: np.ndarray, axis: int = 1):
    L = S.shape[axis]
    S_hat = np.fft.fft(S, axis=axis)
    cor = np.sum(np.fft.ifft(S_hat * S_hat.conj(), axis=axis), axis=1 - axis)
    return np.real(cor[: L // 2]) / S.size


def metropolis2(S: np.ndarray, prob: np.ndarray, cor_S: np.ndarray = None):
    # Aplica el algoritmo de Metropolis al estado S
    if cor_S is None:
        c_original = cor_fft(S)
    else:
        c_original = cor_S
    # La consigna dice que hay que aplicar el algoritmo de metrópolis por cada
    # sitio en la red, es decir, L cuadrado veces.
    S, dm, _ = metropolis(S, prob)
    dc = cor_fft(S) - c_original
    return S, dm, dc


def funcion_correlacion_exponencial(r, A, C, xi):
    return A * np.exp(-r / xi) + C


def funcion_correlacion_ley_potencias(r, A, C, eta, xi):
    return A * np.exp(-r / xi) * (r / xi) ** (-eta) + C


def cuadratica(x, a, b, c):
    return a * x**2 + b * x + c


def lineal(x, a, b):
    return a * x + b
