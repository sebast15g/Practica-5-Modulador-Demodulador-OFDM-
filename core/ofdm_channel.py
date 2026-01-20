import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True


# ============================================================
# 1) CANAL IDEAL
# ============================================================

def channel_ideal(x):
    return x.copy()


# ============================================================
# 2) CANAL AWGN
# ============================================================

def channel_awgn(x, snr_db):
    Px = np.mean(np.abs(x)**2)
    snr_lin = 10**(snr_db/10)
    noise_var = Px / snr_lin

    noise = np.sqrt(noise_var/2) * (
        np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape)
    )
    return x + noise


# ============================================================
# 3) CANAL RAYLEIGH – ITU PEDESTRIAN A (NO CRÍTICO)
# ============================================================

def channel_rayleigh_pedA(x, fs):
    """
    ITU Pedestrian A suavizado
    - Delay spread pequeño
    - Bc >> delta_f
    → Canal no crítico para OFDM
    """

    #delays_ns = np.array([0, 300, 700, 1200, 2300, 3700])   # ns
    #powers_db = np.array([0, -1, -9, -10, -15, -20])      # dB

    delays_ns = np.array([0, 300, 700, 1200, 2200, 3700])   # ns
    powers_db = np.array([0, -3, -9, -10, -15, -20])      # dB

    delays_s = delays_ns * 1e-9
    powers_lin = 10**(powers_db / 10)
    powers_lin /= np.sum(powers_lin)   # Normalización

    delays_samples = np.round(delays_s * fs).astype(int)
    L = len(delays_samples)

    h = np.zeros(delays_samples[-1] + 1, dtype=complex)

    g = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2)

    for i in range(L):
        h[delays_samples[i]] += g[i] * np.sqrt(powers_lin[i])

    y_full = np.convolve(x, h, mode="full")
    y = y_full[:len(x)]

    return y, h, delays_ns, powers_db

# ============================================================
# 3-1) CANAL RAYLEIGH - Vehicular A / Extended Pedestrian (NO CRÍTICO)
# ============================================================
def channel_rayleigh_critical(x, fs):
    """
    Canal Rayleigh selectivo CRÍTICO
    - Delay spread cercano al CP
    - Selectividad fuerte en frecuencia
    """

    delays_ns = np.array([0, 300, 700, 1200, 2200, 3700])   # ns
    powers_db = np.array([0, -3, -9, -10, -15, -20])      # dB

    delays_s = delays_ns * 1e-9
    powers_lin = 10**(powers_db / 10)
    powers_lin /= np.sum(powers_lin)

    delays_samples = np.round(delays_s * fs).astype(int)
    L = len(delays_samples)

    h = np.zeros(delays_samples[-1] + 1, dtype=complex)

    g = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2)

    for i in range(L):
        h[delays_samples[i]] += g[i] * np.sqrt(powers_lin[i])

    y = np.convolve(x, h, mode="full")[:len(x)]

    return y, h, delays_ns, powers_db


# ============================================================
# 4) GRÁFICAS DEL CANAL
# ============================================================

def plot_pdp(delays_ns, powers_db):
    plt.figure(figsize=(6,4))
    plt.stem(delays_ns, powers_db, basefmt=" ")
    plt.grid(True)
    plt.xlabel("Delay (ns)")
    plt.ylabel("Potencia promedio (dB)")
    plt.title("PDP – ITU Pedestrian A")
    plt.show()


def plot_frequency_response(h, fs, Nfft=8192):
    H = np.fft.fftshift(np.fft.fft(h, Nfft))
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))

    H_db = 20*np.log10(np.abs(H) + 1e-12)
    H_db -= np.max(H_db)

    plt.figure(figsize=(8,4))
    plt.plot(f/1e6, H_db)
    plt.grid(True)
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("Magnitud (dB)")
    plt.title("Respuesta en frecuencia – Canal Rayleigh Ped A")
    plt.ylim([-30, 5])
    plt.show()


def plot_time_signals(x, y, fs, Ns=3000):
    t = np.arange(Ns) / fs * 1e6
    plt.figure(figsize=(12,5))
    plt.plot(t, np.real(x[:Ns]), label="TX")
    plt.plot(t, np.real(y[:Ns]), label="Salida canal", alpha=0.8)
    plt.grid(True)
    plt.xlabel("Tiempo (µs)")
    plt.ylabel("Amplitud")
    plt.title("Señal temporal – antes y después del canal")
    plt.legend()
    plt.show()


# ============================================================
# 5) PRUEBA DEL CANAL CON OFDM REAL
# ============================================================

if __name__ == "__main__":

    print("\n===== PRUEBA CANAL + OFDM TX =====")

    # --------------------------------------------------------
    # 1) Parámetros OFDM
    # --------------------------------------------------------
    from ofdm_tx import OFDMConfig, build_ofdm_tx, load_rgb_image_to_bits

    M = 16                       # QPSK=4, 16QAM=16, 64QAM=64
    IMAGE_PATH = "img2.png"
    IMAGE_SIZE = (64, 64)

    cfg = OFDMConfig()
    fs = cfg.fs

    # --------------------------------------------------------
    # 2) Señal OFDM TX
    # --------------------------------------------------------
    bits, _ = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)

    X_all, x_all, num_syms, pad_bits = build_ofdm_tx(bits, cfg, M)

    # Señal temporal continua (como saldría del DAC)
    x_tx = np.concatenate(x_all)

    print(f"Símbolos OFDM TX : {num_syms}")
    print(f"Fs               : {fs/1e6:.2f} MHz")
    print(f"Longitud señal   : {len(x_tx)} muestras\n")

    # --------------------------------------------------------
    # 3) CANAL AWGN
    # --------------------------------------------------------
    snr_db = 20
    y_awgn = channel_awgn(x_tx, snr_db)

    plot_time_signals(x_tx, y_awgn, fs)

    # --------------------------------------------------------
    # 4) CANAL RAYLEIGH – ITU PEDESTRIAN A
    # --------------------------------------------------------
    y_ray, h, delays_ns, powers_db = channel_rayleigh_pedA(x_tx, fs)

    plot_pdp(delays_ns, powers_db)
    plot_frequency_response(h, fs)
    plot_time_signals(x_tx, y_ray, fs)
