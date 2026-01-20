import numpy as np
import matplotlib.pyplot as plt
from math import log2
from dataclasses import dataclass, field
from PIL import Image

plt.rcParams["figure.autolayout"] = True


# ============================================================
#  PARÁMETROS GENERALES 
# ============================================================

M = 16                                  # Orden de modulación (4, 16, 64...)
IMAGE_PATH = "cuenca.png"                # Imagen a usar
IMAGE_SIZE = (64, 64)                  # Tamaño para convertir en bits


# ============================================================
#  CONFIGURACIÓN OFDM (BW_total = 10 MHz, BW_util = 9 MHz)
# ============================================================

@dataclass
class OFDMConfig:
    bw_mhz: float = 10.0
    delta_f: float = 15e3
    guard_fraction: float = 0.10
    cp_time_us: float = 16.6

    N_used: int = field(init=False)
    Nfft: int = field(init=False)
    fs: float = field(init=False)
    T_u: float = field(init=False)
    T_cp: float = field(init=False)
    T_sym: float = field(init=False)
    Ts: float = field(init=False)
    cp_len: int = field(init=False)

    BW_total: float = field(init=False)
    BW_util: float = field(init=False)

    def __post_init__(self):

        self.BW_total = self.bw_mhz * 1e6
        self.BW_util  = self.BW_total * (1 - self.guard_fraction)

        self.N_used = int(self.BW_util // self.delta_f)
        if self.N_used % 2 == 1:
            self.N_used -= 1

        Nfft_min = int(np.ceil(self.BW_total / self.delta_f))
        self.Nfft = 1 << int(np.ceil(np.log2(Nfft_min)))

        self.fs = self.Nfft * self.delta_f
        self.T_u = 1 / self.delta_f
        self.T_cp = self.cp_time_us * 1e-6
        self.cp_len = int(round(self.T_cp * self.fs))
        self.T_sym = self.T_u + self.T_cp
        self.Ts = 1 / self.fs

        print("\n========= CONFIGURACION OFDM =========")
        print(f"BW TOTAL            : {self.BW_total/1e6:.3f} MHz")
        print(f"BW UTIL             : {self.BW_util/1e6:.3f} MHz")
        print(f"Guard Fraction      : {self.guard_fraction*100:.1f}%")
        print(f"delta_f             : {self.delta_f/1e3:.3f} kHz")
        print(f"N_used (SC activas) : {self.N_used}")
        print(f"Nfft                : {self.Nfft}")
        print(f"Fs                  : {self.fs/1e6:.4f} MHz")
        print(f"T_u                 : {self.T_u*1e6:.2f} us")
        print(f"T_cp                : {self.T_cp*1e6:.2f} us")
        print(f"T_symbol            : {self.T_sym*1e6:.2f} us")
        print(f"CP samples          : {self.cp_len}")
        print("=======================================\n")


# ============================================================
# SUBCARRIERS
# ============================================================

def active_subcarrier_indices(cfg):
    Nc = cfg.N_used
    half = Nc // 2
    return np.concatenate([
        np.arange(1, half+1),
        np.arange(cfg.Nfft - half, cfg.Nfft)
    ])


# ============================================================
# Imagen → Bits
# ============================================================

def load_rgb_image_to_bits(path, resize_to):
    img = Image.open(path).convert("RGB")
    img = img.resize(resize_to, Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits, arr.shape


# ============================================================
# Modulación QAM
# ============================================================

def qam_mod(bits, M):
    k = int(log2(M))
    bits = bits.reshape((-1, k))
    ints = bits.dot(2**np.arange(k-1, -1, -1))
    m = int(np.sqrt(M))
    I = 2*(ints % m) - (m-1)
    Q = 2*(ints // m) - (m-1)
    const = I + 1j*Q
    return const / np.sqrt((2/3)*(M-1))


# ============================================================
# OFDM TX
# ============================================================
def pilot_subcarrier_indices(cfg, sym_idx, spacing=6):
    """
    Devuelve los índices FFT de las subportadoras piloto
    siguiendo un patrón LTE-like en frecuencia y tiempo.
    """
    active_idx = active_subcarrier_indices(cfg)
    offset = 0 if (sym_idx % 2 == 0) else spacing // 2
    return active_idx[offset::spacing]

def build_ofdm_tx(bits, cfg, M):
    Nc = cfg.N_used
    k = int(log2(M))

    # Número de subportadoras de datos (NO pilotos)
    pilot_idx0 = pilot_subcarrier_indices(cfg, sym_idx=0)
    N_data = Nc - len(pilot_idx0)

    bps = N_data * k


    num_sym = int(np.ceil(len(bits) / bps))
    pad = num_sym*bps - len(bits)
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    bits = bits.reshape(num_sym, bps)

    X_all = []
    x_all = []

    idx = active_subcarrier_indices(cfg)

    for b in bits:
        a = qam_mod(b, M)
        X = np.zeros(cfg.Nfft, dtype=complex)
        # === DATOS ===
        pilot_idx = pilot_subcarrier_indices(cfg, sym_idx=len(X_all))
        data_idx = np.setdiff1d(idx, pilot_idx)

        # Ajustar número de símbolos QAM
        a_data = a[:len(data_idx)]

        X[data_idx] = a_data
        X[pilot_idx] = 1 + 0j   # piloto LTE-like

        x = np.fft.ifft(X)
        x_cp = np.concatenate([x[-cfg.cp_len:], x])
        X_all.append(X)
        x_all.append(x_cp)

    return np.array(X_all), np.array(x_all), num_sym, pad


# ============================================================
# Frecuencias reales
# ============================================================

def get_freq_mapping(cfg, X0):
    Nc = cfg.N_used
    idx = active_subcarrier_indices(cfg)

    half = Nc//2
    a_pos = X0[idx[:half]]
    a_neg = X0[idx[half:]]

    k_pos = np.arange(1, half+1)
    k_neg = -np.arange(half, 0, -1)

    a = np.concatenate([a_neg, a_pos])
    k = np.concatenate([k_neg, k_pos])
    f = k * cfg.delta_f
    return f, a


# ============================================================
# SINCS
# ============================================================

def plot_sincs_adjacent(X0, cfg, n_carriers=50, OS=64):

    f_sc, a_sc = get_freq_mapping(cfg, X0)
    Nc = len(f_sc)
    mid = Nc//2
    half = n_carriers//2
    idx = np.arange(mid-half, mid+half+1)

    f_sel = f_sc[idx]
    a_sel = a_sc[idx]

    df = cfg.delta_f
    f = np.linspace(f_sel[0]-5*df, f_sel[-1]+5*df, cfg.Nfft*OS)
    amp = np.max(np.abs(a_sc))

    plt.figure(figsize=(12,4))
    plt.title("Conjunto de Subportadoras (50 SC)")
    plt.grid(True)
    plt.xlabel("Frecuencia (kHz)")
    plt.ylabel("Magnitud")

    for fk, ak in zip(f_sel, a_sel):
        plt.plot(f/1e3, np.abs(ak*np.sinc((f - fk)/df))/amp)

    plt.show(block=False)



# ============================================================
# ESPECTRO CONTINUO
# ============================================================

def plot_total_spectrum_sincs(X0, cfg, OS=32):

    f_sc, a_sc = get_freq_mapping(cfg, X0)
    df = cfg.delta_f

    BWt = cfg.BW_total
    BWu = cfg.BW_util

    f = np.linspace(-BWt/2, BWt/2, cfg.Nfft*OS)
    S = np.zeros_like(f, dtype=complex)

    for fk, ak in zip(f_sc, a_sc):
        S += ak * np.sinc((f - fk)/df)

    mag = np.abs(S)
    mag /= mag.max()

    plt.figure(figsize=(14,4))
    plt.plot(f/1e6, mag)
    plt.grid(True)
    plt.title("Espectro OFDM continuo con banda de guarda")

    plt.axvspan(-BWu/2e6, BWu/2e6, color='green', alpha=0.20, label="Banda útil")
    plt.axvspan(-BWt/2e6, -BWu/2e6, color='red', alpha=0.15)
    plt.axvspan(BWu/2e6, BWt/2e6, color='red', alpha=0.15)

    plt.xlim([-8, 8])
    plt.xlabel("Frecuencia (MHz)")
    plt.legend()
    plt.show(block=False)



# ============================================================
# TIEMPO
# ============================================================

def plot_two_symbols_time(x_all, cfg):

    x1 = x_all[0]
    x2 = x_all[1]
    L = cfg.Nfft
    Cp = cfg.cp_len
    Ts = cfg.Ts

    CP_us = Cp * Ts * 1e6
    TU_us = L * Ts * 1e6
    TSYM_us = (Cp + L) * Ts * 1e6

    x1_u = x1[Cp:]
    x2_u = x2[Cp:]
    two_u = np.concatenate([x1_u, x2_u])
    t_u = np.arange(len(two_u)) * Ts * 1e6

    two = np.concatenate([x1, x2])
    t = np.arange(len(two)) * Ts * 1e6

    plt.figure(figsize=(15,7))

    plt.subplot(2,1,1)
    plt.plot(t_u, np.real(two_u))
    plt.grid(True)
    plt.title("Parte útil de dos símbolos OFDM")
    plt.axvline(TU_us, color='g')
    plt.axvline(2*TU_us, color='g')
    plt.xlim([0, 2*TU_us])

    plt.subplot(2,1,2)
    plt.plot(t, np.real(two))
    plt.grid(True)
    plt.title("Dos símbolos completos (CP + útil)")
    plt.xlabel("Tiempo (µs)")
    plt.axvline(CP_us, color='r')
    plt.axvline(CP_us + TU_us, color='g')
    plt.axvline(TSYM_us + CP_us, color='r')
    plt.axvline(TSYM_us + CP_us + TU_us, color='g')
    plt.xlim([0, 2*TSYM_us])

    plt.show(block=False)



# ============================================================
# MAIN
# ============================================================

def main():

    cfg = OFDMConfig()

    bits, _ = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)
    bits_total = len(bits)

    X_all, x_all, num_ofdm_syms, pad_bits = build_ofdm_tx(bits, cfg, M)
    X0 = X_all[0]

    Nc = cfg.N_used
    k = int(log2(M))
    bits_per_ofdm = Nc * k

    num_qam_syms = int(np.ceil(bits_total / k))

    print("========= INFO DIGITAL =========")
    print(f"Bits totales fuente        : {bits_total}")
    print(f"Bits por símbolo QAM       : {k}")
    print(f"Símbolos QAM totales       : {num_qam_syms}")
    print(f"Bits por símbolo OFDM      : {bits_per_ofdm}")
    print(f"Símbolos OFDM generados    : {num_ofdm_syms}")
    print(f"Bits de padding añadidos   : {pad_bits}")
    print("================================\n")

    # Bits
    plt.figure(figsize=(8,3))
    plt.step(np.arange(300), bits[:300])
    plt.grid(True)
    plt.title("Bits")
    plt.show(block=False)

    # Constelación
    a = qam_mod(bits[:Nc*k], M)
    plt.figure(figsize=(4,4))
    plt.scatter(np.real(a), np.imag(a))
    plt.grid(True)
    plt.title(f"Constelación M={M}")
    plt.show(block=False)

    # Espectro discreto
    fa = np.fft.fftfreq(cfg.Nfft, d=1/cfg.fs)
    Xs = np.fft.fftshift(X0)
    fs = np.fft.fftshift(fa)

    # pilotos (sin fftshift en índices)
    pilot_idx0 = pilot_subcarrier_indices(cfg, 0)
    pilot_freqs = fa[pilot_idx0]

    plt.figure(figsize=(12,4))
    plt.plot(fs/1e6, np.abs(Xs), label='Datos')

    plt.scatter(
        pilot_freqs/1e6,
        np.abs(X0[pilot_idx0]),
        color='red',
        label='Pilotos',
        zorder=3
    )

    plt.grid(True)
    plt.title("Espectro discreto |X[k]| (Pilotos correctos)")
    plt.xlim([-8, 8])
    plt.legend()
    plt.show(block=False)


    # Tiempo
    plot_two_symbols_time(x_all, cfg)

    # Grid TF
    TF = np.abs(X_all[:, active_subcarrier_indices(cfg)]).T

    plt.figure(figsize=(8,5))
    plt.imshow(TF, aspect='auto', origin='lower', cmap='viridis')

    for n in range(TF.shape[1]):
        p = pilot_subcarrier_indices(cfg, n)
        p_local = np.searchsorted(active_subcarrier_indices(cfg), p)
        plt.scatter([n]*len(p_local), p_local, c='red', s=10)

    plt.title("Grid TF (Pilotos en rojo)")
    plt.colorbar()
    plt.show(block=False)


    # Primeros 5 símbolos
    plt.figure(figsize=(8,5))
    plt.imshow(TF[:, :5], aspect='auto', origin='lower')
    plt.title("Grid TF (primeros 5 símbolos)")
    plt.colorbar()
    plt.show(block=False)

    # Sincs
    plot_sincs_adjacent(X0, cfg)

    # Espectro continuo
    plot_total_spectrum_sincs(X0, cfg)

    plt.show()


if __name__ == "__main__":
    main()
