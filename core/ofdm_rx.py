import numpy as np
import matplotlib.pyplot as plt
from math import log2
from PIL import Image

from ofdm_tx import (
    OFDMConfig,
    build_ofdm_tx,
    load_rgb_image_to_bits,
    active_subcarrier_indices,
    pilot_subcarrier_indices
)
from ofdm_channel import (
    channel_awgn,
    channel_rayleigh_pedA,
    channel_rayleigh_critical
)

plt.rcParams["figure.autolayout"] = True


# ============================================================
# QAM DEMOD
# ============================================================
def qam_constellation(M):
    m = int(np.sqrt(M))
    re = np.arange(-(m-1), m, 2)
    im = np.arange(-(m-1), m, 2)
    const = np.array([x + 1j*y for y in im for x in re])

    # Normalización (misma que usas en demod)
    const = const / np.sqrt((2/3)*(M-1))
    return const

def qam_demod(symbols, M):
    k = int(log2(M))
    m = int(np.sqrt(M))

    symbols = symbols * np.sqrt((2/3)*(M-1))

    I = np.real(symbols)
    Q = np.imag(symbols)

    I_idx = np.clip(np.round((I + (m-1)) / 2), 0, m-1)
    Q_idx = np.clip(np.round((Q + (m-1)) / 2), 0, m-1)

    ints = (Q_idx * m + I_idx).astype(int)

    bits = (((ints[:, None] & (1 << np.arange(k-1, -1, -1))) > 0)
            .astype(np.uint8))

    return bits.reshape(-1)


# ============================================================
# ESTIMACIÓN DE CANAL LS
# ============================================================

def estimate_channel_ls(Y, cfg, sym_idx):
    pilot_idx = pilot_subcarrier_indices(cfg, sym_idx)
    H_est = np.zeros(cfg.Nfft, dtype=complex)

    # Pilotos conocidos = 1
    H_est[pilot_idx] = Y[pilot_idx]

    data_idx = active_subcarrier_indices(cfg)
    H_est[data_idx] = np.interp(
        data_idx,
        pilot_idx,
        H_est[pilot_idx]
    )

    return H_est


# ============================================================
# CANAL REAL EN FRECUENCIA (Rayleigh)
# ============================================================

def true_channel_frequency(h, cfg):
    h_pad = np.zeros(cfg.Nfft, dtype=complex)
    h_pad[:len(h)] = h
    return np.fft.fft(h_pad)


# ============================================================
# ECUALIZADOR
# ============================================================

def equalize_zf(Y, H):
    return Y / (H + 1e-12)

# ============================================================
# RX OFDM
# ============================================================

def ofdm_rx(
    y_rx,
    cfg,
    M,
    channel_type="RAYLEIGH",
    snr_db=20,
    h_true=None,
    debug_plots=False,
    return_symbols=False   # 👈 NUEVO
):

    idx_active = active_subcarrier_indices(cfg)
    sym_len = cfg.Nfft + cfg.cp_len
    num_syms = len(y_rx) // sym_len

    bits_rx = []

    Px = np.mean(np.abs(y_rx)**2)
    noise_var = Px / (10**(snr_db/10))

    Y_before = None
    Y_after = None

    for n in range(num_syms):

        y_sym = y_rx[n*sym_len:(n+1)*sym_len]
        y_no_cp = y_sym[cfg.cp_len:]
        Y = np.fft.fft(y_no_cp)

        # -------------------------
        # Estimación / ecualización
        # -------------------------
        if "RAYLEIGH" in channel_type:
            H_est = estimate_channel_ls(Y, cfg, n)
            Yeq = equalize_zf(Y, H_est)

        else:  # IDEAL o AWGN simple
            H_est = np.ones(cfg.Nfft, dtype=complex)
            Yeq = Y

        if n == 0:
            Y_before = Y.copy()
            Y_after = Yeq.copy()
            
        # -------------------------
        # GRÁFICAS (solo símbolo 0)
        # -------------------------
        if debug_plots and n == 0:

            f = np.fft.fftshift(
                np.fft.fftfreq(cfg.Nfft, d=1/cfg.fs)
            )

            # ---- Canal real vs estimado (solo Rayleigh)
            if channel_type == "RAYLEIGH" or channel_type == "RAYLEIGH_AWGN" and h_true is not None:

                H_true = true_channel_frequency(h_true, cfg)

                const_ref = qam_constellation(M)

                plt.figure(figsize=(10,4))

                # ---- Antes de EQ
                plt.subplot(1,2,1)
                plt.scatter(np.real(Y[idx_active]),
                            np.imag(Y[idx_active]),
                            s=5, label="Recibido")

                plt.scatter(np.real(const_ref),
                            np.imag(const_ref),
                            color="red", s=40, marker=".",
                            label="Constelación")

                plt.title("Constelación antes de EQ")
                plt.grid(True)
                plt.legend()

                # ---- Después de EQ
                plt.subplot(1,2,2)
                plt.scatter(np.real(Yeq[idx_active]),
                            np.imag(Yeq[idx_active]),
                            s=5, label="Ecualizado")

                plt.scatter(np.real(const_ref),
                            np.imag(const_ref),
                            color="red", s=40, marker=".",
                            label="Constelación")

                plt.title("Constelación después de EQ")
                plt.grid(True)
                plt.legend()

                plt.show()


        pilot_idx = pilot_subcarrier_indices(cfg, n)
        data_idx = np.setdiff1d(idx_active, pilot_idx)

        bits_rx.append(qam_demod(Yeq[data_idx], M))

    bits_rx = np.concatenate(bits_rx)

    if return_symbols:
        return bits_rx, Y_before, Y_after
    else:
        return bits_rx



# ============================================================
# BITS → IMAGEN
# ============================================================

def bits_to_image(bits, shape):
    bits = bits[:np.prod(shape)*8]
    return np.packbits(bits).reshape(shape)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # =========================
    # CONFIGURACIÓN
    # =========================
    M = 64
    IMAGE_PATH = "cuenca.png"
    IMAGE_SIZE = (936, 702)

    CHANNEL = "RAYLEIGH"    # "IDEAL", "AWGN", "RAYLEIGH", "RAYLEIGH_AWGN"
    SNR_DB = 10

    # =========================
    # TX
    # =========================
    cfg = OFDMConfig()

    bits_tx, img_shape = load_rgb_image_to_bits(
        IMAGE_PATH, IMAGE_SIZE
    )

    X_all, x_all, num_syms, pad_bits = build_ofdm_tx(
        bits_tx, cfg, M
    )

    x_tx = np.concatenate(x_all)

    # =========================
    # CANAL
    # =========================
    h_true = None

    if CHANNEL == "IDEAL":
        y = x_tx.copy()
        h_true = None

    elif CHANNEL == "AWGN":
        y = channel_awgn(x_tx, SNR_DB)
        h_true = None

    elif CHANNEL == "RAYLEIGH":
        y, h_true, _, _ = channel_rayleigh_critical(x_tx, cfg.fs)

    elif CHANNEL == "RAYLEIGH_AWGN":
        y_ray, h_true, _, _ = channel_rayleigh_critical(x_tx, cfg.fs)
        y = channel_awgn(y_ray, SNR_DB)

    else:
        raise ValueError("Canal no válido")


    # =========================
    # RX SIN EQ
    # =========================
    bits_rx_no = ofdm_rx(
        y, cfg, M,
        channel_type="IDEAL",
        snr_db=SNR_DB
    )[:len(bits_tx)]

    # =========================
    # RX CON EQ
    # =========================
    bits_rx_yes = ofdm_rx(
        y, cfg, M,
        channel_type=CHANNEL,
        snr_db=SNR_DB,
        h_true=h_true,
        debug_plots=True
    )[:len(bits_tx)]

    # =========================
    # IMÁGENES
    # =========================
    img_tx = np.array(
        Image.open(IMAGE_PATH)
        .resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    )

    img_no = bits_to_image(bits_rx_no, img_shape)
    img_yes = bits_to_image(bits_rx_yes, img_shape)

    # =========================
    # VISUALIZACIÓN FINAL
    # =========================
    plt.figure(figsize=(16,5))

    plt.subplot(1,3,1)
    plt.imshow(img_tx)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(img_no)
    plt.title("RX sin EQ")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(img_yes)
    plt.title("RX con estimación + EQ")
    plt.axis("off")

    plt.suptitle(f"OFDM – Canal {CHANNEL} – SNR = {SNR_DB} dB")
    plt.show()
