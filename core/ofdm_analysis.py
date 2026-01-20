import numpy as np
import matplotlib.pyplot as plt

from ofdm_tx import OFDMConfig, build_ofdm_tx, load_rgb_image_to_bits
from ofdm_channel import channel_awgn, channel_rayleigh_critical
from ofdm_rx import ofdm_rx


# =============================================================================
# PAPR / CCDF
# =============================================================================

def compute_papr(x):
    power = np.abs(x)**2
    return np.max(power) / np.mean(power)


def papr_ccdf(x, num_blocks=1000):
    papr_vals = []
    block_len = len(x) // num_blocks

    for i in range(num_blocks):
        block = x[i*block_len:(i+1)*block_len]
        papr_vals.append(10*np.log10(compute_papr(block)))

    papr_vals = np.array(papr_vals)
    papr_range = np.linspace(np.min(papr_vals), np.max(papr_vals), 200)
    ccdf = [np.mean(papr_vals > p) for p in papr_range]

    return papr_range, ccdf


# =============================================================================
# BER vs SNR (AWGN)
# =============================================================================

def ber_vs_snr_awgn(bits_tx, cfg, M, snr_db_range, repeats=30):

    ber_avg = []

    X_all, x_all, _, _ = build_ofdm_tx(bits_tx, cfg, M)
    x_tx = np.concatenate(x_all)

    for snr_db in snr_db_range:
        ber_rep = []

        for _ in range(repeats):
            y = channel_awgn(x_tx, snr_db)
            bits_rx = ofdm_rx(
                y, cfg, M,
                channel_type="AWGN",
                snr_db=snr_db
            )[:len(bits_tx)]

            ber_rep.append(np.mean(bits_rx != bits_tx))

        ber_avg.append(np.mean(ber_rep))

    return np.array(ber_avg)


# =============================================================================
# BER vs SNR (Rayleigh + AWGN)
# =============================================================================

def ber_vs_snr_rayleigh(bits_tx, cfg, M, snr_db_range, repeats=30):

    ber_avg = []

    X_all, x_all, _, _ = build_ofdm_tx(bits_tx, cfg, M)
    x_tx = np.concatenate(x_all)

    for snr_db in snr_db_range:
        ber_rep = []

        for _ in range(repeats):
            y_ray, _, _, _ = channel_rayleigh_critical(x_tx, cfg.fs)
            y = channel_awgn(y_ray, snr_db)

            bits_rx = ofdm_rx(
                y, cfg, M,
                channel_type="RAYLEIGH_AWGN",
                snr_db=snr_db
            )[:len(bits_tx)]

            ber_rep.append(np.mean(bits_rx != bits_tx))

        ber_avg.append(np.mean(ber_rep))

    return np.array(ber_avg)


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    plt.rcParams["figure.autolayout"] = True

    # -------------------------------------------------------------------------
    # Parámetros
    # -------------------------------------------------------------------------
    snr_db_range = np.arange(0, 21, 1)
    mods = [4, 16, 64]

    IMAGE_PATH = "cuenca.png"
    IMAGE_SIZE = (936, 702)

    # -------------------------------------------------------------------------
    # Bits TX (imagen pequeña)
    # -------------------------------------------------------------------------
    bits_tx, _ = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)

    # -------------------------------------------------------------------------
    # Configuración OFDM
    # -------------------------------------------------------------------------
    cfg = OFDMConfig()

    # -------------------------------------------------------------------------
    # SIMULACIONES (SIN GRAFICAR)
    # -------------------------------------------------------------------------
    ber_awgn_results = {}
    ber_ray_results = {}
    papr_results = {}

    for M in mods:
        print(f"Simulando {M}-QAM...")

        ber_awgn_results[M] = ber_vs_snr_awgn(
            bits_tx, cfg, M, snr_db_range, repeats=1
        )

        ber_ray_results[M] = ber_vs_snr_rayleigh(
            bits_tx, cfg, M, snr_db_range, repeats=1
        )

        X_all, x_all, _, _ = build_ofdm_tx(bits_tx, cfg, M)
        x_tx = np.concatenate(x_all)
        papr_results[M] = papr_ccdf(x_tx, num_blocks=2000)

    # -------------------------------------------------------------------------
    # GRAFICADO FINAL
    # -------------------------------------------------------------------------

    # --- BER vs SNR – AWGN ---
    plt.figure(figsize=(10, 6))
    for M in mods:
        plt.semilogy(
            snr_db_range,
            ber_awgn_results[M],
            marker='o',
            label=f"AWGN – {M}-QAM"
        )

    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR – AWGN")
    plt.legend()

    # --- BER vs SNR – Rayleigh + AWGN ---
    plt.figure(figsize=(10, 6))
    for M in mods:
        plt.semilogy(
            snr_db_range,
            ber_ray_results[M],
            marker='x',
            label=f"Rayleigh+AWGN – {M}-QAM"
        )

    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR – Rayleigh + AWGN")
    plt.legend()

    # --- CCDF del PAPR ---
    plt.figure(figsize=(10, 6))
    for M in mods:
        papr_db, ccdf = papr_results[M]
        plt.semilogy(papr_db, ccdf, label=f"{M}-QAM")

    plt.grid(True, which="both")
    plt.xlabel("PAPR (dB)")
    plt.ylabel("CCDF")
    plt.title("CCDF del PAPR – OFDM")
    plt.legend()

    # Mostrar TODAS al final
    plt.show()
