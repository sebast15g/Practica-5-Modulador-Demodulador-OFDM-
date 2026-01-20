from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox
)

from math import log2
import numpy as np

from gui_mpl import MplWidget
from core.ofdm_tx import (
    build_ofdm_tx,
    qam_mod,
    active_subcarrier_indices,
    pilot_subcarrier_indices,
    plot_total_spectrum_sincs,
    plot_sincs_adjacent,
    plot_two_symbols_time
)


class TxTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state

        # =====================================================
        # Carrusel de gráficas
        # =====================================================
        self.plot = MplWidget(width=8, height=5)
        self.current_plot = 0
        self.plot_funcs = []

        # =====================================================
        # Botones
        # =====================================================
        btn_run = QPushButton("Ejecutar TX")
        btn_prev = QPushButton("⬅")
        btn_next = QPushButton("➡")

        btn_run.clicked.connect(self.run_tx)
        btn_prev.clicked.connect(self.prev_plot)
        btn_next.clicked.connect(self.next_plot)

        nav = QHBoxLayout()
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        nav.addStretch()

        # =====================================================
        # INFO DIGITAL (barra inferior)
        # =====================================================
        self.lbl_bits = QLabel("Bits: -")
        self.lbl_bps = QLabel("Bits/QAM: -")
        self.lbl_qam = QLabel("QAM syms: -")
        self.lbl_ofdm_bits = QLabel("Bits/OFDM: -")
        self.lbl_ofdm = QLabel("OFDM syms: -")
        self.lbl_pad = QLabel("Padding: -")

        info = QHBoxLayout()
        for lbl in [
            self.lbl_bits, self.lbl_bps, self.lbl_qam,
            self.lbl_ofdm_bits, self.lbl_ofdm, self.lbl_pad
        ]:
            lbl.setStyleSheet("padding:4px")
            info.addWidget(lbl)

        info.addStretch()

        # =====================================================
        # Layout
        # =====================================================
        layout = QVBoxLayout()
        layout.addWidget(btn_run)
        layout.addWidget(self.plot, 1)
        layout.addLayout(nav)
        layout.addLayout(info)

        self.setLayout(layout)

    # =========================================================
    # Ejecutar TX
    # =========================================================
    def run_tx(self):
        try:
            if "bits_tx" not in self.state:
                QMessageBox.warning(
                    self,
                    "Sin configuración",
                    "Primero carga los parámetros en Configuración."
                )
                return

            bits = self.state["bits_tx"]
            cfg = self.state["cfg"]
            M = self.state["M"]

            # -------------------------------
            # TX OFDM
            # -------------------------------
            X_all, x_all, num_syms, pad_bits = build_ofdm_tx(bits, cfg, M)

            self.state.update({
                "X_all": X_all,
                "x_all": x_all
            })

            # -------------------------------
            # INFO DIGITAL
            # -------------------------------
            k = int(log2(M))
            bits_total = len(bits)
            num_qam = bits_total // k
            bits_ofdm = cfg.N_used * k

            self.lbl_bits.setText(f"Bits: {bits_total}")
            self.lbl_bps.setText(f"Bits/QAM: {k}")
            self.lbl_qam.setText(f"QAM syms: {num_qam}")
            self.lbl_ofdm_bits.setText(f"Bits/OFDM: {bits_ofdm}")
            self.lbl_ofdm.setText(f"OFDM syms: {num_syms}")
            self.lbl_pad.setText(f"Padding: {pad_bits}")

            # -------------------------------
            # Registrar plots
            # -------------------------------
            self.plot_funcs = [
                lambda: self.plot_constellation(bits, M, cfg),
                lambda: self.plot_discrete_spectrum(X_all[0], cfg),
                lambda: self.plot_adjacent_subcarriers(X_all[0], cfg),   # NUEVA
                lambda: self.plot_time_continuous(x_all),
                lambda: self.plot_two_symbols(x_all, cfg),               # NUEVA
                lambda: self.plot_tf_grid(X_all, cfg, max_syms=20),       # RECORTADA
                lambda: self.plot_continuous_spectrum(X_all[0], cfg)
            ]
            self.current_plot = 0
            self.plot_funcs[0]()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error TX", str(e))

    # =========================================================
    # Navegación carrusel
    # =========================================================
    def prev_plot(self):
        if not self.plot_funcs:
            return
        self.current_plot = (self.current_plot - 1) % len(self.plot_funcs)
        self.plot_funcs[self.current_plot]()

    def next_plot(self):
        if not self.plot_funcs:
            return
        self.current_plot = (self.current_plot + 1) % len(self.plot_funcs)
        self.plot_funcs[self.current_plot]()

    # =========================================================
    # GRÁFICAS TX
    # =========================================================
    def plot_constellation(self, bits, M, cfg):
        self.plot.clear()
        k = int(log2(M))
        a = qam_mod(bits[:cfg.N_used * k], M)
        self.plot.ax.scatter(a.real, a.imag, s=8)
        self.plot.ax.set_title("Constelación QAM")
        self.plot.ax.grid(True)
        self.plot.canvas.draw()

    def plot_discrete_spectrum(self, X0, cfg):
        self.plot.clear()
        f = np.fft.fftfreq(cfg.Nfft, d=1 / cfg.fs)
        Xs = np.fft.fftshift(X0)
        fs = np.fft.fftshift(f)

        pilot_idx = pilot_subcarrier_indices(cfg, 0)
        self.plot.ax.plot(fs / 1e6, np.abs(Xs), label="Datos")
        self.plot.ax.scatter(
            f[pilot_idx] / 1e6,
            np.abs(X0[pilot_idx]),
            color="red",
            label="Pilotos"
        )

        self.plot.ax.set_title("Espectro discreto |X[k]|")
        self.plot.ax.set_xlabel("Frecuencia (MHz)")
        self.plot.ax.grid(True)
        self.plot.ax.legend()
        self.plot.canvas.draw()

    def plot_time_continuous(self, x_all):
        self.plot.clear()
        x = np.concatenate(x_all[:3])
        self.plot.ax.plot(np.real(x))
        self.plot.ax.set_title("Señal OFDM continua (3 símbolos)")
        self.plot.ax.grid(True)
        self.plot.canvas.draw()

    def plot_tf_grid(self, X_all, cfg, max_syms=20):
        self.plot.clear()

        idx = active_subcarrier_indices(cfg)
        X_cut = X_all[:max_syms, :]   # ⬅ SOLO primeros símbolos

        TF = np.abs(X_cut[:, idx]).T
        self.plot.ax.imshow(TF, aspect="auto", origin="lower")

        for n in range(TF.shape[1]):
            p = pilot_subcarrier_indices(cfg, n)
            p_loc = np.searchsorted(idx, p)
            self.plot.ax.scatter([n]*len(p_loc), p_loc, c="red", s=6)

        self.plot.ax.set_title(f"Grid TF (primeros {max_syms} símbolos)")
        self.plot.canvas.draw()


    def plot_continuous_spectrum(self, X0, cfg):
        self.plot.clear()
        plot_total_spectrum_sincs(X0, cfg)
        self.plot.canvas.draw()

    def plot_adjacent_subcarriers(self, X0, cfg):
        self.plot.clear()
        plot_sincs_adjacent(X0, cfg, n_carriers=50)
        self.plot.canvas.draw()

    def plot_two_symbols(self, x_all, cfg):
        self.plot.clear()
        plot_two_symbols_time(x_all, cfg)
        self.plot.canvas.draw()
