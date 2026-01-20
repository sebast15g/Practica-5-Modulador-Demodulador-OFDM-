from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox
)

import numpy as np

from gui_mpl import MplWidget
from core.ofdm_tx import (
    OFDMConfig,
    build_ofdm_tx
)
from core.ofdm_rx import ofdm_rx
from core.ofdm_channel import channel_awgn


class AnalysisTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state

        self.plot = MplWidget(width=8, height=5)
        self.plot_funcs = []
        self.current_plot = 0

        btn_run = QPushButton("Ejecutar análisis")
        btn_prev = QPushButton("⬅")
        btn_next = QPushButton("➡")

        btn_run.clicked.connect(self.run_analysis)
        btn_prev.clicked.connect(self.prev_plot)
        btn_next.clicked.connect(self.next_plot)

        nav = QHBoxLayout()
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        nav.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(btn_run)
        layout.addWidget(self.plot, 1)
        layout.addLayout(nav)

        self.setLayout(layout)

    # =====================================================
    # ANÁLISIS
    # =====================================================
    def run_analysis(self):
        try:
            if "bits_tx" not in self.state:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Carga primero la configuración."
                )
                return

            bits_tx = self.state["bits_tx"]
            cfg = self.state["cfg"]

            snr_db_range = np.arange(0, 26, 2)
            modulations = [4, 16, 64]

            ber_results = {}
            papr_results = {}

            # =====================================
            # BARRIDO DE MODULACIONES
            # =====================================
            for M in modulations:

                # ---------- TX ----------
                X_all, x_all, _, _ = build_ofdm_tx(
                    bits_tx, cfg, M
                )
                x_tx = np.concatenate(x_all)

                # ---------- PAPR ----------
                papr = 10 * np.log10(
                    np.max(np.abs(x_tx)**2) /
                    np.mean(np.abs(x_tx)**2)
                )
                papr_results[M] = papr

                # ---------- BER (AWGN SIEMPRE) ----------
                ber = []

                for snr_db in snr_db_range:
                    y = channel_awgn(x_tx, snr_db)

                    bits_rx = ofdm_rx(
                        y, cfg, M,
                        channel_type="AWGN",
                        snr_db=snr_db
                    )[:len(bits_tx)]

                    ber.append(
                        np.mean(bits_rx != bits_tx)
                    )

                ber_results[M] = np.array(ber)

            # =====================================
            # CCDF PAPR
            # =====================================
            ccdf = {}
            papr_axis = np.linspace(0, 12, 200)

            for M in modulations:
                ccdf[M] = np.exp(
                    -np.exp(papr_axis - papr_results[M])
                )

            # =====================================
            # Carrusel
            # =====================================
            self.plot_funcs = [
                lambda: self.plot_ber(
                    snr_db_range, ber_results
                ),
                lambda: self.plot_papr_ccdf(
                    papr_axis, ccdf
                )
            ]

            self.current_plot = 0
            self.plot_funcs[0]()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error análisis", str(e))

    # =====================================================
    # Navegación
    # =====================================================
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

    # =====================================================
    # GRÁFICAS
    # =====================================================
    def plot_ber(self, snr_db, ber):
        self.plot.clear()

        for M, ber_vals in ber.items():
            self.plot.ax.semilogy(
                snr_db, ber_vals, marker="o",
                label=f"{M}-QAM"
            )

        self.plot.ax.set_xlabel("SNR (dB)")
        self.plot.ax.set_ylabel("BER")
        self.plot.ax.set_title("BER vs SNR (AWGN)")
        self.plot.ax.grid(True, which="both")
        self.plot.ax.legend()
        self.plot.canvas.draw()

    def plot_papr_ccdf(self, papr_axis, ccdf):
        self.plot.clear()

        for M, ccdf_vals in ccdf.items():
            self.plot.ax.semilogy(
                papr_axis, ccdf_vals,
                label=f"{M}-QAM"
            )

        self.plot.ax.set_xlabel("PAPR (dB)")
        self.plot.ax.set_ylabel("CCDF")
        self.plot.ax.set_title("CCDF de PAPR (OFDM)")
        self.plot.ax.grid(True, which="both")
        self.plot.ax.legend()
        self.plot.canvas.draw()
