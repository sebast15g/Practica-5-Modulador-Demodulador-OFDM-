from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox
)

import numpy as np

from gui_mpl import MplWidget
from core.ofdm_channel import (
    channel_ideal,
    channel_awgn,
    channel_rayleigh_critical
)


class ChannelTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state

        # =====================================================
        # Widget de ploteo (con toolbar)
        # =====================================================
        self.plot = MplWidget(width=8, height=5)
        self.plot_funcs = []
        self.current_plot = 0

        # =====================================================
        # Botones
        # =====================================================
        btn_run = QPushButton("Aplicar Canal")
        btn_prev = QPushButton("⬅")
        btn_next = QPushButton("➡")

        btn_run.clicked.connect(self.run_channel)
        btn_prev.clicked.connect(self.prev_plot)
        btn_next.clicked.connect(self.next_plot)

        nav = QHBoxLayout()
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        nav.addStretch()

        # =====================================================
        # Layout principal
        # =====================================================
        layout = QVBoxLayout()
        layout.addWidget(btn_run)
        layout.addWidget(self.plot, 1)
        layout.addLayout(nav)

        self.setLayout(layout)

    # =========================================================
    # Ejecutar canal
    # =========================================================
    def run_channel(self):
        try:
            if "x_all" not in self.state:
                QMessageBox.warning(
                    self,
                    "TX no ejecutado",
                    "Primero ejecuta el TX."
                )
                return

            x_all = self.state["x_all"]
            cfg = self.state["cfg"]
            channel = self.state["channel"]
            snr_db = self.state["snr_db"]

            x_tx = np.concatenate(x_all)

            h_true = None
            delays_ns = None
            powers_db = None

            # -------------------------------
            # Aplicar canal
            # -------------------------------
            if channel == "IDEAL":
                y = channel_ideal(x_tx)

            elif channel == "AWGN":
                y = channel_awgn(x_tx, snr_db)

            elif channel == "RAYLEIGH":
                y, h_true, delays_ns, powers_db = channel_rayleigh_critical(
                    x_tx, cfg.fs
                )

            elif channel == "RAYLEIGH+AWGN":
                y_ray, h_true, delays_ns, powers_db = channel_rayleigh_critical(
                    x_tx, cfg.fs
                )
                y = channel_awgn(y_ray, snr_db)

            else:
                raise ValueError("Canal no válido")

            # Guardar para RX
            self.state.update({
                "y_rx": y,
                "h_true": h_true
            })

            # -------------------------------
            # Registrar carrusel de gráficas
            # -------------------------------
            self.plot_funcs = []

            # 1) TX vs salida del canal (siempre)
            self.plot_funcs.append(
                lambda: self.plot_tx_vs_rx(x_tx, y)
            )

            # 2) PDP y H(f) SOLO si Rayleigh
            if h_true is not None and delays_ns is not None:
                self.plot_funcs.append(
                    lambda: self.plot_pdp(h_true, delays_ns, powers_db)
                )
                self.plot_funcs.append(
                    lambda: self.plot_freq_response(h_true, cfg)
                )

            # 3) RX zoom (siempre)
            self.plot_funcs.append(
                lambda: self.plot_rx_zoom(y)
            )

            # 4) Overlay TX / RX (siempre)
            self.plot_funcs.append(
                lambda: self.plot_overlay(x_tx, y)
            )


            self.current_plot = 0
            self.plot_funcs[0]()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error en Canal", str(e))

    # =========================================================
    # Navegación del carrusel
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
    # GRÁFICAS DEL CANAL
    # =========================================================
    def plot_tx_vs_rx(self, x, y, Ns=2000):
        self.plot.clear()
        self.plot.ax.plot(np.real(x[:Ns]), label="TX")
        self.plot.ax.plot(np.real(y[:Ns]), label="Salida del canal", alpha=0.7)
        self.plot.ax.set_title("Señal temporal: TX vs Canal")
        self.plot.ax.set_xlabel("Muestras")
        self.plot.ax.set_ylabel("Amplitud")
        self.plot.ax.grid(True)
        self.plot.ax.legend()
        self.plot.canvas.draw()

    def plot_pdp(self, h, delays_ns, powers_db):
        self.plot.clear()

        if h is None or delays_ns is None or powers_db is None:
            self.plot.ax.text(  
                0.5, 0.5,
                "Canal no Rayleigh\n(PDP no aplica)",
                ha="center", va="center",
                fontsize=11
            )


            self.plot.canvas.draw()
            return

        # --- PDP estable (sin stem) ---
        x = delays_ns
        y = powers_db

        # Líneas verticales
        self.plot.ax.vlines(x, ymin=min(y)-5, ymax=y, color="C0", linewidth=2)

        # Puntos
        self.plot.ax.scatter(x, y, color="C0", s=40, zorder=3)

        self.plot.ax.set_xlabel("Delay (ns)")
        self.plot.ax.set_ylabel("Potencia promedio (dB)")
        self.plot.ax.set_title("PDP – Canal Rayleigh")
        self.plot.ax.grid(True)
        self.plot.ax.invert_yaxis()
        self.plot.canvas.draw()


    def plot_freq_response(self, h, cfg):
        self.plot.clear()

        if h is None:
            self.plot.ax.text(
                0.5, 0.5,
                "Canal no selectivo\n|H(f)| no aplica",
                ha="center", va="center",
                fontsize=11
            )
        else:
            Nfft = 8192
            H = np.fft.fftshift(np.fft.fft(h, Nfft))
            f = np.fft.fftshift(
                np.fft.fftfreq(Nfft, d=1 / cfg.fs)
            )

            H_db = 20 * np.log10(np.abs(H) + 1e-12)
            H_db -= H_db.max()

            self.plot.ax.plot(f / 1e6, H_db)
            self.plot.ax.set_xlabel("Frecuencia (MHz)")
            self.plot.ax.set_ylabel("|H(f)| (dB)")
            self.plot.ax.set_title("Respuesta en frecuencia del canal")
            self.plot.ax.set_ylim([-30, 5])
            self.plot.ax.grid(True)

        self.plot.canvas.draw()

    def plot_rx_zoom(self, y, Ns=2000):
        self.plot.clear()
        self.plot.ax.plot(np.real(y[:Ns]))
        self.plot.ax.set_title("Señal RX (zoom temporal)")
        self.plot.ax.set_xlabel("Muestras")
        self.plot.ax.set_ylabel("Amplitud")
        self.plot.ax.grid(True)
        self.plot.canvas.draw()

    def plot_overlay(self, x, y, Ns=2000):
        self.plot.clear()
        self.plot.ax.plot(np.real(x[:Ns]), label="TX", alpha=0.7)
        self.plot.ax.plot(np.real(y[:Ns]), label="RX", alpha=0.7)
        self.plot.ax.set_title("Comparación TX / RX")
        self.plot.ax.set_xlabel("Muestras")
        self.plot.ax.set_ylabel("Amplitud")
        self.plot.ax.grid(True)
        self.plot.ax.legend()
        self.plot.canvas.draw()
