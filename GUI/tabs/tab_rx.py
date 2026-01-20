from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox
)

import numpy as np
from PIL import Image

from gui_mpl import MplWidget
from core.ofdm_rx import ofdm_rx, bits_to_image, qam_constellation
from core.ofdm_tx import active_subcarrier_indices


class RxTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state

        # ===============================
        # Widget de ploteo
        # ===============================
        self.plot = MplWidget(width=8, height=5)
        self.plot_funcs = []
        self.current_plot = 0

        # ===============================
        # Botones
        # ===============================
        btn_run = QPushButton("Ejecutar RX")
        btn_prev = QPushButton("⬅")
        btn_next = QPushButton("➡")

        btn_run.clicked.connect(self.run_rx)
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
    # RX
    # =====================================================
    def run_rx(self):
        try:
            if "y_rx" not in self.state:
                QMessageBox.warning(
                    self, "Error", "Ejecuta el canal primero."
                )
                return

            y = self.state["y_rx"]
            cfg = self.state["cfg"]
            M = self.state["M"]
            channel = self.state["channel"]
            h_true = self.state.get("h_true", None)

            bits_tx = self.state["bits_tx"]
            img_shape = self.state["img_shape"]
            img_path = self.state["image_path"]

            # ===============================
            # RX SIN ecualización
            # ===============================
            bits_no, Y_before, _ = ofdm_rx(
                y, cfg, M,
                channel_type="IDEAL",
                return_symbols=True
            )
            bits_no = bits_no[:len(bits_tx)]

            # ===============================
            # RX CON ecualización
            # ===============================
            bits_eq, _, Y_after = ofdm_rx(
                y, cfg, M,
                channel_type=channel,
                h_true=h_true,
                return_symbols=True
            )
            bits_eq = bits_eq[:len(bits_tx)]

            # ===============================
            # Imágenes (NORMALIZADAS)
            # ===============================
            img_tx = np.array(
                Image.open(img_path)
                .resize(img_shape[:2][::-1], Image.Resampling.NEAREST)
            )

            img_no = bits_to_image(bits_no, img_shape)
            img_eq = bits_to_image(bits_eq, img_shape)

            # Imagen original (referencia)
            img_tx = self._norm_img(img_tx)

            # Forzar RX al mismo tamaño que TX
            img_no = self._norm_img(img_no, target_shape=img_tx.shape)
            img_eq = self._norm_img(img_eq, target_shape=img_tx.shape)


            # ===============================
            # Carrusel RX
            # ===============================
            self.plot_funcs = [
                lambda: self.plot_constellation(
                    Y_before, cfg, "Constelación antes de ecualización"
                ),
                lambda: self.plot_constellation(
                    Y_after, cfg, "Constelación después de ecualización"
                ),
                lambda: self.plot_image(
                    img_no, "RX sin ecualización"
                ),
                lambda: self.plot_image(
                    img_eq, "RX con estimación + ecualización"
                ),
                lambda: self.plot_comparison(
                    img_tx, img_no, img_eq
                )
            ]

            self.current_plot = 0
            self.plot_funcs[0]()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error RX", str(e))

    # =====================================================
    # Navegación carrusel
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
    def plot_constellation(self, Y, cfg, title):
        self.plot.clear()

        idx = active_subcarrier_indices(cfg)

        # Constelación recibida
        self.plot.ax.scatter(
            np.real(Y[idx]),
            np.imag(Y[idx]),
            s=6,
            label="Recibido"
        )

        # Constelación ideal (roja)
        const_ref = qam_constellation(self.state["M"])
        self.plot.ax.scatter(
            np.real(const_ref),
            np.imag(const_ref),
            color="red",
            s=40,
            marker=".",
            label="Ideal"
        )

        self.plot.ax.set_title(title)
        self.plot.ax.grid(True)
        self.plot.ax.legend()
        self.plot.canvas.draw()


    def plot_image(self, img, title):
        self.plot.clear()
        self.plot.ax.imshow(img)
        self.plot.ax.set_title(title)
        self.plot.ax.axis("off")
        self.plot.canvas.draw()

    def plot_comparison(self, img_tx, img_no, img_eq):
        self.plot.clear()

        try:
            combo = np.concatenate([img_tx, img_no, img_eq], axis=1)
        except Exception as e:
            self.plot.ax.text(
                0.5, 0.5,
                "Error mostrando comparación\n"
                f"{e}",
                ha="center", va="center"
            )
            self.plot.ax.axis("off")
            self.plot.canvas.draw()
            return

        self.plot.ax.imshow(combo)
        self.plot.ax.set_title("TX | RX sin EQ | RX con EQ")
        self.plot.ax.axis("off")
        self.plot.canvas.draw()


    # =====================================================
    # Utils
    # =====================================================
    def _norm_img(self, img, target_shape=None):
        """
        Normaliza imagen para comparación segura:
        - fuerza RGB
        - fuerza uint8
        - fuerza mismo tamaño si se pasa target_shape
        """
        if img.ndim == 2:  # grayscale → RGB
            img = np.stack([img]*3, axis=-1)

        if img.ndim == 3 and img.shape[2] == 4:  # RGBA → RGB
            img = img[:, :, :3]

        img = img.astype(np.uint8)

        if target_shape is not None:
            h, w, _ = target_shape
            if img.shape[0] != h or img.shape[1] != w:
                from PIL import Image
                img = np.array(
                    Image.fromarray(img)
                    .resize((w, h), Image.Resampling.NEAREST)
                )

        return img

