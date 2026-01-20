from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QComboBox,
    QSpinBox, QMessageBox, QCheckBox, QTextEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import numpy as np
from PIL import Image

# Core OFDM (NO tocar)
from core.ofdm_tx import OFDMConfig, load_rgb_image_to_bits


class ConfigTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state

        # ---------------------------
        # Widgets
        # ---------------------------
        self.lbl_image = QLabel("No hay imagen cargada")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setFixedHeight(420)
        self.lbl_image.setStyleSheet("border: 1px solid gray")

        btn_load_img = QPushButton("Cargar imagen")
        btn_load_img.clicked.connect(self.load_image)

        # Modulación
        self.combo_mod = QComboBox()
        self.combo_mod.addItems(["QPSK", "16-QAM", "64-QAM"])

        # Canal
        self.combo_channel = QComboBox()
        self.combo_channel.addItems([
            "IDEAL",
            "AWGN",
            "RAYLEIGH",
            "RAYLEIGH+AWGN"
        ])

        # SNR
        self.spin_snr = QSpinBox()
        self.spin_snr.setRange(0, 40)
        self.spin_snr.setValue(15)
        self.spin_snr.setSuffix(" dB")

        # Tamaño de imagen
        self.chk_original_size = QCheckBox("Usar tamaño original")
        self.chk_original_size.setChecked(True)
        self.chk_original_size.stateChanged.connect(self.toggle_image_size)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(32, 2000)
        self.spin_w.setValue(256)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(32, 2000)
        self.spin_h.setValue(256)

        self.spin_w.setEnabled(False)
        self.spin_h.setEnabled(False)

        self.spin_bw = QSpinBox()
        self.spin_bw.setRange(1, 50)
        self.spin_bw.setValue(10)
        self.spin_bw.setSuffix(" MHz")

        self.spin_df = QSpinBox()
        self.spin_df.setRange(5, 60)
        self.spin_df.setValue(15)
        self.spin_df.setSuffix(" kHz")

        self.spin_guard = QSpinBox()
        self.spin_guard.setRange(0, 50)
        self.spin_guard.setValue(10)
        self.spin_guard.setSuffix(" %")

        self.spin_cp = QSpinBox()
        self.spin_cp.setRange(1, 50)
        self.spin_cp.setValue(16)
        self.spin_cp.setSuffix(" us")


        self.txt_ofdm_info = QTextEdit()
        self.txt_ofdm_info.setReadOnly(True)
        self.txt_ofdm_info.setFixedHeight(220)

        btn_apply = QPushButton("Cargar parámetros")
        btn_apply.clicked.connect(self.apply_parameters)
        # ---------------------------
        # Layout
        # ---------------------------
        form = QVBoxLayout()
        form.addWidget(QLabel("Modulación"))
        form.addWidget(self.combo_mod)
        form.addWidget(QLabel("Canal"))
        form.addWidget(self.combo_channel)
        form.addWidget(QLabel("SNR"))
        form.addWidget(self.spin_snr)
        form.addStretch()
        

        left = QVBoxLayout()
        left.addWidget(self.lbl_image)
        left.addWidget(btn_load_img)

        main = QHBoxLayout()
        main.addLayout(left, 3)
        main.addLayout(form, 2)

        form.addWidget(QLabel("Tamaño de imagen"))
        form.addWidget(self.chk_original_size)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("W"))
        size_row.addWidget(self.spin_w)
        size_row.addWidget(QLabel("H"))
        size_row.addWidget(self.spin_h)

        form.addLayout(size_row)


        form.addWidget(QLabel("OFDM – Parámetros"))
        form.addWidget(QLabel("BW total"))
        form.addWidget(self.spin_bw)
        form.addWidget(QLabel("Δf"))
        form.addWidget(self.spin_df)
        form.addWidget(QLabel("Guard fraction"))
        form.addWidget(self.spin_guard)
        form.addWidget(QLabel("CP"))
        form.addWidget(self.spin_cp)
        
        form.addWidget(QLabel("OFDM – Valores calculados"))
        form.addWidget(self.txt_ofdm_info)

        form.addWidget(btn_apply)

        self.setLayout(main)

        # ---------------------------
        # Estado interno
        # ---------------------------
        self.image_path = None
        self.image_shape = None
        self.bits_tx = None

    # =========================================================
    # Cargar imagen
    # =========================================================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen",
            "",
            "Imágenes (*.png *.jpg *.bmp)"
        )

        if not path:
            return

        self.image_path = path

        # Preview
        pix = QPixmap(path)
        pix = pix.scaled(
            self.lbl_image.width(),
            self.lbl_image.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_image.setPixmap(pix)

    # =========================================================
    # Aplicar parámetros
    # =========================================================
    def apply_parameters(self):
        cfg = OFDMConfig(
            bw_mhz=self.spin_bw.value(),
            delta_f=self.spin_df.value() * 1e3,
            guard_fraction=self.spin_guard.value() / 100,
            cp_time_us=self.spin_cp.value()
        )
        
        self.txt_ofdm_info.setText(self.format_ofdm_info(cfg))

        if self.image_path is None:
            QMessageBox.warning(
                self,
                "Falta imagen",
                "Debes cargar una imagen primero."
            )
            return

        if self.chk_original_size.isChecked():
            resize_to = Image.open(self.image_path).size
        else:
            resize_to = (self.spin_w.value(), self.spin_h.value())

        # ---------------------------
        # Modulación
        # ---------------------------
        mod_txt = self.combo_mod.currentText()
        M = {"QPSK": 4, "16-QAM": 16, "64-QAM": 64}[mod_txt]

        # ---------------------------
        # Canal y SNR
        # ---------------------------
        channel = self.combo_channel.currentText()
        snr_db = self.spin_snr.value()

        # ---------------------------
        # Imagen → bits
        # ---------------------------
        img = Image.open(self.image_path).convert("RGB")
        resize_to = img.size
        bits, shape = load_rgb_image_to_bits(
            self.image_path,
            resize_to
        )

        # ---------------------------
        # Configuración OFDM
        # ---------------------------
        cfg = OFDMConfig()

        # ---------------------------
        # Guardar estado compartido
        # ---------------------------
        self.state.clear()
        self.state.update({
            "image_path": self.image_path,
            "bits_tx": bits,
            "img_shape": shape,
            "M": M,
            "channel": channel,
            "snr_db": snr_db,
            "cfg": cfg
        })

        QMessageBox.information(
            self,
            "Configuración lista",
            "Parámetros cargados correctamente.\n"
            "Ya puedes ir a TX / Canal / RX."
        )

    def toggle_image_size(self):
        use_original = self.chk_original_size.isChecked()
        self.spin_w.setEnabled(not use_original)
        self.spin_h.setEnabled(not use_original)

    def format_ofdm_info(self, cfg):
        return (
            f"BW TOTAL        : {cfg.BW_total/1e6:.3f} MHz\n"
            f"BW UTIL         : {cfg.BW_util/1e6:.3f} MHz\n"
            f"Guard Fraction  : {cfg.guard_fraction*100:.1f} %\n"
            f"Δf              : {cfg.delta_f/1e3:.2f} kHz\n"
            f"N_used          : {cfg.N_used}\n"
            f"Nfft            : {cfg.Nfft}\n"
            f"Fs              : {cfg.fs/1e6:.3f} MHz\n"
            f"T_u             : {cfg.T_u*1e6:.2f} us\n"
            f"T_cp            : {cfg.T_cp*1e6:.2f} us\n"
            f"T_symbol        : {cfg.T_sym*1e6:.2f} us\n"
            f"CP samples      : {cfg.cp_len}\n"
        )
