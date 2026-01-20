# 📡 Simulador de Sistema OFDM en Python

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Este repositorio contiene la implementación completa de un sistema de comunicaciones digitales basado en la modulación **Orthogonal Frequency Division Multiplexing (OFDM)**, desarrollado en Python.

El proyecto fue realizado como parte de una práctica de laboratorio de la asignatura **Comunicaciones Móviles**, e incluye los bloques de transmisor, canal y receptor, así como herramientas de análisis de desempeño y una interfaz gráfica de usuario (GUI).

---

## 📝 Descripción del Proyecto

El sistema OFDM implementado sigue el modelo clásico de un enlace de comunicaciones digitales, compuesto por tres bloques principales:

1.  **Transmisor OFDM (TX)**
2.  **Canal de transmisión**
3.  **Receptor OFDM (RX)**

El transmisor genera símbolos OFDM a partir de información digital utilizando modulación **QAM**, transformada inversa rápida de Fourier (**IFFT**) e inserción de prefijo cíclico.

El canal permite simular diferentes condiciones de propagación, incluyendo:
* Canal ideal.
* Ruido aditivo blanco gaussiano (**AWGN**).
* Canal Rayleigh selectivo en frecuencia.

En el receptor se realizan procesos de estimación de canal y ecualización en el dominio de la frecuencia para recuperar la información transmitida.

---

## 📂 Estructura del Repositorio
```text
├── core/              # Implementación del transmisor, canal, receptor y análisis
├── GUI/               # Interfaz gráfica para la simulación interactiva
├── figs_resultados/   # Figuras generadas durante las simulaciones y análisis
├── requirements.txt   # Dependencias del proyecto
└── README.md          # Documentación
```
## 🚀 Funcionalidades Implementadas

### Procesamiento de Señal
* **Modulación QAM:** Soporte para 4-QAM, 16-QAM y 64-QAM.
* **OFDM:** Modulación y demodulación mediante FFT/IFFT.
* **Guard Interval:** Inserción y eliminación de prefijo cíclico (Cyclic Prefix).

### Modelos de Canal
* Canal Ideal.
* Canal AWGN.
* Canal Rayleigh (Selectivo en frecuencia).

### Recepción y Análisis
* **Estimación de Canal:** Uso de subportadoras piloto (Least Squares - LS).
* **Ecualización:** Zero Forcing (ZF) en el dominio de la frecuencia.
* **Métricas:**
    * Análisis **BER vs SNR** mediante simulaciones Monte Carlo.
    * Análisis del **PAPR** mediante CCDF.

### Interfaz
* **GUI:** Interfaz gráfica completa para visualización y configuración de parámetros en tiempo real.

---

## ⚙️ Instalación y Requisitos

Para ejecutar este proyecto, asegúrate de tener Python instalado. Luego, instala las dependencias necesarias ejecutando:

## ▶️ Ejecución del Proyecto

Puedes correr los módulos de manera independiente o utilizar la interfaz gráfica.

### 1. Interfaz Gráfica de Usuario (Recomendado)
Para una experiencia interactiva y visualización inmediata:

```bash
python GUI/gui_main.py
```

Si prefieres ejecutar los scripts paso a paso:

# Transmisor
```bash
python core/ofdm_tx.py
```
# Simulación de Canal
```bash
python core/ofdm_channel.py
```
# Receptor
```bash
python core/ofdm_rx.py
```

## 📊 Resultados

El proyecto genera diferentes resultados gráficos que se almacenan en la carpeta `figs_resultados/`:

* Constelaciones QAM (transmitidas vs recibidas).
* Señales OFDM en el dominio del tiempo y la frecuencia.
* Reconstrucción de imágenes bajo distintos escenarios de canal.
* Curvas de **BER vs SNR**.
* Análisis del **PAPR**.

Estos resultados permiten evaluar el desempeño del sistema OFDM y verificar su robustez frente a canales selectivos en frecuencia.

---

## 👥 Autores

*Pablo Bermeo
* Sebastian Guazhima

---

## 📄 Licencia

Este proyecto fue desarrollado con fines **académicos y educativos**. Siéntete libre de usarlo como referencia para tus propios estudios de telecomunicaciones.
