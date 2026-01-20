# Simulador de Sistema OFDM en Python

Simulador completo de un sistema de comunicaciones digitales basado en la
modulación **Orthogonal Frequency Division Multiplexing (OFDM)**, desarrollado
en Python. El proyecto incluye la implementación del transmisor, canal y
receptor, así como herramientas de análisis de desempeño y una interfaz gráfica
de usuario (GUI).

---

## Descripción del Proyecto

Este proyecto implementa un sistema OFDM siguiendo el modelo clásico de un
enlace de comunicaciones digitales. El sistema se divide en tres bloques
principales:

- **Transmisor OFDM (TX)**
- **Canal de transmisión**
- **Receptor OFDM (RX)**

El transmisor genera símbolos OFDM a partir de información digital utilizando
modulación QAM, transformada inversa rápida de Fourier (IFFT) e inserción de
prefijo cíclico. El canal permite simular diferentes condiciones de
propagación, incluyendo canal ideal, ruido aditivo blanco gaussiano (AWGN) y
canal Rayleigh selectivo en frecuencia. En el receptor se realizan procesos de
estimación de canal y ecualización en el dominio de la frecuencia para
recuperar la información transmitida.

---

## Estructura del Repositorio

core/            → Implementación del transmisor, canal, receptor y análisis  
GUI/             → Interfaz gráfica para la simulación interactiva  
figs_resultados/ → Figuras generadas durante las simulaciones y análisis  
Funcionalidades Implementadas
Modulación QAM (4-QAM, 16-QAM y 64-QAM)

Modulación y demodulación OFDM mediante FFT / IFFT

Inserción y eliminación de prefijo cíclico

Modelos de canal:

Canal ideal

Canal AWGN

Canal Rayleigh selectivo en frecuencia

Estimación de canal mediante subportadoras piloto (LS)

Ecualización en frecuencia (Zero Forcing)

Análisis BER vs SNR mediante simulaciones Monte Carlo

Análisis del PAPR mediante CCDF

Interfaz gráfica (GUI) para visualización y configuración del sistema

Ejecución del Proyecto
Ejecución del núcleo OFDM
python core/ofdm_tx.py
python core/ofdm_channel.py
python core/ofdm_rx.py
Análisis de desempeño
python core/ofdm_analysis.py
Interfaz gráfica de usuario
python GUI/gui_main.py
Requisitos
Instalar las dependencias necesarias con:

pip install -r requirements.txt
Librerías principales utilizadas:

NumPy

Matplotlib

Pillow

Resultados
El proyecto genera diferentes resultados gráficos, incluyendo:

Constelaciones QAM

Señales OFDM en el dominio del tiempo

Espectros en frecuencia

Reconstrucción de imágenes bajo distintos escenarios de canal

Curvas BER vs SNR

Análisis del PAPR

Estos resultados permiten evaluar el desempeño del sistema OFDM y verificar su
robustez frente a canales selectivos en frecuencia.

Autores
Israel Delgado

Anthony Domínguez

Sebastian Guazhima

Licencia
Este proyecto fue desarrollado con fines académicos y educativos.
