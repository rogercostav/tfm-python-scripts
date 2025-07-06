import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import convolve

# -------------------- CONFIGURACIÓN --------------------
sample_rate = 44100  # Frecuencia de muestreo
tipo_reverb = 'plate'  # 'spring' o 'plate'
decay = 0.5  # Duración del decay (segundos)
reverb_mix = 0.1  # Mezcla entre seco y reverberado (0.0 a 1.0)


# --------------------------------------------------------

def generar_impulso_spring(sr, decay):
    t = np.linspace(0, decay, int(sr * decay))
    impulso = np.sin(2 * np.pi * 50 * t) * np.exp(-3 * t)  # 50 Hz base
    impulso += np.sin(2 * np.pi * 120 * t) * np.exp(-4 * t)  # Harmònica
    impulso += np.random.normal(0, 0.05, len(t))  # soroll suau
    return impulso / np.max(np.abs(impulso))


def generar_impulso_plate(sr, decay):
    longitud = int(decay * sr)
    impulso = np.random.randn(longitud)
    impulso *= np.exp(-np.linspace(0, decay, longitud))  # Decaïment exponencial
    impulso = np.convolve(impulso, np.ones(50) / 50, mode='same')  # Suavitza (filtre)
    return impulso / np.max(np.abs(impulso))


def aplicar_reverb(audio, sr, tipo, decay, mix):
    if tipo == 'spring':
        impulso = generar_impulso_spring(sr, decay)
    elif tipo == 'plate':
        impulso = generar_impulso_plate(sr, decay)
    else:
        raise ValueError("Tipo de reverb no válido. Usa 'spring' o 'plate'.")

    reverberado = convolve(audio, impulso, mode='full')[:len(audio)]
    salida = (1 - mix) * audio + mix * reverberado
    return salida / np.max(np.abs(salida))


# Rutas
carpeta_entrada = os.path.expanduser("~/Desktop/Entrada_Reverb")
nombre_salida = f"Salida_Reverb_{tipo_reverb}_{int(decay * 1000)}ms_mix{int(reverb_mix * 100)}"
carpeta_salida = os.path.expanduser(f"~/Desktop/{nombre_salida}")
os.makedirs(carpeta_salida, exist_ok=True)

# Procesamiento
for archivo in os.listdir(carpeta_entrada):
    if archivo.lower().endswith('.wav'):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        audio, sr = librosa.load(ruta_entrada, sr=sample_rate)
        audio_reverberado = aplicar_reverb(audio, sr, tipo_reverb, decay, reverb_mix)

        nombre_archivo_salida = f"{os.path.splitext(archivo)[0]}_{tipo_reverb}_Reverb.wav"
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo_salida)

        sf.write(ruta_salida, audio_reverberado, sr)
        print(f"Guardado: {ruta_salida}")

print("\n✅ Proceso de reverb completado.")
