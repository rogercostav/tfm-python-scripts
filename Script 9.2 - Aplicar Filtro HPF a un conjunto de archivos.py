import os
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

# ---------------------- CONFIGURACIÓN ----------------------
frecuencia_corte = 420   # Hz
orden_filtro = 4          # Orden del filtro HPF (orden 4 ≈ -24 dB/octava)
sample_rate = 44100       # Frecuencia de muestreo a usar o forzar
# -----------------------------------------------------------

def aplicar_hpf(audio, sr, freq_corte, orden):
    nyquist = 0.5 * sr
    normal_corte = freq_corte / nyquist
    b, a = butter(orden, normal_corte, btype='high', analog=False)
    return lfilter(b, a, audio)

# Rutas
carpeta_entrada = os.path.expanduser("~/Desktop/Entrada_HPF")
nombre_salida = f"Salida_HPF_{frecuencia_corte}Hz"
carpeta_salida = os.path.expanduser(f"~/Desktop/{nombre_salida}")
os.makedirs(carpeta_salida, exist_ok=True)

# Procesamiento
for archivo in os.listdir(carpeta_entrada):
    if archivo.lower().endswith('.wav'):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        audio, sr = librosa.load(ruta_entrada, sr=sample_rate)
        audio_filtrado = aplicar_hpf(audio, sr, frecuencia_corte, orden_filtro)

        nombre_archivo_salida = f"{os.path.splitext(archivo)[0]}_HPF_{frecuencia_corte}Hz.wav"
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo_salida)

        sf.write(ruta_salida, audio_filtrado, sr)
        print(f"Guardado: {ruta_salida}")

print("\n✅ Proceso HPF completado.")
