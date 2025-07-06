import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve

# -------------------- CONFIGURACIÓN --------------------
sample_rate = 44100                               # Frecuencia de muestreo
ruta_ir = os.path.expanduser("~/Desktop/IRs/plate_reverb.wav")  # Ruta al archivo IR
reverb_mix = 0.3                                  # Mezcla entre seco y reverberado (0.0 a 1.0)
corte_hpf_hz = 200                                # Filtro pasa-altos aplicado a la IR y al resultado
# --------------------------------------------------------

def filtro_pasaaltos(audio, sr, freq_corte):
    nyquist = 0.5 * sr
    normal_corte = freq_corte / nyquist
    b, a = butter(2, normal_corte, btype='high')
    return lfilter(b, a, audio)

def cargar_ir(ruta_ir, sr_objetivo, freq_corte):
    ir, sr_ir = librosa.load(ruta_ir, sr=None)
    if sr_ir != sr_objetivo:
        ir = librosa.resample(ir, orig_sr=sr_ir, target_sr=sr_objetivo)
    ir = filtro_pasaaltos(ir, sr_objetivo, freq_corte)  # Elimina graves de la IR
    return ir / np.max(np.abs(ir))  # Normaliza

def aplicar_convolucion_reverb(audio, ir, mix, sr, freq_corte):
    reverberado = fftconvolve(audio, ir, mode='full')[:len(audio)]
    reverberado = filtro_pasaaltos(reverberado, sr, freq_corte)  # Elimina graves de la reverb
    salida = (1 - mix) * audio + mix * reverberado
    return salida / np.max(np.abs(salida))  # Normaliza la mezcla final
# Rutas
carpeta_entrada = os.path.expanduser("~/Desktop/Entrada_Reverb")
nombre_salida = f"Salida_ReverbLimpia_mix{int(reverb_mix*100)}"
carpeta_salida = os.path.expanduser(f"~/Desktop/{nombre_salida}")
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar IR
ir = cargar_ir(ruta_ir, sample_rate, corte_hpf_hz)

# Procesamiento
for archivo in os.listdir(carpeta_entrada):
    if archivo.lower().endswith('.wav'):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        audio, sr = librosa.load(ruta_entrada, sr=sample_rate)
        audio_reverberado = aplicar_convolucion_reverb(audio, ir, reverb_mix, sr, corte_hpf_hz)

        nombre_archivo_salida = f"{os.path.splitext(archivo)[0]}_ReverbLimpia.wav"
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo_salida)

        sf.write(ruta_salida, audio_reverberado, sr)
        print(f"Guardado: {ruta_salida}")

print("\n✅ Proceso completado con reverb limpia.")
