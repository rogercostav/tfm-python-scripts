import librosa
import numpy as np
import soundfile as sf
import csv
import os
from librosa.sequence import dtw

# Parámetros
sample_rate = 22050
hop_length = 512
window_duration_sec = 3.0
step_fast_ms = 100
step_slow_ms = 10
semitones_list = [-1, 3, -6, 12]  # Cambios de pitch


def ms_to_frames(ms, sr=sample_rate, hop=hop_length):
    return int((ms / 1000) * sr / hop)


# Rutas
escritorio = os.path.expanduser("~/Desktop")
carpeta_entrada = os.path.join(escritorio, "muestras_pitchshift")
carpeta_resultados = os.path.join(escritorio, "Resultados_PitchShift")
os.makedirs(carpeta_resultados, exist_ok=True)

obra_path = os.path.join(escritorio, "obra.wav")
obra, _ = librosa.load(obra_path, sr=sample_rate)
mfcc_obra = librosa.feature.mfcc(y=obra, sr=sample_rate, hop_length=hop_length)
obra_frames = mfcc_obra.shape[1]

step_fast = ms_to_frames(step_fast_ms)
step_slow = ms_to_frames(step_slow_ms)

# Archivo CSV
csv_path = os.path.join(carpeta_resultados, "resultados.csv")
with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Archivo original", "Semitonos", "Inicio (min:seg)", "Distancia DTW"])

    # Itera sobre todos los archivos .wav en la carpeta de entrada
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".wav"):
            ruta_original = os.path.join(carpeta_entrada, archivo)
            muestra_original, _ = librosa.load(ruta_original, sr=sample_rate)

            for semitonos in semitones_list:
                # Aplica pitch shift
                muestra_shifted = librosa.effects.pitch_shift(muestra_original, sr=sample_rate, n_steps=semitonos)
                nombre_modificado = f"{os.path.splitext(archivo)[0]}_Pitch{semitonos:+d}.wav"
                ruta_modificada = os.path.join(carpeta_resultados, nombre_modificado)
                sf.write(ruta_modificada, muestra_shifted, sample_rate)

                # Calcula MFCCs
                mfcc_muestra = librosa.feature.mfcc(y=muestra_shifted, sr=sample_rate, hop_length=hop_length)
                muestra_frames = mfcc_muestra.shape[1]

                # Búsqueda de mejor coincidencia en "obra"
                mejor_distancia = float('inf')
                mejor_coincidencia = None
                i = 0
                while i + muestra_frames <= obra_frames:
                    fragmento = mfcc_obra[:, i:i + muestra_frames]

                    # DTW entre MFCCs (transpuesta requerida: shape [frames, features])
                    D, wp = dtw(mfcc_muestra.T, fragmento.T, subseq=True)
                    distancia = D[-1, -1] / len(wp)  # Normalización por longitud del camino

                    if distancia < mejor_distancia:
                        mejor_distancia = distancia
                        mejor_coincidencia = i
                    i += step_fast

                # Extrae el fragmento encontrado en "obra"
                if mejor_coincidencia is not None:
                    tiempo_inicio_seg = mejor_coincidencia * hop_length / sample_rate
                    tiempo_fin_seg = tiempo_inicio_seg + len(muestra_original) / sample_rate
                    muestra_inicio = int(tiempo_inicio_seg * sample_rate)
                    muestra_fin = int(tiempo_fin_seg * sample_rate)
                    fragmento_audio = obra[muestra_inicio:muestra_fin]

                    # Guarda fragmentos de audio
                    nombre_base = os.path.splitext(archivo)[0]
                    ruta_original_out = os.path.join(carpeta_resultados, f"{nombre_base}_Original.wav")
                    ruta_fragmento_out = os.path.join(carpeta_resultados,
                                                      f"{nombre_base}_Coincidencia_Pitch{semitonos:+d}.wav")
                    sf.write(ruta_original_out, muestra_original, sample_rate)
                    sf.write(ruta_fragmento_out, fragmento_audio, sample_rate)

                    # Escribe resultados en CSV
                    minutos = int(tiempo_inicio_seg // 60)
                    segundos = int(tiempo_inicio_seg % 60)
                    writer.writerow([archivo, semitonos, f"{minutos}:{segundos:02d}", f"{mejor_distancia:.4f}"])

print(f"✅ Proceso completado. Resultados guardados en: {csv_path}”)
