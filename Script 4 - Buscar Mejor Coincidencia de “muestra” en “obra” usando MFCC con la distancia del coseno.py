import librosa
import numpy as np
import soundfile as sf
import csv
from scipy.spatial.distance import cosine
import os

# Parámetros
threshold = 0.2  # Umbral de similitud (0 = máxima similitud)
sample_rate = 22050
hop_length = 512
window_duration_sec = 3.0
step_fast_ms = 100
step_slow_ms = 10

def ms_to_frames(ms, sr=sample_rate, hop=hop_length):
    return int((ms / 1000) * sr / hop)

# Rutas
desktop = os.path.expanduser("~/Desktop")
muestra_path = os.path.join(desktop, "muestra.wav")
obra_path = os.path.join(desktop, "obra.wav")
output_folder = os.path.join(desktop, "Coincidencias_BuscarMejorMuestraMFCC")
os.makedirs(output_folder, exist_ok=True)

# Carga los archivos
muestra, _ = librosa.load(muestra_path, sr=sample_rate)
obra, _ = librosa.load(obra_path, sr=sample_rate)

mfcc_muestra = librosa.feature.mfcc(y=muestra, sr=sample_rate, hop_length=hop_length)
mfcc_obra = librosa.feature.mfcc(y=obra, sr=sample_rate, hop_length=hop_length)

muestra_frames = mfcc_muestra.shape[1]
obra_frames = mfcc_obra.shape[1]

# Vector medio de MFCC para la muestra
mfcc_muestra_mean = np.mean(mfcc_muestra, axis=1)

step_fast = ms_to_frames(step_fast_ms)
step_slow = ms_to_frames(step_slow_ms)

mejor_coincidencia = None
mejor_distancia = float('inf')

i = 0
while i + muestra_frames <= obra_frames:
    fragmento = mfcc_obra[:, i:i+muestra_frames]
    mfcc_fragmento_mean = np.mean(fragmento, axis=1)
    dist = cosine(mfcc_muestra_mean, mfcc_fragmento_mean)

    if dist < mejor_distancia:
        mejor_distancia = dist
        mejor_coincidencia = i

    if dist < threshold:
        j = i
        while j > 0:
            frag_prev = mfcc_obra[:, max(j-step_slow, 0):max(j-step_slow, 0)+muestra_frames]
            frag_prev_mean = np.mean(frag_prev, axis=1)
            dist_prev = cosine(mfcc_muestra_mean, frag_prev_mean)
            if dist_prev < dist:
                dist = dist_prev
                mejor_coincidencia = max(j-step_slow, 0)
                j -= step_slow
            else:
                break
        break  # Elimina si quieres buscar más coincidencias

    i += step_fast

if mejor_coincidencia is not None:
    tiempo_inicio_seg = mejor_coincidencia * hop_length / sample_rate
    tiempo_fin_seg = tiempo_inicio_seg + len(muestra) / sample_rate
    muestra_inicio = int(tiempo_inicio_seg * sample_rate)
    muestra_fin = int(tiempo_fin_seg * sample_rate)
    fragmento_audio = obra[muestra_inicio:muestra_fin]

    # Guarda el audio del fragmento
    ruta_fragmento = os.path.join(output_folder, "fragmento_coincidente.wav")
    sf.write(ruta_fragmento, fragmento_audio, sample_rate)

    # Guarda CSV
    ruta_csv = os.path.join(output_folder, "coincidencia.csv")
    with open(ruta_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Inicio (minutos:segundos)", "Distancia Coseno"])
        minutos = int(tiempo_inicio_seg // 60)
        segundos = int(tiempo_inicio_seg % 60)
        writer.writerow([f"{minutos}:{segundos:02d}", f"{mejor_distancia:.4f}"])

    print(f"Fragmento guardado en: {ruta_fragmento}")
    print(f"Mejor coincidencia en {minutos}:{segundos:02d} con distancia coseno = {mejor_distancia:.4f}")
else:
    print("No se ha encontrado ninguna coincidencia que supere el umbral.")
