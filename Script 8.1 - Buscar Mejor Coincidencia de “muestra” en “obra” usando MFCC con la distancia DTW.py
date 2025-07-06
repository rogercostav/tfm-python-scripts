import librosa
import numpy as np
import soundfile as sf
import os
import csv
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ====== PARÁMETROS ======
sample_rate = 22050
hop_length = 512
window_duration_sec = 3.0
step_fast_ms = 1000
step_slow_ms = 10
threshold = 300  # Umbral DTW para buscar retroactivamente
entrada_folder = os.path.expanduser("~/Desktop/Salida_HPF_420Hz")
obra_path = os.path.expanduser("~/Desktop/obra.wav")
output_base = os.path.expanduser("~/Desktop/Coincidencias_Batch_420_DTW")
os.makedirs(output_base, exist_ok=True)

# ====== FUNCIONES AUXILIARES ======
def ms_to_frames(ms, sr=sample_rate, hop=hop_length):
    return int((ms / 1000) * sr / hop)

def dtw_distance(mfcc1, mfcc2):
    dist, _ = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
    return dist

# ====== CARGA DE OBRA ======
obra, _ = librosa.load(obra_path, sr=sample_rate)
mfcc_obra = librosa.feature.mfcc(y=obra, sr=sample_rate, hop_length=hop_length)
obra_frames = mfcc_obra.shape[1]

# ====== PROCESAMIENTO POR ARCHIVO ======
distancias_dtw = []

for nombre_archivo in os.listdir(entrada_folder):
    if not nombre_archivo.endswith(".wav"):
        continue

    ruta_archivo = os.path.join(entrada_folder, nombre_archivo)
    muestra, _ = librosa.load(ruta_archivo, sr=sample_rate)
    mfcc_muestra = librosa.feature.mfcc(y=muestra, sr=sample_rate, hop_length=hop_length)
    muestra_frames = mfcc_muestra.shape[1]

    step_fast = ms_to_frames(step_fast_ms)
    step_slow = ms_to_frames(step_slow_ms)

    mejor_coincidencia = None
    mejor_distancia = float('inf')
    i = 0
    while i + muestra_frames <= obra_frames:
        fragmento = mfcc_obra[:, i:i+muestra_frames]
        dist = dtw_distance(mfcc_muestra, fragmento)

        if dist < mejor_distancia:
            mejor_distancia = dist
            mejor_coincidencia = i

        if dist < threshold:
            j = i
            while j > 0:
                anterior = mfcc_obra[:, max(j-step_slow, 0):max(j-step_slow, 0)+muestra_frames]
                dist_anterior = dtw_distance(mfcc_muestra, anterior)
                if dist_anterior < dist:
                    dist = dist_anterior
                    mejor_distancia = dist_anterior
                    mejor_coincidencia = max(j-step_slow, 0)
                    j -= step_slow
                else:
                    break
            break
        i += step_fast

    # === GUARDADO DE RESULTADOS ===
    nombre_base = os.path.splitext(nombre_archivo)[0]
    output_folder = os.path.join(output_base, nombre_base)
    os.makedirs(output_folder, exist_ok=True)

    if mejor_coincidencia is not None:
        tiempo_inicio_seg = mejor_coincidencia * hop_length / sample_rate
        tiempo_fin_seg = tiempo_inicio_seg + len(muestra) / sample_rate
        muestra_inicio = int(tiempo_inicio_seg * sample_rate)
        muestra_fin = int(tiempo_fin_seg * sample_rate)
        fragmento_audio = obra[muestra_inicio:muestra_fin]

        # Guarda el fragmento coincidente
        ruta_fragmento = os.path.join(output_folder, f"correspondencia_{nombre_base}.wav")
        sf.write(ruta_fragmento, fragmento_audio, sample_rate)

        # Guarda los audios original y muestra
        sf.write(os.path.join(output_folder, f"original_{nombre_base}.wav"), muestra, sample_rate)

        # CSV
        ruta_csv = os.path.join(output_folder, "coincidencia.csv")
        with open(ruta_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Inicio (min:seg)", "Distancia DTW"])
            minutos = int(tiempo_inicio_seg // 60)
            segundos = int(tiempo_inicio_seg % 60)
            writer.writerow([f"{minutos}:{segundos:02d}", f"{mejor_distancia:.2f}"])

        print(f"[{nombre_base}] Coincidencia encontrada en {minutos}:{segundos:02d} - DTW = {mejor_distancia:.2f}")
        distancias_dtw.append(mejor_distancia)
    else:
        print(f"[{nombre_base}] No se encontró coincidencia.")
        distancias_dtw.append(None)

# ====== DIAGRAMA DE BARRAS ======
distancias_validas = [d for d in distancias_dtw if d is not None]
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
hist, edges = np.histogram(distancias_validas, bins=bins)

plt.figure(figsize=(10, 5))
plt.bar([f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))], hist, color='skyblue')
plt.title("Distribución de distancias DTW")
plt.xlabel("Rango de distancia DTW")
plt.ylabel("Número de archivos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_base, "histograma_distancias.png"))
plt.show()

