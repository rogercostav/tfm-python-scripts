import os
import librosa
import numpy as np
import soundfile as sf
import csv
from sklearn.metrics.pairwise import cosine_similarity
import random

# Parámetros
sample_rate = 22050
duraciones = [1, 3, 7, 15, 30]
num_por_duracion = 20
hop_length = 512

# Rutas
desktop = os.path.expanduser("~/Desktop")
ruta_obra = os.path.join(desktop, "obra.wav")
carpeta_resultados = os.path.join(desktop, "Resultados")
os.makedirs(carpeta_resultados, exist_ok=True)

# Cargar audio principal
obra, _ = librosa.load(ruta_obra, sr=sample_rate)

def cosine_distance(mfcc1, mfcc2):
    mfcc1_mean = np.mean(mfcc1, axis=1, keepdims=True)
    mfcc2_mean = np.mean(mfcc2, axis=1, keepdims=True)
    sim = cosine_similarity(mfcc1_mean.T, mfcc2_mean.T)[0][0]
    return 1 - sim

def extraer_fragmento(audio, duracion_s, sr):
    max_inicio = len(audio) - int(sr * duracion_s)
    if max_inicio <= 0:
        return None, None
    inicio = random.randint(0, max_inicio)
    fin = inicio + int(sr * duracion_s)
    return audio[inicio:fin], inicio / sr

# Crear CSV
csv_path = os.path.join(carpeta_resultados, "resultados.csv")
with open(csv_path, mode='w', newline='') as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["Duración_s", "Archivo_muestra", "Archivo_fragmento", "Inicio_minuto", "Inicio_segundo", "Distancia_coseno"])

    for duracion in duraciones:
        carpeta_duracion = os.path.join(carpeta_resultados, f"Fragmentos {duracion}")
        os.makedirs(carpeta_duracion, exist_ok=True)

        for i in range(num_por_duracion):
            muestra_audio, _ = extraer_fragmento(obra, duracion, sample_rate)
            if muestra_audio is None:
                continue

            nombre_muestra = f"muestra_{duracion}s_{i+1}.wav"
            ruta_muestra = os.path.join(carpeta_duracion, nombre_muestra)
            sf.write(ruta_muestra, muestra_audio, sample_rate)

            mfcc_muestra = librosa.feature.mfcc(y=muestra_audio, sr=sample_rate, hop_length=hop_length)
            muestra_frames = mfcc_muestra.shape[1]

            mejor_distancia = float('inf')
            mejor_inicio = 0
            mejor_fragmento_audio = None

            for j in range(0, len(obra) - len(muestra_audio), int(sample_rate * 0.1)):
                fragmento = obra[j:j+len(muestra_audio)]
                mfcc_fragmento = librosa.feature.mfcc(y=fragmento, sr=sample_rate, hop_length=hop_length)

                if mfcc_fragmento.shape[1] != muestra_frames:
                    continue

                dist = cosine_distance(mfcc_muestra, mfcc_fragmento)
                if dist < mejor_distancia:
                    mejor_distancia = dist
                    mejor_inicio = j
                    mejor_fragmento_audio = fragmento

            nombre_fragmento = f"coincidencia_{duracion}s_{i+1}.wav"
            ruta_fragmento = os.path.join(carpeta_duracion, nombre_fragmento)
            sf.write(ruta_fragmento, mejor_fragmento_audio, sample_rate)

            minutos = int(mejor_inicio / sample_rate // 60)
            segundos = (mejor_inicio / sample_rate) % 60

            writer.writerow([duracion, nombre_muestra, nombre_fragmento, minutos, round(segundos, 2), round(mejor_distancia, 4)])

print(f"Se han guardado los resultados en {csv_path}”)
