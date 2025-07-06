import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import os

# --- Ruta al escritorio ---
desktop = "/Users/rogercostavendrell/Desktop"
audio1_path = os.path.join(desktop, "so1.wav")
audio2_path = os.path.join(desktop, "so2.wav")

# --- Función para extraer descriptores ---
def extraer_descriptores(ruta_audio):
    y, sr = librosa.load(ruta_audio)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spread = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    attack_time = librosa.frames_to_time(np.argmax(onset_env > 0.5), sr=sr)

    return {
        'centroid': centroid,
        'spread': spread,
        'flatness': flatness,
        'rms': rms,
        'zero_crossing': zero_crossing,
        'attack_time': attack_time
    }

# --- Normalización de los descriptores ---
def normalizar_vector_descriptores(desc):
    valores_maximos = {
        'centroid': 8000,       # Hz
        'spread': 4000,         # Hz
        'flatness': 1,          # ya está entre 0-1
        'rms': 0.5,             # valor típico de RMS normalizado
        'zero_crossing': 0.5,   # valor típico normalizado
        'attack_time': 2.0      # segundos (ajustable)
    }
    return np.array([desc[k] / valores_maximos[k] for k in desc])

# --- Comparación entre descriptores ---
def comparar_descriptores(desc1, desc2):
    vec1 = normalizar_vector_descriptores(desc1)
    vec2 = normalizar_vector_descriptores(desc2)
    distancia = euclidean(vec1, vec2)
    similitud = 1 - distancia
    similitud = max(0, min(1, similitud))  # limitar entre [0, 1]
    return similitud, distancia

# --- Ejecución principal ---
desc1 = extraer_descriptores(audio1_path)
desc2 = extraer_descriptores(audio2_path)

similitud, distancia = comparar_descriptores(desc1, desc2)

# --- Resultados por pantalla ---
print("🟦 Descriptores del sonido 1:")
for k, v in desc1.items():
    print(f"  {k}: {v:.4f}")

print("\n🟩 Descriptores del sonido 2:")
for k, v in desc2.items():
    print(f"  {k}: {v:.4f}")

print(f"\n📏 Distancia normalizada: {distancia:.4f}")
print(f"🔁 Similitud estimada (0 = distintos, 1 = idénticos): {similitud:.4f}")
