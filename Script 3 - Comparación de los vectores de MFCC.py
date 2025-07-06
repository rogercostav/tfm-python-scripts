import librosa
import numpy as np
from scipy.spatial.distance import cosine
import os

# Parámetros
sample_rate = 22050
hop_length = 512
n_mfcc = 13
similarity_method = 'cosine'  # Cambia a 'dot' para usar el producto escalar

# Rutas de los archivos
desktop = os.path.expanduser("~/Desktop")
so1_path = os.path.join(desktop, "so1.wav")
so2_path = os.path.join(desktop, "so2.wav")

# Cargar audio
so1, _ = librosa.load(so1_path, sr=sample_rate)
so2, _ = librosa.load(so2_path, sr=sample_rate)

# Extraer MFCCs
mfcc1 = librosa.feature.mfcc(y=so1, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
mfcc2 = librosa.feature.mfcc(y=so2, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)

# Alinear en longitud mínima
min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
mfcc1 = mfcc1[:, :min_frames]
mfcc2 = mfcc2[:, :min_frames]

# Calcular similitud frame a frame
similarities = []
for i in range(min_frames):
    v1 = mfcc1[:, i]
    v2 = mfcc2[:, i]

    if similarity_method == 'cosine':
        sim = 1 - cosine(v1, v2)  # 1 - distancia del coseno = similitud
    elif similarity_method == 'dot':
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        raise ValueError("Método no reconocido. Usa 'cosine' o 'dot'.")

    similarities.append(sim)

# Promedio de similitudes
mean_similarity = np.mean(similarities)

# Mostrar resultado
if similarity_method == 'cosine':
    print(f"Similitud coseno promedio entre 'so1.wav' y 'so2.wav': {mean_similarity:.4f} (1=igual)")
else:
    print(f"Producto escalar normalizado promedio: {mean_similarity:.4f} (1=igual)")
