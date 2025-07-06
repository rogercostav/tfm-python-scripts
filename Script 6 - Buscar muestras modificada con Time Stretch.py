import librosa
import numpy as np
import soundfile as sf
import csv
import os
from scipy.spatial.distance import cosine

# Parámetros
sample_rate = 22050
hop_length = 512
step_fast_ms = 100
step_slow_ms = 10
stretch_factors = [0.2, 0.5, 2.0, 3.5]
threshold = 0.001

def ms_to_frames(ms, sr=sample_rate, hop=hop_length):
    return int((ms / 1000) * sr / hop)

step_fast = ms_to_frames(step_fast_ms)
step_slow = ms_to_frames(step_slow_ms)

# Rutas
desktop = os.path.expanduser("~/Desktop")
obra_path = os.path.join(desktop, "obra.wav")
muestras_folder = os.path.join(desktop, "muestras_timestretch")
output_root = os.path.join(desktop, "Resultados_Multistretch")
os.makedirs(output_root, exist_ok=True)

# CSV único para todos los resultados
csv_path = os.path.join(output_root, "resultados_totales.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Archivo original", "Factor", "Inicio (min:seg)", "Distancia con original"])

    # Carga la obra una vez
    obra, _ = librosa.load(obra_path, sr=sample_rate)
    mfcc_obra = librosa.feature.mfcc(y=obra, sr=sample_rate, hop_length=hop_length)

    # Procesa cada factor de time stretch
    for factor in stretch_factors:
        factor_str = str(factor).replace(".", "_")
        output_folder = os.path.join(output_root, f"Stretch_{factor_str}")
        os.makedirs(output_folder, exist_ok=True)

        # Procesa cada archivo .wav dentro de muestras_timestretch
        for filename in os.listdir(muestras_folder):
            if filename.endswith(".wav"):
                ruta_original = os.path.join(muestras_folder, filename)
                y_orig, _ = librosa.load(ruta_original, sr=sample_rate)

                # Aplica time stretch
                try:
                    y_stretched = librosa.effects.time_stretch(y_orig, rate=1.0/factor)
                except Exception as e:
                    print(f"⚠️ Error al aplicar stretch a {filename} (factor {factor}): {e}")
                    continue

                # Guarda el audio estirado
                stretched_name = filename.replace(".wav", f"_stretch{factor_str}.wav")
                ruta_stretched = os.path.join(output_folder, stretched_name)
                sf.write(ruta_stretched, y_stretched, sample_rate)

                # Calcula MFCCs
                mfcc_muestra = librosa.feature.mfcc(y=y_stretched, sr=sample_rate, hop_length=hop_length)
                muestra_frames = mfcc_muestra.shape[1]
                mfcc_muestra_mean = np.mean(mfcc_muestra, axis=1)

                # Busca la mejor coincidencia dentro de la obra
                mejor_coincidencia = None
                mejor_distancia = float('inf')
                i = 0
                while i + muestra_frames <= mfcc_obra.shape[1]:
                    fragmento = mfcc_obra[:, i:i + muestra_frames]
                    frag_mean = np.mean(fragmento, axis=1)
                    dist = cosine(mfcc_muestra_mean, frag_mean)
                    if dist < mejor_distancia:
                        mejor_distancia = dist
                        mejor_coincidencia = i
                    i += step_fast

                if mejor_coincidencia is not None:
                    seg_inicio = mejor_coincidencia * hop_length / sample_rate
                    muestra_inicio = int(seg_inicio * sample_rate)
                    muestra_fin = muestra_inicio + len(y_stretched)
                    fragmento_obra = obra[muestra_inicio:muestra_fin]

                    # Guarda el fragmento encontrado en la obra
                    fragment_name = filename.replace(".wav", f"_fragmento_obra_stretch{factor_str}.wav")
                    ruta_fragmento = os.path.join(output_folder, fragment_name)
                    sf.write(ruta_fragmento, fragmento_obra, sample_rate)

                    # Calcula la distancia con el original (sin stretch)
                    mfcc_fragmento = librosa.feature.mfcc(y=fragmento_obra, sr=sample_rate, hop_length=hop_length)
                    mfcc_fragmento_mean = np.mean(mfcc_fragmento, axis=1)
                    mfcc_original = librosa.feature.mfcc(y=y_orig, sr=sample_rate, hop_length=hop_length)
                    mfcc_original_mean = np.mean(mfcc_original, axis=1)
                    dist_original = cosine(mfcc_original_mean, mfcc_fragmento_mean)

                    # Guarda el archivo original
                    nombre_original_guardado = filename.replace(".wav", "_original.wav")
                    sf.write(os.path.join(output_folder, nombre_original_guardado), y_orig, sample_rate)

                    # Escribe en el CSV
                    minutos = int(seg_inicio // 60)
                    segundos = int(seg_inicio % 60)
                    writer.writerow([filename, factor, f"{minutos}:{segundos:02d}", f"{dist_original:.4f}"])
                    print(f"✅ {filename} (x{factor}) → Coincidencia en {minutos}:{segundos:02d} con distancia = {dist_original:.4f}")
                else:
                    print(f"❌ No se encontró coincidencia para {filename} con stretch {factor}")
