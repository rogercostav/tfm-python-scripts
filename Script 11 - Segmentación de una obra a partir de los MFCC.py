import librosa
import soundfile as sf
import numpy as np
import os
import csv

# === CONFIGURACIÓN ===
input_path = os.path.expanduser("~/Desktop/reveries.wav")
output_dir = os.path.expanduser("~/Desktop/secciones_agrupadas")
DURACION_MINIMA_SEGUNDOS = 15.0  # ⬅️ Fácil de modificar
mfcc_distance_threshold = 30.0

os.makedirs(output_dir, exist_ok=True)
y, sr = librosa.load(input_path, sr=None)

onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

def get_mfcc_mean(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Agrupación inicial por timbre
sections = []
start_idx = 0
prev_mfcc = None

for i in range(1, len(onset_times)):
    start_sample = int(onset_times[start_idx] * sr)
    end_sample = int(onset_times[i] * sr)
    segment = y[start_sample:end_sample]
    if len(segment) < 2048:
        continue
    mfcc_mean = get_mfcc_mean(segment)
    if prev_mfcc is None:
        prev_mfcc = mfcc_mean
        continue
    distance = np.linalg.norm(mfcc_mean - prev_mfcc)
    if distance > mfcc_distance_threshold:
        section_audio = y[int(onset_times[start_idx] * sr):int(onset_times[i - 1] * sr)]
        sections.append((onset_times[start_idx], onset_times[i - 1], section_audio))
        start_idx = i - 1
        prev_mfcc = mfcc_mean

if start_idx < len(onset_times) - 1:
    section_audio = y[int(onset_times[start_idx] * sr):int(onset_times[-1] * sr)]
    sections.append((onset_times[start_idx], onset_times[-1], section_audio))

# Fusión de secciones cortas
def fusionar_secciones(sections):
    i = 0
    resultado = []
    while i < len(sections):
        start, end, audio = sections[i]
        duracion = end - start
        if duracion >= DURACION_MINIMA_SEGUNDOS:
            resultado.append((start, end, audio))
            i += 1
        else:
            mfcc = get_mfcc_mean(audio)
            if i == 0:
                next_start, next_end, next_audio = sections[i + 1]
                fused_audio = np.concatenate((audio, next_audio))
                resultado.append((start, next_end, fused_audio))
                i += 2
            elif i == len(sections) - 1:
                prev_start, prev_end, prev_audio = resultado[-1]
                fused_audio = np.concatenate((prev_audio, audio))
                resultado[-1] = (prev_start, end, fused_audio)
                i += 1
            else:
                prev_start, prev_end, prev_audio = resultado[-1]
                next_start, next_end, next_audio = sections[i + 1]
                dist_prev = np.linalg.norm(mfcc - get_mfcc_mean(prev_audio))
                dist_next = np.linalg.norm(mfcc - get_mfcc_mean(next_audio))
                if dist_prev <= dist_next:
                    fused_audio = np.concatenate((prev_audio, audio))
                    resultado[-1] = (prev_start, end, fused_audio)
                    i += 1
                else:
                    fused_audio = np.concatenate((audio, next_audio))
                    resultado.append((start, next_end, fused_audio))
                    i += 2
    return resultado

# Aplicar fusión hasta que todas las secciones cumplan la duración mínima
merged = sections
while True:
    merged = fusionar_secciones(merged)
    cortas = [end - start for start, end, _ in merged if end - start < DURACION_MINIMA_SEGUNDOS]
    if not cortas:
        break

# Guardar audios y CSV
descriptores = []
for idx, (start_time, end_time, audio_seg) in enumerate(merged):
    filename = f"seccion_{idx+1:03d}.wav"
    file_path = os.path.join(output_dir, filename)
    sf.write(file_path, audio_seg, sr)

    mfcc = librosa.feature.mfcc(y=audio_seg, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=audio_seg, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio_seg)
    rms = librosa.feature.rms(y=audio_seg)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio_seg)

    descriptores.append({
        "archivo": filename,
        "inicio (min:seg)": f"{int(start_time // 60)}:{int(start_time % 60):02d}",
        "final (min:seg)": f"{int(end_time // 60)}:{int(end_time % 60):02d}",
        "duración (s)": round(end_time - start_time, 2),
        "mfcc_mean_0": np.mean(mfcc[0]),
        "mfcc_mean_1": np.mean(mfcc[1]),
        "mfcc_mean_2": np.mean(mfcc[2]),
        "centroid_mean": np.mean(centroid),
        "flatness_mean": np.mean(flatness),
        "rms_mean": np.mean(rms),
        "zero_cross_rate": np.mean(zero_crossings)
    })

csv_path = os.path.join(output_dir, "descriptores_agrupados.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=descriptores[0].keys())
    writer.writeheader()
    for row in descriptores:
        writer.writerow(row)

print(f"✅ Secciones guardadas en: {output_dir}")
print(f"✅ CSV generado en: {csv_path}")
