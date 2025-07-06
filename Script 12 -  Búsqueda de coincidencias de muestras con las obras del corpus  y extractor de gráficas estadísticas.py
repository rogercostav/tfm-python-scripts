import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import re
import shutil
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
import pandas as pd

# === CONFIGURACIÓN ===
paso_frames_cos = 20   # Paso para la búsqueda por coseno (aprox 50ms)
paso_frames_dtw_gros = 40   # Paso para la búsqueda inicial DTW (aprox 100ms)
paso_frames_dtw_fi = 4      # Paso de refinamiento DTW (aprox 10ms)

# === FUNCIONES AUXILIARES ===
def distancia_mfcc(mfcc1, mfcc2):
    mfcc1_13 = mfcc1[:13, :]
    mfcc2_13 = mfcc2[:13, :]
    min_cols = min(mfcc1_13.shape[1], mfcc2_13.shape[1])
    mfcc1_13 = mfcc1_13[:, :min_cols]
    mfcc2_13 = mfcc2_13[:, :min_cols]
    return cosine(mfcc1_13.flatten(), mfcc2_13.flatten())

def distancia_dtw(mfcc1, mfcc2):
    distancia, _ = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
    return distancia

def calcular_descriptores(y, sr):
    return {
        "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spread": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "flatness": np.mean(librosa.feature.spectral_flatness(y=y)),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y))
    }

def orden_natural(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

# === PROGRAMA PRINCIPAL ===
def main():
    print(f"Iniciando script a las {datetime.now().strftime('%H:%M:%S')}")
    carpeta_muestras = os.path.expanduser('~/Desktop/muestras')
    carpeta_obras = os.path.expanduser('~/Desktop/obras')
    carpeta_resultados = os.path.expanduser('~/Desktop/resultados_coincidencias')
    figuras_carpeta = os.path.join(carpeta_resultados, "figuras")
    os.makedirs(carpeta_resultados, exist_ok=True)
    os.makedirs(figuras_carpeta, exist_ok=True)

    print("Cargando muestras...")
    muestras_files = sorted([f for f in os.listdir(carpeta_muestras) if f.lower().endswith('.wav')], key=orden_natural)
    muestras = []
    for mf in muestras_files:
        ruta = os.path.join(carpeta_muestras, mf)
        print(f"  Cargando muestra: {mf}")
        y, sr = librosa.load(ruta, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        descriptores = calcular_descriptores(y, sr)
        muestras.append((mf, y, sr, mfcc, descriptores))

    print("\nCargando obras...")
    obras_files = sorted([f for f in os.listdir(carpeta_obras) if f.lower().endswith('.wav')], key=orden_natural)
    resultados_csv = os.path.join(carpeta_resultados, 'resultados_coincidencias.csv')
    with open(resultados_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Muestra", "Obra", "Método", "Tiempo (min:seg)", "Distancia", "Δ Centroid", "Δ Spread", "Δ Flatness", "Δ RMS", "Δ Zero Crossing Rate"])

        for obra_file in obras_files:
            print(f"\nAnalizando obra: {obra_file}")
            ruta_obra = os.path.join(carpeta_obras, obra_file)
            y_obra, sr_obra = librosa.load(ruta_obra, sr=None)
            mfcc_obra = librosa.feature.mfcc(y=y_obra, sr=sr_obra, n_mfcc=20)

            for muestra_file, y_muestra, sr_muestra, mfcc_muestra, desc_muestra in muestras:
                carpeta_muestra = os.path.join(carpeta_resultados, os.path.splitext(muestra_file)[0])
                os.makedirs(carpeta_muestra, exist_ok=True)
                ruta_muestra_original = os.path.join(carpeta_muestras, muestra_file)
                destino_muestra = os.path.join(carpeta_muestra, muestra_file)
                if not os.path.exists(destino_muestra):
                    shutil.copy(ruta_muestra_original, destino_muestra)

                # ==== COSENO ====
                print(f" Procesando muestra (coseno): {muestra_file}")
                ventana_frames = mfcc_muestra.shape[1]
                max_pos = mfcc_obra.shape[1] - ventana_frames
                coincidencias = [(pos, distancia_mfcc(mfcc_muestra, mfcc_obra[:, pos:pos+ventana_frames])) for pos in range(0, max_pos, paso_frames_cos)]
                mejor_pos, mejor_dist = sorted(coincidencias, key=lambda x: x[1])[0]

                start_sample = int(mejor_pos * 512)
                end_sample = start_sample + len(y_muestra)
                if end_sample > len(y_obra):
                    end_sample = len(y_obra)
                    start_sample = max(0, end_sample - len(y_muestra))

                fragmento_audio = y_obra[start_sample:end_sample]
                ruta_fragmento = os.path.join(carpeta_muestra, f"{os.path.splitext(obra_file)[0]}_cos.wav")
                sf.write(ruta_fragmento, fragmento_audio, sr_obra)

                desc_frag = calcular_descriptores(fragmento_audio, sr_obra)
                delta = {k: desc_muestra[k] - desc_frag[k] for k in desc_muestra}
                tiempo_min_seg = f"{int(start_sample / sr_obra // 60)}:{int(start_sample / sr_obra % 60):02d}"

                writer.writerow([muestra_file, obra_file, "coseno", tiempo_min_seg, mejor_dist, delta["centroid"], delta["spread"], delta["flatness"], delta["rms"], delta["zcr"]])

                # ==== DTW ====
                print(f" Procesando muestra (DTW): {muestra_file}")
                ventana_frames = mfcc_muestra.shape[1]
                max_pos = mfcc_obra.shape[1] - ventana_frames
                coarse = [(pos, distancia_dtw(mfcc_muestra, mfcc_obra[:, pos:pos+ventana_frames])) for pos in range(0, max_pos, paso_frames_dtw_gros)]
                mejor_pos_gros, _ = sorted(coarse, key=lambda x: x[1])[0]

                refined_positions = range(max(0, mejor_pos_gros - 5), min(max_pos, mejor_pos_gros + 6))
                refinadas = [(p, distancia_dtw(mfcc_muestra, mfcc_obra[:, p:p+ventana_frames])) for p in refined_positions if p % paso_frames_dtw_fi == 0]
                mejor_pos_fino, mejor_dtw = sorted(refinadas, key=lambda x: x[1])[0]

                start_sample = int(mejor_pos_fino * 512)
                end_sample = start_sample + len(y_muestra)
                if end_sample > len(y_obra):
                    end_sample = len(y_obra)
                    start_sample = max(0, end_sample - len(y_muestra))

                fragmento_audio = y_obra[start_sample:end_sample]
                ruta_fragmento = os.path.join(carpeta_muestra, f"{os.path.splitext(obra_file)[0]}_dtw.wav")
                sf.write(ruta_fragmento, fragmento_audio, sr_obra)

                desc_frag = calcular_descriptores(fragmento_audio, sr_obra)
                delta = {k: desc_muestra[k] - desc_frag[k] for k in desc_muestra}
                tiempo_min_seg = f"{int(start_sample / sr_obra // 60)}:{int(start_sample / sr_obra % 60):02d}"

                writer.writerow([muestra_file, obra_file, "dtw", tiempo_min_seg, mejor_dtw, delta["centroid"], delta["spread"], delta["flatness"], delta["rms"], delta["zcr"]])

    print("\nGenerando gráficas resumen...")
    df = pd.read_csv(resultados_csv)
    df_cos = df[df["Método"] == "coseno"]
    min_distancias = df_cos.groupby('Muestra')['Distancia'].min().reset_index().sort_values(by='Distancia')

    plt.figure(figsize=(12, 6))
    plt.bar(min_distancias['Muestra'], min_distancias['Distancia'], color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Muestra')
    plt.ylabel('Distancia coseno mínima')
    plt.title('Similitud mínima por muestra con alguna obra')
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_carpeta, 'similitud_minima_por_muestra.png'))
    plt.close()

    bins = np.arange(0, 1.05, 0.05)
    for muestra, grupo in df_cos.groupby('Muestra'):
        distancias = grupo['Distancia']
        hist, _ = np.histogram(distancias, bins=bins)
        plt.figure(figsize=(10, 5))
        plt.bar(bins[:-1], hist, width=0.045, align='edge', color='cornflowerblue')
        plt.xticks(bins, rotation=45)
        plt.xlabel('Distancia coseno')
        plt.ylabel('Número de coincidencias')
        plt.title(f'Histograma de coincidencias por distancia - {muestra}')
        plt.tight_layout()
        plt.savefig(os.path.join(figuras_carpeta, f"histograma_{muestra.replace('.wav', '')}.png"))
        plt.close()

    umbral = 0.02
    coincidencias_filtradas = df_cos[df_cos['Distancia'] < umbral]
    conteo = coincidencias_filtradas['Muestra'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(conteo.index, conteo.values, color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Muestra')
    plt.ylabel('Número de coincidencias (< 0.02)')
    plt.title(f'Muestras con más coincidencias por debajo del umbral {umbral}')
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_carpeta, 'muestras_mas_coincidencias.png'))
    plt.close()

if __name__ == "__main__":
    main()

