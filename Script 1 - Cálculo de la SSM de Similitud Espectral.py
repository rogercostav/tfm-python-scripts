import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Ruta al escritorio y al archivo de audio
escritorio = os.path.join(os.path.expanduser('~'), 'Desktop')
ruta_audio = os.path.join(escritorio, 'sonido1sonido2.wav')

# 2. Cargar el audio
audio, sr = librosa.load(ruta_audio)

# 3. Espectrograma de magnitud (STFT)
S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

# 4. Normalización de los espectros
S_normalized = S / np.linalg.norm(S, axis=0, keepdims=True)

# 5. Cálculo de la matriz de autosimilitud (similitud espectral)
matriz_similitud = np.dot(S_normalized.T, S_normalized)

# 6. Visualización y guardado con colormap tipo 'inferno' (cálido, similar a iAnalyse5)
plt.figure(figsize=(10, 8))
plt.imshow(matriz_similitud, origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='Similitud espectral (producto escalar)')
plt.title('Matriz de autosimilitud espectral')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.tight_layout()

# 7. Guardar la imagen en el escritorio
ruta_imagen = os.path.join(escritorio, 'matriz_similitud_iAnalyse5.png')
plt.savefig(ruta_imagen, dpi=300)
plt.close()

print(f'Imagen guardada en: {ruta_imagen}’)
