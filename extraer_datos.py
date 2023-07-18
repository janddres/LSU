import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Importar el módulo de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar el detector de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []  # Lista para almacenar los datos de las características de las manos
labels = []  # Lista para almacenar las etiquetas de clase

# Recorrer los directorios dentro de DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    # Recorrer las imágenes dentro de cada directorio
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista auxiliar para almacenar las características de una imagen

        x_ = []  # Lista auxiliar para almacenar las coordenadas x de las características
        y_ = []  # Lista auxiliar para almacenar las coordenadas y de las características

        # Leer la imagen y convertirla a RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con el detector de manos
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Recorrer las landmarks de las manos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                # Almacenar las coordenadas x e y de cada landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalizar las coordenadas en relación con la posición mínima
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                #print(data)

            # Agregar los datos auxiliares y la etiqueta a las listas principales
            data.append(data_aux)
            labels.append(dir_)
            print(data)

# Guardar los datos y etiquetas en un archivo pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
