import cv2
import mediapipe as mp
import numpy as np
import pickle

# Cargar el modelo entrenado desde el archivo pickle
model = pickle.load(open('./model2.p', 'rb'))['model2']

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar el detector de manos de MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario para mapear las etiquetas predichas a caracteres
labels_dict = {"P": 'P', "R": 'R', "I": 'I', "A": 'A'}

while True:
    # Listas auxiliares para almacenar las coordenadas de las landmarks
    data_aux = []
    x_ = []
    y_ = []

    # Leer un frame del video
    ret, frame = cap.read()

    # Obtener el alto (H) y ancho (W) del frame
    H, W, _ = frame.shape

    # Convertir el frame a formato RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame utilizando el detector de manos de MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Dibujar las landmarks de las manos en el frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # imagen en la que dibujar
                hand_landmarks,  # salida del modelo
                mp_hands.HAND_CONNECTIONS,  # conexiones entre puntos de la mano
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Recorrer las landmarks de las manos detectadas
        for hand_landmarks in results.multi_hand_landmarks:
            # Almacenar las coordenadas de las landmarks en las listas auxiliares x_ y y_
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalizar las coordenadas de las landmarks en relación con la posición mínima en x_ e y_
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Restrictir los datos a 42 características
        data_aux = data_aux[:42]

        # Calcular las coordenadas del rectángulo que encierra las manos detectadas
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predecir el carácter utilizando el modelo entrenado
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[prediction[0]]

        # Dibujar el rectángulo y mostrar el carácter predicho en el frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Mostrar el frame en una ventana
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
