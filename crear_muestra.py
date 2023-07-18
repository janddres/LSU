import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

alfabeto= "PRIA"
dataset_size = 100

# Abrir la cámara
cap = cv2.VideoCapture(0)

# Bucle para cada clase
for j in alfabeto:
    # Crear un directorio para la clase actual
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Bucle para esperar la tecla "c" y comenzar la captura
    done = False
    while True:
        ret, frame = cap.read()
        # Mostrar un mensaje en la ventana de la cámara
        cv2.putText(frame, 'Capturar "C", Letra{}'.format(j), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('c'):
            break

    counter = 0
    # Bucle para capturar las imágenes hasta alcanzar el tamaño deseado
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Guardar la imagen en el directorio correspondiente a la clase actual
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
