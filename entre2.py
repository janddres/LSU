import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar los datos desde el archivo pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convertir los datos a matrices NumPy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar el modelo de clasificación Gradient Boosting
model = GradientBoostingClassifier()

# Entrenar el modelo utilizando los datos de entrenamiento
model.fit(x_train, y_train)

# Predecir las etiquetas de los datos de prueba utilizando el modelo entrenado
y_predict = model.predict(x_test)

# Calcular la precisión del modelo comparando las etiquetas predichas con las etiquetas reales
score = accuracy_score(y_predict, y_test)

# Imprimir el porcentaje de muestras clasificadas correctamente
print('{}% of samples were classified correctly!'.format(score * 100))

# Guardar el modelo entrenado en un archivo pickle
with open('model2.p', 'wb') as f:
    pickle.dump({'model2': model}, f)
