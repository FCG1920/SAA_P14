import os
import glob

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.optimizers import Adam

_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
path_to_zip = tf.keras.utils.get_file('flower_photos.zip', origin=_URL, untar=True)

# Contenido de la carpeta descomprimida
path_images = os.path.join(os.path.dirname(path_to_zip), 'flower_photos')
print(os.listdir(path_images))

# Número de imágenes
image_count = len(list(glob.glob(os.path.join(path_images, '*/*.jpg'))))
print('Numero de imágenes : ', image_count)

BATCH_SIZE = 32
IMG_SIZE = (150, 150)

# Dado que no tenemos partición específica de validación, vamos a partir la base de datos en entrenamiento (70%) y validación (30%)
train_dataset = image_dataset_from_directory(path_images,
                                             validation_split=0.3,
                                             subset="training",
                                             seed=123,
                                             image_size=IMG_SIZE,
                                             batch_size=BATCH_SIZE)

validation_dataset = image_dataset_from_directory(path_images,
                                                  validation_split=0.3,
                                                  subset="validation",
                                                  seed=123,
                                                  image_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE)

# Visualizamos el dataset
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#  Generamos partición de test
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Número de lotes de validación: ', tf.data.experimental.cardinality(validation_dataset).numpy())
print('Número de lotes de test: ', tf.data.experimental.cardinality(test_dataset).numpy())

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Le añadimos la dimensión de los 3 canales (RGB)
IMG_SHAPE = IMG_SIZE + (3,)


print(IMG_SHAPE)




# Pregunta 1

# Crear el modelo
model = Sequential([
        # Capa de Reescalado
        Rescaling(1./255, input_shape=IMG_SHAPE),
        # Capa convolucional
        Conv2D(16, (3, 3), padding='same', strides=2, activation='relu'),
        # Capa de MaxPooling
        MaxPooling2D(pool_size=(2, 2)),
        # Capa convolucional
        Conv2D(32, (5, 5), padding='same', activation='relu'),
        # Capa de MaxPooling
        MaxPooling2D(),
        # Capa convolucional
        Conv2D(32, (5, 5), padding='same', activation='relu'),
        # Capa de MaxPooling
        MaxPooling2D(),
        # Capa de Flatten
        Flatten(),
        # Capa densa
        Dense(50, activation='relu'),
        # Capa densa
        Dense(50, activation='relu'),
        # Capa densa
        Dense(len(class_names), activation='softmax')  # Capa de salida
])



# Compilar el modelo
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()
