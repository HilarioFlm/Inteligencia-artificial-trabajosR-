# 2_train_model.py
"""
Script 2: Entrenamiento con Transfer Learning (MobileNetV2)
- Usa imágenes RGB y ImageDataGenerator con validation_split.
- Calcula class weights para balancear el entrenamiento.
- Opcional: fine-tuning de las últimas capas.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

DATASET_PATH = "dataset_rostros"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
NUM_EPOCHS = 25
SEED = 42
MODEL_OUT = "face_recognition_mobilenetv2.h5"
LABELS_OUT = "class_labels.txt"
FINE_TUNE = True  # Si True, descongelar últimas capas después de entrenamiento inicial
FINE_TUNE_AT = 100  # número de capas desde el final para descongelar (ajustable)

# Detect classes
CLASES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASES = len(CLASES)
print(f"Clases detectadas: {CLASES} | Total: {NUM_CLASES}")

# Data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

# Calcular class weights para balanceo
# class_counts basado en estructura de carpetas
class_counts = {}
for cls in CLASES:
    cls_path = os.path.join(DATASET_PATH, cls)
    class_counts[cls] = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])

total_samples = sum(class_counts.values())
class_indices = train_generator.class_indices  # mapping name->index
class_weight = {
    class_indices[name]: total_samples / (NUM_CLASES * count) if count > 0 else 1.0
    for name, count in class_counts.items()
}
print("Class counts:", class_counts)
print("Class weight:", class_weight)

# Modelo base MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Head personalizado
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASES, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss')
]

# Entrenamiento inicial
history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weight,
    callbacks=callbacks
)

# Fine-tuning opcional
if FINE_TUNE:
    print("Iniciando fine-tuning: descongelando últimas capas del backbone.")
    base_model.trainable = True
    # congelar hasta la capa FINE_TUNE_AT
    for layer in base_model.layers[: -FINE_TUNE_AT]:
        layer.trainable = False
    for layer in base_model.layers[-FINE_TUNE_AT:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    fine_history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        class_weight=class_weight,
        callbacks=callbacks
    )

# Guardar modelo final y etiquetas
model.save(MODEL_OUT)
print(f"Modelo guardado en: {MODEL_OUT}")

with open(LABELS_OUT, 'w') as f:
    for label, idx in train_generator.class_indices.items():
        f.write(f"{idx}:{label}\n")
print(f"Etiquetas guardadas en: {LABELS_OUT}")

# Graficar historial
try:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
except Exception as e:
    print('No se pudo graficar:', e)
