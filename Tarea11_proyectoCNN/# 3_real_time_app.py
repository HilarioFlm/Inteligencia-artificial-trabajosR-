# 3_real_time_app.py
"""
Script 3: Aplicación en tiempo real
- Usa MTCNN (si disponible) para detección.
- Usa el modelo entrenado (MobileNetV2 head) para clasificación.
- Mide FPS y muestra probabilidades.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os

MODEL_PATH = "face_recognition_mobilenetv2.h5"
LABELS_PATH = "class_labels.txt"
IMG_SIZE = (150,150)
CAMERA_INDEX = 0

# Cargar modelo
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print("Error al cargar el modelo:", e)
    raise SystemExit

# Cargar etiquetas
class_labels = {}
with open(LABELS_PATH, 'r') as f:
    for line in f:
        idx, lab = line.strip().split(':')
        class_labels[int(idx)] = lab

# Detector MTCNN (fallback a Haar si no está)
use_mtcnn = True
try:
    from mtcnn import MTCNN
    detector = MTCNN()
    print("Usando MTCNN para detección en tiempo real.")
except Exception:
    detector = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("MTCNN no disponible: usando Haar Cascade.")

cap = cv2.VideoCapture(CAMERA_INDEX)

# FPS measurement
fps_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rostros = []
    if detector is not None:
        results = detector.detect_faces(rgb)
        for r in results:
            x, y, w, h = r['box']
            x, y = max(0, x), max(0, y)
            rostros.append((x, y, w, h))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        rostros = found

    for (x, y, w, h) in rostros:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Preprocesamiento: RGB -> resize -> preprocess_input
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)

        # Predicción
        preds = model.predict(img)
        idx = np.argmax(preds[0])
        prob = np.max(preds[0])
        label = class_labels.get(idx, 'unknown')

        color = (0,255,0) if label == 'yo' else (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        texto = f"{label} ({prob*100:.1f}%)"
        cv2.putText(frame, texto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostrar FPS
    fps_counter += 1
    if time.time() - start_time >= 1.0:
        fps = fps_counter / (time.time() - start_time)
        fps_counter = 0
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow('Reconocimiento Facial en Vivo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
