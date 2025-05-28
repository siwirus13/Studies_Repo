## TASK 2

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns


FOLDER = '../dataset_task2'
MODEL_PATH = 'model.h5'
IMG_SIZE = (28, 28)
GRAYSCALE = True

model = load_model(MODEL_PATH)

X_test = []
y_test = []
file_paths = []

for filename in os.listdir(FOLDER):
    if filename.endswith('.png') and filename.startswith('digit_'):
        path = os.path.join(FOLDER, filename)
        file_paths.append(path)

        digit_str = filename.split('_')[1].split('.')[0]
        digit_label = int(digit_str) % 10

        y_test.append(digit_label)

        img = load_img(path, color_mode='grayscale', target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((IMG_SIZE[0], IMG_SIZE[1], 1))
        X_test.append(img_array)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import precision_score, recall_score

classes = list(range(10))

conf_matrix = confusion_matrix(y_test, y_pred_classes)


conf_matrix_log = np.where(conf_matrix > 0, np.log(conf_matrix), 0)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_log, annot=True, fmt='.2f', cmap='YlGn', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Logarithmic Scale)')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

for i in range(len(y_test)):
    if y_test[i] != y_pred_classes[i]:
        print(f"File: {file_paths[i]} | True: {y_test[i]} | Predicted: {y_pred_classes[i]}")
