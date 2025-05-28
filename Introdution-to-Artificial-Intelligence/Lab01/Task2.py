## TASK 2



import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns


matplotlib.use("TkAgg")

# === KONFIGURACJA ===
FOLDER = 'dataset_task2'
MODEL_PATH = 'model.h5'
IMG_SIZE = (28, 28)
GRAYSCALE = True

# === 1. Wczytaj model ===
model = load_model(MODEL_PATH)

# === 2. Wczytaj dane testowe ===
X_test = []
y_test = []
file_paths = []

for filename in os.listdir(FOLDER):
    if filename.endswith('.png') and filename.startswith('digit_'):
        path = os.path.join(FOLDER, filename)
        file_paths.append(path)

        # Wyciągnij cyfrę z nazwy pliku, np. digit_32.png -> 2
        digit_label = int(filename.split('_')[1]) % 10
        y_test.append(digit_label)

        # Wczytaj i przeskaluj obrazek
        img = load_img(path, color_mode='grayscale', target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((IMG_SIZE[0], IMG_SIZE[1], 1))
        X_test.append(img_array)

X_test = np.array(X_test)
y_test = np.array(y_test)

# === 3. Predykcja ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# === 4. Raport tekstowy ===
print("=== RAPORT ===")
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# === 5. Confusion Matrix ===
conf_mat = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()

# Zapis do pliku — działa w terminalu
plt.savefig("confusion_matrix_task2.png")
print("Zapisano confusion matrix jako confusion_matrix_task2.png")

# === 6. Opcjonalnie: błędne klasyfikacje ===
print("\n=== BŁĘDNE KLASYFIKACJE ===")
for i in range(len(y_test)):
    if y_test[i] != y_pred_classes[i]:
        print(f"Plik: {file_paths[i]} | Prawidłowo: {y_test[i]} | Przewidziano: {y_pred_classes[i]}")
