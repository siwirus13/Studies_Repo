# Fashion MNIST Classification with Convolutional Neural Network
# Klasyfikacja Fashion MNIST przy użyciu konwolucyjnej sieci neuronowej

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Ustawienia dla lepszej wizualizacji
plt.style.use('default')
sns.set_palette("husl")

print("Fashion MNIST - Klasyfikacja obrazów odzieży")
print("=" * 50)

# 1. Wczytanie danych Fashion MNIST
print("1. Wczytywanie danych Fashion MNIST...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Nazwy klas w Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Dane treningowe: {X_train.shape}")
print(f"Etykiety treningowe: {y_train.shape}")
print(f"Dane testowe: {X_test.shape}")
print(f"Etykiety testowe: {y_test.shape}")
print(f"Liczba klas: {len(class_names)}")

# 2. Eksploracja danych
print("\n2. Eksploracja danych...")
print(f"Zakres wartości pikseli: {X_train.min()} - {X_train.max()}")
print(f"Rozkład klas treningowych:")
unique, counts = np.unique(y_train, return_counts=True)
for i, (cls, count) in enumerate(zip(unique, counts)):
    print(f"  {class_names[cls]}: {count} próbek")

# Wizualizacja przykładowych obrazów
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Przykładowe obrazy z Fashion MNIST', fontsize=16)
for i in range(10):
    ax = axes[i//5, i%5]
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'{class_names[y_train[i]]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 3. Przygotowanie danych
print("\n3. Przygotowanie danych...")

# Normalizacja pikseli do zakresu [0, 1]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

# Dodanie wymiaru kanału (dla CNN)
X_train_norm = X_train_norm.reshape(-1, 28, 28, 1)
X_test_norm = X_test_norm.reshape(-1, 28, 28, 1)

# One-hot encoding etykiet
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Kształt danych po przetworzeniu:")
print(f"X_train: {X_train_norm.shape}")
print(f"X_test: {X_test_norm.shape}")
print(f"y_train: {y_train_cat.shape}")
print(f"y_test: {y_test_cat.shape}")

# 4. Budowa modelu CNN
print("\n4. Budowa modelu sieci neuronowej...")

model = keras.Sequential([
    # Pierwsza warstwa konwolucyjna
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Druga warstwa konwolucyjna
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Trzecia warstwa konwolucyjna
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Spłaszczenie i warstwy w pełni połączone
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Regularizacja
    layers.Dense(10, activation='softmax')  # Warstwa wyjściowa
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Podsumowanie architektury
print("\nArchitektura modelu:")
model.summary()

# 5. Trenowanie modelu
print("\n5. Trenowanie modelu...")

# Callbacks dla lepszego trenowania
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

# Trenowanie
history = model.fit(X_train_norm, y_train_cat,
                    epochs=15,
                    batch_size=128,
                    validation_data=(X_test_norm, y_test_cat),
                    callbacks=callbacks,
                    verbose=1)

# Wizualizacja procesu trenowania
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Dokładność
ax1.plot(history.history['accuracy'], label='Trenowanie')
ax1.plot(history.history['val_accuracy'], label='Walidacja')
ax1.set_title('Dokładność modelu')
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Dokładność')
ax1.legend()
ax1.grid(True)

# Strata
ax2.plot(history.history['loss'], label='Trenowanie')
ax2.plot(history.history['val_loss'], label='Walidacja')
ax2.set_title('Strata modelu')
ax2.set_xlabel('Epoka')
ax2.set_ylabel('Strata')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 6. Ocena modelu
print("\n6. Ocena modelu na zbiorze testowym...")

# Ewaluacja na zbiorze testowym
test_loss, test_accuracy = model.evaluate(X_test_norm, y_test_cat, verbose=0)
print(f"Dokładność na zbiorze testowym: {test_accuracy:.4f}")
print(f"Strata na zbiorze testowym: {test_loss:.4f}")

# 7. Predykcje i analiza wyników
print("\n7. Predykcje i analiza wyników...")

# Predykcje
y_pred = model.predict(X_test_norm)
y_pred_classes = np.argmax(y_pred, axis=1)

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred_classes)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Macierz pomyłek')
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Dokładność dla każdej klasy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nDokładność dla poszczególnych klas:")
for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
    print(f"{name:15}: {acc:.4f} ({acc*100:.2f}%)")

# Wizualizacja dokładności klas
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(class_names)), class_accuracy)
plt.xlabel('Klasy')
plt.ylabel('Dokładność')
plt.title('Dokładność klasyfikacji dla poszczególnych klas')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.ylim(0, 1)

# Kolorowanie słupków
for i, bar in enumerate(bars):
    if class_accuracy[i] > 0.9:
        bar.set_color('green')
    elif class_accuracy[i] > 0.8:
        bar.set_color('orange')
    else:
        bar.set_color('red')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Przykłady błędnych klasyfikacji
print("\n8. Analiza błędnych klasyfikacji...")
wrong_predictions = np.where(y_pred_classes != y_test)[0]
print(f"Liczba błędnych predykcji: {len(wrong_predictions)}")

# Wizualizacja kilku błędnych predykcji
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Przykłady błędnych klasyfikacji', fontsize=16)

for i, idx in enumerate(wrong_predictions[:10]):
    ax = axes[i//5, i%5]
    ax.imshow(X_test[idx], cmap='gray')
    pred_class = class_names[y_pred_classes[idx]]
    true_class = class_names[y_test[idx]]
    confidence = np.max(y_pred[idx]) * 100
    ax.set_title(f'Prawda: {true_class}\nPredykcja: {pred_class}\nPewność: {confidence:.1f}%', 
                 fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()

# 8. Wnioski i rekomendacje
print("\n" + "="*60)
print("WNIOSKI I ANALIZA WYNIKÓW")
print("="*60)

print(f"\n1. OGÓLNA WYDAJNOŚĆ MODELU:")
print(f"   - Dokładność na zbiorze testowym: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   - Liczba parametrów modelu: {model.count_params():,}")

print(f"\n2. ANALIZA KLAS:")
best_classes = np.argsort(class_accuracy)[-3:][::-1]
worst_classes = np.argsort(class_accuracy)[:3]

print(f"   Najlepiej klasyfikowane klasy:")
for idx in best_classes:
    print(f"   - {class_names[idx]}: {class_accuracy[idx]:.4f} ({class_accuracy[idx]*100:.2f}%)")

print(f"\n   Najgorzej klasyfikowane klasy:")
for idx in worst_classes:
    print(f"   - {class_names[idx]}: {class_accuracy[idx]:.4f} ({class_accuracy[idx]*100:.2f}%)")

print(f"\n3. REKOMENDACJE DO POPRAWY MODELU:")
print(f"   - Zwiększenie złożoności modelu (więcej warstw konwolucyjnych)")
print(f"   - Zastosowanie data augmentation (obroty, przesunięcia)")
print(f"   - Użycie technik regularizacji (Batch Normalization)")
print(f"   - Transfer learning z pretrenowanymi modelami")
print(f"   - Dostrojenie hiperparametrów (learning rate, batch size)")
print(f"   - Zastosowanie ansambli modeli")

print(f"\n4. OBSERWACJE SPECYFICZNE:")
if class_accuracy[6] < 0.8:  # Shirt
    print(f"   - Koszule często mylone z T-shirt/top - podobne kształty")
if class_accuracy[2] < 0.8:  # Pullover
    print(f"   - Swetry mylone z płaszczami - podobna sylwetka")
if class_accuracy[5] < 0.9:  # Sandal
    print(f"   - Sandały dobrze rozpoznawane - charakterystyczny kształt")

print(f"\n5. NASTĘPNE KROKI:")
print(f"   - Implementacja zaawansowanych architektur (ResNet, DenseNet)")
print(f"   - Analiza błędów na poziomie pikseli")
print(f"   - Optymalizacja modelu pod kątem czasu inferencji")
print(f"   - Testowanie na rzeczywistych danych spoza zbioru Fashion MNIST")

print("\n" + "="*60)
print("KONIEC ANALIZY")
print("="*60)
