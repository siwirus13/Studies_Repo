import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Neural_Network:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, activation_type="sigmoid"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if activation_type == 'sigmoid':
            self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1.0/self.input_size)
            self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1.0/self.hidden_size)
        else:
            self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0/self.input_size)
            self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0/self.hidden_size)
            
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

        if activation_type == 'sigmoid':
            self.hidden_activation = self._sigmoid
            self.hidden_activation_derivative = self._sigmoid_derivative
        elif activation_type == 'relu':
            self.hidden_activation = self._relu
            self.hidden_activation_derivative = self._relu_derivative

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _mse_loss(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted)**2)

    def _mse_loss_derivative(self, y_true, y_predicted):
        return 2 * (y_predicted - y_true) / y_true.shape[0]

    def forward(self, X):
        # Warstwa 1 (ukryta)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.hidden_activation(z1)
        
        # Warstwa 2 (wyjściowa)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.hidden_activation(z2)  

        cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        return a2, cache

    def backward(self, y_true, cache, learning_rate):

        X = cache['X']
        z1, a1, z2, a2 = cache['z1'], cache['a1'], cache['z2'], cache['a2']
        
        m = X.shape[0]

        delta2 = self._mse_loss_derivative(y_true, a2) * self._sigmoid_derivative(z2)
        
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * self.hidden_activation_derivative(z1)
        
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Krok 5: Aktualizacja parametrów
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1 
        self.b1 -= learning_rate * db1

    def train(self, X_train, y_train, epochs, learning_rate):
        print(f"\nStarting training for {epochs} epochs with learning rate {learning_rate}...")
        for epoch in range(epochs):
            predictions, cache = self.forward(X_train)
            loss = self._mse_loss(y_train, predictions)
            self.backward(y_train, cache, learning_rate)
            if (epoch + 1) % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        print("Training complete.")

    def predict(self, X):
        predictions, _ = self.forward(X)
        return predictions

def generate_data(num_samples=1000):
    X = np.random.uniform(-1, 1, size=(num_samples, 2))
    y = np.where(np.sign(X[:, 0]) == np.sign(X[:, 1]), 1, 0).reshape(-1, 1)
    return X, y

def normalize_l1(data):
    norm = np.sum(np.abs(data), axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    return data / norm

def normalize_l2(data):
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    return data / norm

def evaluate_model_with_confusion_matrix(model, X_test, y_test, model_name, threshold=0.5):
    predictions = model.predict(X_test)
    binary_predictions = (predictions >= threshold).astype(int)
    accuracy = np.mean(binary_predictions == y_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test.flatten(), binary_predictions.flatten())
    
    # Metryki dodatkowe
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, cm, precision, recall, f1

if __name__ == "__main__":
    np.random.seed(42)  # Dla reprodukowalności
    num_samples = 2000 
    epochs = 5000
    learning_rate = 0.1

    print("--- Generating Data ---")
    X_raw, y_raw = generate_data(num_samples)
    print(f"Raw data shape: X={X_raw.shape}, y={y_raw.shape}")

    split_idx = int(num_samples * 0.8)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]
    print(f"Training data size: {len(X_train_raw)}, Test data size: {len(X_test_raw)}")

    # Przygotowanie wszystkich wersji danych
    X_l1 = normalize_l1(X_raw)
    X_train_l1, X_test_l1 = X_l1[:split_idx], X_l1[split_idx:]
    
    X_l2 = normalize_l2(X_raw)
    X_train_l2, X_test_l2 = X_l2[:split_idx], X_l2[split_idx:]

    # Lista modeli do trenowania
    models_config = [
        ("Sigmoid + Raw Data", X_train_raw, X_test_raw, "sigmoid"),
        ("ReLU + Raw Data", X_train_raw, X_test_raw, "relu"),
        ("Sigmoid + L1 Normalized", X_train_l1, X_test_l1, "sigmoid"),
        ("ReLU + L1 Normalized", X_train_l1, X_test_l1, "relu"),
        ("Sigmoid + L2 Normalized", X_train_l2, X_test_l2, "sigmoid"),
        ("ReLU + L2 Normalized", X_train_l2, X_test_l2, "relu")
    ]

    trained_models = []
    results = []

    # Trenowanie wszystkich modeli
    for name, X_train, X_test, activation in models_config:
        print(f"Training: {name}")
        
        model = Neural_Network(input_size=2, hidden_size=4, output_size=1, activation_type=activation)
        model.train(X_train, y_train_raw, epochs, learning_rate)
        
        accuracy, cm, precision, recall, f1 = evaluate_model_with_confusion_matrix(model, X_test, y_test_raw, name)
        
        trained_models.append((name, model, X_test))
        results.append((name, accuracy, cm, precision, recall, f1))

    # Tworzenie wykresów z macierzami pomyłek
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices for All Model Configurations', fontsize=16, fontweight='bold')
    
    for idx, (name, accuracy, cm, precision, recall, f1) in enumerate(results):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Heatmapa macierzy pomyłek
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        
        # Tytuł z metrykami
        title = f'{name}\nAcc: {accuracy:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Krótkie podsumowanie tekstowe
    print("\nSUMMARY:")
    print("-" * 70)
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    for name, accuracy, cm, precision, recall, f1 in results:
        print(f"{name:<30} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

    # Demonstracja predykcji - tylko krótkie podsumowanie
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS (4 random test cases):")
    print("="*50)
    
    # Wybieramy 4 losowe indeksy ze zbioru testowego
    num_showcase = 4
    test_size = len(X_test_raw)
    random_indices = np.random.choice(test_size, size=min(num_showcase, test_size), replace=False)
    
    # Pobieramy dane
    showcase_X_raw = X_test_raw[random_indices]
    showcase_y = y_test_raw[random_indices]
    
    # Pokazujemy tylko jeden przykład (najlepszy model)
    best_model_idx = max(range(len(results)), key=lambda i: results[i][1])  # najwyższa accuracy
    best_name, best_model, _ = trained_models[best_model_idx]
    
    print(f"\nBest performing model: {best_name}")
    predictions = best_model.predict(showcase_X_raw)
    for i in range(num_showcase):
        input_vec = showcase_X_raw[i]
        pred_raw = predictions[i][0]
        pred_bin = (pred_raw >= 0.5).astype(int)
        expected = showcase_y[i][0]
        status = "✓" if pred_bin == expected else "✗"
        print(f"{status} Input: [{input_vec[0]:6.3f}, {input_vec[1]:6.3f}] → Pred: {pred_raw:.3f} ({pred_bin}) | Expected: {expected}")
    
    print(f"\nTraining completed. Check the confusion matrix plot above for detailed results.")
