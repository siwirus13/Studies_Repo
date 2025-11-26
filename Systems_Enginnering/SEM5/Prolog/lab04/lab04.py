
import numpy as np
import csv
import random


def generate_xor_dataset(filename, n_samples, noise=0.0):
    base = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]
    data = []
    for _ in range(n_samples):
        x1, x2, y = random.choice(base)
        if random.random() < noise:
            y = 1 - y
        data.append((x1, x2, y))
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        writer.writerows(data)


def load_dataset(filename):
    data = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((int(row["x1"]), int(row["x2"]), int(row["y"])))
    return data


class XORNet:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.5):
        self.lr = lr
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(hidden_size, 1)
        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.random.randn(output_size, 1)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.h = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.h + self.b2
        self.o = self.sigmoid(self.z2)
        return self.o

    def backward(self, x, y):
        error = self.o - y
        delta_o = error * self.sigmoid_deriv(self.z2)
        delta_h = (self.W2.T @ delta_o) * self.sigmoid_deriv(self.z1)
        self.W2 -= self.lr * (delta_o @ self.h.T)
        self.b2 -= self.lr * delta_o
        self.W1 -= self.lr * (delta_h @ x.T)
        self.b1 -= self.lr * delta_h
        return float(np.sum(error**2))

    def train(self, data, epochs=5000, log_file="training_log.csv"):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "error"])
            for ep in range(epochs):
                total = 0
                for x1, x2, y in data:
                    x = np.array([[x1], [x2]])
                    y = np.array([[y]])
                    self.forward(x)
                    total += self.backward(x, y)
                writer.writerow([ep, total])

    def test(self, data, outfile="test_results.csv"):
        correct = 0
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x1", "x2", "expected", "pred_value", "pred_bin"])
            for x1, x2, y in data:
                x = np.array([[x1], [x2]])
                out = self.forward(x)
                pred = 1 if out > 0.5 else 0
                if pred == y:
                    correct += 1
                writer.writerow([x1, x2, y, float(out), pred])
        return correct / len(data)


def main():
    generate_xor_dataset("train.csv", 200, noise=0.1)
    generate_xor_dataset("test.csv", 80, noise=0.0)

    train = load_dataset("train.csv")
    test = load_dataset("test.csv")

    net = XORNet(hidden_size=4, lr=0.5)
    net.train(train, epochs=5000)
    acc = net.test(test)
    print(f"Test accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
