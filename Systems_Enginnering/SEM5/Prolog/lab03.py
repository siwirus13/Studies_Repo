
import random

def perceptron_rule(training_set, eta=0.1, max_epochs=100):
    w1, w2, w0 = [random.uniform(-1, 1) for _ in range(3)]
    for epoch in range(max_epochs):
        error_count = 0
        for [x, d] in training_set:
            x1, x2 = x
            s = w1 * x1 + w2 * x2 + w0
            y = 1 if s >= 0 else 0
            error = d - y
            if error != 0:
                w1 += eta * error * x1
                w2 += eta * error * x2
                w0 += eta * error
                error_count += 1
        if error_count == 0:
            print(f"[Perceptron] Bez błędów po {epoch+1} epokach")
            break
    return (w1, w2, w0, epoch + 1)


def hebb_rule(training_data, eta=0.1):
    w1, w2, w0 = 0, 0, 0
    for (x, d) in training_data:
        x1, x2 = x
        w1 += eta * d * x1
        w2 += eta * d * x2
        w0 += eta * d * 1
    return (w1, w2, w0)


def test(weights, training_data, use_minus_one=False):
    w1, w2, w0 = weights
    print("Testowanie perceptronu:")
    for (x, d) in training_data:
        x1, x2 = x
        s = w1 * x1 + w2 * x2 + w0
        if use_minus_one:
            y = 1 if s >= 0 else -1
        else:
            y = 1 if s >= 0 else 0
        print(f"x1={x1}, x2={x2} -> y={y}, oczekiwane={d}")


def main():
    print("=== Zadanie: Perceptron i reguła Hebba ===\n")

    # --- 1. OR ---
    print(">>> OR (Perceptron rule)")
    training_or = []
    for x1 in range(2):
        for x2 in range(2):
            training_or.append([[x1, x2], x1 or x2])

    w1, w2, w0, epochs_or = perceptron_rule(training_or, eta=0.2)
    test((w1, w2, w0), training_or)
    print()

    print(">>> OR (Hebb rule)")
    w1, w2, w0 = hebb_rule(training_or, eta=0.2)
    test((w1, w2, w0), training_or)
    print()

    # --- 2. AND ---
    print(">>> AND (Perceptron rule)")
    training_and = []
    for x1 in range(2):
        for x2 in range(2):
            training_and.append([[x1, x2], x1 and x2])

    w1, w2, w0, epochs_and = perceptron_rule(training_and, eta=0.2)
    test((w1, w2, w0), training_and)
    print()

    print(">>> AND (Hebb rule)")
    w1, w2, w0 = hebb_rule(training_and, eta=0.2)
    test((w1, w2, w0), training_and)
    print()

    # --- 3. Wpływ współczynnika uczenia ---
    print(">>> Wpływ współczynnika uczenia (Perceptron OR)")
    for eta in [0.1, 0.2, 0.5, 1.0]:
        _, _, _, epochs = perceptron_rule(training_or, eta=eta)
        print(f"η={eta} -> {epochs} epok")
    print()

    # --- 4. Skrócenie zbioru S4→S3 (AND) ---
    print(">>> Skrócenie zbioru S4→S3 (AND, Hebb)")
    s3_training = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 1], 1)
    ]
    w1, w2, w0 = hebb_rule(s3_training, eta=0.2)
    test((w1, w2, w0), training_and)
    print()

    # --- 5. Zastąpienie 0 → -1 ---
    print(">>> Zastąpienie 0 → -1 (Hebb OR)")
    training_minus = [
        ([-1, -1], -1),
        ([-1,  1],  1),
        ([ 1, -1],  1),
        ([ 1,  1],  1)
    ]
    w1, w2, w0 = hebb_rule(training_minus, eta=0.2)
    test((w1, w2, w0), training_minus, use_minus_one=True)
    print()


if __name__ == "__main__":
    main()
