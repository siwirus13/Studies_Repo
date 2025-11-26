

import cv2
import numpy as np
import matplotlib.pyplot as plt

def task1(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img

def task2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def task3(img, kernel_size=(5,7)):
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    return blurred

def task4(img, threshold_value=127):
    _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def task5(img, degrees):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated

def task6(img, scale_x, scale_y):
    scaled = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled

def main():
    path = '../lab01/shuttle.jpeg'
    rotation_angle = float(input("Enter rotation angle (degrees): "))
    scale_x = float(input("Enter scale X factor: "))
    scale_y = float(input("Enter scale Y factor: "))

    # Load and process
    original = task1(path)
    gray = task2(original)
    blurred = task3(gray)
    thresholded = task4(blurred)
    rotated = task5(original, rotation_angle)
    scaled = task6(original, scale_x, scale_y)

    # --- 1. Grayscale vs Original ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 2. Gaussian Blur vs Original ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title("Gaussian Blur")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 3. Threshold vs Original ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(thresholded, cmap='gray')
    plt.title("Binary Threshold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 4. Rotated vs Original ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rotated)
    plt.title(f"Rotated ({rotation_angle}°)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 5. Scaled vs Original ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(scaled)
    plt.title(f"Scaled (x{scale_x}, y{scale_y})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

