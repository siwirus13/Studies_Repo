import cv2
import numpy as np

# ========================
# Task 1: Load the image
# ========================
def task1():
    image = cv2.imread('shuttle.jpeg')  # <-- Podaj swoją ścieżkę do obrazu
    if image is None:
        raise FileNotFoundError("Nie znaleziono pliku obrazu! Podaj poprawną ścieżkę.")
    return image

# ========================
# Task 2: Add noise
# ========================
def add_salt_pepper_noise(image, intensity):
    noisy = np.copy(image)
    prob = intensity / 100.0

    rnd = np.random.rand(*image.shape[:2])
    noisy[rnd < prob / 2] = 0
    noisy[rnd > 1 - prob / 2] = 255
    return noisy

def add_gaussian_noise(image, intensity):
    mean = 0
    sigma = intensity
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = cv2.add(image.astype('float32'), gauss)
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def task2(image, noise_type, noise_intensity):
    if noise_type == 0:
        return add_salt_pepper_noise(image, noise_intensity)
    elif noise_type == 1:
        return add_gaussian_noise(image, noise_intensity)
    else:
        return image

# ========================
# Task 3: Denoise (filter)
# ========================
def apply_filter(noisy_image, filter_type, kernel_size):
    if kernel_size % 2 == 0:  # kernel size must be odd
        kernel_size += 1

    if filter_type == 0:
        # Mean filter
        return cv2.blur(noisy_image, (kernel_size, kernel_size))
    elif filter_type == 1:
        # Median filter
        return cv2.medianBlur(noisy_image, kernel_size)
    elif filter_type == 2:
        # Gaussian filter
        return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
    else:
        return noisy_image

# ========================
# Task 4: Main UI & logic
# ========================
def task4():
    image = task1()

    cv2.namedWindow("Original")
    cv2.namedWindow("Noisy")
    cv2.namedWindow("Denoised")

    # Create Trackbars
    cv2.createTrackbar("Noise Type", "Noisy", 0, 1, lambda x: None)  # 0 - salt&pepper, 1 - Gaussian
    cv2.createTrackbar("Noise Intensity", "Noisy", 10, 100, lambda x: None)

    cv2.createTrackbar("Filter Type", "Denoised", 0, 2, lambda x: None)  # 0 - mean, 1 - median, 2 - gaussian
    cv2.createTrackbar("Kernel Size", "Denoised", 3, 20, lambda x: None)

    while True:
        # Read values from trackbars
        noise_type = cv2.getTrackbarPos("Noise Type", "Noisy")
        noise_intensity = cv2.getTrackbarPos("Noise Intensity", "Noisy")

        filter_type = cv2.getTrackbarPos("Filter Type", "Denoised")
        kernel_size = cv2.getTrackbarPos("Kernel Size", "Denoised")

        noisy = task2(image, noise_type, noise_intensity)
        denoised = apply_filter(noisy, filter_type, kernel_size)

        # Show images
        cv2.imshow("Original", image)
        cv2.imshow("Noisy", noisy)
        cv2.imshow("Denoised", denoised)

        # Exit on ESC
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

# ========================
# Main Function
# ========================
def main():
    task4()

if __name__ == "__main__":
    main()

