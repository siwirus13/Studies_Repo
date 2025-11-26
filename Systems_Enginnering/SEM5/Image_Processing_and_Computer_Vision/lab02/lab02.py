
import cv2
import numpy as np

def show_resized_window(win_name, img, max_width=640, max_height=480):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, resized)


def task1(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    show_resized_window('Original Image', img)
    return img


def task2a(img):
    def update(_=None):
        low = cv2.getTrackbarPos('Low', 'Canny')
        high = cv2.getTrackbarPos('High', 'Canny')
        edges = cv2.Canny(img, low, high)
        show_resized_window('Canny', edges)

    cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Low', 'Canny', 50, 255, update)
    cv2.createTrackbar('High', 'Canny', 150, 255, update)

    update()
    print("Press 'q' to close Canny window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Canny')


def task2b(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def update(_=None):
        thresh_val = cv2.getTrackbarPos('Threshold', 'Threshold')
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        show_resized_window('Threshold', binary)

    cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Threshold', 'Threshold', 127, 255, update)

    update()
    print("Press 'q' to close Threshold window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Threshold')


def task2c(img):
    def update(_=None):
        k = cv2.getTrackbarPos('Kernel', 'Gaussian Blur')
        if k % 2 == 0:
            k += 1  # kernel must be odd
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        show_resized_window('Gaussian Blur', blurred)

    cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Kernel', 'Gaussian Blur', 1, 30, update)

    update()
    print("Press 'q' to close Gaussian Blur window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Gaussian Blur')


def main():
    path = 'shuttle.jpeg'
    img = task1(path)

    print("\n--- TASK 2A: Canny Edge Detection ---")
    task2a(img)

    print("\n--- TASK 2B: Binary Thresholding ---")
    task2b(img)

    print("\n--- TASK 2C: Gaussian Blur ---")
    task2c(img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

