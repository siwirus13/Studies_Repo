import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def otsu_threshold(image_gray):
    hist, _ = np.histogram(image_gray.flatten(), 256, [0, 256])
    total = image_gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0
    w_b = 0
    max_between = 0
    best_thresh = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * (mean_b - mean_f) ** 2
        if between_var > max_between:
            max_between = between_var
            best_thresh = t

    return best_thresh

def compute_markers(binary_img):
    kernel = np.ones((5, 5), np.uint8)
    sure_fg = cv2.erode(binary_img, kernel, iterations=2)
    sure_bg = cv2.dilate(binary_img, kernel, iterations=3)
    sure_bg = cv2.threshold(sure_bg, 1, 255, cv2.THRESH_BINARY)[1]
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers, _ = ndimage.label(sure_fg)
    markers = markers.astype(np.int32)
    markers[unknown == 255] = 0
    return markers

def watershed(image, markers):
    h, w = markers.shape
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    from queue import PriorityQueue
    pq = PriorityQueue()

    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:
                pq.put((image_gray[y, x], y, x))

    neigh = [(-1, -1), (-1, 0), (-1, 1),
             ( 0, -1),          ( 0, 1),
             ( 1, -1), ( 1, 0), ( 1, 1)]

    output = markers.copy()

    while not pq.empty():
        _, y, x = pq.get()
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if output[ny, nx] == 0:
                    output[ny, nx] = output[y, x]
                    pq.put((image_gray[ny, nx], ny, nx))

    return output

def process_image(path, title):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val = otsu_threshold(gray)
    binary = (gray > thresh_val).astype(np.uint8) * 255
    markers = compute_markers(binary)
    labels = watershed(img, markers)
    seg = np.zeros_like(img_rgb)
    rng = np.random.default_rng(42)

    for label in np.unique(labels):
        if label == 0:
            continue
        seg[labels == label] = rng.integers(0, 255, 3)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=18)

    ax[0, 0].imshow(img_rgb)
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(binary, cmap="gray")
    ax[0, 1].set_title(f"Otsu Binary (T={thresh_val})")

    ax[1, 0].imshow(markers, cmap="nipy_spectral")
    ax[1, 0].set_title("Markers")

    ax[1, 1].imshow(seg)
    ax[1, 1].set_title("Watershed Result")

    for a in ax.flatten():
        a.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    process_image("shuttle.jpg", "SHUTTLE – Otsu + Watershed")
    process_image("dices.jpg", "DICES – Otsu + Watershed")

if __name__ == "__main__":
    main()
