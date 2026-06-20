import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

def read_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def otsu_threshold(gray):
    hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0
    w_b = 0
    max_between = 0
    best_t = 0

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

        between = w_b * w_f * (mean_b - mean_f) ** 2
        if between > max_between:
            max_between = between
            best_t = t

    return best_t

def compute_markers(binary):
    struct = np.ones((3, 3))

    sure_fg = ndimage.binary_erosion(binary, structure=struct, iterations=3)
    sure_bg = ndimage.binary_dilation(binary, structure=struct, iterations=5)
    unknown = sure_bg ^ sure_fg

    markers, _ = ndimage.label(sure_fg)
    markers = markers.astype(np.int32)
    markers[unknown] = 0

    return markers

def watershed(image, markers):
    h, w = markers.shape
    gray = to_gray(image)

    from queue import PriorityQueue
    pq = PriorityQueue()

    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:
                pq.put((gray[y, x], y, x))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    out = markers.copy()

    while not pq.empty():
        _, y, x = pq.get()

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if out[ny, nx] == 0:
                    out[ny, nx] = out[y, x]
                    pq.put((gray[ny, nx], ny, nx))

    return out

def random_color_labels(labels):
    unique = np.unique(labels)
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(123)

    for u in unique:
        if u == 0:
            continue
        color = rng.integers(0, 255, 3)
        out[labels == u] = color

    return out

def process_image(path, title):
    img = read_image(path)
    gray = to_gray(img)
    t = otsu_threshold(gray)
    binary = gray > t

    markers = compute_markers(binary)
    labels = watershed(img, markers)
    seg = random_color_labels(labels)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    ax[0, 0].imshow(img)
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(binary, cmap="gray")
    ax[0, 1].set_title(f"Otsu Binary (T={t})")

    ax[1, 0].imshow(markers, cmap="nipy_spectral")
    ax[1, 0].set_title("Markers")

    ax[1, 1].imshow(seg)
    ax[1, 1].set_title("Watershed Result")

    for a in ax.flatten():
        a.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    process_image("dices.jpg", "Kości")
    process_image("shuttle.jpeg", "Wachadłowiec")

if __name__ == "__main__":
    main()
