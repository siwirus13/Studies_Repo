
import cv2
import numpy as np

def task1():
    image = cv2.imread("shuttle.jpeg")
    if image is None:
        raise FileNotFoundError("File shuttle.jpeg not found")
    return image

def task2_prepare_lowres(image):
    # Still degraded enough to show differences, but not ultra blurry
    h, w = image.shape[:2]
    small = cv2.resize(image, (450, 450), interpolation=cv2.INTER_AREA)
    return small

def task3_resize(image, scale, method):
    scale = max(0.1, min(scale, 15.0))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=method)

def task4_show():
    original = task1()
    lowres = task2_prepare_lowres(original)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Scaled", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Difference", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Original", 300, 300)
    cv2.resizeWindow("Scaled", 500, 500)
    cv2.resizeWindow("Difference", 500, 500)

    cv2.createTrackbar("Scale x10", "Scaled", 10, 150, lambda x: None)
    cv2.createTrackbar("Method", "Scaled", 0, 3, lambda x: None)

    methods = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
    ]

    while True:
        scale_val = cv2.getTrackbarPos("Scale x10", "Scaled") / 10.0
        method_id = cv2.getTrackbarPos("Method", "Scaled")
        method = methods[method_id]

        scaled = task3_resize(lowres, scale_val, method)

        # Compute difference against a reference resize of original
        try:
            reference = cv2.resize(original, scaled.shape[1::-1], interpolation=cv2.INTER_AREA)
            diff = cv2.absdiff(reference, scaled)

            # boosting visibility for dark differences
            diff_display = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        except:
            diff_display = np.zeros_like(scaled)

        cv2.imshow("Original", original)
        cv2.imshow("Scaled", scaled)
        cv2.imshow("Difference", diff_display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    task4_show()

if __name__ == "__main__":
    main()
