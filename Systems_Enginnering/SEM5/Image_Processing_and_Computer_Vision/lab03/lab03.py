
import cv2
import numpy as np

def task1(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img


def task2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def update(_=None):
        thresh_val = cv2.getTrackbarPos('Threshold', 'Contours')

        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = img.copy()
        area_text = "Area: 0 px"
        perimeter_text = "Perimeter: 0 px"

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            cv2.drawContours(output, [largest], -1, (0, 255, 0), 4)
            area_text = f"Area: {int(area)} px"
            perimeter_text = f"Perimeter: {int(perimeter)} px"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0  # very large text
        color = (255, 255, 0)
        thickness = 4
        shadow_color = (0, 0, 0)
        shadow_offset = 4

        cv2.putText(output, area_text, (20 + shadow_offset, 60 + shadow_offset),
                    font, scale, shadow_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(output, perimeter_text, (20 + shadow_offset, 130 + shadow_offset),
                    font, scale, shadow_color, thickness + 2, cv2.LINE_AA)

        cv2.putText(output, area_text, (20, 60),
                    font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(output, perimeter_text, (20, 130),
                    font, scale, color, thickness, cv2.LINE_AA)

        h, w = output.shape[:2]
        max_width = 800
        if w > max_width:
            scale_factor = max_width / w
            output = cv2.resize(output, (int(w * scale_factor), int(h * scale_factor)))

        cv2.imshow('Contours', output)

    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Threshold', 'Contours', 127, 255, update)

    update()
    print("Press 'q' to close Contours window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('Contours')


def main():
    path = 'shuttle.jpeg'  # replace with your image path
    img = task1(path)

    print("\n--- TASK 2: Interactive Contour Detection (with big text) ---")
    task2(img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

