import cv2
import numpy as np
def main():
    # 1. Load the house image
    image = cv2.imread('house.jpg')
    if image is None:
        print("Error: Could not load the image. Check the path.")
        return
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. Choose the ROI coordinates for your window (adjust as needed)
    #    (x1, y1) is top-left corner, (x2, y2) is bottom-right corner
    x1, y1 = 40, 30   # move top-left corner further out
    x2, y2 = 180, 130 # move bottom-right corner further out
    # Extract the ROI (the window area)
    window_roi = image[y1:y2, x1:x2]
    # 3. Apply a 13x13 Gaussian blur to that ROI
    blurred_roi = cv2.GaussianBlur(window_roi, (13, 13), 0)
    # Put the blurred ROI back into the original image
    image[y1:y2, x1:x2] = blurred_roi
    # 4. Display the partially blurred image
    cv2.imshow("House with Blurred Window (13x13)", image)
    # 5. Canny edge detection on the original grayscale
    edges_original = cv2.Canny(gray, 100, 200)
    cv2.imshow("Canny Edges (Original)", edges_original)
    # 6. Slightly blur the *entire* grayscale image and apply Canny with the same thresholds
    slightly_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_blurred = cv2.Canny(slightly_blurred, 100, 200)
    cv2.imshow("Canny Edges (Slightly Blurred)", edges_blurred)
    # 7. Wait and clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
