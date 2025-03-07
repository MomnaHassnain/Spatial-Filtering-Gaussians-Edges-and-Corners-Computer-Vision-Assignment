import numpy as np
import cv2
from q1 import convolve2D

# Load a real image
def main():
    image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    if image is None:
        print("Error: Image not found.")
        return
    
    
    image = cv2.resize(image, (512, 512))
    
    # Define multiple filters
    filters = {
        "Edge Detection": np.array([[1, 0, -1],
                                     [1, 0, -1],
                                     [1, 0, -1]]),
        "Sharpen": np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]]),
        "Gaussian Blur": np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]) / 16,
        "Large Kernel Blur": np.ones((5, 5)) / 25
    }
    
    for filter_name, kernel in filters.items():
        convolved_output = convolve2D(image, kernel)
        cv2.imshow(filter_name, convolved_output.astype(np.uint8))
    
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
