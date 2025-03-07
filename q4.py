import numpy as np
import cv2
import math
from q1 import convolve2D
from q3 import gaussian_kernel

def apply_gaussian_convolution(image_path, sigma):
    """
    Load an image, apply Gaussian filter convolution, and display results.
    :param image_path: Path to input image
    :param sigma: Standard deviation of the Gaussian function
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    kernel = gaussian_kernel(sigma)
    print("kernal loaded", kernel.shape)

    convolved_float = convolve2D(image.astype(np.float32), kernel)
    convolved_float = np.clip(convolved_float, 0, 255)
    convolved_image = convolved_float.astype(np.uint8)
    cv2.imshow("Convolved", convolved_image)
    cv2.imshow(f'Original Image', image)
    cv2.imshow(f'Gaussian Convolved Image (σ={sigma})', convolved_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    image_path = 'cat.jpg'  # Replace with your actual image path
    sigma = 1.5  # Small sigma to prevent overflow errors
    apply_gaussian_convolution(image_path, sigma) 
if __name__ == "__main__":
    main()
