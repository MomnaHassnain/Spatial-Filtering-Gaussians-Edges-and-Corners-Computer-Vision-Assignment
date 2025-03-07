import numpy as np
import cv2

def convolve2D(image, kernel):
    """
    Perform a 2D convolution operation on an input image using the given filter (kernel).
    :param image: 2D NumPy array representing the input image
    :param kernel: 2D NumPy array representing the filter/kernel
    :return: 2D NumPy array representing the convolved image
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Flip the kernel (required for convolution, but not for cross-correlation)
    kernel = np.flipud(np.fliplr(kernel))
    
    # Determine the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Create an output array
    output = np.zeros((image_height, image_width))
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(
                padded_image[i:i + kernel_height, j:j + kernel_width] * kernel
            )
    
    return output

# Load a real image
def main():
    image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    if image is None:
        print("Error: Image not found.")
        return
    
    image = cv2.resize(image ,(512 , 512))
    
    kernel = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    
    print("Image loaded")
    convolved_output = convolve2D(image, kernel)
    
    # Display results
    print("Image loaded  and convolved")
    cv2.imshow('Original Image', image)
    cv2.imshow('Convolved Image', convolved_output.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
