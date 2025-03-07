import cv2
import numpy as np

# Define Sobel kernels for horizontal and vertical edge detection
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

def convolve2D(image, kernel):
    # Example of a custom 2D convolution (if needed)
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Pad the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros_like(image, dtype=np.float32)
    
    # Perform convolution
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

def edge_detection_sobel(image_path):
    # 1. Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return
    # 2. Convolve with horizontal and vertical filters
    gx = convolve2D(image, sobel_x)
    gy = convolve2D(image, sobel_y)
    # Convert them to absolute values for display
    gx_display = cv2.convertScaleAbs(gx)
    gy_display = cv2.convertScaleAbs(gy)
    # 3. Combine the two results:
    #    A) Direct sum (gx + gy)
    combined = gx + gy
    combined_display = cv2.convertScaleAbs(combined)
    # Alternatively, B) gradient magnitude (common approach)
    # magnitude = np.sqrt(gx**2 + gy**2)
    # magnitude_display = np.clip(magnitude, 0, 255).astype(np.uint8)
    # 4. Resize images to a smaller dimension, e.g., 400×400
    resized_original = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)
    resized_gx = cv2.resize(gx_display, (400, 400), interpolation=cv2.INTER_AREA)
    resized_gy = cv2.resize(gy_display, (400, 400), interpolation=cv2.INTER_AREA)
    resized_combined = cv2.resize(combined_display, (400, 400), interpolation=cv2.INTER_AREA)
    # 5. Show results in smaller windows
    cv2.imshow("Original (resized)", resized_original)
    cv2.imshow("Horizontal Edges (gx)", resized_gx)
    cv2.imshow("Vertical Edges (gy)", resized_gy)
    cv2.imshow("Combined Edge Map", resized_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    edge_detection_sobel("house.jpg")
