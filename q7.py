import cv2
import numpy as np

def harris_corner_detection(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image: {image_path}")
        return
    
    # We'll convert the image to grayscale and float32 for Harris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Let's define a few parameter sets for Harris corner detection.
    # Feel free to add or modify these to see how the results change.
    param_sets = [
        (2, 3, 0.04),
        (4, 3, 0.04),
        (2, 5, 0.06)
    ]

    for idx, (blockSize, kSize, k) in enumerate(param_sets):
        # 2. Apply Harris corner detection
        #    blockSize - neighborhood size
        #    kSize     - aperture parameter for the Sobel operator
        #    k         - Harris detector free parameter (usually between 0.04 and 0.06)
        dst = cv2.cornerHarris(gray, blockSize, kSize, k)

        # 3. Dilate the corner response image to enhance corner points
        dst = cv2.dilate(dst, None)

        # 4. Threshold to identify strong corners
        #    Mark corners in RED on a copy of the original image
        thresh = 0.01 * dst.max()
        result_img = image.copy()
        result_img[dst > thresh] = [0, 0, 255]  # BGR => Red

        # 5. Display the results
        window_title = f"Harris Corners - blockSize={blockSize}, kSize={kSize}, k={k}"
        cv2.imshow(window_title, result_img)

    # Wait until a key is pressed, then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Replace "house.jpg" with the path to your image
    image_path = "house.jpg"
    harris_corner_detection(image_path)

if __name__ == "__main__":
    main()
