import cv2
import numpy as np
def main():
    # 1. Load the two grayscale images
    sq1 = cv2.imread('black&white.jpg', cv2.IMREAD_GRAYSCALE)
    sq2 = cv2.imread('house.jpg', cv2.IMREAD_GRAYSCALE)   
    if sq1 is None or sq2 is None:
        print("Error: Could not load one of the images. Check the file paths.")
        return
    sq2 = cv2.resize(sq2, (sq1.shape[1], sq1.shape[0]))

    # (Optional) Convert to true binary (0 or 255) if needed
    # _, sq1 = cv2.threshold(sq1, 127, 255, cv2.THRESH_BINARY)
    # _, sq2 = cv2.threshold(sq2, 127, 255, cv2.THRESH_BINARY)

    octagon_or  = cv2.bitwise_or(sq1, sq2)
    octagon_and = cv2.bitwise_and(sq1, sq2)
    star_xor    = cv2.bitwise_xor(sq1, sq2)
    cv2.imshow("black&white (sq1)", sq1)
    cv2.imshow("house (sq2) resized", sq2)
    cv2.imshow("Octagon (OR)", octagon_or)
    cv2.imshow("Octagon (AND)", octagon_and)
    cv2.imshow("8-pointed Star (XOR)", star_xor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
