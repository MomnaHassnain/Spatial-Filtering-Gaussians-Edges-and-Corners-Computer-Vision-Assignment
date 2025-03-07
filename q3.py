import numpy as np
import cv2
import math

def gaussian_kernel(sigma):
    # Choose a kernel size (odd number) based on sigma
    # e.g., 6*sigma rounded up to an odd integer
    kernel_size = int(6 * sigma + 1)  
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Coordinate grids, centered at zero
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)

    # Gaussian formula in float
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    # Normalize so that sum of all values = 1
    kernel /= kernel.sum()

    return kernel  # floating-point kernel


def gaussian_kernel_viz(sigma):
    # Determine the kernel size roughly as d = 2πσ.
    # We adjust to make sure the kernel size is odd for symmetry.
    d = int(2 * np.pi * sigma)
    if d % 2 == 0:
        d += 1

    # Create coordinate grid centered at zero.
    ax = np.linspace(-(d // 2), d // 2, d)
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute the Gaussian function.
    kernel = 1 / (2 * np.pi * sigma**2) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize kernel values to the range 0-255 for visualization.
    kernel_norm = kernel - np.min(kernel)
    kernel_norm = kernel_norm / np.max(kernel_norm)
    kernel_norm = (kernel_norm * 255).astype(np.uint8)
    
    return kernel_norm


def visualize_gaussian(sigma):
    """
    Create and visualize a Gaussian filter.
    :param sigma: Standard deviation of the Gaussian function
    """
    kernel = gaussian_kernel_viz(sigma)
    kernel_visual = ((kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255).astype(np.uint8)
    
    enlarged = cv2.resize(kernel_visual, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(f'Gaussian Filter (σ={sigma})', enlarged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    sigma_values = [1, 3, 5]  # Different standard deviations
    for sigma in sigma_values:
        visualize_gaussian(sigma)
    
if __name__ == "__main__":
    main()
