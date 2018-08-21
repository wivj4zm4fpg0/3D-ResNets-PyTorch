import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_file',
    default=None,
    type=str,
    help='input file'
)
parser.add_argument(
    '--output_file',
    default=None,
    type=str,
    help='saved output file name'
)
args = parser.parse_args()


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, L2gradient=True)

    # return the edged image
    return edged


input_image = cv2.imread(args.input_file)

gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

sobel_image = cv2.Sobel(gray_image, cv2.CV_8U, 1, 1, ksize=3)

gaussian_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
laplacian_image = cv2.Laplacian(gaussian_image, cv2.CV_8U, ksize=3)

canny_image = cv2.Canny(gray_image, 100, 200)

canny_image2 = cv2.Canny(gray_image, 75, 100, L2gradient=True)

canny_image3 = auto_canny(gray_image)

cv2.imshow('original', input_image)
cv2.imshow('gray', gray_image)
cv2.imshow('sobel', sobel_image)
cv2.imshow('laplacian', laplacian_image)
cv2.imshow('canny', canny_image)
cv2.imshow('canny2', canny_image2)
cv2.imshow('canny3', canny_image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(231)
# plt.title('original')
# plt.imshow(input_image)
#
# plt.subplot(234)
# plt.title('gray')
# plt.imshow(gray_image)
#
# plt.subplot(232)
# plt.title('sobel')
# plt.imshow(sobel_image)
#
# plt.subplot(235)
# plt.title('laplacian')
# plt.imshow(laplacian_image)
#
# plt.subplot(233)
# plt.title('canny')
# plt.imshow(canny_image)

# plt.subplot(236)
# plt.title('canny2')
# plt.imshow(canny_image2)
#
# plt.show()
