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


input_image = cv2.imread(args.input_file, 0)

canny_image = auto_canny(input_image)

cv2.imwrite('hoge/hoge.png', canny_image)
