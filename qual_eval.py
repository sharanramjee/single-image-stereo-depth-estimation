import cv2
import numpy as np


if __name__ == '__main__':
    dir1_path = 'data/test/depth/'
    dir2_path = 'output/SGDepth/'
    for i in range(1, 4, 1):
        img_name = 'test_depth_' + str(i) + '.png'
        img2_name = 'test_color_' + str(i) + '.jpg'
        img1 = cv2.imread(dir1_path + img_name, cv2.IMREAD_GRAYSCALE)
        img1_norm = cv2.normalize(img1, np.zeros(img1.shape), 0, 255, cv2.NORM_MINMAX)
        img2 = cv2.imread(dir2_path + img2_name, cv2.IMREAD_GRAYSCALE)
        img2_norm = cv2.normalize(img2, np.zeros(img2.shape), 0, 255, cv2.NORM_MINMAX)
        diff = np.abs(img1 - img2)
        cv2.imwrite(img_name, diff)
