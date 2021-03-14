import os
import cv2


def blur_images(input_path, dest_path):
    img_paths = [f for f in os.listdir(input_path) if not f.startswith('.')]
    count = 0
    for img_path in img_paths:
        img = cv2.imread(input_path + img_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(dest_path + img_path, blurred)
        count += 1
        print('Image', count, 'processed')


if __name__ == '__main__':
    input_dir_path = 'data/test/depth/'
    dest_dir_path = 'output/fine_tuned/'
    blur_images(input_dir_path, dest_dir_path)
