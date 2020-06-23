import numpy as np
import cv2

def insertToFile(path, contents):
    file_object = open(path, 'a')
    file_object.write(contents)
    file_object.close()

def normalize(v):
    max_value = max(v)
    min_value = min(v)
    return np.array([2 * (v[i] - min_value) / (max_value - min_value) - 1 for i in range(len(v))])

def resize_image(img_path, img_width, img_height):
    img = cv2.imread(img_path)
    scale_img = np.full((img_height, img_width, 3), (255, 255, 255))
    h, w, _ = img.shape

    if h == img_height and w == img_width:
        return img
    scale_h = img_height / h
    scale_w = img_width / w
    if scale_h > scale_w:
        scale = scale_w
    else:
        scale = scale_h

    width = int(w * scale)
    height = int(h * scale)

    x_offset = int((scale_img.shape[0] - height) / 2 - 1)
    y_offset = int((scale_img.shape[1] - width) / 2 - 1)

    if x_offset < 0:
        x_offset = 0
    if y_offset < 0:
        y_offset = 0

    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    scale_img[x_offset:x_offset + height, y_offset:y_offset + width] = resized

    return scale_img.astype(np.uint8)
