import cv2
import numpy as np


def reshape_std(img, width, height):

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return img


def red_line_check(img, img_BGR):

    lines = img.shape[0]

    dialta = int(lines/100)

    hsv = img  # 色彩空间转换为hsv，便于分离
    lower_hsv_1 = np.array([0, 45, 48])  # 提取颜色的低值
    high_hsv_1 = np.array([10, 255, 255])  # 提取颜色的高值
    lower_hsv_2 = np.array([156, 43, 48])  # 提取颜色的低值
    high_hsv_2 = np.array([180, 255, 255])  # 提取颜色的高值

    mask = cv2.inRange(hsv, lowerb=lower_hsv_1, upperb=high_hsv_1)
    mask += cv2.inRange(hsv, lowerb=lower_hsv_2, upperb=high_hsv_2)

    a = np.where(mask != 0)

    x = a[0]
    y = a[1]

    x_min = np.min(x) + dialta
    x_max = np.max(x) - dialta
    y_min = np.min(y) + dialta
    y_max = np.max(y) - dialta

    cut = img_BGR[x_min:x_max, y_min:y_max]

    cut = reshape_std(cut, int(479*1.5), int(61*1.5))

    return cut


def cut_tiny6(cutten_all):

    # 分别裁剪6个小线框
    block_number = 6
    lines, cols, _ = cutten_all.shape

    block_length = lines
    gap_length = int((cols - block_length*block_number)/5)
    numbers = []

    for i in range(block_number):

        error = int(lines/15)

        x_begin = i*(block_length+gap_length) + error
        x_end = x_begin+block_length - error

        y_begin = error
        y_end = block_length

        temp = cutten_all[y_begin:y_end, x_begin:x_end]

        temp = cv2.medianBlur(temp, 5)

        error = int(error/2)+2

        numbers.append(temp[error:-error, error:-error])

    return numbers
