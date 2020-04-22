import cv2
import numpy as np



def ad_threshold(img):

    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (3, 3), 0) # 降噪处理

    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 19, 4)

    return th2


def ad_cut_central(img):

    img = ad_threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    lines, cols = img.shape

    result = img.copy()

    lables_x = []
    lables_y = []

    mask = cv2.inRange(result, lowerb=0, upperb=20)

    mask = np.where(mask == np.max(mask))

    lables_x = mask[0]
    lables_y = mask[1]

    x_min = np.min(lables_x)-2 if np.min(lables_x)-2 > 0 else 0
    x_max = np.max(lables_x)+2 if np.max(lables_x)+2 < lines else lines

    y_min = np.min(lables_y)-2 if np.min(lables_y)-2 > 0 else 0
    y_max = np.max(lables_y)+2 if np.max(lables_y)+2 < cols else cols

    test = img[x_min:x_max, y_min:y_max]

    di_x = x_max-x_min
    di_y = y_max-y_min
    xy = [di_x, di_y]

    liss = [x_min, x_max, y_min, y_max]

    for i in range(len(liss)):
        if liss[i] < 0:
            liss[i] = 0
        if liss[i] > lines:
            liss[i] = lines

    side_length = np.max(xy)

    cutten_tem = img[liss[0]:liss[1], liss[2]:liss[3]]

    filling_gap = 20 + side_length

    cutten = np.ones((filling_gap, filling_gap))*255

    error_x = int((filling_gap-di_x)/2)

    error_y = int((filling_gap-di_y)/2)

    cutten[error_x:error_x+cutten_tem.shape[0],
           error_y:error_y+cutten_tem.shape[1]] = cutten_tem

    return cutten
