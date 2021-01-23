import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def find_card(I):
    # 识别出车牌区域并返回该区域的图像
    [y, x, z] = I.shape
    # y取值范围分析
    Blue_y = np.zeros((y, 1))
    for i in range(y):
        for j in range(x):
            # 蓝色rgb范围
            temp = I[i, j, :]
            if (I[i, j, 2] <= 30) and (I[i, j, 0] >= 119):
                Blue_y[i][0] += 1
    MaxY = np.argmax(Blue_y)
    PY1 = MaxY
    while (Blue_y[PY1, 0] >= 5) and (PY1 > 0):
        PY1 -= 1
    PY2 = MaxY
    while (Blue_y[PY2, 0] >= 5) and (PY2 < y - 1):
        PY2 += 1
    # x取值
    Blue_x = np.zeros((1, x))
    for i in range(x):
        for j in range(PY1, PY2):
            if (I[j, i, 2] <= 30) and (I[j, i, 0] >= 119):
                Blue_x[0][i] += 1
    PX1 = 0
    while (Blue_x[0, PX1] < 3) and (PX1 < x - 1):
        PX1 += 1
    PX2 = x - 1
    while (Blue_x[0, PX2] < 3) and (PX2 > PX1):
        PX2 -= 1
    # 对车牌区域的修正
    PX1 -= 2
    PX2 += 2
    return I[PY1:PY2, PX1 - 2: PX2, :]


def divide(I):
    [y, x, z] = I.shape
    White_x = np.zeros((x, 1))
    for i in range(x):
        for j in range(y):
            if I[j, i, 1] > 176:
                White_x[i][0] += 1
    return White_x


def divide_each_character(I):
    [y, x, z] = I.shape
    White_x = np.zeros((x, 1))
    for i in range(x):
        for j in range(y):
            if I[j, i, 1] > 176:
                White_x[i][0] += 1
    res = []
    length = 0
    for i in range(White_x.shape[0]):
        # 使用超参数经验分割
        t = I.shape[1] / 297
        num = White_x[i]
        if num > 8:
            length += 1
        elif length > 20 * t:
            res.append([i - length - 2, i + 2])
            length = 0
        else:
            length = 0
    return res


if __name__ == '__main__':
    I = cv2.imread('Car.jpg')
    Plate = find_card(I)
    # White_x = divide(Plate)
    plt.imshow(Plate)
    plt.show()
    # plt.plot(np.arange(Plate.shape[1]), White_x)
    res = divide_each_character(Plate)
    plate_save_path = './singledigit/'
    for t in range(len(res)):
        plt.subplot(1, 7, t + 1)
        temp = res[t]
        save_img = cv2.cvtColor(Plate[:, temp[0]:temp[1], :],cv2.COLOR_BGR2GRAY)
        ma = max(save_img.shape[0], save_img.shape[1])
        mi = min(save_img.shape[0], save_img.shape[1])
        ans = np.zeros(shape=(ma, ma, 3),dtype=np.uint8)
        start =int(ma/2-mi/2)
        for i in range(mi):
            for j in range(ma):
                if save_img[j,i] > 125:
                    for k in range(3):
                        ans[j,start+i,k]=255
        ans=cv2.merge([ans[:,:,0],ans[:,:,1],ans[:,:,2]])
        ans=cv2.resize(ans,(25,25))
        dir_name=plate_save_path+str(t)
        os.mkdir(dir_name)
        cv2.imwrite(dir_name+'/'+str(t)+'.jpg',ans)
        plt.imshow(ans)
    plt.show()
