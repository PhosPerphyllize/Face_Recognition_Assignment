
import cv2
import os
import numpy as np
import torch


def SmoFuc(ratio:float, vertical:bool):
    # 平滑函数，党ratio较靠近0或1时降低斜率，类似arctan
    if vertical:
        center = [0.45, 0.45]
        if ratio < center[0]:
            ratio = center[0] + (center[0] - ratio)  # 中心点对称
            ratio = ratio - center[0] - 1.8 * (ratio - center[0]) * (ratio - center[0]) + center[0]
            ratio = center[1] - (ratio - center[1])
        else:
            ratio = ratio - center[0] - 1.8 * (ratio - center[0]) * (ratio - center[0]) + center[0]
    else:
        center = [0.5, 0.5]
        if ratio < center[0]:
            ratio = center[0] + (center[0] - ratio)  # 中心点对称
            ratio = ratio - center[0] - 1 * (ratio - center[0]) * (ratio - center[0]) + center[0]
            ratio = center[1] - (ratio - center[1])
        else:
            ratio = ratio - center[0] - 1 * (ratio - center[0]) * (ratio - center[0]) + center[0]
    return ratio

def ExtrVal(path:str, charac_point,flip:bool=False):
    if type(charac_point)==torch.Tensor:
        temp = []
        for i in range(len(charac_point)):
            temp.append(charac_point[i].item())
        charac_point = temp
        print(charac_point)

    img = cv2.imread(path)
    img_h,img_w,_ = img.shape  # H,W,C
    if flip:
        img = cv2.flip(img, 1)

    x = charac_point[:5]
    y = charac_point[5:]
    if x[1]==x[0]:
        x[1] +=1
    # 计算需要旋转的角度
    detal = (y[1] - y[0])/(x[1] - x[0])
    det_theta = -np.arctan(detal)
    # 以鼻子为中心，对特征点进行旋转
    for i in range(len(x)):
        if i == 2:
            x[i] = int(x[2])
            y[i] = int(y[2])
        else:
            r = np.sqrt( np.square(x[i]-x[2]) + np.square(y[i]-y[2]) )
            if x[i]==x[2]:
                x[i] += 1
            if x[i] > x[2]:
                theta_ori = np.arctan((y[i]-y[2])/(x[i]-x[2]))
            else:
                theta_ori = np.arctan((y[i]-y[2])/(x[i]-x[2])) + np.pi
            theta_tran = theta_ori + det_theta
            x[i] = r * np.cos(theta_tran) + x[2]
            y[i] = r * np.sin(theta_tran) + y[2]

    #得到旋转矩阵，（旋转中心，旋转角度，图像缩放比例）
    M = cv2.getRotationMatrix2D((x[2],y[2]), -det_theta/np.pi*180, 1)
    #进行仿射变换，（参数图像，旋转矩阵，变换之后的图像大小(W,H)）
    img = cv2.warpAffine(img, M, (int(img_w*1.1), int(img_h*1.1)))

    # 画五个特征点，（图片对象，中心点坐标，半径大小，颜色，宽度）
    cv2.circle(img,(int(x[0]),int(y[0])),2,(255,111,111),-1)
    cv2.circle(img,(int(x[1]),int(y[1])),2,(255,111,111),-1)
    cv2.circle(img,(int(x[2]),int(y[2])),2,(255,111,111),-1)
    cv2.circle(img,(int(x[3]),int(y[3])),2,(255,111,111),-1)
    cv2.circle(img,(int(x[4]),int(y[4])),2,(255,111,111),-1)
    #计算框脸的框的长宽
    rec_w = abs(x[1]-x[0])*2.2
    rec_h = abs(y[0]-y[3])*2.2

    rec_left = abs(x[0]-x[2]) + abs(x[3]-x[2])
    rec_right = abs(x[1]-x[2]) + abs(x[4]-x[2])
    ratio_h = rec_left/(rec_left + rec_right)  # 以鼻子到两边特征点进行加权计算比例
    print(ratio_h)
    ratio_h = SmoFuc(ratio_h, vertical=False)
    print(ratio_h)

    rec_up = abs(y[0]-y[2]) + abs(y[1]-y[2])
    rec_down = abs(y[3]-y[2]) + abs(y[4]-y[2])
    ratio_v = rec_up/(rec_up + rec_down)
    print(ratio_v)
    ratio_v = SmoFuc(ratio_v, vertical=True)  # （嘴离鼻子近，微笑）时进行降低
    print(ratio_v)

    # 画矩形框脸，（图片对象，左上角坐标，右下角坐标，颜色，宽度）
    cv2.rectangle(img,(int(x[2]-rec_w*ratio_h),int(y[2]-rec_h*ratio_v)),
                  (int(x[2]+rec_w*(1-ratio_h)),int(y[2]+rec_h*(1-ratio_v))),
                  (0,0,255),1)


    cv2.imshow("img", img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    root = "../Dataset"
    path = os.path.join(root, "lfw_5590\Adriana_Perez_Navarro_0001.jpg")
    output = [107.250000, 149.750000, 123.250000, 96.250000, 135.250000, 110.250000, 120.250000, 144.250000, 147.250000, 157.250000]
    ExtrVal(path,output)


# AFLW/0001-image20056.jpg
# 56.607985, 101.326679, 79.548094, 43.831216, 93.196007, 43.250454, 58.059891, 69.965517, 86.226860, 103.940109 1 1 2 3
# lfw_5590\Aaron_Sorkin_0001.jpg
# 102.250000, 144.750000, 130.750000, 100.250000, 143.250000, 113.250000, 112.750000, 138.750000, 156.250000, 155.750000 1 1 2 3
# lfw_5590\Abel_Aguilar_0001.jpg
# 106.750000, 147.750000, 128.250000, 110.750000, 138.750000, 116.750000, 118.750000, 139.750000, 160.250000, 160.750000 1 2 2 3
# lfw_5590\Adriana_Perez_Navarro_0001.jpg
# 107.250000, 149.750000, 123.250000, 96.250000, 135.250000, 110.250000, 120.250000, 144.250000, 147.250000, 157.250000 2 1 2 3
# lfw_5590\Kevin_Marshall_0001.jpg
# 105.750000, 148.250000, 123.250000, 111.250000, 140.750000, 111.750000, 111.750000, 137.750000, 161.750000, 162.250000 1 2 2 3
# net_7876/7242_0_0.jpg
# 169.000000, 276.000000, 228.000000, 184.000000, 254.000000, 149.000000, 152.000000, 193.000000, 244.000000, 247.000000 1 1 2 3
#  net_7876\9026_0_0.jpg
#  166.000000, 260.000000, 214.000000, 149.000000, 239.000000, 146.000000, 164.000000, 213.000000, 240.000000, 261.000000 2 1 2 3
# net_7876\9037_0_0.jpg
# 135.000000, 212.000000, 132.000000, 134.000000, 201.000000, 165.000000, 145.000000, 205.000000, 261.000000, 251.000000 2 2 2 1
# net_7876\_-60_6059_0.jpg
# 67.000000, 88.000000, 67.000000, 66.000000, 89.000000, 64.000000, 69.000000, 83.000000, 89.000000, 94.000000, 2 1 2 2
# net_7876\_-60_6063_0.jpg
# 66.000000, 96.000000, 70.000000, 67.000000, 97.000000, 60.000000, 59.000000, 81.000000, 97.000000, 97.000000 1 2 2 2
# AFLW/0023-image25562.jpg
# 46.444646, 92.905626, 62.125227, 59.511797, 99.874773, 62.415608, 43.831216, 74.611615, 110.328494, 97.842105
# net_7876\_-60_5936_0.jpg
# 65.000000, 98.000000, 70.000000, 70.000000, 94.000000, 66.000000, 60.000000, 82.000000, 106.000000, 101.000000
# AFLW/0026-image41552.jpg
# 51.671506, 99.003630, 81.871143, 64.448276, 99.294011, 52.542650, 47.315789, 91.163339, 100.455535, 94.647913