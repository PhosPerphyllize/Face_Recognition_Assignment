
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


def ExtrVal(path:str, charac_point, flip:bool=False, resize_target:tuple=None, crop:float=None):
# 利用五个特征点画脸框，path图片路径 flip 是否翻转图片
# charac_point：五个特征点 可以为数组元组，也可为Tensor
# lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
# resize_target为进行resize的目标值（h，w）

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

    if type(crop) == float:
        # for i in [1, 3, 5, 7, 9]:
        #     charac_point[i] -= crop
        img = img[int(crop):int(img_w + crop), :]
        img_h = img_w

    if type(resize_target) == tuple:
        # charac_point = onetResize((img_h,img_h), charac_point, resize_target)
        img_h, img_w = resize_target
        img = cv2.resize(img, (img_w, img_h))     # 输入参数 w,h

    x = charac_point[0:9:2]
    y = charac_point[1:10:2]
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
    # print(ratio_h)
    ratio_h = SmoFuc(ratio_h, vertical=False)
    # print(ratio_h)

    rec_up = abs(y[0]-y[2]) + abs(y[1]-y[2])
    rec_down = abs(y[3]-y[2]) + abs(y[4]-y[2])
    ratio_v = rec_up/(rec_up + rec_down)
    # print(ratio_v)
    ratio_v = SmoFuc(ratio_v, vertical=True)  # （嘴离鼻子近，微笑）时进行降低
    # print(ratio_v)

    # 画矩形框脸，（图片对象，左上角坐标，右下角坐标，颜色，宽度）
    cv2.rectangle(img,(int(x[2]-rec_w*ratio_h),int(y[2]-rec_h*ratio_v)),
                  (int(x[2]+rec_w*(1-ratio_h)),int(y[2]+rec_h*(1-ratio_v))),
                  (0,0,255),1)


    cv2.imshow("img", img)
    # print(type(img))
    # print(img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def onetResize(image, charac_point, target_size):
# 输入相关信息与 五点坐标值，返回resize压缩后的五点坐标值
# 输入 image可以是path 或 （img_h, img_w）  charac_point 为五点坐标值 target_size 为目标图片宽度 target_h, target_w
    if type(charac_point) == torch.Tensor:
        temp = []
        for i in range(len(charac_point)):
            temp.append(charac_point[i].item())
        charac_point = temp
        print(charac_point)

    if type(image) == tuple:
        img_h, img_w = image
    elif type(image) == str:
        img = cv2.imread(image)
        img_h, img_w, _ = img.shape  # H,W,C
    else:
        raise ValueError("image must be tuple or path.")

    target_h, target_w = target_size
    for i in [0, 2, 4, 6, 8]:
        charac_point[i] = charac_point[i] * target_w / img_w
    for i in [1, 3, 5, 7, 9]:
        charac_point[i] = charac_point[i] * target_h / img_h
    return charac_point

def onetUnCrop(charac_point, crop):
    if type(charac_point) == torch.Tensor:
        temp = []
        for i in range(len(charac_point)):
            temp.append(charac_point[i].item())
        charac_point = temp
        print(charac_point)

    for i in [1, 3, 5, 7, 9]:
        charac_point[i] += crop
    return charac_point

if __name__ == '__main__':
    root = "../../CeleDataset/Img"
    path = os.path.join(root, "000001.jpg")
    output = [69,  109,  106,  113,   77,  142,   73,  152,  108,  154]
    # ExtrVal(path,output)
    # ExtrVal(path,output,crop=50.0)

    resize_target = (300,300)
    crop = 20.0

    ExtrVal(path,output,resize_target=resize_target,crop=crop)
