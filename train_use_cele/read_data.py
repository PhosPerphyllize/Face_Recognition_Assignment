import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import sys
from readtxt import readTxt
from onet_val import ExtrVal, onetResize, onetUnCrop
from PIL import Image

class MyData(Dataset):
    def __init__(self, root:str, train:bool=False, resize:tuple=(48,48),
                 val:bool=False, flip:bool=False):  # val 测试用接口
        self.root = root # 文件目录
        self.trans = torchvision.transforms.ToTensor()
        self.resize = resize
        self.flip = flip
        self.val = val
        if train:
            txt_root = os.path.join(self.root, "training.txt")
        else:
            txt_root = os.path.join(self.root, "testing.txt")
        if val:
            txt_root = self.root
            self.root = "../../CeleDataset"

        self.img_list, self.target_tab = readTxt(txt_root)
        self.root = os.path.join(self.root, "Img")
        if not os.path.exists(self.root):
            raise ValueError("path not exist.")

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_target = self.target_tab[idx]

        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path)  # 灰度图：cv2.IMREAD_GRAYSCALE

        nose_y = img_target[5]
        img_h, img_w,_ = img.shape
        # 以鼻子为中心裁剪成正方形
        detel_y = 0.0
        if nose_y < img_w/2:
            detel_y = 0
        elif nose_y + img_w/2 > img_h:
            detel_y = img_h - img_w
        else:
            detel_y = (nose_y - img_w / 2)
        img = img[int(detel_y):int(img_w + detel_y), :]
        for i in [1,3,5,7,9]:
            img_target[i] -= detel_y

        # 图像缩放
        img_target = onetResize((img_w, img_w), img_target, self.resize)
        img_h, img_w = self.resize
        img = cv2.resize(img, (img_w, img_h))  # 输入参数 w,h

        # 图像翻转
        if self.flip:
            img = cv2.flip(img,1)

            temp = img_target[:]
            temp[0] = img_w - img_target[2]
            temp[1] = img_target[3]
            temp[2] = img_w - img_target[0]
            temp[3] = img_target[1]

            temp[4] = img_w - img_target[4]

            temp[6] = img_w - img_target[8]
            temp[7] = img_target[9]
            temp[8] = img_w - img_target[6]
            temp[9] = img_target[7]
            img_target = temp

        if self.val:
            return self.trans(img), torch.Tensor(img_target), img_path, detel_y
        return self.trans(img), torch.Tensor(img_target)  # 在这里确定返回，只要调用 类名[i]就返回

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    img_resize = (160,160)

    trainset = MyData(root="test.txt", train=True, resize=img_resize,flip=False, val=True)
    testset = MyData(root="../../CeleDataset", train=False)

    img, output, img_path, detel_y = trainset[0]
    print(img.shape)
    print(output.shape)
    print(img)
    print(output)

    print(detel_y)
    print(type(detel_y))

    print(img_path)
    # ExtrVal(img_path, output, flip=True, resize_target=img_resize,crop=float(detel_y))

    output = onetResize(img_resize, output, (178,178))
    output = onetUnCrop(output, detel_y)
    ExtrVal(img_path, output, flip=False)

    tran_pil = torchvision.transforms.ToPILImage()
    img_pil = tran_pil(img)
    img_pil.show()

    loader = DataLoader(testset, batch_size=4, shuffle=True)  # 注意batch_size 不要超过8，这是验证集，大小只有9
    for data in loader:
        img, output = data
        print(img.shape)
        print(output.shape)
        sys.exit(0)


