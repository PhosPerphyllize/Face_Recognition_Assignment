import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
from onet_val import ExtrVal, onetResize, onetUnCrop
from torch import nn

class ReadFolder(Dataset):
    def __init__(self, root:str):
        self.root = root # 文件目录
        if not os.path.exists(self.root):
            raise ValueError("path not exist.")
        self.trans = torchvision.transforms.ToTensor()
        self.resize = (48,48)
        self.img_list = os.listdir(self.root)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root, img_name)

        img = cv2.imread(img_path)  # 灰度图：cv2.IMREAD_GRAYSCALE
        img_h, img_w, img_c = img.shape
        print(img.shape)

        # 以图片中心裁剪成正方形
        detel_y = 0.0
        if img_h > img_w:
            detel_y = 0.5*(img_h-img_w)
            img = img[int(detel_y):int(img_w + detel_y), :]
            img_h = img_w
            print("Crop")
        img_ori_shape = (img_h, img_w)
        # 图像缩放
        img_h, img_w = self.resize
        img = cv2.resize(img, (178, 178))
        img = cv2.resize(img, (img_w, img_h))  # 输入参数 w,h
        return self.trans(img), img_path, detel_y, img_ori_shape

    def __len__(self):
        return len(self.img_list)


model_path = "nnONet_save_new/nnONet_model30.pth"
nn_model = torch.load(model_path)
print(nn_model)

nn_model.eval()  # 将网络设置成测试模式
nn_model.to("cpu")

valset = ReadFolder("valpic")
print(len(valset))

person = 10
img,img_path,detel_y,img_ori_shape = valset[person]
print(img.shape)
img = img.reshape(1,3,48,48)
print(img.shape)

output = nn_model(img)
print(output)
print(output.shape)

output = output.reshape(-1)
print(output.shape)

print(img_ori_shape)
print(detel_y)
output = onetResize((48,48), output, img_ori_shape)
print(output)
output = onetUnCrop(output, detel_y)
print(output)
ExtrVal(img_path, output)



