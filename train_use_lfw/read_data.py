import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import sys
from readtxt import readTxt
from extr_val import ExtrVal

class MyData(Dataset):
    def __init__(self, root:str, train:bool=False, transforms=torchvision.transforms.ToTensor(),
                 val:bool=False, flip:bool=False):  # 输入的root_dir为路径名：用于找到训练集（图片）  fold_n 文件夹名为标签
        self.root = root
        if train:
            txt_root = os.path.join(self.root, "training.txt")
        else:
            txt_root = os.path.join(self.root, "testing.txt")
        if val:
            txt_root = self.root
            self.root = "../../Dataset"

        self.img_list, self.target_tab = readTxt(txt_root)
        self.trans = transforms
        self.flip = flip
        self.val = val

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_h, img_w = img.shape
        img = img[0:img_w, :]  # 裁剪上半段成正方形

        if self.flip:
            img = cv2.flip(img,1)
            img_target = self.target_tab[idx]
            temp = img_target[:]
            temp[0] = img_w - img_target[1]
            temp[5] = img_target[6]
            temp[1] = img_w - img_target[0]
            temp[6] = img_target[5]

            temp[2] = img_w - img_target[2]

            temp[3] = img_w - img_target[4]
            temp[8] = img_target[9]
            temp[4] = img_w - img_target[3]
            temp[9] = img_target[8]
            img_target = temp
        else:
            img_target = self.target_tab[idx]
        if self.val:
            return self.trans(img), torch.Tensor(img_target), img_path
        return self.trans(img), torch.Tensor(img_target)  # 在这里确定返回，只要调用 类名[i]就返回

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    dataset_trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((64, 64)),
    ])

    trainset = MyData(root="valset.txt", train=True, transforms=dataset_trans, flip=True, val=True)
    testset = MyData(root="../Dataset", train=False, transforms=dataset_trans)

    img, output, img_path = trainset[-1]
    print(img.shape)
    print(output.shape)
    print(img)
    print(output)
    print(img_path)
    ExtrVal(img_path, output, flip=True)

    loader = DataLoader(testset, batch_size=4, shuffle=True)  # 注意batch_size 不要超过8，这是验证集，大小只有9
    for data in loader:
        img, output = data
        print(img.shape)
        print(output.shape)
        sys.exit(0)


