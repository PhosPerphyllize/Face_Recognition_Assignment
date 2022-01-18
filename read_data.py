import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import sys
from readtxt import readTxt

class MyData(Dataset):
    def __init__(self, root:str, train:bool=False, transforms=torchvision.transforms.ToTensor()):  # 输入的root_dir为路径名：用于找到训练集（图片）  fold_n 文件夹名为标签
        self.root = root
        if train:
            txt_root = os.path.join(self.root, "training.txt")
        else:
            txt_root = os.path.join(self.root, "testing.txt")

        self.img_list, self.target_tab = readTxt(txt_root)
        self.trans = transforms

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_target = self.target_tab[idx]
        return self.trans(img), torch.Tensor(img_target)  # 在这里确定返回，只要调用 类名[i]就返回

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    dataset_trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((64, 64)),
    ])

    trainset = MyData(root="../Dataset", train=True, transforms=dataset_trans)
    testset = MyData(root="../Dataset", train=False, transforms=dataset_trans)

    img, output = trainset[0]
    print(img.shape)
    print(output.shape)
    print(img)
    print(output)

    loader = DataLoader(testset, batch_size=4, shuffle=True)  # 注意batch_size 不要超过8，这是验证集，大小只有9
    for data in loader:
        img, output = data
        print(img.shape)
        print(output.shape)
        sys.exit(0)


