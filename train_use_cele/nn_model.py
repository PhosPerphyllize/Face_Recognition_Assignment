import torch
from torch import nn

class nnONet(nn.Module):
    def __init__(self):
        super(nnONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False),          # 46*46*32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),               # 23*23*32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False),         # 21*21*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                          # 10*10*64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),         # 8*8*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # 4*4*64
            nn.Conv2d(64, 128, kernel_size=2, stride=1, bias=False),        # 3*3*128
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    mynn = nnONet()
    imgs_test = torch.ones((1, 3, 48, 48))
    output = mynn(imgs_test)
    print(output.shape)
    # print(output)