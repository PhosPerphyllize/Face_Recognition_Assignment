import torch
from torch import nn

class nnExtr(nn.Module):
    def __init__(self):
        super(nnExtr, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    mynn = nnExtr()
    imgs_test = torch.ones((1, 1, 48, 48))
    output = mynn(imgs_test)
    print(output.shape)
    # print(output)