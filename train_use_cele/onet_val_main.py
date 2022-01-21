
import torch
from torch.utils.data import DataLoader

from nn_model import nnONet
from read_data import MyData
from onet_val import ExtrVal, onetResize, onetUnCrop
from torch import nn

# model_path = "nnONet_save/nnextr_model500.pth"
# nn_model = torch.load(model_path)
nn_model = nnONet()

nn_model.eval()  # 将网络设置成测试模式
nn_model.to("cpu")

valset = MyData("test.txt", train=False, val=True)
valset_input1 = MyData("test.txt", train=False,val=True)
valset_input2 = MyData("test.txt", train=False, flip=True,val=True)
valset_input = valset_input1 + valset_input2
print(len(valset))

val_loader = DataLoader(valset_input, batch_size=8,shuffle=True)
loss_fn = nn.MSELoss()

for i in range(2):
    print("===================")
    loss_test = 0
    for data in val_loader:
        imgs,targets,a,b = data

        outputs = nn_model(imgs)
        print(outputs)
        print(targets)
        loss_test += loss_fn(outputs, targets)
    print("ValSet test loss: {}".format(loss_test))

person = 3
img,target,img_path,detel_y = valset[person]
img,target,img_path,detel_y = valset[person]
print(target)
print(img.shape)
img = img.reshape(1,3,48,48)
print(img.shape)

output = nn_model(img)
print(output)

print(output.shape)
output = output.reshape(-1,10)
target = target.reshape(-1,10)
print(output.shape)

loss = loss_fn(output, target)
print(loss)

print(output.shape)
output = output.reshape(-1)
target = target.reshape(-1)
print(output.shape)

output = target
output = onetResize((48,48), output, (178,178))
output = onetUnCrop(output, detel_y)
ExtrVal(img_path, output)









