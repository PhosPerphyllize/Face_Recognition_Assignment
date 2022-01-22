
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from read_data import *
from nn_model import *

writer = SummaryWriter("logs/nnONet_new")
model_save_path = "nnONet_save_new"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "../../CeleDataset"
trainset1 = MyData(root=root, train=True)
trainset2 = MyData(root=root, train=True, flip=True)
trainset = trainset1 + trainset2
print("Train set read: successful.")
testset = MyData(root=root, train=False)
print("Test set read: successful.")

train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=32, shuffle=True)

nn_model = nnONet()
nn_model.to(device)

# 损失函数与优化器
loss_fun = nn.MSELoss()
loss_fun.to(device)

learn_rate = 0.001
nn_optim = torch.optim.Adam(nn_model.parameters(), lr=learn_rate)

train_num = 0
test_num = 0
epoch = 200

start_time = time.time()   # 记录时间
for i in range(epoch):
    nn_model.train()
    print("--------EPOCH {}--------".format(i+1))
    loss_train = 0
    for data in train_dataloader:
        imgs, target = data
        imgs= imgs.to(device)
        target = target.to(device)

        output = nn_model(imgs)
        loss = loss_fun(output, target)
        loss_train += loss

        # 优化器
        nn_optim.zero_grad()
        loss.backward()
        nn_optim.step()

        train_num += 1
        if train_num % 50 == 0:
            print("In train num {}, loss: {}".format(train_num, loss))
            writer.add_scalar(tag="train_num vs loss", scalar_value=loss, global_step=train_num)

    print("In epoch {}, TrainSet train loss: {}".format(i+1, loss_train))  # tensor数据类型也可以正常打印，如果想变为普通类型，可使用xxx.item()
    writer.add_scalar(tag="epoch(TrainSet) vs loss", scalar_value=loss_train, global_step=i+1)

    nn_model.eval()  # 将网络设置成测试模式
    with torch.no_grad():
        loss_test = 0
        for data in test_dataloader:
            imgs, target = data
            imgs = imgs.to(device)
            target = target.to(device)

            output = nn_model(imgs)
            loss_test += loss_fun(output, target)

            test_num += 1

        print("In epoch {}, TestSet test loss: {}".format(i+1, loss_test))
        writer.add_scalar(tag="epoch(TestSet) vs loss", scalar_value=loss_test, global_step=i + 1)

    if i != 0 and (i + 1) % 10 == 0:
        path = os.path.join(model_save_path, ("nnONet_model{}.pth".format(i + 1)))
        torch.save(nn_model, path)  # 自动保存

    end_time = time.time()
    print("Time consume: {}".format(end_time - start_time))

writer.close()