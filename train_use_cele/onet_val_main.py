
import torch
from read_data import MyData
from onet_val import ExtrVal
import torchvision

model_path = "nnextr_save/nnextr_model4000.pth"
nn_model = torch.load(model_path)
nn_model.to("cpu")

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((48, 48)),
])
valset = MyData("test.txt", train=False, transforms=dataset_trans, val=True,flip=True)
print(len(valset))

person = 9
img,target,img_path = valset[person]
print(img.shape)
img = img.reshape(1,1,48,48)
print(img.shape)

nn_model.eval()
output = nn_model(img)
output = output.reshape(-1)
print(output.shape)
print(output)
print(target)

ExtrVal(img_path, output, flip=True)
# ExtrVal(img_path, target, flip=True)




