from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import nn, optim
import numpy as np
import torch
import time
import sys
import cv2
import os

#车牌字符数组
match = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
            13: '川', 14: 'D', 15: 'E', 16: '鄂',
            17: 'F', 18: 'G', 19: '赣', 20: '甘', 21: '贵', 22: '桂', 23: 'H', 24: '黑', 25: '沪', 26: 'J', 27: '冀', 28: '津',
            29: '京', 30: '吉', 31: 'K', 32: 'L', 33: '辽',
            34: '鲁', 35: 'M', 36: '蒙', 37: '闽', 38: 'N', 39: '宁', 40: 'P', 41: 'Q', 42: '青', 43: '琼', 44: 'R', 45: 'S',
            46: '陕', 47: '苏', 48: '晋', 49: 'T', 50: 'U',
            51: 'V', 52: 'W ', 53: '皖', 54: 'X', 55: '湘', 56: '新', 57: 'Y', 58: '豫', 59: '渝', 60: '粤', 61: '云', 62: 'Z',
            63: '藏', 64: '浙'
            }


data_path = './characterData/data'
data_transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
dataset=ImageFolder(data_path,transform=data_transform)



validation_split=.1
shuffle_dataset = True
random_seed= 42
batch_size=100
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_iter = DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_iter = DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)


#lenet训练
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 65)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def predict(img,net,device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    res=''
    with torch.no_grad():
        for X,y in img:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                temp=net(X.to(device)).argmax(dim=1)
                x=np.array(X)
                temp=np.array(temp).tolist()
                for i in temp:
                    res+=str(match[i])
                net.train() # 改回训练模式
    return res

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            x=np.array(X)
            Y=np.array(y)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


net = LeNet()
print(net)
lr, num_epochs = 0.001, 20
batch_size=256
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
checkpoint_save_path = "./LeNet5.pth"
if os.path.exists(checkpoint_save_path ):
    print('load the model')
    net.load_state_dict(torch.load(checkpoint_save_path))
else:
    train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    torch.save(net.state_dict(),checkpoint_save_path)

#识别车牌内容
# plate_paths=os.listdir('./singledigit')
# ans=''
# for plate in plate_paths:
#     #ans+=predict(cv2.imread(plate),net)
#     img=cv2.imread('./singledigit/'+plate)
#     img2=np.zeros(shape=(3,20,20),dtype=torch.float32)
#
#     a=net(torch.from_numpy(img))
# print(ans)

pre_path = './singledigit'
pre_transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.CenterCrop(size=(20,20)),
    transforms.ToTensor()
])
preset=ImageFolder(pre_path,transform=pre_transform)
pre_iter = DataLoader(preset)
ans=predict(pre_iter,net)
print('预测结果为：'+ans)

