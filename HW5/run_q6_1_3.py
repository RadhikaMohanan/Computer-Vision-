import numpy as np
import scipy.io
import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import skimage
import skimage.io
import skimage.transform
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import matplotlib.patches
from nn import *
from q4 import *
import torch.nn as nn
import torch.nn.functional as F
import warnings

max_iters = 20
learning_rate = 0.01
hidden_size = 64
batch_size=32


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
examples = len(trainset)
test_examples = len(testset)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
training_loss_data = []
test_loss_data = []
training_acc_data = []
test_acc_data = []

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

for itr in range(max_iters):
    print(itr)
    total_loss = 0
    total_acc = 0
    test_total_loss = 0
    test_total_acc = 0
    for batch_idx, (x, target) in enumerate(trainloader):
        out =  model(x)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        total_acc += ((target==predicted).sum().item())
        total_loss+=loss
    for batch_idx, (x, target) in enumerate(testloader):
        out =  model(x)
        test_loss = criterion(out,target)
        optimizer.zero_grad()
        test_loss.backward()

        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        test_total_acc += ((target==predicted).sum().item())
        test_total_loss+=loss
    ave_acc = total_acc/examples
    test_ave_acc = test_total_acc/test_examples
    print('total loss: ' + str(total_loss))
    print('accuracy: ' + str(ave_acc))
    print('test loss: ' + str(test_total_loss))
    print('test accuracy: ' + str(test_ave_acc))
    training_loss_data.append(total_loss / (examples / batch_size))
    training_acc_data.append(ave_acc)
    test_loss_data.append(test_total_loss / (test_examples / batch_size))
    test_acc_data.append(test_ave_acc)
plt.figure(0)
plt.xlabel('max_iters')
plt.ylabel('loss')
plt.plot(np.arange(max_iters), training_loss_data, 'r')
plt.plot(np.arange(max_iters), test_loss_data, 'b')
plt.legend(['training loss','valid loss'])

plt.show()
plt.figure(1)
plt.xlabel('max_iters')
plt.ylabel('Accuracy')
plt.plot(np.arange(max_iters),training_acc_data,'r')
plt.plot(np.arange(max_iters),test_acc_data,'b')
plt.legend(['training accuracy','valid accuracy'])

plt.show()

