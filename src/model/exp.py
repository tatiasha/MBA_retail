# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def convert_to_list(a):
    t = [i[1:len(i) - 1] for i in a]
    r = [i.split(', ') for i in t]
    q = [list(map(int, i)) for i in r]
    return q


class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.l1 = nn.Linear(51, 64)
       self.l2 = nn.Linear(64, 32)
       self.l3 = nn.Linear(32, 51)

       # self.conv1 = nn.Conv2d(3, 6, 5)
       # self.pool = nn.MaxPool2d(2, 2)
       # self.conv2 = nn.Conv2d(6, 16, 5)
       # self.fc1 = nn.Linear(16 * 5 * 5, 120)
       # self.fc2 = nn.Linear(120, 84)
       # self.fc3 = nn.Linear(84, 10)

   def forward(self, x):
       # x = self.pool(F.relu(self.conv1(x)))
       # x = self.pool(F.relu(self.conv2(x)))
       # x = x.view(-1, 16 * 5 * 5)
       # x = F.relu(self.fc1(x))
       # x = F.relu(self.fc2(x))
       # x = self.fc3(x)
       x = F.relu(self.l1(x))
       x = F.relu(self.l2(x))
       x = F.relu(self.l3(x))
       return x

if __name__ == '__main__':
    data_path = "E:\Projects\MBA_retail\\tmp"
    global_data_path = "E:\Projects\MBA_retail\\tmp\datasets"
    data = pd.read_csv('{0}\\x_target.csv'.format(data_path))
    data = data.drop(columns=['Unnamed: 0'])
    purchasing = data['Purchasing'].values
    target = data['Target'].values
    X = convert_to_list(purchasing)
    y = convert_to_list(target)
    batch_size = 20

    inputs_train = X[:1500]
    labels_train = y[:1500]

    inp = []
    lbl = []
    for i in range(len(inputs_train)):
        tmp_inp = []
        tmp_lbl = []
        idx = [np.random.randint(0, 1500) for j in range(batch_size)]
        for h in idx:
            tmp_inp.append(inputs_train[h])
            tmp_lbl.append(labels_train[h])
        lbl.append(tmp_lbl)
        inp.append(tmp_inp)
    inputs_train = torch.tensor(inp, dtype=torch.float32)
    labels_train = torch.tensor(lbl, dtype=torch.float32)

    inputs_test = X[1500:]
    inputs_test = torch.tensor(inputs_test, dtype=torch.float32)

    labels_test = y[1500:]
    labels_test = torch.tensor(labels_test, dtype=torch.float32)

    net = Net()
    # CosineEmbeddingLoss
    # CrossEntropyLoss
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.LBFGS(net.parameters(), lr=0.0001)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(len(inputs_train)):
            # get the inputs
            inputs = inputs_train[i]
            labels = labels_train[i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            N = 100
            if (1 + i) % N == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / N))
                running_loss = 0.0

    print('Finished Training')


    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    images = inputs_test
    labels = labels_test
    for i in range(len(images)):
        outputs = net(images[i])
        out = []
        for h in outputs:
            if h > 0:
                out.append(1)
            else:
                out.append(0)
        print("nn ", out)
        print("re ", labels[i].numpy().astype(int).tolist())
        if sum(out == labels[i].numpy()) == 51:
            correct += 1
            print("Ok")
        print("=====")



    print('Accuracy of the network on the {0} test clients: {1}%'.format(len(inputs_test),
                                                                         (100*correct / len(inputs_test))))
