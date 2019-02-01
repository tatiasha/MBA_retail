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
from sklearn.decomposition import TruncatedSVD, PCA


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
   def __init__(self, s):
       super(Net, self).__init__()
       self.l1 = nn.Linear(s, 256)
       self.l2 = nn.Linear(256, 512)
       self.l3 = nn.Linear(512, s)

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
    data = pd.read_csv('{0}\\x_target_3.csv'.format(data_path))
    data = data.drop(columns=['Unnamed: 0'])
    purchasing = data['Purchasing'].values
    target = data['Target'].values
    X = convert_to_list(purchasing)
    y = convert_to_list(target)
    batch_size = 20

    inputs_train = X[:5000]
    labels_train = y[:5000]
    inputs_test = X[5000:]
    labels_test = y[5000:]
    labels_test = torch.tensor(labels_test, dtype=torch.float32)
    components = []
    accuracy = []
    for number_of_components in range(1, 52, 2):
        inp = []
        lbl = []
        train_size = 30000

        if number_of_components != 51:
            # svd = TruncatedSVD(n_components=number_of_components, n_iter=100, random_state=51)
            svd = TruncatedSVD(n_components=number_of_components, random_state=51)
            transform_inputs = svd.fit_transform(inputs_train)
            transform_targets = svd.transform(labels_train)
        else:
            transform_inputs = inputs_train
            transform_targets = labels_train

        for i in range(train_size):
            tmp_inp = []
            tmp_lbl = []
            idx = [np.random.randint(0, 5000) for j in range(batch_size)]
            for h in idx:
                tmp_inp.append(transform_inputs[h])
                tmp_lbl.append(transform_targets[h])
            lbl.append(tmp_lbl)
            inp.append(tmp_inp)
        transform_inputs = torch.tensor(inp, dtype=torch.float32)
        transform_targets = torch.tensor(lbl, dtype=torch.float32)

        if number_of_components != 51:
            transform_inputs_test = svd.transform(inputs_test)
            transform_inputs_test = torch.tensor(transform_inputs_test, dtype=torch.float32)
        else:
            transform_inputs_test = torch.tensor(inputs_test, dtype=torch.float32)

        net = Net(number_of_components)
        # CosineEmbeddingLoss
        # CrossEntropyLoss
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        # optimizer = optim.LBFGS(net.parameters(), lr=0.0001)

        for epoch in range(7):  # loop over the dataset multiple times

            running_loss = 0.0
            for i in range(train_size):
                # get the inputs
                inputs = transform_inputs[i]
                labels = transform_targets[i]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                #outputs = svd.inverse_transform(outputs.detach().numpy())
                #outputs = torch.tensor(outputs, dtype=torch.float32)

                ####################################
                ##########################
                # print(outputs)
                loss = criterion(outputs, target=labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                N = 100
                if (1 + i) % N == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / N))
                    running_loss = 0.0

        print('Finished Training')

        images = transform_inputs_test
        labels = labels_test
        a = 0
        for rep in range(5):
            print(rep)
            for i in range(len(images)):
                outputs = net(images[i])
                if number_of_components != 51:
                    outputs = svd.inverse_transform([outputs.detach().numpy()])
                else:
                    outputs = [outputs]

                out = []
                for p in outputs:
                    correct = 0
                    for h in p:
                        if h > np.random.uniform(0, 1):
                            out.append(1)
                        else:
                            out.append(0)
                # print("nn ", out)
                # print("re ", labels[i].numpy().astype(int).tolist())
                    if sum(out == labels[i].numpy()) == 51:
                        correct += 1
                    # print("Ok")
                # print("=====")
                    a += 100*correct / len(inputs_test)
        a /= 5
        accuracy.append(a)
        components.append(number_of_components)

        print('Accuracy of the network on the {0} test clients with {1} components: {2}%'.format(len(inputs_test),number_of_components, a))

    plt.plot(components, accuracy)
    plt.xlabel("Number of components")
    plt.ylabel("Accuracy")
    plt.show()

    print(components)
    print(accuracy)
