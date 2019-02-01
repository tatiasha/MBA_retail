from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
import keras.losses
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# инициализируем Tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# инициализируем Keras
from keras import backend as K

K.set_session(sess)

def convert_to_list(a):
    tmp = a[1:-1]
    tmp2 = tmp.split(",")
    tmp2 = list(map(int, tmp2))
    return tmp2

def convert_to_list2(a):
    tmp = a
    tmp = tmp[2:-2]
    tmp2 = tmp.split('\r')
    tmp2 = [item for sublist in tmp2 for item in sublist]
    tmp2 = [x for x in tmp2 if (x != ' ' and x != '\n')]
    tmp2 = list(map(int, tmp2))
    return tmp2

class Model:
    def __init__(self, state_size, output_size):
        self.INPUT_SIZE = state_size
        self.OUTPUT_SIZE = output_size
        self.LEARNING_RATE = 1e-4
        self.model = self.buildmodel()

    # структура сети
    def buildmodel(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.INPUT_SIZE, activation=keras.activations.relu))
        model.add(Dense(2048, activation=keras.activations.relu))
        model.add(Dense(1024, activation=keras.activations.relu))
        model.add(Dense(512, activation=keras.activations.relu))

        # model.add(Dense(256, input_dim=self.INPUT_SIZE, activation=keras.activations.relu))
        # model.add(Dense(512, activation=keras.activations.relu))
        # model.add(Dense(256, activation=keras.activations.relu))
        model.add(Dense(self.OUTPUT_SIZE, activation=keras.activations.relu))
        # optimizer
        optimizer = Adam(lr=self.LEARNING_RATE)
        # loss in model
        model.compile(loss=keras.losses.logcosh, optimizer=optimizer)
        return model



data_path = "E:\Projects\MBA_retail\\tmp"
train = pd.read_csv("{0}/train_input_target_dunnhumby.csv".format(data_path))
test = pd.read_csv("{0}/test_input_target_dunnhumby.csv".format(data_path))

inputs_train = list(map(convert_to_list, train["input"].tolist()))
labels_train = list(map(convert_to_list2, train["target"].tolist()))

inputs_test = list(map(convert_to_list, test["input"].tolist()))
labels_test = list(map(convert_to_list2, test["target"].tolist()))

batch_size = 32
input_size = len(inputs_train[0])
output_size = len(labels_train[0])
model = Model(input_size, output_size)

inp = []
lbl = []
train_size = len(inputs_train)*5

for i in range(train_size):
    tmp_inp = []
    tmp_lbl = []
    idx = [np.random.randint(0, len(inputs_train)-1) for j in range(batch_size)]
    for h in idx:
        tmp_inp.append(inputs_train[h])
        tmp_lbl.append(labels_train[h])
    lbl.append(tmp_lbl)
    inp.append(tmp_inp)

l = []
for e in range(4):
    for batch in range(train_size):

        input_batch = [inp[batch]]
        target_batch = [lbl[batch]]

        # for prediction
        #output_batch = model.model.predict(input_batch)
        # for training
        model_loss = model.model.train_on_batch(input_batch, target_batch)
        # loss = model_loss.history['loss'][0]
        if batch % 100 == 0:
            print(e, batch, train_size)
            print("Loss = {}".format(model_loss))
        l.append(model_loss)
plt.plot(l)
plt.show()



correct = 0
test_batch = 5
for i in range(int(len(inputs_test)/test_batch)):
    l = [inputs_test[i*test_batch:i*test_batch+test_batch]]
    outputs = model.model.predict_on_batch(l)
    out = []
    outs = []
    eps = 0.1
    for k in outputs:
        out = []
        for h in k:
            if h > np.random.uniform(0, 1):
                out.append(1)
            else:
                out.append(0)
        outs.append(out)
    for u in range(test_batch):
        print("nn ", outs[u])
        print("re ", labels_test[i*test_batch+u])
        if outs[u] == labels_test[i*test_batch+u]:
            correct += 1
            print("Ok")
        print("=====")

print('Accuracy of the network on the {0} test clients: {1}%'.format(len(inputs_test),
                                                                         (100*correct / len(inputs_test))))



