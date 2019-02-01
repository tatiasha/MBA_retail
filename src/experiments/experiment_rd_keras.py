from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
import keras.losses
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD, PCA

import matplotlib.pyplot as plt
# инициализируем Tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# инициализируем Keras
from keras import backend as K

K.set_session(sess)

def convert_to_list(a):
    t = [i[1:len(i) - 1] for i in a]
    r = [i.split(', ') for i in t]
    q = [list(map(int, i)) for i in r]
    return q


class Model:
    def __init__(self, state_size, output_size):
        self.INPUT_SIZE = state_size
        self.OUTPUT_SIZE = output_size
        self.LEARNING_RATE = 1e-4
        self.model = self.buildmodel()

    # структура сети
    def buildmodel(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.INPUT_SIZE, activation=keras.activations.relu))
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
global_data_path = "E:\Projects\MBA_retail\\tmp\datasets"
data = pd.read_csv('{0}\\x_target_3.csv'.format(data_path))
data = data.drop(columns=['Unnamed: 0'])
purchasing = data['Purchasing'].values
target = data['Target'].values
X = convert_to_list(purchasing)
y = convert_to_list(target)
inputs_train = X[:5000]
labels_train = y[:5000]

inputs_test = X[5000:]
labels_test = y[5000:]

inp = []
lbl = []
train_size = 30000
accuracy = []
components = []
for number_of_components in range(1, 52, 2):
    batch_size = 20
    input_size = number_of_components
    output_size = number_of_components
    model = Model(input_size, output_size)
    if number_of_components != 51:
        # svd = TruncatedSVD(n_components=number_of_components, n_iter=100, random_state=51)
        svd = PCA(n_components=number_of_components, random_state=51)
        transform_inputs = svd.fit_transform(inputs_train)
        transform_targets = svd.transform(labels_train)
    else:
        transform_inputs = inputs_train
        transform_targets = labels_train

    inp = []
    lbl = []
    for i in range(train_size):
        tmp_inp = []
        tmp_lbl = []
        idx = [np.random.randint(0, 5000) for j in range(batch_size)]
        for h in idx:
            tmp_inp.append(transform_inputs[h])
            tmp_lbl.append(transform_targets[h])
        lbl.append(tmp_lbl)
        inp.append(tmp_inp)

    l = []
    epochs = 7
    for e in range(epochs):
        for batch in range(train_size):
            print("components = {0} epoch = {1}/{2} {3}/{4}".format(number_of_components, e+1, epochs, batch+1, train_size))
            input_batch = [inp[batch]]
            target_batch = [lbl[batch]]
            # for prediction
            #output_batch = model.model.predict(input_batch)
            # for training
            model_loss = model.model.train_on_batch(input_batch, target_batch)
            # loss = model_loss.history['loss'][0]
            # print("Loss = {}".format(model_loss))
            # l.append(model_loss)

    # model_loss = model.model.fit([inputs_train], [labels_train], batch_size=32, epochs=8)
    # loss = model_loss.history['loss'][0]

    test_batch = 5
    ac = 0
    for rep in range(epochs):
        correct = 0
        for i in range(int(len(inputs_test)/test_batch)):
            l = [inputs_test[i*test_batch:i*test_batch+test_batch]]

            if number_of_components != 51:
                transform_test_inputs = svd.transform(l[0])
                outputs = model.model.predict_on_batch(transform_test_inputs)
                outputs = svd.inverse_transform(outputs)
            else:
                transform_test_inputs = l
                outputs = model.model.predict_on_batch(transform_test_inputs)

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
                # print("nn ", outs[u])
                # print("re ", labels_test[i*test_batch+u])
                if outs[u] == labels_test[i*test_batch+u]:
                    correct += 1
                    # print("Ok")
                # print("=====")
        ac += 100*correct / len(inputs_test)
    components.append(number_of_components)
    accuracy.append(ac/float(epochs))
    print('Accuracy of the network on the {0} test clients with {1} components: {2}%'.format(len(inputs_test), number_of_components,
                                                                                             ac / float(epochs)))

plt.plot(components, accuracy)
plt.xlabel("Number of components")
plt.ylabel("Accuracy")
plt.show()

print(components)
print(accuracy)
