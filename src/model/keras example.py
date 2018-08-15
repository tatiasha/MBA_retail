

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

class Model:
    def __init__(self, state_size, output_size):
        self.INPUT_SIZE = state_size
        self.OUTPUT_SIZE = output_size

        self.LEARNING_RATE = 1e-4
        self.model = self.buildmodel()

    # структура сети
    def buildmodel(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.INPUT_SIZE, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.OUTPUT_SIZE, activation='relu'))
        # optimizer
        optimizer = Adam(lr=self.LEARNING_RATE)
        # loss in model
        model.compile(loss='cross', optimizer=optimizer)
        return model

input_size = 50
output_size = 50
model = Model(input_size, output_size)

input = # data
target = # data

for e in epochs:
    for batch in batches:
        input_batch = # from data
        target_batch = # from data

        # for prediction
        output_batch = model.predict(input_batch)
        # for training
        model_loss = model.train_on_batch(input_batch, target_batch)
        print("Loss = {}".format(model_loss))



