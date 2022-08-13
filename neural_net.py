from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt

class Neural_net:
    def __init__(self, input_size, topology, output_size) -> None:
        self.topology = topology
        self.input_size = input_size
        self.output_size = output_size

        self.model = keras.Sequential()
        for i, size in enumerate(topology):
            if i == 0:
                #first layer and second layer
                act = 'relu'
                self.model.add(keras.layers.Dense(units = topology[i], activation = 'linear', input_shape=[input_size]))
                self.model.add(keras.layers.Dense(units = topology[i], activation = act)) #, kernel_initializer = keras.initializers.GlorotUniform()))
            elif i > 0 and i < len(topology)-1:
                #core layers
                self.model.add(keras.layers.Dense(units = topology[i], activation = act))
            else:
                #last layer
                self.model.add(keras.layers.Dense(units = output_size, activation = 'linear'))

        #opt = tf.keras.optimizers.SGD(lr=0.000001, momentum=0.9, clipnorm=0.5)
        self.model.compile(loss='mape', optimizer='adam', metrics=['mape'])
        #loss='mse' tf.keras.metrics.MeanAbsolutePercentageError()
        #keras.metrics.RootMeanSquaredError()
    
        self.model.summary()
                
    def load_data(self, data_file, test_size):
        #input vector
        with open(data_file) as input:
            file = csv.reader(input, delimiter=',')
            self.input_data = []
            for row in file:
                input_data_row = []
                for i in range(self.input_size):
                    input_data_row.append(float(row[i]))
                self.input_data.append(input_data_row)
        #output vector
        with open(data_file) as output:
            file = csv.reader(output, delimiter=',')
            self.output_data = []
            for row in file:
                output_data_row = []
                for i in range(self.input_size, self.input_size + self.output_size):
                    output_data_row.append(float(row[i]))
                self.output_data.append(output_data_row)

        if test_size is None:
            #do not split training data
            self.input_data_train, self.output_data_train= self.input_data, self.output_data      
            self.input_data_train, self.output_data_train = shuffle(self.input_data_train, self.output_data_train)

        elif test_size is not None:
            #split training data
            self.input_data_train, self.input_data_test, self.output_data_train, self.output_data_test = train_test_split(self.input_data, self.output_data, test_size=test_size)
            
            self.input_data_train, self.output_data_train = shuffle(self.input_data_train, self.output_data_train)
            self.input_data_test, self.output_data_test = shuffle(self.input_data_test, self.output_data_test)


    def train_model(self, epochs, error_limit = None, savefig_name = None):
        if error_limit is not None:
            callback=my_treshold_callback(threshold=error_limit)
            self.history = self.model.fit(self.input_data_train, self.output_data_train, epochs=epochs, verbose=1, callbacks=[callback], batch_size=16)
        else:
            self.history = self.model.fit(self.input_data_train, self.output_data_train, epochs=epochs, verbose=1, batch_size=16)

        #plot training
        plt.plot(self.history.history['mape']) #'mape'
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Percentage Error [%]')
        #plt.ylim((10**0,40))
        plt.yscale('log')
        plt.grid()
        if savefig_name is not None:
            plt.savefig(savefig_name + str(self.topology) + '.jpg')
        else:
            plt.show()

    def predict(self, custom_data):
        # if custom_data is None:
        #     return self.model.predict(self.input_data_test)
        # else:
        return self.model.predict(custom_data)

    def load_model(self, filename):
        self.model = keras.models.load_model(filename)
        print(f'Loaded model {filename}')

    def save_model(self, filename):
        model_name = filename + '_'+ str(self.input_size) + str(self.topology) + str(self.output_size)
        self.model.save(model_name)
        print(f'Saved model as {model_name}')

#custom class for stopping when loss metric ('mape') reaches certain treshold value
class my_treshold_callback(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(my_treshold_callback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["mape"]
        if accuracy <= self.threshold:
            self.model.stop_training = True

if __name__ == '__main__':

    nn = Neural_net(10000, (32, 32), 9604) 
    nn.load_data('input.csv', None)
    nn.train_model(150_000, 1., 'he_whole')
    nn.save_model('he_whole')
