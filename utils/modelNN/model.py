from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Conv2D
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

def get_architecture():
    '''
    The network should accept a 96x96 grayscale image as input, and it should
    output a vector with entries corresponding to the predicted (horizontal and
    vertical) locations of each facial keypoint.
    '''
    model = Sequential()
    #model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add( Conv2D( filters=32,
                        kernel_size=(5,5),
                        padding='same',
                        data_format='channels_last',
                        input_shape=(96,96,1),
                        activation = 'relu',
                        name = 'firstConv2D'
                    ) )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(34, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(42))

    return model;

def compile(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def train(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)

def save(model, fileName):
    model.save(fileName)

def load(fileName):
    return load_model(fileName)
