from utils import load_data
from model import *
import cv2
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: train <dataset> <model>')

    # Load training set
    x_train, y_train = load_data(os.path.abspath(sys.argv[1]))

    # Setting the CNN architecture
    model = get_architecture()

    # Compiling the CNN model with an appropriate optimizer and loss and metrics
    compile(model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    # Training the model
    hist = train(model, x_train, y_train)
    # train returns a History object. History.history attribute is a record of training loss values and metrics
    # values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

    # Saving the model
    save(model, os.path.abspath(sys.argv[2]))
