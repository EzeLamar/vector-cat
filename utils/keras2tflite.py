import tensorflow as tf
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: train <keras_model> <tflite_model>')

    converter = tf.lite.TFLiteConverter.from_keras_model_file(os.path.abspath(sys.argv[1]))
    tflite_model = converter.convert()
    open(os.path.abspath(sys.argv[2]), "wb").write(tflite_model)
