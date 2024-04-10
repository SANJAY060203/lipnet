import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.initializers import Orthogonal
import tensorflow as tf

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    # model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    model.add(Dropout(.5))

    # model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # weights_path = os.path.join('..', 'models', 'checkpoint.h5')  # Adjust the extension as necessary
    # # weights_path = 'C:/haidun.li/NSCC/LipNet/models/checkpoint'
    # print('weights_path:', weights_path)
    # if os.path.exists(weights_path):
    #     model.load_weights(weights_path)
    # else:
    #     raise FileNotFoundError(f"Weight file {weights_path} does not exist.")

    # return model

     # Create a checkpoint instance that points to the folder where the checkpoints are saved
    checkpoint_dir = os.path.join('..', 'models')
    checkpoint = tf.train.Checkpoint(model=model)

    # Restore the latest checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        checkpoint.restore(latest)
    else:
        raise FileNotFoundError("No checkpoint found in {}".format(checkpoint_dir))

    return model