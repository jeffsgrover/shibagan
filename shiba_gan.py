from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100
# Size vector to generate images from
NOISE_SIZE = 100
# Configuration
EPOCHS = 10 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3

training_data = np.load('shibas.npy')

Spell run python shiba_gan.py -t V100 -m uploads/shiba_gan/shiba_data.npy