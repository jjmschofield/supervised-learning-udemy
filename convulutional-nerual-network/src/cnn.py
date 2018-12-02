# Data pre-processing is not required as Karis can handle this for us
# Based on the folder structure of the dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # mac issue

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Structure
# Convolution -> Pooling -> Flattening -> Input -> Hidden -> Output

# Config
image_dimension = 64  # can be bigger but will slow processing


# The convolution layer creates feature maps from an image
def create_convolution_layer():
    n_feature_maps = 32  # can be bigger but often this is in subsequent layers
    feature_map_dimension = 3
    image_color_channels = 3  # rbg
    return Convolution2D(n_feature_maps,
                         feature_map_dimension,
                         input_shape=(image_dimension, image_dimension, image_color_channels),
                         activation='relu'
                         )


# Pooling is equiv to down sampling reducing data size and handling rotation of images whilst maintaining spacial difs
def create_pooling_layer():
    step_size = 2  # a 2 by 2 is normally recommend, feature maps reduce in size by 2 and will handle rotation
    return MaxPool2D(pool_size=(step_size, step_size))


# Flattening creates a flattened input vector for the input layer of the ANN
def create_flattening_layer():
    return Flatten()


def create_fully_connected_hidden_layer():
    return Dense(
        units=128,  # Driven by experimentation to derive best hyper parameter
        activation='relu'
    )


def create_fully_connected_output_layer():
    return Dense(
        units=1,
        activation='sigmoid'  # Gives us probability based on softmax
    )


# Init
classifier = Sequential()

# Convert image into input vector
classifier.add(create_convolution_layer())
classifier.add(create_pooling_layer())
classifier.add(create_flattening_layer())

# Create ANN
classifier.add(create_fully_connected_hidden_layer())
classifier.add(create_fully_connected_output_layer())

# Compile
# Binary cross entropy is used as we have a binary outcome (cat vs dog), otherwise we would use cross_entropy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../course-resources/dataset'

# Image augmentation
# Prevent over fitting by creating lots of batches and adding random transformations (like rotating, flipping etc)
# This improves accuracy with the same amount of data!
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    dataset_path + '/training_set',
    target_size=(image_dimension, image_dimension),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    dataset_path + '/test_set',
    target_size=(image_dimension, image_dimension),
    batch_size=32,
    class_mode='binary')

# Train
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,  # Number of images
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)  # Number of images
