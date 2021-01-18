import numpy as np
import os
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, AveragePooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import config


# CONSTANTS
LR = 0.0001
BATCH_SIZE = 32
EPOCHS = 5
OPT = Adam(LR)

# DATA GENERATOR
generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
	height_shift_range=0.2,
    brightness_range=[0.75, 1],
    zoom_range=[0.85, 1.05],
    fill_mode='nearest',
    validation_split=0.15
)

train_generator = generator.flow_from_directory(
    config.TRAIN_BASE_PATH, 
    target_size=config.INPUT_SHAPE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset='training'
)

validation_generator = generator.flow_from_directory(
    config.TRAIN_BASE_PATH, 
    target_size=config.INPUT_SHAPE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset='validation'
)

# MODEL
base_model = MobileNetV3Small((*config.INPUT_SHAPE, 3), include_top=False)
base_model.trainable = False

x = AveragePooling2D(pool_size=(7, 7))(base_model.output)
x = Flatten(name="flatten")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(base_model.input, x)
model.compile(OPT, 'binary_crossentropy', ['accuracy'])

### TRAIN
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

model.save(config.MODEL_PATH)