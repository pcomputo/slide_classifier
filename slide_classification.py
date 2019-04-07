from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np



DATASET_DIR = 'image_classifier'
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, 
	horizontal_flip = True, validation_split=VALIDATION_SPLIT)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(DATASET_DIR, 
	target_size = (IMAGE_HEIGHT, IMAGE_WIDTH), 
	batch_size = BATCH_SIZE, class_mode = 'binary', subset='training')
validation_generator = train_datagen.flow_from_directory(DATASET_DIR, 
	target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
	class_mode='binary', subset='validation')