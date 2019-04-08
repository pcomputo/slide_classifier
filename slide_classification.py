from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import numpy as np


DATASET_DIR = 'image_classifier'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 1
STEPS_PER_EPOCH = 3
VALIDATION_STEPS = 32


class SlideClassification():


	def generate(self):
		#Labelled dataset creation
		self.train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, 
			horizontal_flip = True, validation_split=VALIDATION_SPLIT)
		self.test_datagen = ImageDataGenerator(rescale = 1./255)
		self.training_set = self.train_datagen.flow_from_directory(DATASET_DIR, 
			target_size = (IMAGE_HEIGHT, IMAGE_WIDTH), 
			batch_size = BATCH_SIZE, class_mode = 'binary', subset='training')
		self.validation_generator = self.train_datagen.flow_from_directory(DATASET_DIR, 
			target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
			class_mode='binary', subset='validation')


	def train(self):
		#classifier architecture
		self.classifier = Sequential()
		self.classifier.add(Conv2D(32, (3, 3), input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), 
			activation = 'relu'))
		self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
		self.classifier.add(Flatten())
		self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
		self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


		self.classifier.fit_generator(self.training_set, steps_per_epoch = STEPS_PER_EPOCH, 
			validation_data = self.validation_generator, validation_steps = VALIDATION_STEPS, 
			epochs = EPOCHS)

		self.save(self.classifier, "model_e"+str(EPOCHS)+"_spe_"+str(STEPS_PER_EPOCH)+".h5")


	def save(self, classifier, model):
		#save model passed as param
		classifier.save(model)


	def load(self, model_name):
		#load model passed as param
		model = load_model(model_name)
		return model


classification = SlideClassification()
classification.generate()
classification.train()