from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import keras_metrics
from sklearn import metrics

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


DATASET_DIR = 'image_classifier'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
EPOCHS = 5
STEPS_PER_EPOCH = 300
#VALIDATION_STEPS = 32


class SlideClassification():


	def generate(self):
		#Labelled dataset creation
		self.train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, 
			horizontal_flip = True,  validation_split=VALIDATION_SPLIT)
		#self.test_datagen = ImageDataGenerator(rescale = 1./255)
		self.training_set = self.train_datagen.flow_from_directory(DATASET_DIR, 
			target_size = (IMAGE_HEIGHT, IMAGE_WIDTH), 
			batch_size = BATCH_SIZE,  class_mode = 'binary', subset='training')
		self.validation_generator = self.train_datagen.flow_from_directory(DATASET_DIR, 
			target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=1,
			class_mode='binary', subset='validation', shuffle=False)


	def remove_truncated_image(self, gen):
		#checks if image is corrupt or not
		while True:
			try:
				features, label = next(gen)
				yield features, label
			except:
				pass
            	#TODO write corrupted images to txt


	def train(self):
		#classifier architecture
		self.classifier = Sequential()
		self.classifier.add(Conv2D(32, (3, 3), input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), 
			activation = 'relu'))
		self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
		self.classifier.add(Flatten())
		self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
		self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
			metrics = ['accuracy'])

		STEP_SIZE_TRAIN=self.training_set.n/BATCH_SIZE #train_generator.batch_size
		STEP_SIZE_VALID=self.validation_generator.n/BATCH_SIZE #valid_generator.batch_size
		print STEP_SIZE_VALID

		self.classifier.fit_generator(self.remove_truncated_image(self.training_set),
		 steps_per_epoch = STEP_SIZE_TRAIN, validation_data = self.validation_generator, 
		 validation_steps = STEP_SIZE_VALID, epochs = EPOCHS)

		#self.classifier.evaluate_generator(generator=self.train_datagen)

		self.save(self.classifier, "model_e"+str(EPOCHS)+"_spe"+str(STEP_SIZE_TRAIN)+".h5")


	def evaluate_metrics(self, model):
		#self.validation_generator.reset()
		#print model.evaluate_generator(generator=self.validation_generator, steps=1, verbose = 1)
		#print model.metrics_names
		STEP_SIZE_VALID=self.validation_generator.n/1
		predictions = model.predict_generator(self.validation_generator, 
			steps=STEP_SIZE_VALID, verbose = 1)
		#print type(predictions)
		#print predictions.shape
		val_preds = np.where(predictions>0.5,1,0).flatten()
		#print val_preds.shape
		val_trues = self.validation_generator.classes
		labels = self.validation_generator.class_indices.keys()
		precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, 
			val_preds)

		return precisions, recall, f1_score


	def save(self, classifier, model):
		#save model passed as param
		classifier.save(model)


	def load(self, model_name):
		#load model passed as param
		model = load_model(model_name)
		return model


	def inference(self, test_image, model):
		#Infer on input image, for now takes one image
		test_image = image.load_img(test_image, 
			target_size = (IMAGE_HEIGHT, IMAGE_WIDTH))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = model.predict(test_image)
		self.training_set.class_indices
		if result[0][0] == 1:
			prediction = 'Presentation'
		else:
			prediction = 'Not Presentation'

		return prediction



classification = SlideClassification()
classification.generate()
#classification.train()
model = classification.load("model_e5_spe233.h5")
precision, recall, f1 = classification.evaluate_metrics(model)
print precision, recall, f1
#prediction = classification.inference('single_inference/test_single.jpg', model)
#print "Classified as: ", prediction