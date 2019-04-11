# slide_classifier
Classifier to help classify images that contain slides


ImageDataGenerator of Keras was used to create the labelled dataset (images, labelled as presentation or not). Some images in the dataset are truncated, a helper function helps remove these corrupted images. A validation set is created from this training set with a 80:20 percent split. train() helps to train the Neural Network, and save() and load() help with saving and loading the model. inference() helps with the predictions and evaluate_metrics() helps plot a classification report.
There was no feature extraction since the model figures out its own features. Convolutional Neural Network was chosen based on intuition to do well on images. If there was more time, there could be other models which could be prototyped as well. binary_crossentropy was chosen as the loss function since its a binary classification problem. Adam optimizer is useful on account of its momentum. 5 epochs were chosen after hyperparamter optimization phase.
The accuracy metrics for class presentation versus not presentation:
precision: [0.91395793 0.97393894] 
recall: [0.93177388 0.96674058] 
f1: [0.92277992 0.97032641]
support: [513 1353]

Since in general the dataset had more data for presentation than no presentation, hence the model did better for the presentation class. Otherwise, these metrics look good for the data. They could possibly be tuned more with a deeper network.
One of the challenges was the long waiting times during training epochs due to system architecture constraints.


