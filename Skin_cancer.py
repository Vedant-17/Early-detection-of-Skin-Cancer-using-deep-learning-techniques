import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random
import zipfile
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
import PIL

import zipfile
zip_ref = zipfile.ZipFile('/content/drive/MyDrive/Skin Cancer Datasets /archive.zip','r')
zip_ref.extractall('/content/Skin Cancer Datasets')
zip_ref.close()

#Dataset used: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
#contains images of skin lesions
path='/content/Skin Cancer Datasets/hmnist_28_28_RGB.csv'

#reads the CSV file located at the specified path
df=pd.read_csv(path)

#display the last few rows of the DataFrame
df.tail()

# this part of the code shuffles the DataFrame df and then splits it into training and testing
fractions=np.array([0.8,0.2])
df=df.sample(frac=1)
train_set, test_set = np.array_split(df, (fractions[:-1].cumsum() * len(df)).astype(int))

print(len(train_set))
print(len(test_set))

#used to retrieve the unique values present in the column named 'label'
df.label.unique()

# reference: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/discussion/183083
classes={
    0:('akiec', 'actinic keratoses and intraepithelial carcinomae'),

    1:('bcc' , 'basal cell carcinoma'),

    2:('bkl', 'benign keratosis-like lesions'),

    3:('df', 'dermatofibroma'),

    4:('nv', ' melanocytic nevi'),

    5:('vasc', ' pyogenic granulomas and hemorrhage'),

    6:('mel', 'melanoma'),
}

#This code separates the features (input) and labels (output) for both training and testing sets.

y_train=train_set['label']

x_train=train_set.drop(columns=['label'])

y_test=test_set['label']

x_test=test_set.drop(columns=['label'])

columns=list(x_train)


#this part of the code determines whether the system has a GPU available for use with PyTorch. 
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Random oversampling is performed on the training data
#The purpose of this part of the code is to ensure that the classes in the training dataset are balanced before training the machine learning models.
#Balanced classes can lead to better model performance
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()
x_train,y_train  = oversample.fit_resample(x_train,y_train)


#the code displays multiple random images from the training data,
#allowing for visual inspection of the images to get an understanding of the dataset.
import matplotlib.pyplot as plt
import random

num=random.randint(0,8000)
x_train=np.array(x_train, dtype=np.uint8).reshape(-1,28,28,3)

plt.imshow(x_train[num].reshape(28,28,3))
plt.title("Random image from training data")
plt.show()
num=random.randint(0,8000)
plt.imshow(x_train[num].reshape(28,28,3))
plt.title("Random image from training data")
plt.show()

num=random.randint(0,8000)
plt.imshow(x_train[num].reshape(28,28,3))
plt.title("Random image from training data")
plt.show()

#this part of the code defines a CNN architecture for classifying images. 
#The architecture consists of multiple convolutional layers, max pooling layers,
# batch normalization layers, dropout layers, and dense layers. 
#The model is designed to take input images of size 28x28 pixels with 3 color channels (RGB) and output probabilities for 7 different classes.

#Convolutional Layers: applies filter to the data, Detect patterns like edges or textures.
#Max Pooling Layers: Reduce image size, keeping important features.
#Batch Normalization Layers: Stabilize and speed up training by normalizing inputs.
#Dropout Layers: Prevent overfitting by randomly turning off neurons.
#Dense Layers: Connect all neurons, making final predictions.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
import tensorflow as tf
%time

model = Sequential()

model.add(Conv2D(16,
                 kernel_size = (3,3),
                 input_shape = (28, 28, 3),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(32,
                 kernel_size = (3,3),
                 activation = 'relu'))

model.add(Conv2D(64,
                 kernel_size = (3,3),
                 activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(128,
                 kernel_size = (3,3),
                 activation = 'relu'))

model.add(Conv2D(256,
                 kernel_size = (3,3),
                 activation = 'relu'))

model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(256,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(128,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(64,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(7,activation='softmax'))

model.summary()



# ModelCheckpoint callback is used to save the best model based on validation accuracy.
callback = tf.keras.callbacks.ModelCheckpoint(filepath='/content/best_model.h5',
                                              monitor='val_acc',
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True)
             
 
# prepares the model for training by setting up the optimizer and 
# compiling it with the specified loss function, optimizer, and evaluation metric.                                              
%time

optimizer=tf.keras.optimizers.Adam(lr=0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer =optimizer,
              metrics = ['accuracy'])
              
     
# training the model and measure the duration of the training process
from datetime import datetime

start_time = datetime.now()

history = model.fit(x_train,
                    y_train,
                    validation_split=0.2,
                    batch_size = 128,
                    epochs = 50,
                    shuffle=True,
                    callbacks=[callback])

end_time = datetime.now()

print('Duration: {}'.format(end_time - start_time))


model.load_weights('/content/best_model.h5')


# Evaluate CNN model
x_test=np.array(x_test).reshape(-1,28,28,3)

loss, acc = model.evaluate(x_test, y_test, verbose=2)

print("CNN Accuracy:", acc)



# Prepare data for Random Forest and KNN and SVM

x_train_rf_knn_svm = train_set.drop(columns=['label']).values
y_train_rf_knn_svm = train_set['label'].values
x_test_rf_knn_svm = test_set.drop(columns=['label']).values
y_test_rf_knn_svm = test_set['label'].values

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_rf_knn_svm, y_train_rf_knn_svm)

# KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_rf_knn_svm, y_train_rf_knn_svm)


#svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
#svm_model.fit(x_train.reshape(len(x_train), -1), y_train)

# SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(x_train_rf_knn_svm, y_train_rf_knn_svm)

# Predictions
# CNN predictions
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Random Forest predictions
y_pred_rf = rf_model.predict(x_test_rf_knn_svm)

# KNN predictions
y_pred_knn = knn_model.predict(x_test_rf_knn_svm)

# SVM predictions
y_pred_svm = svm_model.predict(x_test_rf_knn_svm)

# Ensemble predictions (simple averaging)
y_pred_ensemble = np.round((y_pred + y_pred_rf + y_pred_knn) / 3).astype(int)


# Accuracy
accuracy_cnn = accuracy_score(y_test, y_pred)
accuracy_rf = accuracy_score(y_test_rf_knn_svm, y_pred_rf)
accuracy_knn = accuracy_score(y_test_rf_knn_svm, y_pred_knn)
accuracy_svm = accuracy_score(y_test_rf_knn_svm, y_pred_svm)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

print("CNN Accuracy:", accuracy_cnn * 100, "%")
print("Random Forest Accuracy:", accuracy_rf * 100, "%")
print("KNN Accuracy:", accuracy_knn * 100, "%")
print("SVM Accuracy:", accuracy_svm * 100, "%")
print("Ensemble Accuracy:", accuracy_ensemble * 100, "%")

# Predict probabilities using CNN model
y_pred_prob = model.predict(x_test)
cnn_probs = np.zeros_like(y_pred_prob)
cnn_probs[np.arange(len(y_pred_prob)), y_pred_prob.argmax(1)] = 1  # One-hot encoding

# Predict probabilities using Random Forest, KNN, and SVM models
rf_probs = rf_model.predict_proba(x_test_rf_knn_svm)
knn_probs = knn_model.predict_proba(x_test_rf_knn_svm)
svm_probs = svm_model.predict_proba(x_test_rf_knn_svm)

# Combine the predicted probabilities from each model
ensemble_probs = (rf_probs + knn_probs + svm_probs + cnn_probs) / 4

# Take the average probability for each class label
ensemble_pred = np.argmax(ensemble_probs, axis=1)

# Evaluate accuracy
accuracy_ensemble = accuracy_score(y_test_rf_knn_svm, ensemble_pred)
print("Hybrid Model Accuracy:", accuracy_ensemble * 100, "%")



