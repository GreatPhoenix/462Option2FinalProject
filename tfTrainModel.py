import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

epochsVal = 100

class_names = ['Like','Dislike']

dataSet = pd.read_csv("C:/Users/bcwhi/Documents/GitHub/462Option2FinalProject/tfTrainData.csv", names=["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature","liked"])
testDataSet = pd.read_csv("C:/Users/bcwhi/Documents/GitHub/462Option2FinalProject/tfTestData.csv", names=["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature","liked"])
#print(dataSet.head())

dataSetFeatures = dataSet.copy()
dataSetLabels = dataSetFeatures.pop('liked')

testdataSetFeatures = testDataSet.copy()
testdataSetLabels = testdataSetFeatures.pop('liked')

dataSetFeatures = np.array(dataSetFeatures)

testdataSetFeatures = np.array(testdataSetFeatures)

spotModel = tf.keras.Sequential([
    layers.Dense(32, activation = 'relu', input_dim =  13),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1)
])

spotModel.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(), metrics= 'accuracy')

spotModel.fit(dataSetFeatures, dataSetLabels, epochs = epochsVal, batch_size = 100)

#spotModel.save('spotModel')

loss, acc = spotModel.evaluate(testdataSetFeatures, testdataSetLabels, verbose=1)
print('Model, accuracy: {:5.2f}%'.format(100 * acc))

spotModel.save('C:/Users/bcwhi/Documents/GitHub/462Option2FinalProject/spot')

predTest = spotModel.predict(testdataSetFeatures)
predTest = [0 if val < 0.5 else 1 for val in predTest]
print(predTest)
#print(dataSetFeatures)
#print(spotModel.summary())