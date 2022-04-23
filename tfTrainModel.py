import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

epochsVal = 100

class_names = ['Like','Dislike']

dataSet = pd.read_csv("C:/Users/bcwhi/Documents/GitHub/462Option2FinalProject/data.csv")

dataSetFeatures = dataSet.copy()
dataSetLabels = dataSetFeatures.pop('liked')

dataSetFeatures = np.array(dataSetFeatures)

spotModel = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

spotModel.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam())

spotModel.fit(dataSetFeatures, dataSetLabels, epochs = epochsVal)

spotModel.save("spot")