from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import keras
from keras import models
from keras import layers
import pandas as pd

'''
***************************************************************************************
REFERNCE
*    Title: Music Genre Classification with Python
*    Author: Parul Pandey
*    Date: 2018
*    Availability: https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
***************************************************************************************
'''

data = pd.read_csv('data.csv')
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)
