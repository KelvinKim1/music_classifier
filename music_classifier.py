from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import keras
from keras import models
from keras import layers
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler


'''
***************************************************************************************
REFERNCE
*    Title: Music Genre Classification with Python
*    Author: Parul Pandey
*    Date: 2018
*    Availability: https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
***************************************************************************************
'''
file_to_classify = 'Monologue.wav'

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
#print('test_acc: ',test_acc)

with open('data_to_predict.csv', mode='r') as csv_reader:
    readcsv = csv.reader(csv_reader)
    Xnew = []
    for row in readcsv:
        Xnew.append(row)
csv_reader.close()
Xnew = scaler.transform(Xnew)
ynew = model.predict_classes(Xnew)
for i in range(len(Xnew)):
    if ynew[i] == 0:
        print("Predicted = classical")
    elif ynew[i] == 1:
        print("Predicted = country")
    elif ynew[i] == 2:
        print("Predicted = hiphop")
    elif ynew[i] == 3:
        print("Predicted = jazz")
    elif ynew[i] == 4:
        print("Predicted = metal")
    elif ynew[i] == 5:
        print("Predicted = pop")
    elif ynew[i] == 6:
        print("Predicted = rock")



