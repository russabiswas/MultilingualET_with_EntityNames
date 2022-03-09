from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Conv1D, GlobalMaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.layers.convolutional import MaxPooling1D
from pandas import read_csv
#from load_embedding import load_embeddings
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print('Loading data...')

path_x = "X_wiki2vec"
path_y = "Y_data_coarse"

def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

x_data = load_file(path_x)
y_data = load_file(path_y)

print(len(x_data), 'x sequences')
print(len(y_data), 'y sequences')

#print(len(x_dev), 'val sequences')

input_dim, output_dim = 300, 1

x_data = x_data.astype('float32')
y_data = y_data.astype('float32')


x_train, x_dev_test, y_train, y_dev_test = train_test_split(x_data, y_data, test_size=0.50)
x_dev, x_test, y_dev, y_test = train_test_split(x_dev_test, y_dev_test, test_size=0.60)

y_train_hot = np_utils.to_categorical(y_train)
print('New y_data shape: ', y_train_hot.shape)


print('Xtrain shape', x_train.shape)
print('Xdev+test shape', x_dev_test.shape)
print('ytrain shape', y_train.shape)
print('Ydev+test shape', y_dev_test.shape)
print('Xdev shape', x_dev.shape)
print('Xtest shape', x_test.shape)
print('Ydev shape', y_dev.shape)
print('Ytest shape', y_test.shape)



y_test_hot = np_utils.to_categorical(y_test)
print('New y_test shape: ', y_test_hot.shape)
y_dev_hot = np_utils.to_categorical(y_dev)
print('New y_dev shape: ', y_dev_hot.shape)

print('Build model...')

model_m = Sequential()

model_m.add(Reshape((300,1), input_shape=(300,)))
model_m.add(Flatten())
model_m.add(Dense(256,activation='relu'))
model_m.add(Dense(128, activation='relu'))
model_m.add(Dense(45, activation='softmax'))

model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_m.summary())

model_m.fit(x_train, y_train_hot, 
          validation_data=(x_dev, y_dev_hot),
          batch_size =32,
          epochs=100)



#print(classification_report(y_test, pred))
print('TEST SHAPE')
print('X', x_test.shape)
print('Y', y_test.shape)
y_pred = model_m.predict_classes(x_test)
#y_pred = model_m.predict(x_test)

print('prediction shape', y_pred.shape)



loss = model_m.evaluate(x_test, y_test_hot)[0]
accuracy = model_m.evaluate(x_test, y_test_hot)[1]
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_normal = f1_score(y_test, y_pred, average=None)

print('loss', loss)
print('accuracy', accuracy)
print('f1_macro', f1_macro)
print('f1_micro', f1_micro)
print('f1_normal', f1_normal)

model_m.save("model.h5")
print("Saved model to disk")

f = open("predictions.txt", "w")
for i in range(len(x_test)):
    f.write(str(y_test[i]))
    f.write("\t")
    f.write(str(y_pred[i]))
    f.write("\n")
f.close()
