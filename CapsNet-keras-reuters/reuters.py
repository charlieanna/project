'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# 8083/8083 [==============================] - 1s - loss: 1.4334 - acc: 0.6759 - val_loss: 1.0897 - val_acc: 0.7664
# Epoch 2/5
# 8083/8083 [==============================] - 1s - loss: 0.7876 - acc: 0.8173 - val_loss: 0.9397 - val_acc: 0.7853
# Epoch 3/5
# 8083/8083 [==============================] - 1s - loss: 0.5477 - acc: 0.8647 - val_loss: 0.8997 - val_acc: 0.7964
# Epoch 4/5
# 8083/8083 [==============================] - 1s - loss: 0.4160 - acc: 0.8984 - val_loss: 0.8812 - val_acc: 0.8109
# Epoch 5/5
# 8083/8083 [==============================] - 1s - loss: 0.3250 - acc: 0.9170 - val_loss: 0.9113 - val_acc: 0.8009
# 1952/2246 [=========================>....] - ETA: 0sTest score: 0.88778052623
# Test accuracy: 0.792074799644
