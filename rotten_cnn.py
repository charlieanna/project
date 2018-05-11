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

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, Input
from sklearn.model_selection import train_test_split
from data_helpers import load_data

max_words = 1000
batch_size = 32
epochs = 2

max_features = 5000
maxlen = 1000
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250




x, y, vocabulary, vocabulary_inv = load_data()

# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

# this returns a tensor
print("Creating Model...")
x = Input(shape=(sequence_length,), dtype='int32')
embed = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(x)

print('Building model...')


# x = Input(shape=(maxlen, ))
# embed = Embedding(max_features, embedding_dims, input_length=maxlen)(x)
print(embed.shape)
dropout = Dropout(0.2)(embed)

conv1 = Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(dropout)
pool = GlobalMaxPooling1D()(conv1)
#lstm = LSTM(32)(pool)
x_out = Dense(hidden_dims, )(pool)
dropout = Dropout(0.2)(x_out)

out = Activation('relu')(dropout)
x_out = Dense(1, activation='sigmoid')(out)

model = Model(x, x_out)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)
y_pred = model.predict(x_test, batch_size=batch_size)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


