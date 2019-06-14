from keras import Input, layers
from keras import optimizers
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, \
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization

from keras.models import Model
from keras.layers.merge import add
import data_generator
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam, RMSprop


def train_the_data(class_name, max_length, vocab_size, embedding_dim, embedding_matrix, train_descriptions, train_features, wordtoix):
    image_model = Sequential([
        Dense(embedding_dim, input_shape=(1536,), activation='relu'),
        RepeatVector(max_length)
    ])

    caption_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])

    model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    epochs = 10
    number_pics_per_bath = 3
    steps = len(train_descriptions) // number_pics_per_bath

    for i in range(epochs):
        generator = data_generator.data_generator(train_descriptions, train_features, wordtoix, max_length,
                                                  number_pics_per_bath, vocab_size)
        print(generator)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    model.save('Model/model_'+ class_name + '.h5')


