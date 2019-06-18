
import os
import numpy as np
"""
Generates 200 length vector for each word
If the word is not present in the index, it returns all zeros for taht word.

This requires glove directory to be downloaded and installed in Glove Directory
"""


def word_embedding(vocabulary_size, word_to_index):
    # Loading all the  Glove vectors
    glove_directory_location = 'glove'
    embeddings_index = {}  # empty

    with open(os.path.join(glove_directory_location, 'glove.6B.200d.txt'), encoding="utf-8") as glove_file:
        for each_line in glove_file:
            values = each_line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients

    embedding_vector_length = 200

    embedding_matrix = np.zeros((vocabulary_size, embedding_vector_length))

    for word, i in word_to_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_vector_length, embedding_matrix
