
import os
import numpy as np


def word_embedding(vocab_size,wordtoix):
    # Load Glove vectors
    glove_dir = 'glove'
    embeddings_index = {}  # empty dictionary
    f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_dim = 200

    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoix.items():
        # if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector

    return embedding_dim,embedding_matrix
