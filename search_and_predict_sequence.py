import numpy as np
from keras.preprocessing import sequence

"""
This functions runs the model multiple times, meanwhile generating the caption one word at a a time
"""


def beamSearch(image_feature_vector, caption_generator_model, max_length, word_to_index, index_to_word, graph):
    beam_index = 3
    start = [word_to_index["startseq"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            partial_caption = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            with graph.as_default():
                predictions = caption_generator_model.predict([image_feature_vector, partial_caption])

            word_predictions = np.argsort(predictions[0])[-beam_index:]

            for w in word_predictions:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += predictions[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting the words according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Get the words with highest probabilities
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_partial_caption = [index_to_word[i] for i in start_word]

    generated_caption = []

    for i in intermediate_partial_caption:
        print(i)
        if i != 'endseq':
            generated_caption.append(i)
        else:
            break

    generated_caption = ' '.join(generated_caption[1:])
    return generated_caption
