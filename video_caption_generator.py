

import pre_processing_caption
import pre_processing_image
import train_the_data
import word_embeddings
import time
import pickle as pickle
import predict_caption

from flask import Flask
app = Flask(__name__)


@app.route("/")
def test():
    return "Working!!"

# Add a flask API to upload Images - and train them and add a model
def upload_and_train(class_name):
    if class_name == "general":
        print("In General")
        start = time.time()
        max_length, vocab_size, train_descriptions , train_img, wordtoix= pre_processing_caption.pre_process_captions("general")
        end = time.time()
        print("Caption Pre processing completed in : "+str(end-start))
        start = time.time()
        # train_features = pre_processing_image.pre_process_images("general", train_img)
        # Load the bottleneck train features from the disk
        with open("Pickle/general_trained.pkl", "rb") as encoded_pickle:
            train_features = pickle.load(encoded_pickle)
        end = time.time()
        print("Image Pre processing completed in : "+str(end-start))

        embedding_dim, embedding_matrix = word_embeddings.word_embedding(vocab_size, wordtoix)
        start = time.time()
        train_the_data.train_the_data(class_name, max_length, vocab_size, embedding_dim, embedding_matrix,
                                      train_descriptions, train_features, wordtoix)
        end = time.time()
        print("Training completed in : " + str(end - start))


# Add a flask API to fetch a model and Generate Caption for an Input Image


def generate_caption(class_name, img_path):
    predict_caption.predict_caption(class_name, img_path)

# Add a flask API to fetch a model and Generate Caption for a Video. An asynchronous function?


# Write an API to fetch a list of models


# app.run(port=5000)

# upload_and_train("general")
generate_caption("general","Dataset/general_images/3289893683_d4cc3ce208.jpg")










