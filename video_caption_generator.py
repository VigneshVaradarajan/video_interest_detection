

import pre_processing_caption
import pre_processing_image
import train_the_data_old
import word_embeddings
import time
import pickle as pickle
import predict_caption
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.models import load_model
import os
from keras import backend as K
import tensorflow as tf

from flask_cors import CORS

from flask import Flask, request
app = Flask(__name__)
CORS(app)


inception_model = None
graph = None
caption_model = None

@app.route("/update")
def download_model():
    K.clear_session()
    global inception_model
    global caption_generator_model
    model = InceptionResNetV2(weights='imagenet')
    # returns all the vectors before classification
    inception_model = Model(model.input, model.layers[-2].output)
    print("Model Created")
    pickle.dump(inception_model, open('Model/inception_model.pkl', 'wb'))

    print("Inception Model created and Save")

    return "Completed", 200


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
        train_the_data_old.train_the_data(class_name, max_length, vocab_size, embedding_dim, embedding_matrix,
                                          train_descriptions, train_features, wordtoix)
        end = time.time()
        print("Training completed in : " + str(end - start))


# Add a flask API to fetch a model and Generate Caption for an Input Image

@app.route("/generate", methods=['POST'])
def generate_caption():
    global inception_model, caption_model
    input_img = request.files['file']
    input_img.save("input.jpg")
    data = request.form
    print(data)
    class_name = data['class_name']
    img_path = "input.jpg"
    return predict_caption.predict_caption(class_name, img_path, inception_model, caption_model, graph), 200
    # return predict_caption.predict_caption(class_name, img_path), 200



# Add a flask API to fetch a model and Generate Caption for a Video. An asynchronous function?


# Write an API to fetch a list of models


# app.run(port=5000)

# upload_and_train("general")
# generate_caption("general","Dataset/general_images/3289893683_d4cc3ce208.jpg")
# download_model()


if __name__ == '__main__':
    # global inception_model

    model = InceptionResNetV2(weights='imagenet')
    # returns all the vectors before classification
    inception_model = Model(model.input, model.layers[-2].output)

    caption_model = load_model("Model/model_"+"general"+".h5")

    inception_model._make_predict_function()


    # inception_model = pickle.load(open('Model/inception_model.pkl', 'rb'))

    # global graph
    graph = tf.get_default_graph()

    print("Inception Model created and Saved")
    app.run(debug=False, host='0.0.0.0', port=5000)








