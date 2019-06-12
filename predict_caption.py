import common_functions as common_functions
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
import pickle as pickle
import search_and_predict_sequence
from keras.models import load_model
import matplotlib.pyplot as plt



"""
Function that returns all the images of the dataset
Input : Image path
"""


def fetch_image(img_path):
    # Default Target size for inception restnet v2 is 299,299
    img = image.load_img(img_path, target_size=(299, 299))
    # Conver image to an array
    image_array = image.img_to_array(img)
    # Insert a new axis that will appear at the axis position in the expanded array shape.
    image_array = np.expand_dims(image_array, axis=0)

    # Use the inbuilt function of inception resnet to pre-process the data
    pre_processed_image = preprocess_input(image_array)
    return pre_processed_image


"""
Function that runs InceptionRestNet
Input : Image path
"""


def run_inceptionRestNet(img_path,updated_model):

    pre_processed_image = fetch_image(img_path)
    # use the updated model to predict the vectors for each of the image
    vectors = updated_model.predict(pre_processed_image)
    vectors = np.reshape(vectors, vectors.shape[1])
    return vectors


"""
Function that pre-processes all the images
Input : dataset_name - Dataset name to work on.
"""


def predict_caption(class_name, img):
    print(class_name)

    model = InceptionResNetV2(weights='imagenet')
    # returns all the vectors before classification
    updated_model = Model(model.input, model.layers[-2].output)
    print("Model created")
    feature_vector = run_inceptionRestNet(img, updated_model)

    model = load_model("Model/model_"+class_name+".h5")

    with open("Pickle/"+class_name+"_max_length.pkl", "rb") as encoded_pickle:
        max_length = pickle.load(encoded_pickle)

    with open("Pickle/"+class_name+"_ixtoword.pkl", "rb") as encoded_pickle:
        ixtoword = pickle.load(encoded_pickle)

    with open("Pickle/"+class_name+"_wordtoix.pkl", "rb") as encoded_pickle:
        wordtoix = pickle.load(encoded_pickle)

    photo = feature_vector.reshape((1,1536))
    x = plt.imread(img)
    plt.imshow(x)
    # print(search_and_predict_sequence.greedySearch(photo, model, max_length, wordtoix, ixtoword))
    print(search_and_predict_sequence.beamSearch(photo, model, max_length, wordtoix, ixtoword))


