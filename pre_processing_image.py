import glob
import constants as constants
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input


import pickle as pickle

"""
Function that gets the list of image sin teh directory
Input : datsset_name - Datset to fetch
"""
def fetch_image_list(dataset_name):
    image_location = constants.dataset_location + "\\"+dataset_name+"_images"
    return glob.glob(image_location+'/*')


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
    # model = models.inception_v3()


    # use the updated model to predict the vectors for each of the image
    vectors = updated_model.predict(pre_processed_image)
    vectors = np.reshape(vectors, vectors.shape[1])
    return vectors


"""
Function that pre-processes all the images
Input : dataset_name - Dataset name to work on.
"""


def pre_process_images(dataset_name,train_img):
    print(dataset_name)
    # # Load all Images from the data set
    # images = fetch_image_list(dataset_name)
    model = InceptionResNetV2(weights='imagenet')
    # returns all the vectors before classification
    updated_model = Model(model.input, model.layers[-2].output)
    print("Model created")
    images = 'Dataset/general_images/'

    train_features = {}
    for img in train_img:
        feature_vector = run_inceptionRestNet(img,updated_model)
        train_features[img[len(images):]] = feature_vector

    # Save the bottleneck train features to disk
    with open("Pickle/general_trained.pkl", "wb") as encoded_pickle:
        pickle.dump(train_features, encoded_pickle)

    return train_features




