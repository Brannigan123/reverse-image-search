import pickle

import numpy as np
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

import dataset
import paths

model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    f = model.predict(x)
    return f.flatten()


def extract_dataset_features():
    file_list = dataset.get_file_list(paths.images_folder_path)
    feature_list = []

    for filename in tqdm(file_list):
        feature_list.append(extract_features(filename))

    return file_list, feature_list


def update_features():
    file_list, features = extract_dataset_features()

    pickle.dump(file_list, open(paths.filenames_path, 'wb'))
    pickle.dump(features, open(paths.features_path, 'wb'))

    return file_list, features


if __name__ == "__main__":
    update_features()
