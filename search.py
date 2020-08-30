import sys
from random import shuffle
import matplotlib.pyplot as plt
from scipy import spatial
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

import dataset
import features
import paths


def visualize_similar_images(img_paths, max_query_imgs=7, max_matches=5):
    img_paths = img_paths[:min(max_query_imgs, len(img_paths))]
    fig, axs = plt.subplots(len(img_paths), max_matches + 1, figsize=(10, 10))

    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        similar = similar_images_paths(img_path, max_imgs=max_matches)

        __plot_similarities__(axs[i], img_path, similar)

    plt.tight_layout(h_pad=2)
    plt.show()


def __plot_similarities__(ax, img_path, similar):
    ax[0].set_title('Query image', size=7)
    ax[0].imshow(img_to_array(load_img(img_path)) / 255)
    ax[0].axis('off')
    ax[0].autoscale()

    cnt = 1
    for path, similarity in similar:
        ax[cnt].imshow(img_to_array(load_img(path)) / 255)
        ax[cnt].set_title('Related image\n similarity %f' % (similarity,), size=7)
        ax[cnt].axis('off')
        ax[cnt].autoscale()
        cnt += 1


def similar_images_paths(img_path, max_imgs=4):
    query_features = features.extract_features(img_path)
    stored_features = dataset.get_stored_features()

    max_imgs = min(max_imgs, len(stored_features[0]))
    similarities = []

    for filename, encoding in list(zip(*stored_features)):
        h_distance = spatial.distance.hamming(query_features, encoding)
        c_distance = spatial.distance.cosine(query_features, encoding)
        similarity = 1 - (h_distance + c_distance) / 2
        similarities.append((filename, similarity))

    similarities.sort(key=lambda tup: -tup[1])
    return similarities[:max_imgs]


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        visualize_similar_images(args[1:])
    else:
        paths = dataset.get_file_list(paths.query_images_folder_path)
        shuffle(paths)
        visualize_similar_images(paths)
