import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

import paths

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def get_file_list(root_dir):
    file_list = []
    counter = 1

    for root, dirs, filenames in os.walk(root_dir):
        for filename in tqdm(filenames):
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return sorted(file_list)


def get_stored_features():
    stored_filenames = pickle.load(open(paths.filenames_path, 'rb'))
    stored_feature_list = pickle.load(open(paths.features_path, 'rb'))
    return stored_filenames, stored_feature_list


def visualize_features():
    filenames, features = get_stored_features()

    num_feature_dimensions = 100  # Set the number of features
    pca = PCA(n_components=num_feature_dimensions)
    pca.fit(features)
    feature_list_compressed = pca.transform(features)

    tsne = TSNE(n_components=2, verbose=1, n_iter=4000, metric='cosine', init='pca')
    tsne_results = tsne.fit_transform(feature_list_compressed)
    tsne_results = StandardScaler().fit_transform(tsne_results)

    size = (45, 45)
    imgs = [img_to_array(load_img(path, target_size=size)) / 255 for path in filenames]
    visualize_scatter_with_images(tsne_results, imgs=imgs, size=size, zoom=0.7)


def visualize_scatter_with_images(data, imgs, size=(28, 28), zoom=1):
    fig, ax = plt.subplots(figsize=size)
    artist = []
    for xy, i in tqdm(zip(data, imgs)):
        x, y = xy
        img = OffsetImage(i, zoom=zoom)
        ab = AnnotationBbox(img, (x, y), xycoords='data', frameon=False)
        artist.append(ax.add_artist(ab))
    ax.update_datalim(data)
    ax.autoscale()
    ax.axis('off')
    plt.tight_layout(pad=1.2)
    plt.show()


if __name__ == "__main__":
    visualize_features()
