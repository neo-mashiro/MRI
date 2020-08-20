import os
import numpy as np
import pandas as pd

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


class Extractor:
    """Image features extractor on a specified layer"""
    def __init__(self, layer='block5_pool'):
        self._base = VGG19(include_top=False, weights='imagenet')
        self._model = Model(inputs=self._base.input,
                            outputs=self._base.get_layer(layer).output)
        self._seed = np.random.seed(2501)  # important! must extract the same positions for each image
        self._indices = np.random.permutation(7*7*512)  # shuffle indices

    def __str__(self):
        return self._model.summary()

    def extract(self, path, n_features=1000):
        """Return features as an ndarray of shape (n_features,)"""
        assert n_features < 25088, 'features amount out of bound'  # total = 7x7x512 = 25088
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self._model.predict(x)
        features = features.reshape(-1, 7*7*512)
        features = features[0][self._indices]
        return features[:n_features]


def extract_features(model, folder, n_features=1000, outfile=None):
    if os.path.exists(outfile):
        os.remove(outfile)

    for root, dirs, files in os.walk(folder):
        features = []
        for f in files:
            feat = model.extract(os.path.join(root, f), n_features=n_features)
            features.append(feat)

        df = pd.DataFrame({'conv5': features}, index=files)
        df.to_pickle(outfile)


def run():
    n_features = 1000
    work_dir = 'lab_final'

    dir1 = os.path.join(work_dir, 'images/train')
    dir2 = os.path.join(work_dir, 'images/test')
    dir3 = os.path.join(work_dir, 'images/artificial')
    dir4 = os.path.join(work_dir, 'images/letter')
    dirs = [dir1, dir2, dir3, dir4]

    out = os.path.join(work_dir, 'features')
    if not os.path.exists(out):
        os.makedirs(out)

    opener = Extractor('block5_pool')

    for folder in dirs:
        filename = os.path.split(folder)[1] + '.pkl'
        outfile = os.path.join(out, filename)
        extract_features(opener, folder, n_features, outfile)


def test():
    out = os.path.join('lab_final', 'features')
    for root, dirs, files in os.walk(out):
        for f in files:
            data = pd.read_pickle(os.path.join(root, f))
            print(data)
            print(data.shape)
