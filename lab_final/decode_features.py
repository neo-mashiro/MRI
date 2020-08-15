import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bdpy.ml import add_bias
from bdpy.stats import corrcoef
from bdpy.preproc import select_top
from scipy import stats
from slir import SparseLinearRegression


def corr2_coeff(x, y):
    """A magic function for computing correlation between matrices and arrays.
       This code is 640x+ faster on large dataset compared to np.corrcoef().
       ------------------------------------------------------------------
       author:  Divakar (https://stackoverflow.com/users/3293881/divakar)
       url:     https://stackoverflow.com/questions/42677677
       ------------------------------------------------------------------
    """
    # input arrays subtract row-wise mean
    x_sub_mx = x - x.mean(1)[:, None]
    y_sub_my = y - y.mean(1)[:, None]

    # sum of squares across rows
    ssx = (x_sub_mx ** 2).sum(1)
    ssy = (y_sub_my ** 2).sum(1)

    return np.dot(x_sub_mx, y_sub_my.T) / np.sqrt(np.dot(ssx[:, None], ssy[None]))


class Decoder:
    """Decode features from fmri bold signals, unit by unit, each using a different SLR model.
       for each feature unit (column), select a subset of voxels that contribute most to the unit,
       this 2D subset is then compared to the 1D unit vector to train the model in that unit.
       -------------------------------------------
       fmri_subset:  shape = (n_samples, n_voxels)
       feature_unit: shape = (n_samples, 1)
       -------------------------------------------
    """
    def __init__(self, n_voxel=1500, n_iter=200):
        self.x_train = None  # array_like, shape = (n_sample, n_voxel)
        self.y_train = None  # array_like, shape = (n_sample, n_unit)
        self.n_voxel = n_voxel  # number of voxels to keep (control the sparsity level)
        self.n_iter = n_iter  # number of iterations

        # recall that units are randomly taken from the last convolutional layer
        self.n_unit = 0  # number of units in a feature vector

        self.models = []          # list of sparse linear regression models for each feature unit
        self.subset_indices = []  # list of selected voxel indices for each feature unit
        self.normal_terms = []    # list of (mu, std) tuples for each feature unit (to de-normalize y)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.n_unit = y_train.shape[1]

        # normalize mri data x (along each voxel)
        mu = np.mean(self.x_train, axis=0)
        std = np.std(self.x_train, axis=0, ddof=1)
        std[std == 0] = 1
        self.x_train = (self.x_train - mu) / std

        # for each feature unit train a sparse linear regression
        for i in range(self.n_unit):
            feature_unit = self.y_train[:, i]
            print(f'start training on unit {i}')

            # normalize image features y (along each unit)
            # this must be done separately because we need to de-normalize later
            mu = np.mean(feature_unit, axis=0)
            std = np.std(feature_unit, axis=0, ddof=1)
            std = 1 if std == 0 else std
            feature_unit = (feature_unit - mu) / std
            self.normal_terms.append((mu, std))

            # voxel selection, select the top `n_voxel` voxels with highest correlations
            # corr = corrcoef(feature_unit, x_train, var='col')
            corr = corr2_coeff(feature_unit.reshape((-1, 1)).T, x_train.T).ravel()
            corr[np.isnan(corr)] = 0  # mark np.nan values as 0 contribution
            x_train_subset, subset_index = select_top(x_train, np.abs(corr), self.n_voxel, axis=1, verbose=False)

            # add bias terms
            x_train_subset = add_bias(x_train_subset, axis=1)

            # set up the model
            model = SparseLinearRegression(n_iter=self.n_iter, prune_mode=1)

            # train model parameters
            try:
                model.fit(x_train_subset, feature_unit)
                self.models.append(model)
                self.subset_indices.append(subset_index)
            except:
                self.models.append(0)
                self.subset_indices.append(0)

    def predict(self, x_test, y_test):
        y_predict = np.zeros_like(y_test)  # shape = (n_sample, n_unit)
        corrs = []  # list of pearson correlations for each unit

        for i in range(self.n_unit):
            true_features = y_test[:, i]
            model = self.models[i]
            unit_status = 'valid' if model != 0 else 'invalid*****'
            print(f'start predicting on unit {i} ({unit_status})')

            # feature prediction
            if unit_status != 'valid':  # SLR failed in this unit
                prediction = np.zeros(true_features.shape)
            else:
                x_test_subset = x_test[:, self.subset_indices[i]]
                x_test_subset = add_bias(x_test_subset, axis=1)
                prediction = model.predict(x_test_subset)
                mu, std = self.normal_terms[i]
                prediction = prediction * std + mu  # de-normalize

            corr, p_value = stats.pearsonr(prediction, true_features)
            corr = 0 if np.isnan(corr) else corr
            corrs.append(corr)

            y_predict[:, i] = prediction

        return y_predict, corrs


def run():
    # configuration
    work_dir = 'lab_final'
    # work_dir = '/content/drive/My Drive'  # if run in Google Colab

    out = os.path.join(work_dir, 'features')
    if not os.path.exists(out):
        os.makedirs(out)

    mri = os.path.join(work_dir, 'data/train.bold.pkl')
    features = os.path.join(work_dir, 'features/train.pkl')

    # load dump files
    x = pd.read_pickle(mri).sort_index(axis=0)
    y = pd.read_pickle(features).sort_index(axis=0)

    def validate(x, y, label='train'):
        """Make sure the two pandas dataframes exactly match in size and order"""
        n_samples = len(x)
        indices = x.index.to_numpy()
        suffix = {'train': '.JPEG', 'test': '.JPEG', 'artificial': '.tiff', 'letter': '.tif'}
        indices += suffix[label]

        assert len(x) == len(y), 'number of samples does not match...'
        assert np.sum(y.index.to_numpy() == indices) == n_samples, 'order of indices does not match...'
        return n_samples, indices

    n_samples, indices = validate(x, y)

    # this lambda function de-serializes a pandas dataframe in the specified column
    # previously in the dumped dataframe, each feature vector has been wrapped as a single object,
    # so that the dataframe only has 1 column 'feature', instead of thousands of columns.
    # but now we need to unwrap the object and expand the features to obtain an ndarray.
    de_serialize = lambda df, col: np.vstack([row for row in df[col].to_numpy()])

    x_train = de_serialize(x, 'mri')    # array of bold signal vectors
    y_train = de_serialize(y, 'conv5')  # array of feature vectors

    # train model
    decoder = Decoder(n_voxel=200, n_iter=10)
    decoder.fit(x_train, y_train)

    # main
    test_combo = ['test', 'artificial', 'letter']

    for label in test_combo:
        mri = os.path.join(work_dir, f'data/{label}.bold.pkl')
        features = os.path.join(work_dir, f'features/{label}.pkl')
        outfile = os.path.join(work_dir, f'features/{label}.decode.pkl')

        dfx = pd.read_pickle(mri).sort_index(axis=0)
        dfy = pd.read_pickle(features).sort_index(axis=0)

        n_samples, indices = validate(dfx, dfy, label=label)
        print(f'number of {label} samples: {n_samples}')

        x = de_serialize(dfx, 'mri')
        y = de_serialize(dfy, 'conv5')

        # decode features
        y_predict, corrs = decoder.predict(x, y)

        # evaluate accuracy
        f, ax = plt.subplots()
        ax.plot(corrs)
        ax.set(xlabel='feature unit', ylabel='Pearson correlation',
               title=f'overall accuracy on {label}: {np.mean(corrs):.4f}')
        ax.grid()
        plt.savefig(os.path.join(work_dir, f'result/{label}_feat_corr.png'))
        plt.show()

        # serialize and dump to disk
        if os.path.exists(outfile):
            os.remove(outfile)

        y_predict = [[row] for row in y_predict]
        df = pd.DataFrame({'conv5': y_predict}, index=indices)
        df.to_pickle(outfile)


def test():
    pass

