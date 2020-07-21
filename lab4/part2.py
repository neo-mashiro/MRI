import numpy as np
import matplotlib.pyplot as plt
from .part1 import ExecutionTimer


CONVERGE_THRESHOLD = 0.1
CLUSTER_THRESHOLD = 10


@ExecutionTimer
def mean_shift(im, bandwidth):
    """Mean-shift clustering algorithm implemented using expectation–maximization approach"""
    # reference slide: http://www.cs.cmu.edu/~aarti/SMLRG/miguel_slides.pdf
    # normal approach: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TUZEL1/MeanShift.pdf
    n_row, n_col = im.shape
    pixels = np.ravel(im).astype('float64')
    clusters = np.copy(pixels)
    var = bandwidth ** 2

    # vectorize to speed up exponent computation
    fast_exp = lambda x: np.exp(x) if x > -5 else 0.0
    vec_exp = np.vectorize(fast_exp)

    # obtain (flattened) indices of neighbor pixels given a flattened index
    def neighbor_indices(index, offset):
        row, col = index // n_col, index % n_col
        u = max(row - offset, 0)          # neighbor window top row
        d = min(row + offset, n_row - 1)  # neighbor window bottom row
        l = max(col - offset, 0)          # neighbor window leftmost column
        r = min(col + offset, n_col - 1)  # neighbor window rightmost column
        x, y = np.meshgrid(np.arange(u, d + 1), np.arange(l, r + 1), indexing='ij')
        return np.ravel(x * n_col + y)

    # compute shift vector for each pixel
    for i in range(len(pixels)):
        pixel = clusters[i].copy()  # initial value
        neighbor = neighbor_indices(i, 10)  # speed up by choosing a window, window size tunable

        # shift pixel until a cluster peak is hit
        while True:
            neighbor_pixel = pixels[neighbor]
            diff = (pixel - neighbor_pixel) ** 2
            power = -diff / (2 * var)
            prob = vec_exp(power)
            prob *= (1 / np.sum(prob))  # normalize

            center = np.sum(np.multiply(neighbor_pixel, prob))  # compute window center
            update = np.abs(center - pixel)
            pixel = center  # shift pixel
            # neighbor = ...  # should shift window as well, but ignored for now

            # spatial discretization speedup not implemented, see slide 26
            # if n is not large enough, speedup will introduce segmentation errors
            if update < CONVERGE_THRESHOLD:
                clusters[i] = pixel  # peak of the i-th pixel
                break

    clusters = (clusters / CLUSTER_THRESHOLD).astype(int) * CLUSTER_THRESHOLD  # merge similar clusters
    clusters = np.minimum(clusters + 30, 254) % 255  # for better visualization
    return clusters.reshape((n_row, n_col))


def test():
    from skimage import io
    from sklearn.cluster import MeanShift

    house = io.imread('lab4/images/house.jpg', as_gray=True)
    n_row, n_col = house.shape
    data = house.reshape(-1, 1)  # transform to feature space (1D for grayscale)

    # segmentation using scikit-image library function
    ms = MeanShift(bandwidth=20, bin_seeding=True)
    clusters = ms.fit_predict(data).reshape(-1, 1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    [ax.set_axis_off() for ax in (ax1, ax2)]
    ax1.imshow(house, cmap='gray')
    ax1.set_title('original')
    ax2.imshow(clusters.reshape((n_row, n_col)), cmap='gray')
    ax2.set_title('segmentation')
    f.suptitle(f"scikit-image library function", fontsize=16)
    plt.show()

    # segmentation using our algorithm
    bandwidth = 20
    clusters = mean_shift(house, bandwidth=bandwidth)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    [ax.set_axis_off() for ax in (ax1, ax2)]
    ax1.imshow(house, cmap='gray')
    ax1.set_title('original')
    ax2.imshow(clusters.reshape((n_row, n_col)), cmap='gray')
    ax2.set_title('segmentation')
    f.suptitle(f"our algorithm (bandwidth = {bandwidth})", fontsize=16)
    plt.show()


def run():
    from skimage.io import imread
    # read denoised images
    i1 = imread('lab4/images/t1.png', as_gray=True)
    i2 = imread('lab4/images/t1_v2.png', as_gray=True)
    i3 = imread('lab4/images/t1_v3.png', as_gray=True)
    i4 = imread('lab4/images/t2.png', as_gray=True)
    i5 = imread('lab4/images/flair.png', as_gray=True)

    i_vec = [i1, i2, i3, i4, i5]
    titles = {0: 't1', 1: 't1_v2', 2: 't1_v3', 3: 't2', 4: 'flair'}

    def tune_param(im_index, bandwidth, figsize):
        image = i_vec[im_index]
        clusters = mean_shift(image, bandwidth=bandwidth)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        [ax.set_axis_off() for ax in (ax1, ax2)]
        ax1.imshow(image, cmap='gray')
        ax1.set_title('original')
        ax2.imshow(clusters, cmap='gray')
        ax2.set_title('segmentation')
        f.subplots_adjust(top=0.6, bottom=0)
        f.suptitle(f"Mean-shift clustering on {titles[im_index]} (bandwidth = {bandwidth})", fontsize=16)
        plt.show()

    tuned_radius = [8, 5, 6.5, 3.5, 5]
    f_size = [(10, 8), (10, 9), (10, 6), (10, 8), (10, 8)]

    for i in range(len(i_vec)):
        tune_param(i, tuned_radius[i], f_size[i])
