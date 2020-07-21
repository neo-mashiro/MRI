import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma


class ExecutionTimer:
    def __init__(self, f):
        self._f = f

    def __call__(self, *args, **kwargs):
        self._start = time.time()
        self._result = self._f(*args, **kwargs)
        self._end = time.time()
        print("execution time in seconds: {}".format(self._end - self._start))
        return self._result


def bilateral_filter(im):
    """Denoise the input image using bilateral filter"""
    raise NotImplementedError("This algorithm will be implemented later!")


def autoencoder(im):
    """Denoise the input image using denoising autoencoder (based on Keras)"""
    raise NotImplementedError("This algorithm will be implemented later!")


def patch_distance(p1, p2, kernel):
    """Compute the sum of kernel-weighted Euclidean distances between two patches"""
    assert p1.shape == p2.shape, "Patch sizes are not equal ..."
    assert p1.shape[0] == p1.shape[1], "Input patch must be a square matrix ..."
    assert p1.shape[0] % 2 != 0, "Patch size must be odd ..."
    diff = (p1 - p2) ** 2
    return np.sum(np.multiply(kernel, diff))


def gaussian_kernel(x, y, sigma):
    """Compute the filter matrix of meshgrid x and y given a 2D Gaussian kernel"""
    var = sigma ** 2
    return np.exp(-((x ** 2 + y ** 2) / (2 * var))) / (2 * np.pi * var)


def oracle_kernel(x, y):
    """Compute the filter matrix of meshgrid x and y given the Oracle kernel"""
    # the Oracle filter may produce better performance for non-local means
    # reference: https://hal.inria.fr/hal-01575918/document
    #            see formula (26), (27) on page 7 for the kernel definition

    offset = x.shape[0] // 2
    dist = np.maximum(np.abs(x), np.abs(y))  # orthogonal distance from the patch center
    dist[offset, offset] = 1  # patch center has the same weight as pixels with distance = 1
    filt = np.zeros_like(x).astype('float64')
    for d in range(offset, 0, -1):
        mask = (dist <= d)
        filt[mask] += (1 / (2 * d + 1) ** 2)

    return filt * (1 / offset)


@ExecutionTimer
def nl_means_filter(im, patch_size=3, window_size=5, h=0.6, sigma=1.0, kernel='Gaussian'):
    """Denoise the input image using non-local means filter
       h:      parameter that controls the degree of filtering
       sigma:  standard deviation of the Gaussian noise in the image
    """
    # use a symmetric patch and search window whose size is odd
    if patch_size % 2 == 0:
        patch_size += 1
    if window_size % 2 == 0:
        window_size += 1

    n_row, n_col = im.shape
    offset = patch_size // 2  # offset from the patch center

    new_im = np.zeros_like(im)  # initialize the denoised image
    pad_im = np.pad(im, ((offset, offset), (offset, offset)), mode='reflect')  # pad the image

    # compute the patch filter (the weighted matrix defined by the kernel)
    patch_range = np.arange(-offset, offset + 1)
    x, y = np.meshgrid(patch_range, patch_range, indexing='ij')

    if kernel == 'Oracle':
        filt = oracle_kernel(x, y)
    else:
        filt = gaussian_kernel(x, y, sigma)

    # iterate over each pixel in the original image
    for row in range(n_row):
        u = row - min(window_size, row)          # search window top row (up)
        d = row + min(window_size, n_row - row)  # search window bottom row + 1 (down)

        for col in range(n_col):
            l = col - min(window_size, col)          # search window leftmost column
            r = col + min(window_size, n_col - col)  # search window rightmost + 1 column

            p1 = pad_im[row:row+patch_size, col:col+patch_size]

            Z = 0.0
            pixel_value = 0.0

            # iterate over every other pixel in the search window
            for i in range(u, d):
                for j in range(l, r):
                    p2 = pad_im[i:i+patch_size, j:j+patch_size]

                    # compute distance and weight
                    distance = patch_distance(p1, p2, filt)
                    power = -distance / (h ** 2)
                    if power < -5:  # exp cutoff
                        weight = 0  # exp of a large negative number is close to 0
                    else:
                        weight = np.exp(power)

                    Z += weight  # the normalization term (sum of weight)
                    pixel_value += weight * im[i, j]

            # normalize the result
            if Z == 0:  # this happens only when p1 = every p2 in the search window (probably in the background)
                new_im[row, col] = im[row, col]
            else:
                new_im[row, col] = pixel_value / Z

    return new_im


def test():
    from skimage import io
    from skimage.util import random_noise
    from skimage.restoration import denoise_nl_means, estimate_sigma

    lena = io.imread('lab4/images/lena512.bmp', as_gray=True)
    noisy = random_noise(lena, var=0.004)

    # denoise using scikit-image library function
    sigma = estimate_sigma(noisy, multichannel=False)
    h = 0.6 * sigma
    kwargs = dict(patch_size=7,      # 3x3 patches
                  patch_distance=10,  # 5x5 search window
                  multichannel=False)
    denoised = denoise_nl_means(noisy, h=h, sigma=sigma, fast_mode=True, **kwargs)
    noise = noisy - denoised

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.8))
    [ax.set_axis_off() for ax in (ax1, ax2, ax3, ax4)]
    ax1.imshow(lena, cmap='gray')
    ax1.set_title('original')
    ax2.imshow(noisy, cmap='gray')
    ax2.set_title('noisy')
    ax3.imshow(denoised, cmap='gray')
    ax3.set_title('denoised')
    ax4.imshow(noise, cmap='gray')
    ax4.set_title('noise')
    f.subplots_adjust(top=0.5, bottom=0.0)
    f.suptitle(f"scikit-image library function", fontsize=16)
    plt.show()

    # denoise using our own algorithm
    h = 1.5 * sigma
    denoised = nl_means_filter(noisy, h=h, sigma=sigma, patch_size=3, window_size=5, kernel='Oracle')
    noise = noisy - denoised

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.8))
    [ax.set_axis_off() for ax in (ax1, ax2, ax3, ax4)]
    ax1.imshow(lena, cmap='gray')
    ax1.set_title('original')
    ax2.imshow(noisy, cmap='gray')
    ax2.set_title('noisy')
    ax3.imshow(denoised, cmap='gray')
    ax3.set_title('denoised')
    ax4.imshow(noise, cmap='gray')
    ax4.set_title('noise')
    f.subplots_adjust(top=0.5, bottom=0.0)
    f.suptitle(f"our algorithm", fontsize=16)
    plt.show()


def run(i_vec):
    titles = {0: 't1', 1: 't1_v2', 2: 't1_v3', 3: 't2', 4: 'flair'}

    # estimate snr for each brain image
    for i, brain in enumerate(i_vec):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.5))

        # do not use ax.set_axis_off() since we need to color the axis splines
        for ax in (ax1, ax2, ax3):
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

        im1 = ax1.imshow(brain, cmap='gray')
        ax1.set_title(titles[i])
        f.colorbar(im1, ax=ax1)

        overlay = np.zeros((brain.shape[0], brain.shape[1], 4))  # RGBA overlay
        overlay[-36:, 0:36, 0] = 1         # red channel (noise patch)
        overlay[-36:, 0:36, 3] = 0.8       # alpha channel
        overlay[150:190, 90:130, 1] = 1    # green channel (signal patch)
        overlay[150:190, 90:130, 3] = 0.4  # alpha channel

        ax1.imshow(overlay)

        noise_patch = brain[-36:, 0:36]
        im2 = ax2.imshow(noise_patch, cmap='gray')
        ax2.set_title(f'noise patch, std = {np.std(noise_patch):.4f}')
        f.colorbar(im2, ax=ax2, format='%.2f', shrink=0.7, ticks=np.arange(0, 5.5, 0.5))
        for pos in ('top', 'bottom', 'left', 'right'):
            ax2.spines[pos].set_color('red')
            ax2.spines[pos].set_linewidth(3)

        signal_patch = brain[150:190, 90:130]
        im3 = ax3.imshow(signal_patch, cmap='gray')
        ax3.set_title(f'signal patch, mean = {np.mean(signal_patch):.4f}')
        f.colorbar(im3, ax=ax3, shrink=0.7)
        for pos in ('top', 'bottom', 'left', 'right'):
            ax3.spines[pos].set_color('#50C878')
            ax3.spines[pos].set_linewidth(3)

        snr = np.mean(signal_patch) / np.std(noise_patch)
        f.suptitle(f'SNR = mean(signal) / std(noise) = {snr:.2f}', fontsize=16, y=0.95)
        f.tight_layout()
        plt.show()

    # evaluate the denoised effect
    def tune_param(im_index, figsize, h, kernel='Oracle'):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        [ax.set_axis_off() for ax in (ax1, ax2, ax3)]

        image = i_vec[im_index]
        sigma = estimate_sigma(image, multichannel=False)
        h *= sigma
        denoised = nl_means_filter(image, h=h, sigma=sigma, kernel=kernel)
        noise = image - denoised
        noise = np.subtract(noise, 128)

        ax1.imshow(image, cmap='gray')
        ax1.set_title('noisy brain')
        ax2.imshow(np.abs(noise), cmap='gray')
        ax2.set_title('noise')
        ax3.imshow(denoised, cmap='gray')
        ax3.set_title('denoised = noisy - noise')

        f.suptitle(f"Non-local means on {titles[im_index]} ($\sigma$ = {sigma:.4f})", fontsize=18)
        f.subplots_adjust(top=0.6, bottom=0)
        plt.show()

    tuned_h = [0.8125, 1.5625, 1.875, 1.625, 7.5]
    f_size = [(14, 7.5), (14, 8.5), (14, 6), (14, 7.5), (14, 7.5)]

    for i in range(len(i_vec)):
        tune_param(i, f_size[i], tuned_h[i])
