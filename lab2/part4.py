import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from .part2 import ssd


class ExecutionTimer:
    def __init__(self, f):
        self._f = f

    def __call__(self, *args, **kwargs):
        self._start = time.time()
        self._result = self._f(*args, **kwargs)
        self._end = time.time()
        print("execution time in seconds: {}".format(self._end - self._start))
        return self._result


def translation(i, p, q):
    """Return a new image corresponding to image i translated by vector (p, q)"""
    # construct a 2D grid
    len_x, len_y = i.shape
    x, y = np.mgrid[0:len_x, 0:len_y]

    # translation matrix
    t = np.array([[1, 0, p],
                  [0, 1, q],
                  [0, 0, 1]])

    # axis values in homogeneous coordinates
    axes = np.vstack([np.ravel(x), np.ravel(y), np.ones(len_x * len_y)])

    # data points in cartesian coordinates
    data_points = axes[:-1].T

    # new axis values in cartesian coordinates
    new_axes = np.apply_along_axis(lambda col: t @ col.T, 0, axes)[:-1]

    # new grid
    new_x, new_y = new_axes[0].reshape(len_x, len_y), new_axes[1].reshape(len_x, len_y)

    # interpolate using griddata for better performance
    new_i = interpolate.griddata(data_points, np.ravel(i), (new_x, new_y), method='cubic', fill_value=0)

    return new_i


def rotation(i, theta):
    """Return a new image corresponding to image i rotated by theta around origin (top left)"""
    # construct a 2D grid
    len_x, len_y = i.shape
    x, y = np.mgrid[0:len_x, 0:len_y]

    # rotation matrix
    angle = np.deg2rad(theta)
    s, c = np.sin(angle), np.cos(angle)
    r = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])

    # axis values in homogeneous coordinates
    axes = np.vstack([np.ravel(x), np.ravel(y), np.ones(len_x * len_y)])

    # data points in cartesian coordinates
    data_points = axes[:-1].T

    # new axis values in cartesian coordinates (this column-wise step is slow)
    new_axes = np.apply_along_axis(lambda col: r @ col.T, 0, axes)[:-1]

    # new grid
    new_x, new_y = new_axes[0].reshape(len_x, len_y), new_axes[1].reshape(len_x, len_y)

    # interpolate using griddata (this step is slow, but much faster than interp2d)
    new_i = interpolate.griddata(data_points, np.ravel(i), (new_x, new_y), method='cubic', fill_value=0)

    return new_i


def rigid_transform(i, p, q, theta):
    """Return a new image corresponding to image i both translated and rotated"""
    len_x, len_y = i.shape
    x, y = np.mgrid[0:len_x, 0:len_y]

    t = np.array([[1, 0, p],
                  [0, 1, q],
                  [0, 0, 1]])

    angle = np.deg2rad(theta)
    s, c = np.sin(angle), np.cos(angle)
    r = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])

    T = t @ r

    axes = np.vstack([np.ravel(x), np.ravel(y), np.ones(len_x * len_y)])
    data_points = axes[:-1].T
    new_axes = np.apply_along_axis(lambda col: T @ col.T, 0, axes)[:-1]
    new_x, new_y = new_axes[0].reshape(len_x, len_y), new_axes[1].reshape(len_x, len_y)
    new_i = interpolate.griddata(data_points, np.ravel(i), (new_x, new_y), method='cubic', fill_value=0)

    return new_i


def test(im):
    # test rigid transform functions
    im1 = rigid_transform(im, 20, -20, 0)
    im2 = rigid_transform(im, 0, 0, 30)
    im3 = rigid_transform(im, 20, -20, 30)

    f, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax1.imshow(im, cmap='gray')
    ax1.set_title("original image")
    ax2.imshow(im1, cmap='gray')
    ax2.set_title("translated")
    ax3.imshow(im2, cmap='gray')
    ax3.set_title("rotated")
    ax4.imshow(im3, cmap='gray')
    ax4.set_title("translated and rotated")
    plt.show()


@ExecutionTimer
def regis_trans(i1, i2, stepsize, n_iter=100):
    """Register 2D image i2 onto i1 by minimizing SSD (consider only translations)"""
    ssd_list = []  # save the SSD for each iteration
    p = q = 0

    for _ in range(n_iter):
        # move i2 and save SSD
        i2_moved = translation(i2, p, q)
        i_diff = i2_moved - i1
        ssd_list.append(ssd(i1, i2_moved))

        # compute gradient
        x_grad, y_grad = np.gradient(i2_moved)
        ssd_p = 2 * np.sum(np.multiply(i_diff, x_grad))
        ssd_q = 2 * np.sum(np.multiply(i_diff, y_grad))

        # update parameters
        p -= stepsize * ssd_p
        q -= stepsize * ssd_q

    return translation(i2, p, q), p, q, ssd_list


def regis_rotat(i1, i2, stepsize, n_iter=100):
    """Register 2D image i2 onto i1 by minimizing SSD (consider only rotations)"""
    ssd_list = []  # save the SSD for each iteration
    theta = 0

    for _ in range(n_iter):
        # move i2 and save SSD
        i2_moved = rotation(i2, theta)
        i_diff = i2_moved - i1
        ssd_list.append(ssd(i1, i2_moved))

        # compute gradient
        x_grad, y_grad = np.gradient(i2_moved)
        x, y = np.mgrid[0:i2.shape[0], 0:i2.shape[1]]
        angle = np.deg2rad(theta)
        s, c = np.sin(angle), np.cos(angle)
        ssd_theta = 2 * np.sum(np.multiply(i_diff,
                                           np.multiply(x_grad, -(x * s + y * c)) +
                                           np.multiply(y_grad, x * c - y * s)))
        print("ssd_theta: ", ssd_theta)
        print("theta: ", theta)

        # update parameters
        theta -= stepsize * ssd_theta * 0.01

    return rotation(i2, theta), theta, ssd_list


def run(images):
    # test registration for translation
    # f = plt.figure(figsize=(16, 12), dpi=120)
    # spec = f.add_gridspec(nrows=30, ncols=20)
    # plt.suptitle("2D registration minimizing SSD (translation)", fontsize=20)
    #
    # ax00 = f.add_subplot(spec[1:16, :6])
    # ax01 = f.add_subplot(spec[1:16, 7:13])
    # ax02 = f.add_subplot(spec[1:16, 14:])
    # ax10 = f.add_subplot(spec[16:, 5:-5])
    # axes = [ax00, ax01, ax02, ax10]
    #
    # ssd_vec = []
    # for i in range(1, len(images)):
    #     registered_image, p, q, ssd_i = regis_trans(images[0], images[i], stepsize=0.000005, n_iter=200)
    #     ssd_vec.append(ssd_i)
    #     axes[i-1].imshow(registered_image, cmap='gray')
    #     axes[i-1].set_title(f"BrainMRI_{i+1}")
    #
    # for i, ssd_i in enumerate(ssd_vec):
    #     axes[-1].plot(ssd_i, label=f"BrainMRI_{i+1}")
    #
    # axes[-1].set_xlabel('number of iterations')
    # axes[-1].set_ylabel('sum squared difference')
    # axes[-1].set_title('SSD curve (translation)')
    # axes[-1].legend()
    #
    # plt.show()

    """Describe the SSD curve, is it strictly decreasing, and if not, why?
       discuss the quality of your registration
       stepsize or local optimum?
    """

    # test registration for rotation
    f = plt.figure(figsize=(16, 12), dpi=120)
    spec = f.add_gridspec(nrows=30, ncols=20)
    plt.suptitle("2D registration minimizing SSD (rotation)", fontsize=20)

    ax00 = f.add_subplot(spec[1:16, :6])
    ax01 = f.add_subplot(spec[1:16, 7:13])
    ax02 = f.add_subplot(spec[1:16, 14:])
    ax10 = f.add_subplot(spec[16:, 5:-5])
    axes = [ax00, ax01, ax02, ax10]

    ssd_vec = []
    for i in range(1, len(images)):
        registered_image, theta, ssd_i = regis_rotat(images[0], images[i], stepsize=0.0000005, n_iter=20)
        ssd_vec.append(ssd_i)
        axes[i-1].imshow(registered_image, cmap='gray')
        axes[i-1].set_title(f"BrainMRI_{i+1}")

    for i, ssd_i in enumerate(ssd_vec):
        axes[-1].plot(ssd_i, label=f"BrainMRI_{i+1}")

    axes[-1].set_xlabel('number of iterations')
    axes[-1].set_ylabel('sum squared difference')
    axes[-1].set_title('SSD curve (rotation)')
    axes[-1].legend()

    plt.show()

    """Describe the SSD curve, is it strictly decreasing, and if not, why?
       discuss the quality of your registration
       stepsize or local optimum
    """
