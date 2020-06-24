import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
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


def test_transform(im):
    # test rigid transform functions
    im1 = rigid_transform(im, 20, -20, 0)   # translated
    im2 = rigid_transform(im, 0, 0, 30)     # rotated
    im3 = rigid_transform(im, 20, -20, 30)  # translated and rotated

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
    pq_list = []  # let's also save (p, q) for each step, for diagnosis

    for _ in range(n_iter):
        # move i2 and save SSD, p, q
        i2_moved = translation(i2, p, q)
        i_diff = i2_moved - i1
        pq_list.append((p, q))
        ssd_list.append(ssd(i1, i2_moved))

        # compute gradient
        x_grad, y_grad = np.gradient(i2_moved)
        ssd_p = 2 * np.sum(np.multiply(i_diff, x_grad))
        ssd_q = 2 * np.sum(np.multiply(i_diff, y_grad))

        # update parameters
        p -= stepsize * ssd_p * 3
        q -= stepsize * ssd_q * 3

    return translation(i2, p, q), p, q, pq_list, ssd_list


@ExecutionTimer
def regis_rotat(i1, i2, stepsize, n_iter=100):
    """Register 2D image i2 onto i1 by minimizing SSD (consider only rotations)"""
    ssd_list = []  # save the SSD for each iteration
    theta = 0
    theta_list = []  # let's also save theta for each step, for diagnosis

    for _ in range(n_iter):
        # move i2 and save SSD, theta
        i2_moved = rotation(i2, theta)  # counterclockwise
        i_diff = i2_moved - i1
        theta_list.append(theta)
        ssd_list.append(ssd(i1, i2_moved))

        # compute gradient (important: +/- signs before sine and cosine depend on the rotation function!)
        x_grad, y_grad = np.gradient(i2_moved)
        x, y = np.mgrid[0:i2.shape[0], 0:i2.shape[1]]
        angle = np.deg2rad(theta)
        s, c = np.sin(angle), np.cos(angle)
        ssd_theta = 2 * np.sum(np.multiply(i_diff,
                                           np.multiply(x_grad, -x * s + y * c) +  # signs for counterclockwise
                                           np.multiply(y_grad, -x * c - y * s)))  # signs for counterclockwise

        # update parameters
        theta -= stepsize * ssd_theta * 0.01

    return rotation(i2, theta), theta, theta_list, ssd_list


@ExecutionTimer
def regis_rigid(i1, i2, stepsize, n_iter=100):
    """Register 2D image i2 onto i1 by minimizing SSD (translations and rotations)"""
    p = q = theta = 0
    ssd_list = []
    pq_list = []
    theta_list = []

    for _ in range(n_iter):
        # move i2 and save SSD, p, q, theta
        i2_moved = rigid_transform(i2, p, q, theta)
        i_diff = i2_moved - i1
        pq_list.append((p, q))
        theta_list.append(theta)
        ssd_list.append(ssd(i1, i2_moved))

        # compute gradient
        x_grad, y_grad = np.gradient(i2_moved)
        x, y = np.mgrid[0:i2.shape[0], 0:i2.shape[1]]
        angle = np.deg2rad(theta)
        s, c = np.sin(angle), np.cos(angle)
        ssd_p = 2 * np.sum(np.multiply(i_diff, x_grad))
        ssd_q = 2 * np.sum(np.multiply(i_diff, y_grad))
        ssd_theta = 2 * np.sum(np.multiply(i_diff,
                                           np.multiply(x_grad, -x * s + y * c) +
                                           np.multiply(y_grad, -x * c - y * s)))

        # update parameters
        p -= stepsize * ssd_p * 5
        q -= stepsize * ssd_q * 5
        theta -= stepsize * ssd_theta * 0.01

    registered_image = rigid_transform(i2, p, q, theta)
    return registered_image, p, q, theta, pq_list, theta_list, ssd_list


@ExecutionTimer
def gradient_descent(i1, i2, stepsize, n_iter=100):
    """Gradient descent image registration using normalized SSD similarity metric"""
    p = q = theta = 0
    ssd_list = []

    for _ in range(n_iter):
        i2_moved = rigid_transform(i2, p, q, theta)
        i_diff = i2_moved - i1
        ssd_list.append(ssd(i1, i2_moved))

        # compute gradient
        x_grad, y_grad = np.gradient(i2_moved)
        x, y = np.mgrid[0:i2.shape[0], 0:i2.shape[1]]
        angle = np.deg2rad(theta)
        s, c = np.sin(angle), np.cos(angle)
        ssd_p = 2 * np.sum(np.multiply(i_diff, x_grad))
        ssd_q = 2 * np.sum(np.multiply(i_diff, y_grad))
        ssd_theta = 2 * np.sum(np.multiply(i_diff,
                                           np.multiply(x_grad, -x * s + y * c) +
                                           np.multiply(y_grad, -x * c - y * s)))

        # adaptive learning rate
        if _ > 0 and ssd_list[-2] - ssd_list[-1] < 1000:
            alpha = stepsize * 10
        else:
            alpha = stepsize

        # update parameters
        p -= alpha * ssd_p * 5
        q -= alpha * ssd_q * 5
        theta -= alpha * ssd_theta * 0.01

    return p, q, theta, ssd_list


def test_registration(im):
    # translation registration
    i1 = im                        # the ground truth image
    i2 = translation(im, 20, -20)  # translated by 20 along x, y axis

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(i1, cmap='gray')
    ax[1].imshow(i2, cmap='gray')
    plt.show()

    registered_image, p, q, pqs, ssds = regis_trans(i1, i2, stepsize=3e-08, n_iter=500)
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(registered_image, cmap='gray')
    ax[1].plot(ssds)
    ax[1].set_title(f"registered (p,q): ({round(p, 4)},{round(q, 4)}), true (p,q): (20,20)")
    plt.show()

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ps, qs = list(zip(*pqs))
    x, y = np.mgrid[min(ps):max(ps)+1, min(qs):max(qs)+1]
    surface = interpolate.griddata(pqs, np.array(ssds), (x, y), method='nearest', fill_value=0)
    ax.plot_surface(x, y, surface, rstride=1, cstride=1, cmap='hot')
    ax.set_xlabel('p')
    ax.set_ylabel('q')
    ax.set_zlabel('ssd')
    plt.show()

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.scatter(np.array(ps), np.array(qs), np.array(ssds), facecolor='tomato', marker='o', s=10, alpha=0.5)
    plt.show()

    # rotation registration
    i1 = im                # the ground truth image
    i2 = rotation(im, 15)  # rotated by 15 degrees

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(i1, cmap='gray')
    ax[1].imshow(i2, cmap='gray')
    plt.show()

    registered_image, theta, theta_list, ssds = regis_rotat(i1, i2, stepsize=1e-08, n_iter=500)
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(registered_image, cmap='gray')
    ax[1].plot(ssds)
    ax[1].set_title(f"registered theta: {theta} (true theta: 15)")
    plt.show()

    f, ax = plt.subplots()
    ax.scatter(theta_list, ssds, c='r', marker='o', s=10, alpha=0.5)  # plot the cost function
    ax.set_xlabel('theta')
    ax.set_ylabel('SSD')
    plt.show()

    # rigid registration
    i1, i2 = im, rigid_transform(im, 20, -20, 15)

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(i1, cmap='gray')
    ax[1].imshow(i2, cmap='gray')
    plt.show()

    i2_regis, p, q, theta, _, _, ssds = regis_rigid(i1, i2, stepsize=1e-08, n_iter=500)
    p, q, theta = round(p, 4), round(q, 4), round(theta, 4)
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(i2_regis, cmap='gray')
    ax[1].plot(ssds)
    ax[1].set_title(f"registered (p,q,theta): ({p},{q},{theta}), true (p,q,theta): (20,20,15)")
    plt.show()


def test_gradient_descent(im):
    i1, i2 = im, rigid_transform(im, 20, -20, 15)

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(i1, cmap='gray')
    ax[1].imshow(i2, cmap='gray')
    plt.show()

    p, q, theta, ssds = gradient_descent(i1, i2, stepsize=1e-08, n_iter=500)
    p, q, theta = round(p, 4), round(q, 4), round(theta, 4)
    i2_regis = rigid_transform(i2, p, q, theta)
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(i2_regis, cmap='gray')
    ax[1].plot(ssds[1:])
    ax[1].set_title(f"registered (p,q,theta): ({p},{q},{theta}), true (p,q,theta): (20,20,15)")
    plt.show()


def run(images):
    # test registration for translation
    f = plt.figure(figsize=(16, 12), dpi=120)
    spec = f.add_gridspec(nrows=30, ncols=20)
    plt.suptitle("2D registration minimizing SSD (translation)", fontsize=20)

    ax00 = f.add_subplot(spec[1:16, :6])
    ax01 = f.add_subplot(spec[1:16, 7:13])
    ax02 = f.add_subplot(spec[1:16, 14:])
    ax10 = f.add_subplot(spec[16:, 5:-5])
    axes = [ax00, ax01, ax02, ax10]

    ssd_vec = []
    for i in range(1, len(images)):
        registered_image, p, q, _, ssd_i = regis_trans(images[0], images[i], stepsize=3e-08, n_iter=500)
        ssd_vec.append(ssd_i)
        axes[i-1].imshow(registered_image, cmap='gray')
        axes[i-1].set_title(f"BrainMRI_{i+1}, p={round(p,2)}, q={round(q,2)}")

    for i, ssd_i in enumerate(ssd_vec):
        axes[-1].plot(ssd_i, label=f"BrainMRI_{i+2}")

    axes[-1].set_xlabel('number of iterations')
    axes[-1].set_ylabel('sum squared difference')
    axes[-1].set_title('SSD curve (translation)')
    axes[-1].legend()

    plt.show()

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
        registered_image, theta, _, ssd_i = regis_rotat(images[0], images[i], stepsize=1e-08, n_iter=500)
        ssd_vec.append(ssd_i)
        axes[i-1].imshow(registered_image, cmap='gray')
        axes[i-1].set_title(f"BrainMRI_{i+1}, theta={round(theta,2)}")

    for i, ssd_i in enumerate(ssd_vec):
        axes[-1].plot(ssd_i, label=f"BrainMRI_{i+2}")

    axes[-1].set_xlabel('number of iterations')
    axes[-1].set_ylabel('sum squared difference')
    axes[-1].set_title('SSD curve (rotation)')
    axes[-1].legend()

    plt.show()

    # test registration for rigid transform
    f = plt.figure(figsize=(16, 12), dpi=120)
    spec = f.add_gridspec(nrows=30, ncols=20)
    plt.suptitle("2D registration minimizing SSD (rigid)", fontsize=20)

    ax00 = f.add_subplot(spec[1:16, :6])
    ax01 = f.add_subplot(spec[1:16, 7:13])
    ax02 = f.add_subplot(spec[1:16, 14:])
    ax10 = f.add_subplot(spec[16:, 5:-5])
    axes = [ax00, ax01, ax02, ax10]

    ssd_vec = []
    for i in range(1, len(images)):
        registered_image, p, q, theta, _, _, ssd_i = regis_rigid(images[0], images[i], stepsize=1e-08, n_iter=500)
        ssd_vec.append(ssd_i)
        axes[i-1].imshow(registered_image, cmap='gray')
        axes[i-1].set_title(f"BrainMRI_{i+1}, p={round(p,2)}, q={round(q,2)}, theta={round(theta,2)}")

    for i, ssd_i in enumerate(ssd_vec):
        axes[-1].plot(ssd_i, label=f"BrainMRI_{i+2}")

    axes[-1].set_xlabel('number of iterations')
    axes[-1].set_ylabel('sum squared difference')
    axes[-1].set_title('SSD curve (rotation)')
    axes[-1].legend()

    plt.show()

    # test improved gradient descent
    f = plt.figure(figsize=(16, 12), dpi=120)
    spec = f.add_gridspec(nrows=30, ncols=20)
    plt.suptitle("Gradient descent with adaptive stepsize", fontsize=20)

    ax00 = f.add_subplot(spec[1:16, :6])
    ax01 = f.add_subplot(spec[1:16, 7:13])
    ax02 = f.add_subplot(spec[1:16, 14:])
    ax10 = f.add_subplot(spec[16:, 5:-5])
    axes = [ax00, ax01, ax02, ax10]

    ssd_vec = []
    for i in range(1, len(images)):
        p, q, theta, ssd_i = gradient_descent(images[0], images[i], stepsize=1e-08, n_iter=500)
        i2_regis = rigid_transform(images[i], p, q, theta)
        ssd_vec.append(ssd_i)
        axes[i-1].imshow(i2_regis, cmap='gray')
        axes[i-1].set_title(f"BrainMRI_{i+1}, p={round(p, 2)}, q={round(q, 2)}, theta={round(theta, 2)}")

    for i, ssd_i in enumerate(ssd_vec):
        axes[-1].plot(ssd_i, label=f"BrainMRI_{i+2}")

    axes[-1].set_xlabel('number of iterations')
    axes[-1].set_ylabel('sum squared difference')
    axes[-1].set_title('SSD curve (rotation)')
    axes[-1].legend()

    plt.show()