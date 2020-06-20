import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_3d_grid():
    """Generate a 3d grid of evenly spaced points"""
    return np.mgrid[0:21, 0:21, 0:5]


def homo_coordinate(g):
    """Transform a numpy 3D meshgrid to 3D data points in homogeneous coordinates"""
    axes = np.vstack(list(map(np.ravel, g)))
    data_points = axes.T
    return np.insert(data_points, 3, 1, axis=1)


def plot_3d_grid(m3):
    """Plot a numpy array of 3D data points in homogeneous coordinates"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = m3.T[0], m3.T[1], m3.T[2]  # ignore the extra scale dimension
    ax.scatter(x, y, z, facecolor=(0, 0, 0, 0.5), marker='o', s=20, edgecolor='black')  # (R,G,B,A)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_zlim(0, 15)
    ax.set_title("3D Grid of evenly spaced points")
    plt.show()


def rigid_transform(m3, theta, omega, phi, p, q, r):
    """
    Return the matrix (in homogeneous coordinates) of the rigid transform corresponding to
       1. rotation of angle theta around the x-axis
       2. rotation of angle omega around the y-axis
       3. rotation of angle phi around the z-axis
       4. translation of vector (p, q, r)
    """
    # 3D translation matrix
    t = np.array([[1, 0, 0, p],
                  [0, 1, 0, q],
                  [0, 0, 1, r],
                  [0, 0, 0, 1]])

    # 3D rotation matrix
    rxz = np.array([[np.cos(omega), 0, np.sin(omega), 0],
                    [0, 1, 0, 0],
                    [-np.sin(omega), 0, np.cos(omega), 0],
                    [0, 0, 0, 1]])

    ryz = np.array([[1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])

    rxy = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                    [np.sin(phi), np.cos(phi), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    T = t @ ryz @ rxz @ rxy

    m3_transformed = np.apply_along_axis(lambda row: T @ row.T, 1, m3)
    return m3_transformed


def affine_transform(m3, s, theta, omega, phi, p, q, r):
    """Add a scaling factor s on top of the rigid_transform()"""
    # 3D scaling matrix
    sx = sy = sz = s
    sm = np.array([[sx, 0, 0, 0],
                   [0, sy, 0, 0],
                   [0, 0, sz, 0],
                   [0, 0, 0, 1]])

    m3_transformed = rigid_transform(m3, theta, omega, phi, p, q, r)
    m3_scaled = np.apply_along_axis(lambda row: sm @ row.T, 1, m3_transformed)
    return m3_scaled


def run():
    """Test the transform functions on the 3D grid, plot to show the result"""
    data = homo_coordinate(make_3d_grid())
    plot_3d_grid(data)

    dat1 = rigid_transform(data, 0, 0, 0, 5, 5, 8)  # test translation
    dat2 = rigid_transform(data, np.pi/12, np.pi/12, -np.pi/9, 0, 0, 15)  # test rotation
    dat3 = affine_transform(data, 0.5, np.pi/2, -np.pi/4, 0, 30, 15, 30)  # test scaling

    # plot translation
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data.T[0], data.T[1], data.T[2], facecolor=(0, 0, 0, 0.5), marker='o', s=20, edgecolor='black')
    ax.scatter(dat1.T[0], dat1.T[1], dat1.T[2], facecolor='#9400D3', marker='o', s=20, edgecolor='#9400D3', alpha=0.35)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_zticks([0, 5, 10, 15, 20, 25])
    ax.set_zlim(0, 20)
    ax.set_title("test translation")
    plt.show()

    # plot rotation
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data.T[0], data.T[1], data.T[2], facecolor=(0, 0, 0, 0.5), marker='o', s=20, edgecolor='black')
    ax.scatter(dat2.T[0], dat2.T[1], dat2.T[2], facecolor='tomato', marker='o', s=20, edgecolor='r', alpha=0.3)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_zticks([0, 5, 10, 15, 20, 25])
    ax.set_zlim(0, 20)
    ax.set_title("test rotation")
    plt.show()

    # plot scaling
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data.T[0], data.T[1], data.T[2], facecolor=(0, 0, 0, 0.5), marker='o', s=20, edgecolor='black')
    ax.scatter(dat3.T[0], dat3.T[1], dat3.T[2], facecolor='r', marker='o', s=20, edgecolor='r', alpha=0.1)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_zticks([0, 5, 10, 15, 20, 25])
    ax.set_zlim(0, 20)
    ax.set_title("test scaling")
    plt.show()

    m1 = [[0.9045, -0.3847, -0.1840, 10.0000],
          [0.2939, 0.8750, -0.3847, 10.0000],
          [0.3090, 0.2939, 0.9045, 10.0000],
          [0, 0, 0, 1.0000]]

    m2 = [[-0.0000, -0.2598, 0.1500, -3.0000],
          [0.0000, -0.1500, -0.2598, 1.5000],
          [0.3000, -0.0000, 0.0000, 0],
          [0, 0, 0, 1.0000]]

    m3 = [[0.7182, -1.3727, -0.5660, 1.8115],
          [-1.9236, -4.6556, -2.5512, 0.2873],
          [-0.6426, -1.7985, -1.6285, 0.7404],
          [0, 0, 0, 1.0000]]