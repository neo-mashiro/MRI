# to reconstruct, refer to the example scripts at:
# https://github.com/KamitaniLab/icnn

from skimage import io
import os
import matplotlib.pyplot as plt


def run():
    source_dir = 'result'
    f, axes = plt.subplots(nrows=10, ncols=5, figsize=(10, 20))
    for i, num in enumerate(range(10, 510, 10)):
        file_name = os.path.join(source_dir, f'{num:05d}.jpg')
        image = io.imread(file_name)
        row, col = int(i / 5), int(i % 5)
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f'iteration {i+1}')

    truth = io.imread(os.path.join(source_dir, 'truth.jpg'))
    plt.imshow(truth)
    plt.title('Ground truth')
    plt.show()
