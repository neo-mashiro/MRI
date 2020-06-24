import numpy as np
import matplotlib.pyplot as plt


def jointhist(i, j, bins=256):
    """Calculate the joint histogram of two grayscale images of the same size"""
    li, lj = np.ravel(i), np.ravel(j)  # unravel ndarrays into lists
    assert len(li) == len(lj), "images have different sizes, please scale your data first!"

    nrows = min(int(np.ceil(max(li))), 255)
    ncols = min(int(np.ceil(max(lj))), 255)

    H = np.zeros((bins, bins))  # initially black everywhere, (0~255) = (black~white)
    x_bin_width = (nrows + 1) / bins
    y_bin_width = (ncols + 1) / bins

    for index in range(len(li)):
        x_value, y_value = int(li[index]), int(lj[index])  # grayscale intensity
        x_bin_id = int(x_value / x_bin_width)
        y_bin_id = int(y_value / y_bin_width)
        H[x_bin_id, y_bin_id] += 1

    return H


def test(i_vec, j_vec):
    """Test the joint histogram against the numpy built-in function"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 11))
    fig.suptitle("Numpy built-in Joint histograms of (I1,J1) ~ (I6,J6)", fontsize=24)

    for index in range(6):
        H, _, _ = np.histogram2d(np.ravel(i_vec[index]), np.ravel(j_vec[index]), bins=256)

        axis = (index // 3, index % 3)  # axes(subplot) index
        ax[axis].imshow(np.log(H + 0.000001), origin='low')
        ax[axis].set_xlabel(f"I{index + 1}")
        ax[axis].set_ylabel(f"J{index + 1}")

    fig.subplots_adjust(top=1)
    fig.tight_layout()
    plt.show()


def run(i_vec, j_vec):
    """Verify the number of pixels and draw the joint histograms"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 11))
    fig.suptitle("Joint histograms of (I1,J1) ~ (I6,J6)", fontsize=24)

    for index in range(6):
        nrows, ncols = i_vec[index].shape
        print(f"I{index + 1}.shape = J{index + 1}.shape = {i_vec[index].shape}")
        print(f"{nrows} * {ncols} = {nrows * ncols}")

        H = jointhist(i_vec[index], j_vec[index])  # calculate histogram
        H_sum = H.sum()
        print(f"Joint histogram's total sum is H.sum() = {H_sum}")
        print("---------------------------------------------------")
        assert H_sum == nrows * ncols, "verification failed..."

        axis = (index // 3, index % 3)  # axes(subplot) index
        ax[axis].imshow(np.log(H + 0.000001), origin='low')
        ax[axis].set_xlabel(f"I{index + 1}")
        ax[axis].set_ylabel(f"J{index + 1}")

    fig.subplots_adjust(top=1)
    fig.tight_layout()
    plt.show()