import numpy as np
from .part1 import jointhist


def ssd(i, j):
    """Calculate the sum squared difference between two images of the same size"""
    li, lj = np.ravel(i), np.ravel(j)
    assert len(li) == len(lj), "images have different sizes, please scale your data first!"
    return np.sum(np.square(li - lj))


def corr(i, j):
    """Calculate the pearson correlation coefficient between two images of the same size"""
    li, lj = np.ravel(i), np.ravel(j)
    assert len(li) == len(lj), "images have different sizes, please scale your data first!"
    i_bar, j_bar = np.mean(li), np.mean(lj)
    i_dev, j_dev = li - i_bar, lj - j_bar
    numerator = np.sum(i_dev * j_dev)
    denominator = np.sqrt(np.sum(i_dev ** 2)) * np.sqrt(np.sum(j_dev ** 2))
    return numerator / denominator


def mi(i, j):
    """Calculate the mutual information between two images of the same size"""
    H = jointhist(i, j)  # joint histogram
    H_norm = H / np.sum(H)  # normalized joint histogram
    row_m = np.sum(H_norm, axis=1, keepdims=True, dtype=np.float64)
    col_m = np.sum(H_norm, axis=0, keepdims=True, dtype=np.float64)
    P = row_m * col_m  # product matrix
    P[P == 0] = 1  # avoid the divide by zero error
    log_term = np.log(H_norm / P)
    log_term[~np.isfinite(log_term)] = 0  # avoid the divide by zero error in log
    return np.sum(H_norm * log_term)


def test(i_vec, j_vec):
    """Test the three functions"""
    def r_corr(i, j):
        numerator = np.mean((i - i.mean()) * (j - j.mean()))
        denominator = i.std() * j.std()
        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def r_mi(i, j):
        from sklearn.metrics import mutual_info_score
        H, _, _ = np.histogram2d(np.ravel(i), np.ravel(j), bins=256)
        return mutual_info_score(None, None, contingency=H)

    for index in range(6):
        print(f"(I{index + 1}, J{index + 1}) real pearson corr = {r_corr(i_vec[index], j_vec[index])}")
        print(f"(I{index + 1}, J{index + 1}) real mutual info = {r_mi(i_vec[index], j_vec[index])}")
        print("---------------------------------------------------")


def run(i_vec, j_vec):
    """Compare the results of the three functions above on (I1,J1) ~ (I6,J6)"""
    for index in range(6):
        print(f"(I{index + 1}, J{index + 1}) sum sqr diff = {ssd(i_vec[index], j_vec[index])}")
        print(f"(I{index + 1}, J{index + 1}) pearson corr = {corr(i_vec[index], j_vec[index])}")
        print(f"(I{index + 1}, J{index + 1}) mutual info = {mi(i_vec[index], j_vec[index])}")
        print("---------------------------------------------------")
