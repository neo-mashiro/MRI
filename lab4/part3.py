from dipy.denoise.nlmeans import nlmeans_3d, nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import cv2 as cv
import numpy as np
import nibabel as nib


def preprocess(nifti, name):
    """Preprocess the 3D MRI image before image segmentation"""
    image = nifti.get_fdata()
    sigma = estimate_sigma(image, N=16)  # N: number of coils in the receiver of the MRI scanner
    denoised = nlmeans(image, sigma)
    denoised_nifti = nib.Nifti1Image(denoised, nifti.affine)
    nib.save(denoised_nifti, f'lab4/data/clean_{name}.nii.gz')


def cluster(nifti, name):
    """Segment the 3D image slice by slice, then merge all slices and save as nifti"""
    n_cluster = 7  # number of clusters
    image = nifti.get_fdata(dtype=np.float32)

    for i, slice in enumerate(image):
        data = slice.reshape((-1, 1))
        vessel, vessel_id = max(data), np.argmax(data)  # vessel is the brightest pixel
        if vessel < 10:  # slice has no vessels (perhaps outside the brain)
            image[i, ...] = 0  # enforce binary property so as to view polygon model in imeka
            continue

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1)  # (type, max_iter, epsilon)
        _, labels, _ = cv.kmeans(data, n_cluster, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        cluster_id = labels[vessel_id]  # cluster id of all vessels

        data[labels == cluster_id] = 255
        data[labels != cluster_id] = 0
        image[i, ...] = data.reshape(slice.shape)

    output = nib.Nifti1Image(image, nifti.affine)
    nib.save(output, f'lab4/data/out_{name}.nii.gz')


def run():
    swi = nib.load('lab4/data/invert_swi.nii.gz')
    tof = nib.load('lab4/data/bet_tof.nii.gz')

    # preprocess(swi, 'swi')
    # preprocess(tof, 'tof')
    cluster(nib.load('lab4/data/clean_swi.nii.gz'), "swi")
    # cluster(nib.load('lab4/data/clean_tof.nii.gz'), "tof")
