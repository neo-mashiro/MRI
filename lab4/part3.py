from dipy.denoise.nlmeans import nlmeans_3d, nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import nibabel as nib
import numpy as np
import skimage as ski
import time
import cv2
import matplotlib.pyplot as plt

base = '/home/yee/Documents/CS516/A4/img/anats/'

def pre_process(img, name):
    """Pre process to the image (denoise) , run this function only once before running cluster"""
    # sigma: standard deviation of background piexels
    sigma = estimate_sigma(img, N=16);
    print("sigma = ", sigma)
    print(name, "...nl means running...")
    st = time.time()
    # denoise with nl-means
    clear_img = nlmeans(img, sigma);
    et = time.time();
    print("pre-process time = ", (et - st), " s")
    nifti_img = nib.Nifti1Image(clear_img, swi.affine)
    path = base + name
    nib.save(nifti_img, path)


def cluster(image, msk, slice, isTof):
    """Segment the veins and arteries from a slice of 3d image
       Parameters:
           image: the source image to be segmented
           msk: the 3d mask of the entire brain
           slice: index of slice to be deal with
           isTof: if true, the image will be sliced according to the third dimension
    """
    # num_mask: amount of mask pixels in the image
    # num_tissue: amount of tissue detected
    num_mask = 0
    num_tissue = 0
    # this variable indicates the number of clusters for kmeans
    tof_cluster = 4
    # A list record the amount of pixels in each cluster
    label_list = [0 for i in range(tof_cluster)]
    # slice the image and mask according to isTof
    img = image[:, :, slice] if isTof else image[slice, :, :]
    mask = msk[:, :, slice] if isTof else msk[slice, :, :]
    # check if image and mask have the same shape
    if image.shape != msk.shape:
        print("error : shape of image not consistent with mask")
        return
    rows, cols = img.shape
    size = rows * cols
    # reshape the data to 1d array
    data_img = img.reshape((size, 1))
    data_mask = mask.reshape((size, 1))
    # format transform for kmeans function
    data_img = np.float32(data_img)
    # parameters: type, max iteration times, accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # initialize random centers
    flag = cv2.KMEANS_RANDOM_CENTERS
    # cluster amount for kmeans, for the swi.nii, 2 is enough, for tof.nii, 4 lead to better performance
    cluster = tof_cluster if isTof else 2
    # Kmeans from opencv library
    compactness, labels, centers = cv2.kmeans(data_img, cluster, None, criteria, 20, flag)
    # loop all pixels in the slice
    for i in range(size):
        # mask[i] == 1 means the certain pixel is in the brain area
        if data_mask[i] == 1:
            num_mask += 1
            # if isTof, figure out the amount of pixels in each cluster
            if isTof:
                label_list[int(labels[i])] += 1
            # if not, figure out the amount of tissue pixels
            if labels[i] == 1:
                num_tissue += 1
        else:
            # Set every piexl outside the brain area,.
            # if isTof, to 1 (0 lead to unknown error when visualized in imeka); if not, to 0
            labels[i] = 1 if isTof else 0
    print("slice:", slice, ".... mask:", num_mask, ".... white:", num_tissue)
    print("labels:", label_list)
    if isTof:
        # index indicates the cluster which has min pixel amount
        index = label_list.index(min(label_list))
        # set tissue pixels white
        for i in range(size):
            if data_mask[i] == 1:
                labels[i] = 0 if labels[i] == index else 1
    else:
        # for a normal image, if the color of tissue pixels is black, invert the color of tissue and non-tissue pixels
        if num_tissue > (num_mask * 0.3):
            print("Inverting...")
            for i in range(size):
                if data_mask[i] == 1:
                    if labels[i] == 1:
                        labels[i] = 0
                    else:
                        labels[i] = 1
        # check the result of invertion
            num_mask = num_tissue = 0
            for i in range(size):
                if data_mask[i] == 1:
                    num_mask += 1
                    num_tissue += 1 if labels[i] == 1 else 0
            print("Inverted slice:", slice, ".... mask:", num_mask, ".... white:", num_tissue)
    # reshape the array
    result = labels.reshape((img.shape[0], img.shape[1]))
    # save data to image
    if isTof:
        image[:, :, slice] = result
    else:
        image[slice, :, :] = result


def save_image(image, name):
    """save image with name in nifti format"""
    nifti_img = nib.Nifti1Image(image, affine=None)
    nib.save(nifti_img, name)
    print("image saved...")


def run(img, msk, name, isTof):
    """start segmentation
    Parameters:
           img: the source image to be segmented
           msk: the 3d mask of the entire brain
           name: filename of the final result
           isTof: if true, the image will be sliced according to the third dimension"""
    # amount of slices
    r = img.shape[2] if isTof else img.shape[0]
    ts = time.time()
    print(name, "...kmeans running...")
    for i in range(r):
        cluster(img, msk, i, isTof)
    te = time.time()
    print("total time: ", (te - ts), " s")
    save_image(img, name)


def show_image(img, slice1, slice2, slice3):
    """show image"""
    plt.imshow(img[:, :, slice])
    plt.figure("Show image")
    plt.subplot(4, 1, 1)
    plt.imshow(img[slice1, :, :])
    plt.title('slice: ', slice1)
    plt.subplot(4, 1, 2)
    plt.imshow(img[:, slice2, :])
    plt.title('slice: ', slice2)
    plt.subplot(4, 2, 1)
    plt.imshow(img[:, :, slice3])
    plt.title('slice: ', slice3)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def init():
    # load data
    swi = nib.load(base + 'bet_swi.nii.gz')
    tof = nib.load(base + 'bet_tof.nii.gz')
    mask_swi = nib.load(base + 'mask_swi.nii.gz')
    mask_tof = nib.load(base + 'mask_tof.nii.gz')
    img_swi = swi.get_fdata()
    img_tof = tof.get_fdata()
    img_mask_swi = mask_swi.get_fdata()
    img_mask_tof = mask_tof.get_fdata()

    # pre_process(img_swi, 'clear_swi.nii.gz')
    # pre_process(img_tof,'clear_tof.nii.gz')

    # run(img_swi, img_mask_swi, "kmeans_swi.nii", False)
    run(img_tof, img_mask_tof, "kmeans_tof_inverted.nii", True)
