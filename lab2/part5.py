import os, subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def fsl(i1, i2, slice):
    """Register 3D medical image i2 onto i1 at the given slice using FSL libraries
       https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
    """
    # load NIfTI data
    nifti_f = nib.load(f'lab2/data/{i1}.nii')  # fixed image
    nifti_m = nib.load(f'lab2/data/{i2}.nii')  # moved image

    # print('voxel size:')
    # print(nifti_f.header.get_zooms())
    # print(nifti_m.header.get_zooms())

    # reset voxel size
    # nifti_f.header.set_zooms((0.6, 0.6, 0.6))
    # nifti_m.header.set_zooms((0.6, 0.6, 0.6))

    # print('changed voxel size:')
    # print(nifti_f.header.get_zooms())
    # print(nifti_m.header.get_zooms())

    # os.system(f"flirt -in data/{i2}.nii -ref data/{i1}.nii -out data/{i2}_in_{i1}.nii -omat matrix.mat")
    os.system(f"flirt -in lab2/data/{i2}.nii -ref lab2/data/{i1}.nii -out lab2/data/{i2}_in_{i1}.nii")

    f = nifti_f.get_fdata()[slice, :, :]  # fixed image
    m = nifti_m.get_fdata()[slice, :, :]  # moved image
    r = nib.load(f'lab2/data/{i2}_in_{i1}.nii.gz').get_fdata()[slice, :, :]  # registered image

    return f, m, r


def run():
    t1, tof, tof_in_t1 = fsl('t1', 'tof', 150)
    masked = np.ma.masked_where(tof_in_t1 == 0, tof_in_t1)

    f, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.imshow(t1.T, origin='lower', cmap='gray')
    ax0.set_title('t1')
    ax1.imshow(tof, cmap='gist_rainbow')
    ax1.set_title('tof')
    plt.show()

    plt.figure()
    plt.imshow(t1.T, cmap='gray', origin='lower')
    plt.imshow(masked.T, cmap='gnuplot2', origin='lower', alpha=0.75)
    plt.title('FSL')
    plt.show()