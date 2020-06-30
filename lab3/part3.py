import numpy as np
import pandas as pd
import nibabel as nib
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def corr_volume(im, cv):
    """Compute the correlation between an f-MRI image and a convolution, voxel by voxel"""
    ci = im - np.expand_dims(np.mean(im, 3), 3)
    cc = cv - np.mean(cv)
    corr = np.sum(ci * cc, 3) / (np.sqrt(np.sum(ci * ci, 3) + 1e-14) *
                                 np.sqrt(np.sum(cc * cc) + 1e-14))
    return corr


def run():
    """Localize task activation: save each subject's correlation map"""
    hrf = pd.read_csv('data/hrf.csv', header=None)
    hrf = hrf.to_numpy().reshape(len(hrf), )

    for i in range(16):
        fmri = nib.load(f'data/sub{(i+1):02d}/clean_bold.nii.gz')
        task = pd.read_csv(f'data/sub{(i+1):02d}/events.tsv', delimiter='\t').to_numpy()

        tr = fmri.header.get_zooms()[3]
        ts = np.zeros(int(tr * fmri.shape[3]))

        mask = np.isin(task[:, 3], ('FAMOUS', 'UNFAMILIAR', 'SCRAMBLED'))
        for onset in task[mask][:, 0]:
            ts[int(onset)] = 1

        convolved = signal.convolve(ts, hrf, mode='full')[0:len(ts)][0::2]
        corr_map = corr_volume(fmri.get_fdata(), convolved)

        corr_nii = nib.Nifti1Image(corr_map, fmri.affine)
        nib.save(corr_nii, f'data/sub{(i+1):02d}/corrs.nii.gz')


def view_corr_in_t1():
    """Plot correlation map overlaid on T1 for each subject"""
    views = {1: 'axial', 2: 'sagittal', 3: 'coronal'}
    f, ax = plt.subplots(nrows=16, ncols=3, figsize=(16, 120))
    plt.rcParams.update({'font.size': 20})

    for i in range(1, 17):
        for v in range(1, 4):
            im = mpimg.imread(f"lab3/images/{i:02d}{v}.png")
            ax[i-1, v-1].imshow(im)
            ax[i-1, v-1].set_axis_off()
            ax[i-1, v-1].set_title(f'subject {i:02d} ({views[v]})')

    plt.show()


def view_average():
    f, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))

    for i in range(1, 17):
        image = mpimg.imread(f"lab3/data/sub{i:02d}/s{i:02d}.png")
        row, col = (i - 1) // 4, (i - 1) % 4
        ax[row, col].imshow(image)
        ax[row, col].set_axis_off()
        ax[row, col].set_title(f'subject {i}')

    plt.show()

    for i in range(16):
        corr_in_tmp = nib.load(f'data/sub{(i+1):02d}/corrs_in_tmp.nii.gz').get_fdata()
        if i == 0:
            sum_corr = corr_in_tmp
        else:
            sum_corr = np.add(sum_corr, corr_in_tmp)

    avg_corr = np.multiply(sum_corr, 1/16)  # how to save?


if __name__ == '__main__':
    run()
