import numpy as np
import pandas as pd
import nibabel as nib
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


n_runs = 8   # 8 runs per session
n_sess = 15  # 15 sessions
# exception:
"""
   session 07 has 4 runs
   session 08 has 10 runs
   session 09 has 10 runs

   there are 4 types of sessions:
   - natural images (training): 8 runs per session x 15 sessions = 120 runs in total, 55 stimulus blocks (5 randomly interspersed repetition blocks), duration per run 7 min 58 s
   - natural images (test): 8 runs per session x 3 sessions = 24 runs in total, 55 stimulus blocks (5 randomly interspersed repetition blocks), duration per run 7 min 58 s
   - artificial images: 10 runs per session x 2 sessions = 20 runs in total, 44 stimulus blocks (4 randomly interspersed repetition blocks), duration per run 6 min 30 s
   - letter images: 12 runs per session x 1 sessions = 12 runs in total, 11 stimulus blocks (1 randomly interspersed repetition block), duration per run 5 min 2 s

   Additional 32- and 6-s rest periods were added to the beginning and end of each run respectively. (first and last row in .tsv)
   so that the first stimulus appears at 32s

   TR, 2000 ms
   voxel size, 2 × 2 × 2 mm

   must shifting the data by 4 s (two volumes) to compensate for hemodynamic delays.

   for letter images, each stimulus block was 12s, followed by a 12s rest period
   for other images, each stimulus block was 8s, with no rest period followed




   .tsv
   'onset': onset time of an event (sec)
   'duration': duration of the event (sec)
   'event_type': 1: Stimulus presentation block, 2: Repetition block, -1, -2, and -3: Initial, inter-trial, and post rest blocks without visual stimulus
   'stimulus_id': stimulus ID of the image presented in a stimulus block ('n/a' in rest blocks)
   'stimulus_name': stimulus file name of the image presented in a stimulus block ('n/a' in rest blocks)
"""
base_dir = "./data/NaturalImageTraining"


def extract_sample(im):
    """Extract samples corresponding to the stimulus from the bold time series
       detailed explanation: 4s hrf delay + average every 4 slices
    """
    # load data files
    fmri = nib.load(f'lab3/data/{img}.nii.gz')  # f-mri image after pre-processing
    task = pd.read_csv('lab3/data/events.tsv', delimiter='\t').to_numpy()
    hrf = pd.read_csv('lab3/data/hrf.csv', header=None)  # hemodynamic response function
    hrf = hrf.to_numpy().reshape(len(hrf),)

    tr = fmri.header.get_zooms()[3]  # repetition time (time interval between 2 volumes)
    n = int(tr * fmri.shape[3])      # number of seconds in the time series
    ts = np.zeros(n)                 # initialize the time series (a function of seconds)

    # create an ideal time series
    stimuli = ('FAMOUS', 'UNFAMILIAR', 'SCRAMBLED')  # stimulus type we are interested in
    mask = np.isin(task[:, 3], stimuli)              # filter out the tasks of interest
    for onset in task[mask][:, 0]:                   # time when the stimulus is shown to the subject
        ts[int(onset)] = 1

    # convolve the time series with hrf
    convolved = signal.convolve(ts, hrf, mode='full')
    convolved = convolved[0:len(ts)]

    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax[0].plot(ts)
    ax[0].set_xlabel('time(seconds)')
    ax[1].plot(ts)
    ax[1].plot(convolved * 2.6, c='r', lineWidth=0.7)
    ax[1].set_xlabel('time(seconds)')
    plt.show()

    # correlate convolved triggers with signal in each voxel
    convolved = convolved[0::2]  # tr = 2 seconds per volume
    im = fmri.get_fdata()
    corr_map = corr_volume(im, convolved)

    f, ax = plt.subplots(nrows=7, ncols=7, figsize=(18, 22))
    for z in range(7, 53):
        row, col = (z - 7) // 7, (z - 7) % 7
        ax[row, col].imshow(np.rot90(corr_map[:, :, z]), origin='lower', vmin=-0.25, vmax=0.25)

    f.delaxes(ax[6, -1])
    f.delaxes(ax[6, -2])
    f.delaxes(ax[6, -3])
    plt.show()


def corr_volume(im, cv):
    """Compute the correlation between an f-MRI image and a convolution, voxel by voxel"""
    ci = im - np.expand_dims(np.mean(im, 3), 3)
    cc = cv - np.mean(cv)
    # np.sqrt() uses division inside, so we add a small constant to prevent division by zero error
    corr = np.sum(ci * cc, 3) / (np.sqrt(np.sum(ci * ci, 3) + 1e-14) *
                                 np.sqrt(np.sum(cc * cc) + 1e-14))
    return corr


def compute_corr(im):
    """Compute the correlation map voxel-wise"""
    # hemodynamic response function
    hrf = pd.read_csv('data/hrf.csv', header=None)
    hrf = hrf.to_numpy().reshape(len(hrf), )

    for ses in range(1, n_sess + 1):
        for run in range(1, n_runs + 1):
                    fmri = nib.load(f'base_dir/reg_{(ses):02d}{(run):02d}.nii.gz')
                    task = pd.read_csv(f'base_dir/{(ses):02d}{(run):02d}.tsv', delimiter='\t').to_numpy()

                    tr = fmri.header.get_zooms()[3]  # repetition time
                    ts = np.zeros(int(tr * fmri.shape[3]))

                    mask = np.isin(task[:, 3], ('FAMOUS', 'UNFAMILIAR', 'SCRAMBLED'))
                    for onset in task[mask][:, 0]:
                        ts[int(onset)] = 1

                    convolved = signal.convolve(ts, hrf, mode='full')[0:len(ts)][0::2]
                    corr_map = corr_volume(fmri.get_fdata(), convolved)

                    corr_nii = nib.Nifti1Image(corr_map, fmri.affine)
                    nib.save(corr_nii, f'data/sub{(i+1):02d}/corrs.nii.gz')
