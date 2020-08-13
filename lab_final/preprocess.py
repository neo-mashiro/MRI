import os
import numpy as np
import pandas as pd
import nibabel as nib


HRF_DELAY = 4  # shift f-mri by 4 secs to compensate for hrf delay
VC = 1 / 3     # use the last 1/3 of the brain along y axis as visual cortex


def clean(folder, outfile):
    """This function slices the f-MRI bold signals
       1. for each stimuli, average across all time slices to obtain a 3D ndarray
       2. flatten out the ndarray into one-dimensional vector
       3. return all vectors as a pandas dataframe, indexed by the stimuli name
       4. serialize the dataframe into binary format, dump to local disk
    """
    if os.path.exists(outfile):
        os.remove(outfile)

    for root, dirs, files in os.walk(folder):
        mri_dict = {}  # a dictionary of {stimuli : slice} pairs

        for file in files:
            if not file.endswith('nii.gz'):
                continue
            else:
                tsv = file[4:8] + '.tsv'

            nii = nib.load(os.path.join(root, file))
            dx, dy, dz, dt = nii.shape
            tr = nii.header.get_zooms()[3]  # repetition time

            y = int(dy * (1 - VC))
            mri = nii.get_fdata()[:, y:, ...]
            print(mri.shape)

            events = pd.read_csv(os.path.join(root, tsv), delimiter='\t').to_numpy()
            events = events[1:-1, ...]  # exclude the first and last row (32s and 6s rest periods)

            for onset, duration, block, _, stimuli, _, _, _ in events:
                if block < 0:  # rest blocks without visual stimulus
                    continue

                a = int((onset + HRF_DELAY) / tr)  # first slice
                z = int(a + duration / tr)         # last slice

                slice = np.mean(mri[..., a:z], axis=3)
                slice = slice.ravel()

                if stimuli in mri_dict.keys():
                    updated_slice = (mri_dict[stimuli][0] + slice) / 2
                    mri_dict.update({stimuli: [updated_slice]})
                else:
                    mri_dict.update({stimuli: [slice]})

        mri_df = pd.DataFrame.from_dict(mri_dict, orient='index', columns=['mri'])
        mri_df.to_pickle(outfile)


def run():
    base_dir = "lab_final/data"

    dir1 = os.path.join(base_dir, 'train')
    dir2 = os.path.join(base_dir, 'test')
    dir3 = os.path.join(base_dir, 'artificial')
    dir4 = os.path.join(base_dir, 'letter')
    dirs = [dir1, dir2, dir3, dir4]

    for folder in dirs:
        filename = os.path.split(folder)[1] + '.bold.pkl'
        outfile = os.path.join(base_dir, filename)
        clean(folder, outfile)


def test():
    out = os.path.join('lab_final', 'data')
    for root, dirs, files in os.walk(out):
        for f in files:
            if not f.endswith('.bold.pkl'):
                continue

            df = pd.read_pickle(os.path.join(root, f))
            print("file name: ", f)
            print(df)
            print(df.shape)

            print(df['mri'].iloc[0].shape)  # 96x32x76 = 233472
