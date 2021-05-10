""" Medical Image Analysis lab sessions and project """

__uid__ = '0022763557355608'

import os, subprocess, sys
from skimage import io

if __name__ == '__main__':
    if sys.argv[1] == 'lab1':
        print("Lab1 is inside the Jupyter notebook")

    elif sys.argv[1] == 'lab2':
        # sys.stdout = open("lab2/output.out", 'w')
        i1 = io.imread('lab2/data/I1.png', as_gray=True)
        i2 = io.imread('lab2/data/I2.jpg')
        i3 = io.imread('lab2/data/I3.jpg')
        i4 = io.imread('lab2/data/I4.jpg')
        i5 = io.imread('lab2/data/I5.jpg')
        i6 = io.imread('lab2/data/I6.jpg')
        j1 = io.imread('lab2/data/J1.png', as_gray=True)
        j2 = io.imread('lab2/data/J2.jpg')
        j3 = io.imread('lab2/data/J3.jpg')
        j4 = io.imread('lab2/data/J4.jpg')
        j5 = io.imread('lab2/data/J5.jpg')
        j6 = io.imread('lab2/data/J6.jpg')
        m1 = io.imread('lab2/data/BrainMRI_1.jpg')
        m2 = io.imread('lab2/data/BrainMRI_2.jpg')
        m3 = io.imread('lab2/data/BrainMRI_3.jpg')
        m4 = io.imread('lab2/data/BrainMRI_4.jpg')

        i_vec = [i1, i2, i3, i4, i5, i6]
        j_vec = [j1, j2, j3, j4, j5, j6]
        m_vec = [m1, m2, m3, m4]

        from lab2 import part1, part2, part3, part4, part5

        part1.test(i_vec, j_vec)
        part1.run(i_vec, j_vec)
        part2.test(i_vec, j_vec)
        part2.run(i_vec, j_vec)
        part3.run()
        part4.test_transform(m1)
        part4.test_registration(m1)
        part4.test_gradient_descent(m1)
        part4.run(m_vec)
        part5.run()

    elif sys.argv[1] == 'lab3':
        # sys.stdout = open("lab3/output.out", 'w')
        from lab3 import part2, part3

        part2.run('clean_bold')
        part2.run('bold')

        # first run part3.sh in the sub folder, which will run part3.py as a stand-alone program
        # next view each subject's correlation map superimposed on T1 in afni (threshold of 0.15)
        # then save each image as a .png file to disk
        part3.view_corr_in_t1()
        # next continue with part3.sh

    elif sys.argv[1] == 'lab4':
        # sys.stdout = open("lab4/output.out", 'w')
        i1 = io.imread('lab4/data/t1.png', as_gray=True)
        i2 = io.imread('lab4/data/t1_v2.png', as_gray=True)
        i3 = io.imread('lab4/data/t1_v3.png', as_gray=True)
        i4 = io.imread('lab4/data/t2.png', as_gray=True)
        i5 = io.imread('lab4/data/flair.png', as_gray=True)

        i_vec = [i1, i2, i3, i4, i5]

        from lab4 import part1, part2, part3

        part1.test()
        part1.run(i_vec)
        part2.test()
        part2.run()
        part3.run()

    elif sys.argv[1] == 'final':
        # sys.stdout = open("lab_final/output.out", 'w')

        # download data from openneuro
        os.system("chmod +x lab_final/download.sh && lab_final/download.sh")  # ~ 6 hours
        # preprocess bold signals
        os.system("chmod +x lab_final/pipeline.sh && lab_final/pipeline.sh")  # ~ 8 hours

        from lab_final import preprocess, extract_features, decode_features, reconstruct

        preprocess.run()
        preprocess.test()
        extract_features.run()
        extract_features.test()
        decode_features.run()
        decode_features.test()
        reconstruct.run()
