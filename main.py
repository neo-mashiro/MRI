''' Medical Image Analysis lab sessions and project '''

__author__ = 'Wentao Lu, Yi Ren'
__bu_id__ = '002276355, 002269013'

import sys
from skimage import io

if __name__ == '__main__':
    if sys.argv[1] == 'lab1':
        pass
    elif sys.argv[1] == 'lab2':
        i1 = io.imread('data/I1.png', as_gray=True)
        i2 = io.imread('data/I2.jpg')
        i3 = io.imread('data/I3.jpg')
        i4 = io.imread('data/I4.jpg')
        i5 = io.imread('data/I5.jpg')
        i6 = io.imread('data/I6.jpg')

        j1 = io.imread('data/J1.png', as_gray=True)
        j2 = io.imread('data/J2.jpg')
        j3 = io.imread('data/J3.jpg')
        j4 = io.imread('data/J4.jpg')
        j5 = io.imread('data/J5.jpg')
        j6 = io.imread('data/J6.jpg')

        i_vec = [i1, i2, i3, i4, i5, i6]
        j_vec = [j1, j2, j3, j4, j5, j6]

        m1 = io.imread('data/BrainMRI_1.jpg')
        m2 = io.imread('data/BrainMRI_2.jpg')
        m3 = io.imread('data/BrainMRI_3.jpg')
        m4 = io.imread('data/BrainMRI_4.jpg')

        m_vec = [m1, m2, m3, m4]

        from lab2 import part1, part2, part3, part4

        # part1.test(i_vec, j_vec)
        # part1.run(i_vec, j_vec)
        # part2.test(i_vec, j_vec)
        # part2.run(i_vec, j_vec)

        # part3.run()

        # part4.test(m1)
        part4.run(m_vec)



