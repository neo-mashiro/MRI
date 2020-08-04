import os
from .preprocess import ...
from .predict import ...
import lbfgs



def reconstruct(im):
    """Reconstruct images"""
    pass


def run():
    a = os.system("ls images/ArtificialImage")
    b = os.system("ls images/LetterImage")
    c = os.system("ls images/test")
    rec_artificial = []  # reconstructed artificial images
    rec_letter = []
    rec_natural = []

    for img in a:
        rec_artificial.append(reconstruct(img))

    for img in b:
        rec_letter.append(reconstruct(img))

    for img in c:
        rec_natural.append(reconstruct(img))

    # plot
    f, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
    pass

    # plot bar plots
    pass
