#!/bin/bash

export LC_ALL=C
CWD=$(pwd)
cd "$CWD/data"

# remove skull using bet (brain extraction tool) and generate binary brain mask
bet swi.nii bet_swi.nii.gz -f 0.72 -m
bet tof.nii bet_tof.nii.gz -f 0.05 -m  # dark tof needs a small threshold

# invert the swi image (black <-> white) so that veins appear bright like arteries
fslmaths bet_swi.nii.gz -sub 255 -mul -1 -mul mask_swi.nii.gz invert_swi.nii.gz
