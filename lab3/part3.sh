#!/bin/bash

# download data from all 16 subjects
# run the pre-processing pipeline on each subject
# apply correlation analysis (from part2) on each subject
# save the output as a .nii file
# align the correlation map with subject's T1 space
# then visualize the alignment using afni and compute the grand average in python

export LC_ALL=C

# create a separate data folder for each subject
echo "Creating data folders for multiple subjects......"
CWD=$(pwd)
cd "$CWD/data"
mkdir sub{01..16}

# copy the pre-process pipeline script to each folder
echo "Copying pipeline.sh to each folder......"
for i in {01..16}
do
  cp "$CWD"/data/pipeline.sh "$CWD"/data/sub$i/pipeline.sh
done

# use `httpie` to download T1-weighted and f-MRI images from openneuro
url_folder="https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files"

function download_file {
    printf "\nDownloading t1.nii.gz ..."
    time http $url_folder/sub-$1:ses-mri:anat:sub-$1_ses-mri_acq-mprage_T1w.nii.gz > "$CWD"/data/sub$1/t1.nii.gz
    printf "\nDownloading bold.nii.gz ..."
    time http $url_folder/sub-$1:ses-mri:func:sub-$1_ses-mri_task-facerecognition_run-01_bold.nii.gz > "$CWD"/data/sub$1/bold.nii.gz
    printf "\nDownloading events.tsv ..."
    time http $url_folder/sub-$1:ses-mri:func:sub-$1_ses-mri_task-facerecognition_run-01_events.tsv > "$CWD"/data/sub$1/events.tsv
}

# pre-process each subject
function preprocess {
    printf "\nPre-processing bold.nii.gz for subject $i ..."
    bash "$CWD"/data/sub$1/pipeline.sh
}

# handle the interrupt signal in case download or processing is slow
signal_handler() {
        echo
        read -p 'Interrupt? (y/n) [N] > ' answer
        case $answer in
                [yY])
                    kill -TERM -$$  # kill the process id of the script
                    ;;
        esac
}

trap signal_handler INT  # catch signal

# __main__
for i in {01..16}
do
  echo "Downloading data for subject $i"
  download_file $i
  echo "++++++++++++++++++++++++++++++++++++++++++++++"
done

for i in {01..16}
do
  (  # pipeline.sh is dependent on path, so must run in a subshell
  cd "$CWD/data/sub$i"
  preprocess $i
  )
done

cd "$CWD"
source "$HOME"/py_36_env/bin/activate  # load python virtual environment
python3 "$CWD"/part3.py

for i in {01..16}
do
  (
  cd "$CWD/data/sub$i"
  # source "$HOME/.zprofile"
  printf "\nRegistering subject $i correlation map into T1 space ..."
  flirt -in corrs.nii.gz -ref t1.nii.gz -applyxfm -init epireg.mat -out corrs_in_t1.nii.gz
  # printf "\nRegistering subject $i T1 into template (template is in T1 space) ..."
  # flirt -in t1.nii.gz -ref "$CWD"/data/MNI152_2009_template.nii.gz -out t1_in_tmp.nii -omat transform.mat
  # printf "\nRegistering subject $i correlation map into template ..."
  # flirt -in corrs_in_t1.nii.gz -ref "$CWD"/data/MNI152_2009_template.nii.gz -applyxfm -init transform.mat -out corrs_in_tmp.nii.gz
  )
done

wait
