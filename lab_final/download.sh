#!/bin/bash

export LC_ALL=C
CWD=$(pwd)

mkdir data && cd data
mkdir ArtificialImage
mkdir LetterImage
mkdir NaturalImageTest
mkdir NaturalImageTraining

url_folder="https://openneuro.org/crn/datasets/ds001506/snapshots/1.3.1/files"
path1="sub-01:ses-perception"
path2=":func:sub-01_ses-perception"
path3="_task-perception_run-"

# download f-MRI images from openneuro
function download_file {
    session=$1
    run=$2
    img=$3  # ArtificialImage / LetterImage / NaturalImageTest / NaturalImageTraining
    echo "--------------------------------------------------------------------"
    printf "Downloading bold.nii.gz ..."
    time http --download "$url_folder/$path1$img$session$path2$img$session$path3${run}_bold.nii.gz" > $CWD/data/$img/$session$run.nii.gz
    printf "\nDownloading events.tsv ..."
    time http --download "$url_folder/$path1$img$session$path2$img$session$path3${run}_events.tsv" > $CWD/data/$img/$session$run.tsv
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

# download ArtificialImage
for ses in {01..02}  # 2 sessions
do
    for run in {01..10}  # 10 runs
    do
        download_file $ses $run "ArtificialImage"
    done
done

# download LetterImage
for run in {01..12}  # 1 session 12 runs
do
    download_file "01" $run "LetterImage"
done

# download NaturalImageTest
for ses in {01..03}  # 3 sessions
do
    for run in {01..08}  # 8 runs
    do
        download_file $ses $run "NaturalImageTest"
    done
done

# download NaturalImageTraining
# ! for some reason, session 07 has 4 runs, session 08 and 09 each has 10 runs
# ! must fix these downloads manually
for ses in {01..15}  # 15 sessions
do
    for run in {01..08}  # 8 runs
    do
        download_file $ses $run "NaturalImageTraining"
    done
done

wait
