#!/bin/bash

############################################################################
# to prevent exhausted hardware storage, we will delete images as we go
# so, it is recommended to backup the images before running this script
############################################################################

export LC_ALL=C
CWD=$(pwd)

function cleanup {
    local image=$1
    printf "__________(0%%)\r"

    # remove large spikes in time series
    3dDespike -prefix despike_$image $image 2>&1  # redirect stderr to stdout
    [[ $? -eq 0 ]] && printf "++ #########_____________________(33%% completed)\r" || exit 1

    # motion correction
    mcflirt -in despike_$image -out mc_$image 2>&1
    [[ $? -eq 0 ]] && printf "++ #####################_________(67%% completed)\r" || exit 2

    # compute mean (will be later used to obtain the transform matrix w.r.t. the template)
    3dTstat -prefix mean_$image mc_$image 2>&1
    [[ $? -eq 0 ]] && printf "++ ##############################(99%% completed)\r" || exit 3
    printf "\n\n"

    # free disk space to prevent exhausted hardware (only 40 GB available on my laptop)
    # important! must delete these 2 images on the spot, otherwise 97 GB is required
    # if only 1 image is deleted, we still need 60+ GB, hence, we must delete both
    # we then create a fake $image file, which will be used as dummy iterator in the registration loop
    rm $image despike_$image
    touch $image  # dummy file
}

# despike, motion correct the raw bold images
for dir in $(ls -d data/*/)  # command substitution $()
do
    cd $CWD/$dir
    for img in *.nii.gz
    do
        echo "cleaning ${dir}/$img"
        cleanup $img
    done
done

cd $CWD
cp "$CWD/data/ArtificialImage/mean_0101.nii.gz" "$CWD/data/template.nii.gz"
template="$CWD/data/template.nii.gz"  # use the first mean artificial image as template

function register {
    local mean=mean_$1
    local mc=mc_$1
    local xfm=${mean/.nii.gz/.m}  # one-time pattern substitution: ${var/old_str/new_str}

    # for the same subject, a degree of freedom of 6 would suffice (translation + rotation)
    flirt -in $mean -ref $template -out dispose.nii.gz -dof 6 -omat ${mean/.nii.gz/.m}
    flirt -in $mc -ref $template -applyxfm -init $xfm -out reg_$1

    # free disk space again after coregistration to prevent exhausted hardware
    rm $1 $mean $mc $xfm dispose.nii.gz
}

# register all bold images to the template space
for dir in $(ls -d data/*/)  # command substitution $()
do
    cd $CWD/$dir
    for img in [0-9]*.nii.gz
    do
        printf "registering ${dir}${img}\n"
        register $img
    done
done

# later try some other preprocessing steps ...
cd $CWD

# for dir in $(ls -d data/*/)
# do
#     cd $CWD/$dir
#     for img in reg_*.nii.gz
#     do
#         # bandpass filtering
#         3dTproject -prefix clean_$img -input $img -passband 0.01 0.1 >/dev/null
#         rm $img
#     done
# done
