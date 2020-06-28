# must have both 'afni' and 'fsl' software packages installed to run

# deoblique to get rid of warnings
3dWarp -deoblique -prefix bold.nii.gz -overwrite bold.nii.gz
# despike - remove large spikes in time series
3dDespike -prefix despike.nii.gz bold.nii.gz
# motion correction - align all volumes
3dvolreg -prefix volreg.nii.gz -1Dfile motion_params.1D bold.nii.gz
# Brain Extraction Tool (bet) - removes skull
bet t1.nii.gz bet.nii.gz -f 0.3
# FMRIBs Automated Segmentation Tool (fast) - segment tissue types
fast -g -o fast bet.nii.gz
3dTstat -prefix mean_bold.nii.gz volreg.nii.gz # get mean of BOLD (for epi_reg)
# epi_reg - does the multimodal registration between BOLD image and T1 (uses white matter from 'fast')
epi_reg --epi=mean_bold.nii.gz --t1=t1.nii.gz --t1brain=bet.nii.gz --out=epireg --wmseg=fast_seg_2.nii.gz
# invert the bold->T1 matrix (so can bring white matter into BOLD space
convert_xfm -inverse -omat t1_2_epi.m epireg.mat
# bring white matter (from fast) into BOLD space:
flirt -in fast_seg_2.nii.gz -ref mean_bold.nii.gz -applyxfm -init t1_2_epi.m -interp nearestneighbour -out white_matter_in_bold.nii.gz
# get the average BOLD signal in the white matter mask:
3dmaskave -quiet -mask white_matter_in_bold.nii.gz volreg.nii.gz > white_matter_signal.1D
# create a mask over the entire brain (for input to 3dTproject below)
3dAutomask -prefix mask.nii.gz mean_bold.nii.gz
# final step: regress out nuisance (white matter signal, motion) and do bandpass filtering + blurring
3dTproject -prefix clean_bold.nii.gz -input volreg.nii.gz -ort white_matter_signal.1D -ort motion_params.1D -passband 0.01 0.1 -mask mask.nii.gz -blur 4

