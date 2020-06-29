# pre-process to clean up the BOLD fMRI images
# EPI distortion correction
# Rigid body motion correction
# Nuisance regression + bandpass filtering
# Spatial smoothing
# Alignment of BOLD to T1-weighted anatomy

chmod +x data/pipeline.sh
data/pipeline.sh
