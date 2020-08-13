# Lab Final - Visual cortex decoding (mind reading)

Image enhancement and segmentation using mathematical approaches. (deep learning techniques may produce better results)

## Step 1

**Implement a non-local denoising algorithm such as bilateral filtering, non-local means or a denoising autoencoder, estimate SNR and test your algorithm on the images provided. Do not call functions from libraries such as `scipy` or `dipy`.**

Write sth here.

Due to limited storage on my laptop, we have to use data from a single subject (30 GB) instead of the whole dataset (97 GB)

https://doi.org/10.1371/journal.pcbi.1006633



Feature extraction is used to reduce the dimension of the original data space to a new feature space

vanilla version

squashes it to range between 0 and 1, to a vector



**TLDR;**

**VGG19 as fixed feature extractor**

objects may be more suitably represented using mid- or high-level visual features which are invariant to such image differences, so we use the last conv layer



usually, we normalize across every individual *feature* in the data, that is, take the average of every single dimension across all training samples, so as to remove outliers in that dimension and center data around 0. However, here the fMRI signals are different, we must normalize along the time series. note that it makes no sense to normalize voxel-wisely across all runs since stimuli blocks vary in each run)

pixels/voxels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.



 noise unrelated to visual stimuli were effectively removed by averaging over also help to increase the SNR







slr does variable selection very much like how Lasso regression penalizes variables. the solution is sparse, meaning that many of the many coefficients would become 0. Here, the level of sparsity is controlled by explicitly choosing the number of bold voxels we would like keep, and slr will automatically select that many voxels with the highest correlation coefficients. If the number is too small, we are at the risk of considerable feature loss, if it's too large, the model is susceptible to underfitting since we only have 1,200 samples, meanwhile the millions of parameters can easily lead the model to explode with infinite computational cost.





tsv

```
.tsv
'onset': onset time of an event (sec)
'duration': duration of the event (sec)
'event_type': 1: Stimulus presentation block, 2: Repetition block, -1, -2, and -3: Initial, inter-trial, and post rest blocks without visual stimulus
'stimulus_id': stimulus ID of the image presented in a stimulus block ('n/a' in rest blocks)
'stimulus_name': stimulus file name of the image presented in a stimulus block ('n/a' in rest blocks)
```

```
shifting the data by 4 s (two volumes) to compensate for hemodynamic delays.
```

```
data
there are 4 types of sessions:
- natural images (training): 8 runs per session x 15 sessions = 120 runs in total, 55 stimulus blocks (5 randomly interspersed repetition blocks), duration per run 7 min 58 s
- natural images (test): 8 runs per session x 3 sessions = 24 runs in total, 55 stimulus blocks (5 randomly interspersed repetition blocks), duration per run 7 min 58 s
- artificial images: 10 runs per session x 2 sessions = 20 runs in total, 44 stimulus blocks (4 randomly interspersed repetition blocks), duration per run 6 min 30 s
- letter images: 12 runs per session x 1 sessions = 12 runs in total, 11 stimulus blocks (1 randomly interspersed repetition block), duration per run 5 min 2 s
```

<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>



## Step 2

**Implement a segmentation algorithm such as Otsu’s method, watershed transform, region growing, mean-shift clustering and graph-cut, or a neural network model. Test on the images provided and comment on the performance. Do not call functions from libraries such as `scipy` or `dipy`.**

Write sth here.

<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>

## Step 3

**Segment arteries from the TOF modality image, veins from the SWI modality image and visualize the result.**

Write sth here.

<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


<details>
<summary>View code</summary>

```python
sss
```
</details>


## Reference

[1] Shen G, Horikawa T, Majima K, Kamitani Y (2019) Deep image reconstruction from human brain activity. PLoS Comput Biol 15(1): e1006633. https://doi.org/10.1371/journal.pcbi.1006633

[2] VGG16 and VGG19 from Keras - https://keras.io/api/applications/vgg/

[3] Horikawa, T., Kamitani, Y. Generic decoding of seen and imagined objects using hierarchical visual features. _Nat Commun_**8,** 15037 (2017). https://doi.org/10.1038/ncomms15037

[4] Code for L-BFGS Deep Image Reconstruction @ [Kamitani Lab](https://github.com/KamitaniLab/DeepImageReconstruction)