# Lab Final - Visual cortex decoding (mind reading)

Image enhancement and segmentation using mathematical approaches. (deep learning techniques may produce better results)

## Step 1

**Implement a non-local denoising algorithm such as bilateral filtering, non-local means or a denoising autoencoder, estimate SNR and test your algorithm on the images provided. Do not call functions from libraries such as `scipy` or `dipy`.**

Write sth here.

Due to limited storage on my laptop, we have to use data from a single subject (30 GB) instead of the whole dataset (97 GB)

https://doi.org/10.1371/journal.pcbi.1006633





vanilla version

squashes it to range between 0 and 1, to a vector

thresholded at zero

**TLDR;**

**ConvNet as fixed feature extractor**

https://cs231n.github.io/transfer-learning/

resize input bold image shapes before feeding to VGG:

https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/

https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

usually, we normalize across every individual *feature* in the data, that is, take the average of every single dimension across all training samples, so as to remove outliers in that dimension and center data around 0. However, here the fMRI signals are different, we must normalize along the time series. note that it makes no sense to normalize voxel-wisely across all runs since stimuli blocks vary in each run)

normalize data along the time dimension:

https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/programs/3dTstat_sphx.html

pixels/voxels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.



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