# Lab 4

Image enhancement and segmentation using mathematical approaches. (adopting deep learning techniques may produce better results)

## Part 1: image enhancement - denoising

**Implement a non-local denoising algorithm such as bilateral filtering, non-local means or a denoising autoencoder, estimate SNR and test your algorithm on the images provided. Do not call functions from libraries such as `scipy` or `dipy`.**

Given the 5 MRI brain images, we extract a square patch from both the background and the within the brain. While the patch inside the brain area is a signal patch, the patch in the background does not receive MRI signals so it can be viewed as a noise patch. We can roughly estimate the signal-to-noise ratio by taking the division of the two patches, where numerator is the mean of the signal and denominator is the standard deviation of the noise. Depending on where we extract the patches, the estimate is going to vary a bit, but overall this gives us a reasonable measure of SNR.

In the figure below, each noise patch is highlighted in red and signal patch in green, zoomed in to show the details inside the patch. As we can see, `t1_v2.png` only contains little noise so SNR is high (766.75), `flair.png` contains much more noise so SNR is relatively low (183.52).

![img](images/11.png)

![img](images/12.png)

![img](images/13.png)

![img](images/14.png)

![img](images/15.png)

The other three images have a perfectly clean noise patch in the background, so the standard deviation is constant 0 (that's why colorbar fails), yielding a SNR of infinity, which is a little bit counterintuitive. In fact, in this case, the above estimate approach does not work, we cannot simply take a square from the background as the noise patch. Instead, we must zoom in the brain to pick a meaningful noise patch. Ideally, such a patch should be taken from a large flat tissue where all pixels have almost identical intensity values, so that most variations within the patch are attributed to noise. However, it may be difficult to choose such a flat patch, we are likely to end up with a patch being too small in size, which comes at the price of losing information.

---

Image enhancement aims to improve the visual quality of a digital image, which can be measured in metrics such as MSE, SNR or PSNR. Here to denoise the image, we implement the non-local means algorithm as a nonlinear filter. As a unit test, we compared our algorithm to the library function from `skimage`, the result is shown below. Our denoised image is less cleaner, with some noise blurred rather than completely eliminated, and some features removed as well, but the overall performance is acceptable.

![img](images/test1.png)

![img](images/test2.png)

Below is the code snippet of our implementation. Here we are using a 3x3 small patch and a 5x5 search window. According to the experimental studies in [this paper](http://web.csulb.edu/~ssodagar/non%20local%20means.pdf), these parameters tend to produce the optimum results. In brief, we simply iterate over each pixel in the original image, decide the search window position, then for each pixel in that window, we compute the weight based on the patch distance between the two pixels, and finally update the original pixel value as a weighted sum of all pixels within its search window.

<details>
<summary>View code</summary>

```python
@ExecutionTimer
def nl_means_filter(im, patch_size=3, window_size=5, h=0.6, sigma=1.0, kernel='Oracle'):
    """Denoise the input image using non-local means filter
       h:      parameter that controls the degree of filtering
       sigma:  standard deviation of the Gaussian noise in the image
    """
    # use a symmetric patch and search window whose size is odd
    if patch_size % 2 == 0:
        patch_size += 1
    if window_size % 2 == 0:
        window_size += 1

    n_row, n_col = im.shape
    offset = patch_size // 2  # offset from the patch center

    new_im = np.zeros_like(im)  # initialize the denoised image
    pad_im = np.pad(im, ((offset, offset), (offset, offset)), mode='reflect')  # pad the image

    # compute the patch filter (the weighted matrix defined by the kernel)
    patch_range = np.arange(-offset, offset + 1)
    x, y = np.meshgrid(patch_range, patch_range, indexing='ij')

    if kernel == 'Oracle':
        filt = oracle_kernel(x, y)
    else:
        filt = gaussian_kernel(x, y, sigma)

    # iterate over each pixel in the original image
    for row in range(n_row):
        u = row - min(window_size, row)          # search window top row (up)
        d = row + min(window_size, n_row - row)  # search window bottom row + 1 (down)

        for col in range(n_col):
            l = col - min(window_size, col)          # search window leftmost column
            r = col + min(window_size, n_col - col)  # search window rightmost + 1 column

            p1 = pad_im[row:row+patch_size, col:col+patch_size]

            Z = 0.0
            pixel_value = 0.0

            # iterate over every other pixel in the search window
            for i in range(u, d):
                for j in range(l, r):
                    p2 = pad_im[i:i+patch_size, j:j+patch_size]

                    # compute distance and weight
                    distance = patch_distance(p1, p2, filt)
                    power = -distance / (h ** 2)
                    if power < -5:  # exp cutoff
                        weight = 0  # exp of a large negative number is close to 0
                    else:
                        weight = np.exp(power)

                    Z += weight  # the normalization term (sum of weight)
                    pixel_value += weight * im[i, j]

            # normalize the result
            if Z == 0:  # this happens only when p1 = every p2 in the search window (probably in the background)
                new_im[row, col] = im[row, col]
            else:
                new_im[row, col] = pixel_value / Z

    return new_im
```
</details>

In concrete, the patch distance is a weighted sum of all Euclidean distances between two corresponding pixels, this is exactly the essence of non-local means, whose idea is to weigh each pixel in the patch differently, taking into account the nonlinear relationship between nearby pixels.

<details>
<summary>View code</summary>

```python
def patch_distance(p1, p2, kernel):
    """Compute the sum of kernel-weighted Euclidean distances between two patches"""
    assert p1.shape == p2.shape, "Patch sizes are not equal ..."
    assert p1.shape[0] == p1.shape[1], "Input patch must be a square matrix ..."
    assert p1.shape[0] % 2 != 0, "Patch size must be odd ..."
    diff = (p1 - p2) ** 2
    return np.sum(np.multiply(kernel, diff))
```
</details>

It is worth mentioning that we are not using the default Gaussian kernel in this implementation. Instead, we used an Oracle kernel as the weight filter and it produced better results, as mentioned in [this paper](https://hal.inria.fr/hal-01575918/document) (p.7). In this case, the estimated pixel value is updated by the formula below, which is very similar to the original non-local means formula.

![img](images/f26.png)

The Oracle kernel profile is defined as:

![img](images/f27.png)

which has the shape of a bell-like pyramid:

<details>
<summary>View code</summary>

```python
def oracle_kernel(x, y):
    """Compute the filter matrix of meshgrid x and y given the Oracle kernel"""
    offset = x.shape[0] // 2
    dist = np.maximum(np.abs(x), np.abs(y))  # orthogonal distance from the patch center
    dist[offset, offset] = 1  # patch center has the same weight as pixels with distance = 1
    filt = np.zeros_like(x).astype('float64')
    for d in range(offset, 0, -1):
        mask = (dist <= d)
        filt[mask] += (1 / (2 * d + 1) ** 2)

    return filt * (1 / offset)
```
</details>

![img](images/oracle.png)

Applying our non-local means on the provided noisy brain images, we obtain the following result. If we zoom in, we can clearly see that the denoised images are much better in terms of visual perceptibility.

![img](images/111.png)

![img](images/112.png)

![img](images/113.png)

![img](images/114.png)

![img](images/115.png)

## Part 2: image segmentation

**Implement a segmentation algorithm such as Otsu’s method, watershed transform, region growing, mean-shift clustering and graph-cut, or a neural network model. Test on the images provided and comment on the performance. Do not call functions from libraries such as `scipy` or `dipy`.**

In this part, we implemented the mean-shift clustering algorithm for the purpose of image segmentation. This is a nonparametric clustering technique that does not require prior knowledge of the number of clusters, so the algorithm is highly automatic. In contrast, k-means usually needs to predefine the number of clusters using the elbow method, watershed requires a number of manually labelled markers to achieve best result. Here's a snapshot of our code.

<details>
<summary>View code</summary>

```python
@ExecutionTimer
def mean_shift(im, bandwidth):
    n_row, n_col = im.shape
    pixels = np.ravel(im).astype('float64')
    clusters = np.copy(pixels)
    var = bandwidth ** 2

    # vectorize to speed up exponent computation
    fast_exp = lambda x: np.exp(x) if x > -5 else 0.0
    vec_exp = np.vectorize(fast_exp)

    # obtain (flattened) indices of neighbor pixels given a flattened index
    def neighbor_indices(index, offset):
        row, col = index // n_col, index % n_col
        u = max(row - offset, 0)          # neighbor window top row
        d = min(row + offset, n_row - 1)  # neighbor window bottom row
        l = max(col - offset, 0)          # neighbor window leftmost column
        r = min(col + offset, n_col - 1)  # neighbor window rightmost column
        x, y = np.meshgrid(np.arange(u, d + 1), np.arange(l, r + 1), indexing='ij')
        return np.ravel(x * n_col + y)

    # compute shift vector for each pixel
    for i in range(len(pixels)):
        pixel = clusters[i].copy()  # initial value
        neighbor = neighbor_indices(i, 10)  # speed up by choosing a window, window size tunable

        # shift pixel until a cluster peak is hit
        while True:
            neighbor_pixel = pixels[neighbor]
            diff = (pixel - neighbor_pixel) ** 2
            power = -diff / (2 * var)
            prob = vec_exp(power)
            prob *= (1 / np.sum(prob))  # normalize

            center = np.sum(np.multiply(neighbor_pixel, prob))  # compute window center
            update = np.abs(center - pixel)
            pixel = center  # shift pixel
            # neighbor = ...  # should shift window as well, but ignored for now

            if update < CONVERGE_THRESHOLD:
                clusters[i] = pixel  # peak of the i-th pixel
                break

    clusters = (clusters / CLUSTER_THRESHOLD).astype(int) * CLUSTER_THRESHOLD  # merge similar clusters
    clusters = np.minimum(clusters + 30, 254) % 255  # for better visualization
    return clusters.reshape((n_row, n_col))
```
</details>

In brief, we simply iterate over each pixel in the image, compute its mean shift vector by looking at all other pixels, and then shift (update) the pixel until converge. The mean shift vector always points toward the direction of the maximum increase in the probability density function, so it is guaranteed to converge to a point where the gradient of density is close to zero.

The biggest bottleneck of our algorithm is the running time. When it comes to image segmentation, in our case we want to group pixels into a few clusters based on intensity similarity, so the feature space is only 1-dimensional (the grayscale intensity), but the number of pixels is very large. For example, a 512x512 grayscale image has 262,144 pixels, which would require trillions of pixel-level computations, or hundreds of hours of execution time. This makes scalability a serious issue especially for large images.

To accelerate our algorithm, we have adopted a moving window of reasonable radius. Therefore, in each iteration, we compute the "mass" center of that window, shift the pixel towards the center, and then shift the window as well. A small window size would drastically reduce the running time, but at the cost of introducing segmentation errors. With this improvement, our algorithm takes about 2 minutes to run on the images provided. Another speedup strategy is to cache the cluster of pixels in the basin of attraction, so that once we reached a pixel in the basin that has already been visited, we can break out of the inner loop and assign it to the previously computed cluster, but this is difficult and I haven't yet figured out how.

Again as a unit test, we compared our algorithm to the library function from `skimage`:

![img](images/house1.png)

![img](images/house2.png)

The figures below show the segmentation on the provided images using our naive implementation.

![img](images/21.png)

![img](images/22.png)

![img](images/23.png)

![img](images/24.png)

![img](images/25.png)

## Part 3: vascular segmentation

**Segment arteries from the TOF modality image, veins from the SWI modality image and visualize the result.**

[Imeka](https://www.imeka.ca/mi-brain/)


**1. Pre-process.**  
Before the segmentation, some pre-process steps are necessary, including denoising and brain extraction. Hence, we processed the source image with
nlmeans to remove noise and extract brain from skull using 'bet' command.


**2. Segment veins and arteries**  
In this part, we searched and studied a lot of tutorials and resources about 3d image segmentation. We found that U-Net
is a great method to deal with 3d medical image segmentation. But considering none of us has experience in deep learning, Keras and TensorFlow 
also need much time on studying and environment setup, we finally choose another method to solve this problem.  

The main procedures are as below:  
● Slice the source 3d image into 2d images(array).  
● For each 2d array, segment the vessel with kmeans(from opencv).  
● Restrict the target area with a mask.  
● Figure out the target cluster by comparing the pixel amounts of all clusters.   
● Set all target pixels value 1(white) and other pixels value 0(black).  
● Combine all slices into a 3d image and save that image.   

Image array:
<img src="http://15.222.11.163/wp-content/uploads/2020/07/5b4cd8bff4adbf54dee648ddd8f4c53-1024x596.png"></div>  
 
A single slice of clusted vein image:    
<img src="http://15.222.11.163/wp-content/uploads/2020/07/slice.png"></div>  
  
Result of the combination(.nii):  
<img src="http://15.222.11.163/wp-content/uploads/2020/07/result-1024x330.png"></div>  

<details>
<summary>View code</summary> 

```python
    # num_mask: amount of mask pixels in the image
    # num_tissue: amount of tissue detected
    num_mask = 0
    num_tissue = 0
    # this variable indicates the number of clusters for kmeans
    tof_cluster = 4
    # A list record the amount of pixels in each cluster
    label_list = [0 for i in range(tof_cluster)]
    # slice the image and mask according to isTof
    img = image[:, :, slice] if isTof else image[slice, :, :]
    mask = msk[:, :, slice] if isTof else msk[slice, :, :]
    # check if image and mask have the same shape
    if image.shape != msk.shape:
        print("error : shape of image not consistent with mask")
        return
    rows, cols = img.shape
    size = rows * cols
    # reshape the data to 1d array
    data_img = img.reshape((size, 1))
    data_mask = mask.reshape((size, 1))
    # format transform for kmeans function
    data_img = np.float32(data_img)
    # parameters: type, max iteration times, accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # initialize random centers
    flag = cv2.KMEANS_RANDOM_CENTERS
    # cluster amount for kmeans, for the swi.nii, 2 is enough, for tof.nii, 4 lead to better performance
    cluster = tof_cluster if isTof else 2
    # Kmeans from opencv library
    compactness, labels, centers = cv2.kmeans(data_img, cluster, None, criteria, 20, flag)
    # loop all pixels in the slice
    for i in range(size):
        # mask[i] == 1 means the certain pixel is in the brain area
        if data_mask[i] == 1:
            num_mask += 1
            # if isTof, figure out the amount of pixels in each cluster
            if isTof:
                label_list[int(labels[i])] += 1
            # if not, figure out the amount of tissue pixels
            if labels[i] == 1:
                num_tissue += 1
        else:
            # Set every piexl outside the brain area,.
            # if isTof, to 1 (0 lead to unknown error when visualized in imeka); if not, to 0
            labels[i] = 1 if isTof else 0
    print("slice:", slice, ".... mask:", num_mask, ".... white:", num_tissue)
    print("labels:", label_list)
    if isTof:
        # index indicates the cluster which has min pixel amount
        index = label_list.index(min(label_list))
        # set tissue pixels white
        for i in range(size):
            if data_mask[i] == 1:
                labels[i] = 0 if labels[i] == index else 1
    else:
        # for a normal image, if the color of tissue pixels is black, invert the color of tissue and non-tissue pixels
        if num_tissue > (num_mask * 0.3):
            print("Inverting...")
            for i in range(size):
                if data_mask[i] == 1:
                    if labels[i] == 1:
                        labels[i] = 0
                    else:
                        labels[i] = 1
        # check the result of invertion
            num_mask = num_tissue = 0
            for i in range(size):
                if data_mask[i] == 1:
                    num_mask += 1
                    num_tissue += 1 if labels[i] == 1 else 0
            print("Inverted slice:", slice, ".... mask:", num_mask, ".... white:", num_tissue)
    # reshape the array
    result = labels.reshape((img.shape[0], img.shape[1]))
```  
</details>

**3d Result:**  
Veins:
<img src="http://15.222.11.163/wp-content/uploads/2020/07/swi2-1024x576.png" ></div>  
  
Arteries:
<img src="http://15.222.11.163/wp-content/uploads/2020/07/tof1-1024x576.png"></div>  


## Reference

[1] Alireza Nasiri Avanaki, Abolfazl Diyanat, Shabnam Sodagari, 2008. "Optimum parameter estimation for non-local means image de-noising using corner information". In ICSP2008 Proceedings, pp. 861-863.

[2] Qiyu Jin, Ion Grama, Charles Kervrann, Quansheng Liu. "Non-local means and optimal weights for noise removal". SIAM Journal on Imaging Sciences, Society for Industrial and Applied Mathematics, 2017. hal-01575918
