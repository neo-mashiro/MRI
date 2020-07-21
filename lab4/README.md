# Lab 4

*Make sure you have [keras]() installed before running this lab*

Image enhancement and segmentation using both mathematical approaches and deep learning techniques.

## Part 1: image enhancement - denoising

**Implement a non-local denoising algorithm such as bilateral filtering, non-local means or a denoising autoencoder, estimate SNR and test your algorithm on the images provided. Do not call functions from libraries such as `scipy` or `dipy`.**

Write something here.

![img](images/t1.png)

![img](images/t2.png)

Write something here.

![img](images/11.png)

![img](images/12.png)

![img](images/13.png)

![img](images/14.png)

![img](images/15.png)

Write something here.

## Part 2: image segmentation

**Implement a segmentation algorithm such as Otsu’s method, watershed transform, region growing, mean-shift clustering and graph-cut, or a neural network model. Test on the images provided and comment on the performance. Do not call functions from libraries such as `scipy` or `dipy`.**

Write sth here.

**1** - Create an ideal time series that represents how the brain should react to the stimulus

<details>
<summary>View code</summary>

```python
# code snippet
```
</details>

## Part 3: vascular segmentation

**Segment arteries from the tof image, veins from the swi image and visualize the result.**

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

