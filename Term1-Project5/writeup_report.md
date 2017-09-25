## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_notcar_hog.jpg
[image3]: ./output_images/detect_car.jpg
[image4]: ./output_images/heatmap.jpg
[image5]: ./output_images/vehicle_lane_detection.png
[video]: ./project_video_output.mp4

---
### 1. Histogram of Oriented Gradients (HOG)

#### 1.1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I used the functions from the lesson for features extraction (HOG, binned color and color histogram features). These functions are defined at lesson_functions.py

The code for this step uses the **lesson_functions.py** and an visualizing example is contained in the next code cell.

It's using the RGB color space and the following HOG parameters:
- orientations = 9
- pixels_per_cell = (8, 8)
- cells_per_block = (2, 2)

I then explored different color spaces and different **get_hog_featuers()** parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the **get_hog_featuers()** output looks like.

Here's an example using these HOG parameters:

![alt text][image2]

#### 1.2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and results shows that these parameter performs the best.

- color_space = 'YCrCb'
- spatial_size = (16, 16)
- hist_bins = 32
- orient = 9
- pix_per_cell = 8
- cell_per_block = 2
- hog_channel = 'ALL'
- spatial_feat = True
- hist_feat = True
- hog_feat = True

#### 1.3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I've trained a linear SVM with the default classifier parameters. Dataset was normalized and shuffled before training (see the next 2 code cells below). Using HOG features alone I was able to achieve a test accuracy of 98.47%.

## 2. Sliding Window Search

#### 2.1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I've adapted the sliding window function **find_cars**, from the lessons, to extract features using Hog Sub-sampling Window Search, and to also make some predictions. Later on we generate a list of boxes with predefined parameters and draw the list of boxes on an image. Again, these later functions are from the Udacity's lesson.

Searching area is limited to y position (400, 656), 3 windows scales (small=0.8, medium=1.5, and large=2.3) are used. Instead of overlap, 2 cells per step are used.

All above numbers are settled after testing with different numbers on test image to reach out the best detection.

#### 2.2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

As described above, I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here is an example image:

![alt text][image3]

---

## 3. Video Implementation

#### 3.1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

For the video pipeline, I used a **Window_History** class, that will hold detected hot windows from the past 20 frames. Then a heatmap threshold and draw label boxes were implemented on top of the "history" instance. This process will make the detector more robust and stable. Here's a [link to the output video][video]

#### 3.2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

On a single frame, heatmaps are created by adding up the detected car box.

Then thresholds are set to filter out false positives.

Final box were drew by function **draw_labeled_bboxes**. These code are in cell below and followed by an example of heatmap threshold and final draw box:

![alt text][image4]

Here we load a function from the Project 4 **Advanced Lane Lines** to draw the lane lines along with vehicle detectation.

![alt text][image5]

---

## 4. Discussion

#### 4.1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The final result is satisfied. Most of the problems were mainly concerned about detection accuracy. Balancing the accuracy of the classifier with execution speed was crucial. Here's a few pending challenges:

- When the car is far away, a smaller sampling rate has to be used but it will also bring in more false positives.
- Bounding box not always closely followed the the car image, there were missing detection on a few frames.

Given plenty of time to pursue it, a good approach would be to combine a very high accuracy classifier with high overlap in the search windows. The execution cost could be offset with more intelligent tracking strategies, such as:

- Determine vehicle location and speed to predict its location in subsequent frames
- Begin with expected vehicle locations and nearest (largest scale) search areas, and preclude overlap and redundant detections from smaller scale search areas to speed up execution
- Use a convolutional neural network (like YOLO or SSD), to preclude the sliding window search altogether
