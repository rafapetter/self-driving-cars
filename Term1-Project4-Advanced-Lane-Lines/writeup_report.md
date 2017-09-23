## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/boards_calibrated.png "Boards Calibrated"
[image2]: ./output_images/undist_img.jpg "Undistorted Images"
[image3]: ./output_images/yellow_line_detection.png "Yellow Line Detection"
[image4]: ./output_images/binary_channel.png "Binary Channel"
[image5]: ./output_images/undistorted_straight_lines.png "Undistorted Straight Lines"
[image6]: ./output_images/lane_polynomial.png "Lane Polynomial"
[image7]: ./output_images/lane_plotting.png "Lane Plotting"
[video]: ./video_output.mp4 "Video Output"

---

### Camera Calibration

### 1. Carmera Calibration

#### 1.1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Extract object and image points for camera calibration.

At first, we prepare object points, that are x, y and z points in the real world of the chessboard corners. I assume that they're all on the z plane.

Now we loop through each of the images converting them to greyscale.

Then we use cv2.findChessboardCorners to find the image coordinates of the object points and finally append both to a copy of them.

Finally, we used cv2.calibrateCamera to get the camera matrix and distortion coefficients.

![alt text][image1]

### 2. Pipeline (single images)

#### 2.1. Provide an example of a distortion-corrected image.

Applying the undistortion transformation to a test image yields the following result (left distorted, right corrected):
![alt text][image2]

#### 2.2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

We've performed Histogram normalization, binary Sobel transformation(magnitude and direction), red color channel and saturation channel.

An adaptive threshold was applied on a red channel image for white line finding and on a linear combination of the red and saturation channels for the yellow line. An example of image used for yellow line detection is shown below:

![alt text][image3]

And the binary and channel versions of the image above looks as follows:

![alt text][image4]

#### 2.3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

We'll be getting an image with straight lane lines, to draw a few points to define an isoceles trapezoid where the left and right lines overlay the lane lines. The image I used was test_images/test3.jpg.

The lane lines are supposed to be straight where I created dst points that form a rectangle. I added an offset from the left and right side that leaves room for curving lines. Then I get the transformation using these points with cv2.getPerspectiveTransform, as well as the inverse which will be used for transforming back from top-down view to our normal view.

The warp function uses the transformation matrix to transform the image from our normal view to the top-down view.

![alt text][image5]

#### 2.4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

- First step is to get a histogram of the lower half binary warped image. Which is used to identify where to start looking for lane lines.

- Now we look for parts of the lane lines starting from the bottom.

- Then, we use np.polyfit to fit a 2nd order polynomial to the points we identified as being part of lane lines.

- And finally, we collect the data created in this method and return it.

![alt text][image6]

#### 2.5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For curve radius I first get new coefficients for curvature in the real world, not pixel space. Then calculate the radius with the new polynomial.

#### 2.6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in functions `fit_lanes` and `visualize_fit_lanes`.  Here is an example of my result on a test image:

![alt text][image7]

---

### 3. Pipeline (video)

#### 3.1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Final video output is saved at ![Video Output][video]

---

### 4. Discussion

#### 4.1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When taking the harder_challenge_video.mp4 video I've found a few issues related to:
- the similarity of colors from the road and lane lines,
- shadows over the lines.

Reprojecting into birds-eye view and fitting lane lines by polynomial is promising method which may find not only straight lane lines but also curved ones. But it is not robust enough to deal with complex environment with tree shadows, road defects, brightness/contrast issues. It will be effective on environments where lane lines are bright, contrast, not occluded or overlapped.

Changing the thresholding might be a solution. The video filtering algorithm can be improved in order to make it more robust on the harder_challenge_video.mp4 and other real-world videos.

Also if you can see one line clearly you know the rough estimate for the other, right now my approach identifies one line separately, having information about the other line as well, and assigning a probability to how certain you are this line is correct might help.
