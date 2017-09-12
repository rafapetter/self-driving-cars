
## Writeup - Finding Lane Lines on the Road

### 1. Pipeline

Let's first start by listing the steps I took for creating the pipeline:

1. We need to convert the image to grayscale. The reason is that it will be easier to contrast the lines on the image.

2. And to even get a better contrast, we're blurring the image. This will smooth out the overall numbers in the image matrix, helping the lines to stand out, and getting rid of noisy parts of the image. We're using a kernel_size of 3, which is the amount of blur to be applied on the image.

3. Then to help us detect the edges, we're applying the Canny Transform. It will take two threshold values to determine what are acceptable edges. For this project we're setting the threshold between 70 and 150.

4. After getting a list of edges found on the previous step, we now need to exclude edges that aren't important in our pipeline, so we'll have to mask out. The mask will only focus on the bottom part of the road view.

5. Ok, now we're getting into some interesting work, we run the Hough Transform algorithm to find lines on the image. It will convert the edges into meaningful lines. It considers how straight should a line be and what is the minimum length of the lines. It can also connect consecutive lines, as long as we specify the maximum gap allowed. That will help to group relative similar lines into a single detected lane.

6. Now, for a more optimized and consistent lane detection we're going to have to average and extrapolate how those lines are detected and shown over the image. Here's some brief overview on how to achieve that: a) calculate the slope of the lines; b) choose the highest and lowest slopes; c) iterate through the lines and choose only the lines that are within 80% of the highest and lowest slopes, separating lines in 2 groups: lower slope lines go to left and higher slope lines go to right; d) now we calculate the average of the lines, to extrapolate from the highest point to the lowest edge; e) with that, we can finally draw our lanes on both right and left.

7. The last step is to overlay our line detection with the original image/video.

![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Project-1-Lane-Detection/test_images_output/lane_detection.png)

Challenge Video: https://github.com/rafapetter/self-driving-cars/blob/master/Project-1-Lane-Detection/test_videos_output/challenge.mp4


### 2. Shortcomings

The line detection fails when there's more lines than usual on the road. Other variables and classification methods need to be used here, besides the steepest of the slope. Maybe restrict a range that line slopes can have.

Also, when testing in the videos, sometimes the pipeline lanes change abruptly, especially in the challenge video. Maybe a smooth lane slope variation would solve this.


### 3. Improvements

Consider increasing the grayscale contrast so it's easier to detect the lines.

If we start testing on curves it would definitely need to consider more parameters when detecting lanes, specially on the road upper view from our region of interest. I guess we would have to start thinking about no-linear solutions to this problem.
