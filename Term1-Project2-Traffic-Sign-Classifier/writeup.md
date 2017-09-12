### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

The size of training set is 34799.
The size of the validation set is 4410.
The size of test set is 12630.
The shape of traffic sign image is square 32 x 32 x 3.
The number of classes/labels in the set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset. A bar chart showing the occurrences of each label on each dataset.
We can notice that different signs are not uniformly distributed.

![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/exploratory_1.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it helps isolate different visual attributes. Think about the lane line project, we converted to gray scale and then used Canny edge detection to find the gradient at different lines in the image.

Conversion of images to grayscale using openCV to allow the model to train on traffic sign features Image histogram equalization using the openCV CLAHE (Constrast-Limited Adaptive Histogram Equalization) algorithm to improve the illumination in the images. The poor brightness across some images can be seen in the data set visualization presented earlier in Section 2.2 Conversely, there are images included in the dataset that are over-exposed and also need to be normalized. This equalization is carried out on both the RGB images and grayscale images independently.

Then, all the data is normalized to zero mean and equal variance to be ready for deep learning. X_train = X_train / 255.0-0.5 X_valid = X_valid / 255.0-0.5 X_test = X_test / 255.0-0.5

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The LeNet Lab Solution was the starting part for the traffic detection pipeline I created. Using Tensor Flow, Google's machine learning library to speed up development time. The Input was the german data set provided for free, the data set was already cleaned to only contain 32x32 pixel photos, saving the students from the many hours it takes to clean such a large data set. Data manipulation and conditioning can often be the most difficult part of creating a numeral network pipeline. While this project only required minor data to be loaded from a very clean data set, minor variation where added to go from the RGB color scale to a gray scale to better match the LeNet architecture.

The LeNet network was modified to gain accuracy for sign detection. The input of the network takes 32x32 photos from a data set and create hidden layers to analyze the image for features. Using convolution a mathematical principle to modify the pixels to add more layers and compare changes. Understanding that color images are not the best to analyze because each image has 3 layers, think (32X32)x3 equals 3,702 total pixel the script will have to process for every image. By simplifying the data set to gray scale we can flatten in a senses the color data to a grayscale value between 0 and 255, if the pixel are 8 bits. This input becomes important to the feature map layer where we go from 3 layers of 32x32 to 6 layers of 28x28 pixels. Again we samples the data going form 6 layer to another 6 layers of 14X14 pixel. We do a similar process two more times going to 16 layers at 10x10 and 16 layers going to 5x5 pixel. Ending with 3 passes of a connected layers to output the final answer output.

My final model consisted of the following layers:

|Layer	| Description |
|:---------------------:|:---------------------------------------------:|
|Input	| 32x32x1 gray image|
|Convolution L1 5x5	| 1x1 stride, valid padding, outputs 28x28x6 |
|RELUActivation	| 	 |
|Max pooling|	2x2 stride, Output = 14x14x6 |
|Convolution L2 5x5 |	1x1 stride, valid padding, outputs 10x10x16 |
|RELU Activation | 	  |
|Max pooling	2x2 | stride, valid padding, outputs 5x5x16 |
|Fully Connected Layer L3	| Output = 120 |
|RELU Activation |	|
|Dropout	| keep_prob 0.7|
|Fully Connected Layer L4 |	Output = 84  |
|RELU Activation |	|
|Dropout	| keep_prob 0.7|
|Fully Connected Layer5 | Output = 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The AdamOptimizer was applied. The parameters were: epochs = 20, batch_size = 128, learning rate = 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

After the data is normalized and the current parameters are set, the training accuracy gets closer to 1,
but the validation accuracy kept getting lower than expected, which might mean the model was over-fitting.

Then I added dropout units after two fully connected layers with keep_prob = 0.5, but both training and validation accuracies were low, which means the model might be under-fitting.

Lastly I increased keep_prob to 0.7, to finally get the validation accuracy to over 0.93.

The final model results:
* training set accuracy of 0.995
* validation set accuracy of 0.948
* test set accuracy of 0.920

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the six German traffic signs that I found on the web. It's already processed to be 32x32x3 dimensions.

![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/web-signs/1.png)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/web-signs/2.png)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/web-signs/3.png)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/web-signs/4.png)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/web-signs/5.png)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/web-signs/6.png)

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. Comparing not so favorably to the accuracy on the test set of 92.0%.

The granularity of having only 6 traffic signs is 16%, so perhaps by having more traffic signs from the web, the test set accuracy of 92% would be represented

The third image (No vehicles) might be difficult to classify because there's noting inside, so the model must keep trying to match with something, since most of the signs with this shape has something inside.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image, the model is completely sure that this is a "Right-of-way at the next intersection" sign (probability of 1.0).
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/OutputSoftmax_1.png)

For the second image, the model is also completely sure that this is a "Ahead only" sign (probability of 1.0)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/OutputSoftmax_2.png)

For the third image, the model is struggling to figure it out the "No vehicles" sign (probability of 0.58 for the first guess).
Looking into the other guesses, it does get the circle shape right. The problem is what is inside, or even what it's not inside.
Interesting to notice that on the fifth guess it gets right, but at only a 2% probability.
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/OutputSoftmax_3.png)

For the fourth image, the model is completely sure that this is a "Go straight or left" sign (probability of 1.0)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/OutputSoftmax_4.png)

For the fifth image, the model is completely sure that this is a "General caution" sign (probability of 1.0)
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/OutputSoftmax_5.png)

For the sixth image, the model is quite sure that this is a "Road work" sign (probability of 0.87), though it has some indecisions as we notice
the other guesses probabilities. It might be because the drawing inside the triangle is a bit more complex than, for instance, the previous triangle sign.
![alt text](https://raw.githubusercontent.com/rafapetter/self-driving-cars/master/Term1-Project2-Traffic-Sign-Classifier/project-images/OutputSoftmax_6.png)
