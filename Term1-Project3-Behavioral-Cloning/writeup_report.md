#**Behavioral Cloning**

##Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./resources/center_camera.jpg "Center Camera"
[image2]: ./resources/recovery_1.jpg "Recovery - far"
[image3]: ./resources/recovery_2.jpg "Recovery - mid"
[image4]: ./resources/recovery_3.jpg "Recovery - center"
[image5]: ./resources/normal_image.jpg "Normal"
[image6]: ./resources/flipped_image.jpg "Flipped"
[video]: ./resources/run1.mp4 "First Track"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the nVidia CNN used for End-to-End driverless car. It consists of 5 convolutional layers where the first 3 layers use 2x2 pooling and 5x5 filters with outputs ranging from 24 to 48 and the last 2 use a 3x3 filter with both outputs of 64

The model includes RELU layers to introduce nonlinearity on all 5 convolutional layers, and the data is normalized in the model using a Keras lambda layer.

As a pre-processing task we have used cropping to remove top 70 and bottom 25 from the images.

####2. Attempts to reduce overfitting in the model

The model fully convolutional layers contain dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. It was also tested by running it through the simulator and ensuring that the vehicle would not go off track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track in reverse as well as recordings from both tracks.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I first used a convolution neural network model similar to LeNet, because it had worked well for the "Traffic Sign Classifier" project.

To measure how well the model was working, I have separated my image and steering angle data into training and validation sets. My first model got a low mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was overfitting.

To handle overfitting, I've added dropout layers to the model with a dropout rate of 0.5. Which reduced the overfitting.

Then I've ran the model over in the simulator to see how well the car was driving around track one. But there were a few spots where the vehicle was going off the track. I tried changing the parameters to improve the driving behavior but none solved the problem, it kept going off the track.

So I decided to experiment another model, based on to the nVidia model. The nVidia model is appropriate because it uses multiple convolutional layers which is relevant in our case because we have images as inputs. Following the convolutional layers are multiple fully connected layers which is the standard approach and it ends with a single floating point value which is all that what we need to model our steering angle.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network similar to the model developed by nVidia which was presented in lecture 14 with the following layers and layer sizes:

- convolutional layer with a 5x5 kernel and a depth of 24
- convolutional layer with a 5x5 kernel and a depth of 36
- convolutional layer with a 5x5 kernel and a depth of 48
- convolutional layer with a 3x3 kernel and a depth of 64
- convolutional layer with a 3x3 kernel and a depth of 64
- dropout layer at rate of 0.5
- dense layer with a size of 100
- dropout layer at rate of 0.5
- dense layer with a size of 50
- dropout layer at rate of 0.5
- dense layer with a size of 10
- dropout layer at rate of 0.5
- dense layer with a size of 1

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center of the road if it strays away from it. These images show what a recovery looks like starting from the edge of the road, bringing the car back to the center:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To get my model to generalize better, I also recorded images while driving on the track in reverse. Likewise, I recorded images of the vehicle recovering from left and right to the center while driving in reverse.

After the collection process, I had about 10000 data points.

I used all of the images of the center camera and their belonging steering angels.
Additionally, I have used the images from the left and right cameras and multiplied their respective steering angles with a factor of 3.

Then, I flipped all images (taken by the left, right and center cameras, the left camera and the right camera) and negated the steering angles.

![alt text][image5]
![alt text][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The final video can be found here: ![alt text][video]
