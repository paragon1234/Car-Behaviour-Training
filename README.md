# Behavioral Cloning Project

## Goal

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the track
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dataSetImages/center_2017_09_03_09_51_41_668.jpg "Model Visualization"
[image2]: ./dataSetImages/center_2017_09_03_10_05_15_623.jpg "Recovery Image"
[image3]: ./dataSetImages/center_2017_09_03_10_05_18_024.jpg "Recovery Image"
[image4]: ./dataSetImages/center_2017_09_03_10_05_18_382.jpg "Recovery Image"
[image5]: ./dataSetImages/center_2017_09_03_10_05_18_733.jpg "Recovery Image"


## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Overview

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3/5x5 filter sizes and depths between 32 and 64 (model.py lines 54-58) 

The model includes RELU layers to introduce nonlinearity (code line 54-56), and the data is normalized in the model using a Keras lambda layer (code line 52). 

To increase accuracy of the model, the top 71 pixels and bottom 25 pixels are removed from the image using Keras Cropping2D layer (code line 53). This ensures that the image contains only the track and not mountains at the top and the car at the bottom.

### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 67). The epoches were trimmed to 5, as the validation loss increases beyond that(but training loss keeps on decreasing) indicating overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the track. For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use different layers so that vehicle stays on track without touching the side lines as if it is safe for the person sitting inside the car while doing actual driving.

My first step was to use only a fully connected layer(FCL) and check out the result. It was found that when the vehicle moved on a bridge, whose texture was different from the track, then it hit the side walls.

Then I improved the model by adding a convolution neural network layer(CNN). Using CNN the vehicle successfully traversed the bridge, but it was not able to take sharp turn.

The next improvement was to use 2 layers of CNN with relu, and 3 FCL. In this case the vehicle was able to slightly turn steering angle on sharp turn, but it still went off the track.

Then taking clue from Nvidia architecture I created a trimmed down architecture: 3 CNN (output depth of 24, 36, 64) with relu units, followed by 4 FCL. In this case the vehicle was safely able to take sharp turn, but it moved from one side of the track to another, like a drunken car.

My final archtitecture was the Nvidia Architecture where the car movement from one side of track to another reduced. At the end of the process, the vehicle is able to drive autonomously around the track (at 25 speed) without leaving the track.

### 2. Final Model Architecture

The final model architecture (model.py lines 54-63) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   						| 
| Cropping2D     		| 64x320x3 cropped 71 top pix and 25 bottom pix	|
| Convolution 5x5     	| 2x2 stride, same padding, output 32x160x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, same padding, output 16x80x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, same padding, output 8x40x48 		|
| RELU					|												|
| Convolution 3x3		| output 8x40x64								|
| RELU					|												|
| Convolution 3x3		| output 8x40x64    							|
| RELU					|												|
| Flatten				| outputs 20480 								|
| Fully connected		| Output 100        							|
| Fully connected		| Output 50 									|
| Fully connected		| Output 10        								|
| Fully connected		| Output 1 										|
|						|												|
|						|												|


### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the track back to center so that the vehicle would learn to recover from onse side of the track to the centre. The images below show what a recovery looks like:

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would double the data size and additionally creates training data for clock-wise turn on the track (the track has anti-clockwise turns). Also, I utilized the left and right camera images with steering adjusted by -2 and +2 respectively. These provide data for deviation of the car steering, and creates data that is not heavily biased for zero degree steering.

After the collection process, I had approximately 9000 number of data points. I then preprocessed this data: 
* First by normalizing it, and finally
* By cropping top and bottom corners  so that image contains only the track, and not the mountains and the car itself.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by decreasing training/validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
