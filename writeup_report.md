# Behavioral Cloning
---
This project was done as part of Udacity's Self-Driving Car Nanodegree Program. The model performance has been tested on for resolution of **320x240**, and graphic quality selected as 'fastest'.


To see the model performance click the following links:
* [Model performance on track 1](https://youtu.be/WzMiriCVSTI)
* [Model performance on track 2](https://youtu.be/4SYe9758P_Q)

\* _model was only trained on track 1 data_

[//]: # (Image References)
[model]: ./images/model.png "Model Visualization"
[steering_hist]: ./images/steering_angle_histogram.png "Steering Angle"
[cropped_image]: ./images/cropped_image.png "Cropped Image"
[flipped]: ./images/flipped.png "Flipped Image"
[left_center_right]: ./images/left_center_right.png "Left and Right Camera Image"
[translated]: ./images/translated.png "Translated Image"


## Goals and Objective

The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


## Code Structure

My project includes the following files:
* model.py containing the script to create, train and save the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. _Change the location of `DATA_PATH` and `LABEL_PATH` as per the location of data in your machine_.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* utils.py contains code for Image Generator and data augmentation
* DataVisialization.ipynb contains code for images being used in the writeup
* writeup_report.md (also in README.md) summarizing the reports

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```


## Model Architecture

To begin with, I used AlexNet architecture with last layer as fully connected layer with one unit. I also used experimented with the model as described by comma.ai ([github link](https://github.com/commaai/research/blob/master/train_steering_model.py)). These models performed well, but did not genralize to the the second track. Also, the model parameter space was bigger than my GPU (GeForce 840M) was able to handle, and the training was quite slow on CPU, and hence playing with parameters was quite time inefficient.

Finally, I used model similar to one described in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from nvidia.

The model includes ELU (Exponential Linear Units) layers to introduce nonlinearity, and the data is resized and normalized in the model using a Keras lambda layer, and dropout were used at various stages to avoid overfitting.

Following image describes model architecture (_the `lambda_1` layer is the layer used for resizing the image, and `lambda_2` is the layer used for normalizing images_).

![Model Defination][model]

The complete model defination code can be found at [line 20 in model.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/model.py#L20).


## Training and Validation Data

### Directory Structure
The model was trained using data provided by Udacity.

```ruby
udacity-data/
├── IMG/
└── driving_log.csv
```

**IMG** folder contains central, right and left frame of the driving and each row in **driving_log.csv** sheet correlates these images with the steering angle, throttle, brake, and speed of the car.

### Training and Validation Data Split
The data provided by Udacity was split so as to use **80%** of data as training set and rest of the data was used for validation. The validation data was used to ensure that the hyperparameters chosen for the model does not overfit.

### Data Preprocessing
When the model was trained with the raw data as provided by Udacity, the car had tendency to go straight and lost the track particularly at turnings. This led me to explore for various data processing and data augmentation techniques (_data augmentation_ techniques are discussed in next section).

* #### Cropping Image
	The original image was cropped to remove redundant top portion (sky and other details which is not required to decide steering angle). Also the bottom of the image displaying car hood was cropped out. The code for the same is [at line 18 of utils.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/utils.py#L18).

	![Cropped Image][cropped_image]

* #### Reduction of Low Steering Angle Data
	A quick look at the histogram of the steering angle shows that the data is biased towards low steering angles (which is expected as the car would mostly be driving straight).

	![Steering Angle][steering_hist]

	To combat the issue, about 70% of the randomly selected low steering angle were dropped from the **training data** (check [line 72 in model.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/model.py#L72) and corresponding function defination in [line 7 of utils.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/utils.py#L7)).

### Image Generators
Image generators were used to generate training batches in realtime (this was to combat high memory usage if all the images were pre-cached in memory). The code for the same can be found at [line 71 of utils.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/utils.py#L71). The code is well commented and a few steps involved in image generator are explained in the data augmentation section.


## Data Augmentation

Despite the removal of low steering data, the car would deviate from the track at a few places. Since the training data provided by Udacity is focused on driving down the middle of the road, the models did not learn what to do if it gets off to the side of the road.


### Approach 1: Collecting More Data
To teach the car what to do when it’s off on the side of the road, I generated _**recovery data**_ i.e. collecting data such that it captures the behavior to follow when the car deviates from the track. I recorded data when the car is driving from the side of the road back toward the center line.

The approach didn't work as the data collected did not have smooth steering angle across the laps (_I did not have fine control over the steering angle when running the simulator using keyboard_).

### Approach 2: Image Transformations
Applying image transformation techniques to the existing data can be used to increase the volume of data available for training. Moreover, these makes the model less prone to **overfitting**.

Following techniques were used for image augmentations:

* #### Flipping Image
	Mirroring the image and reversing the steering angle gives equally valid image for training. In the image generator, 50% of the images were flipped.

	![Flipped Images][flipped]

* #### Using Left and Right Camera Images
	The left camera image has to move right to get to center, and right camera has to move left. Adding a small angle .25 to the left camera and subtract a small angle of 0.25 from the right camera does the trick.

	![Corrected Left and Right Camera Images][left_center_right]

* #### Applying Horizontal and Vertical Shifts
	The camera images were horizontally/vertically shifted to simulate the effect of car being at different positions on the road, and an offset corresponding to the shift was added to the steering angle ([line 58 of utils.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/utils.py#L58)).

	![Translated Image][translated]

\* _The approach used for image transformation are provided in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidia and [this blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1xcd9d1vo) by Vivek Yadav._


## Parameter Tuning
The model used an adam optimizer for minimizing **Mean Squared Error** as loss function. The ***initial learning rate** was choosen as **0.001** as the model did not converge well with the default learning rate of 0.01 ([model.py line 25](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/model.py#L55)). The samples per epochs were decided on basis of the lenght of training data, and **epochs** used for training were **50** keeping in account that the model did not overfit (_this was ensured by keeping check on validation loss during training_).

## Model Generalization
The model was trained using images obtained from track 1 alone and it worked without any further tuning for **track2**. The fact that the model worked on a track unseen by it speaks about the generalization of the model.

## What more can be done?
* The training data can be augmented for brightness and hue jitter.
* Random patch of black tiles can be overlayed on the training data to simulate shadow and make data less prone to the effect of the same.
* The image cropping can be done as part of model to utilize CUDA acceleration
* The model does not perform well with higher resolution even if the images are resized to current input size during preprocessing (mostly because of the additional lag due to extra preprocessing or the image resizing requires particular type of interpolation strategy so as to meet current input specifications).