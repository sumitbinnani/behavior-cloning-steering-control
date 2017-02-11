# Behavioral Cloning
---
This project was done as part of Udacity's Self-Driving Car Nanodegree Program.

[//]: # (Image References)
[image1]: ./images/model.png "Model Visualization"


## Goals and Objective

The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


## Code Structure

My project includes the following files:
* model.py containing the script to create, train and save the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* utils.py contains code for Image Generator and data augmentation
* README.md (also in writeup_report.md) summarizing the reports

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```


## Model Architecture

I started with model as described by comma.ai [here](https://github.com/commaai/research/blob/master/train_steering_model.py). However, the model parameter space was bigger than my GPU (GeForce 840M) was able to handle. Finally, I used model similar to one described in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from nvidia.

The model includes ELU (Exponential Linear Units) layers to introduce nonlinearity, and the data is resized and normalized in the model using a Keras lambda layer. The model is trained using dropouts to avoid overfitting.

The model used an adam optimizer, with initial learning rate of 0.001 ([model.py line 25](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/model.py#L55)).

Following image describes model architecture (_the `lambda_1` layer is the layer used for resizing the image, and `lambda_2` is the layer used for normalizing images).

![Model Defination][image1]

The complete model defination code can be found at [line 20 in model.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/model.py#L20).