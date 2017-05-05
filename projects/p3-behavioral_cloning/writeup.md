# **Behavioral Cloning**
## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/initial_image.jpg "Initial image example"
[image2]: ./report_images/processed_image.jpg "Processed image"
[image3]: ./report_images/tiny_processed_image.jpg "Tiny processed image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* data_loader.py containing helper methods for creating generators for the training and validation data
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and data_loader.py files contain the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture of my model is the following:

| Layer | Output Shape | # of parameters |
| :---: | :---: | --- |
| Cropping 2D | (None, 80, 320, 3) | 0 |
| Lambda | (None, 32, 64, 1) | 0 |
| Convolutional2D | (None, 31, 63, 16) | 80 |
| MaxPooling2D | (None, 7, 15, 16) | 0 |
| Convolutional2D | (None, 6, 14, 32) | 2080 |
| MaxPooling2D | (None, 1, 3, 32) | 0 |
| Dropout | (None, 1, 3, 32) | 0 |
| Flatten | (None, 96) | 0 |
| Fully Connected | (None, 256) | 24832 |
| Fully Connected | (None, 128) | 32896 |
| Fully Connected | (None, 16) | 2064 |
| Fully Connected | (None, 1) | 17 |

The convolutional and fully connected layers use the **ReLU** activation function.

The total number of trainable parameters is **61,969**.

#### 2. Attempts to reduce overfitting in the model

After several experiments with dropout layers and various percentages within them, I decided to only use one dropout layer, at the transition between the convolutional layers and the fully connected ones.

After recording all the data by manually driving through the simulator, I split it up into training and validation data, and this is what I used to ensure that the model was not overfitting. Afterwards, I ran the model on the track in autonomous mode to make sure that the vehicle would stay on the track for the entire lap.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Details of model architecture

I started with a model having 3 convolutional layers (with Maxpooling and Dropout in between them) and 3 fully connected layers at the end. Initially I was using cropped RGB images to pass through the model (cropping the top 50 pixels and bottom 20 pixels). This seemed to take longer on my GPU and also stress the computations more. Therefore, I then looked into how I can minimize the images.

I came up with the preprocessing routine that I use in the Lambda layer:
* I first normalize the pixel values to get in the [0, 1] range. This is necessary both due to general normalization concepts but also because of my next preprocessing step (converting to HSV)
* I convert the RGB images to HSV and only keep the S channel because it is the one that contains the information that I mostly need - i.e. the information that clearly differentiates the road from the surroundings, due to the saturation of the road color
    * Tensorflow notes in their documentation that the output of the HSV conversion is only valid if the input image has normalized pixels in the [0, 1] range
* I continue the normalization, bringing the pixel values in the [-0.5, 0.5] range
* I finally resize the image to [32, 64] pixels in order to reduce the workload on the computation; after some tests, this did not negatively impact the performance of my model. I also tried smaller values, but those resized images were losing too much information necessary during the training.

Here is an example of an image that was passed through the model:
* On entry in the model:

![initial image][image1]

* After the preprocessing layer (normal size)

![processed image][image2]

* After resizing as well - tiny image that goes through the model

![tiny processed image][image3]

Once I got my new images, training was faster and did not put that much load on my GPU (actually my GPU was only at around 4% load during training, which I found very interesting; its memory was fully commited due to how Tensorflow likes to eat up the entire memory on load).

Initially I used the dropout layers to avoid overfitting, but I was a bit too aggressive with the percentage used within them. I was dropping with 50% chance, which turned out to be not very productive in terms of learning the steering angles. Therefore, I eventually ended up using 30% with only one dropout layer.

I also decided to drop out one the convolutional layers, ending up with only 2 in the final architecture. This was mostly due to my desire to shrink the network as much as possible. Something to note is that I was able to shrink my current one even more, but I felt that this version, listed above, gives me more flexibility in terms of fine tuning.

#### 5. Creation of the Training Set & Training Process

The training data that I used consists of 3 main batches:
* Pre-recorded training data provided with the project (*24,108 data points*)
* Manually collected training data by myself, driving around the track **once**, **counter-clockwise** and using **keyboard** (*4,953 data points*)
* Manually collected training data by myself, driving around the track **once**, **clockwise** and using **keyboard** (*15,624*)

All of the data that I collected was collected using center lane driving. No recovering driving was recorded. I only considered doing this if necessary, but the model performed well without it.

I also collected data using the more discreet mouse based steering, but this did not perform too well in my training sessions (probably due to the constant variations in steering due to a constant movement of my mouse during driving, even in straight lines).

Overall, I used **44,685 data points**.

In order to split this data into training and validation sets, I used `sklearn`'s `train_test_split`, keeping **20%** for validation.

Moreover, I made use of all 3 cameras that were present on the vehicle. In order to do this, I created a test data generator that uses a probabilistic model in terms of serving up next either a center, a left, or a right camera image. Something worth mentioning is that I wanted to put more emphasis on the side cameras in order to help the model train better for steering and avoiding exiting the track. Therefore, the the probabilistic test data generator has **15%** chance of serving a center camera image and **42.5%** chance of serving a left or right camera image. An interesting side effect of this aggressive way of training the model mostly on side camera images is the fact that the vehicle tends to shift from left to right when driving in a straight line. This is just my hypothesis, but I think this is a fairly good theoretical basis for it. I will conduct more tests to conclude it.

When using the side camera images, I add a deviation factor to the steering angle recorded from the perspective of the center camera. This deviation factor is set to **0.10** and I have not tested any other values for it throughout my tests.