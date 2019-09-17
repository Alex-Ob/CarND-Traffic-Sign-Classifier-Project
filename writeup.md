# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[TrainImagesSet]: ./examples/SignsExplore.png "Training Images Set"
[ToNormalize]: ./examples/to_normalize.png "Image to normalize"
[TrafficSign4]: ./examples/ex4.bmp "Traffic Sign 4"
[TrafficSign9]: ./examples/ex9.bmp "Traffic Sign 9"
[TrafficSign12]: ./examples/ex12.bmp "Traffic Sign 12"
[TrafficSign14]: ./examples/ex14.bmp "Traffic Sign 14"
[TrafficSign17]: ./examples/ex17.bmp "Traffic Sign 17"
[TrafficSign35]: ./examples/ex35.bmp "Traffic Sign 35"
[Softmax4]: ./examples/class_4_prob.png "Softmax output for class 4"
[Softmax9]: ./examples/class_9_prob.png "Softmax output for class 9"
[Softmax12]: ./examples/class_12_prob.png "Softmax output for class 12"
[Softmax14]: ./examples/class_14_prob.png "Softmax output for class 14"
[Softmax35]: ./examples/class_35_prob.png "Softmax output for class 35"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to load main dataset, the csv library to load and parse traffic signs labels list and standard python methods for calculate summary statistics of the data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43 ( 0..42 )

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
This is a PNG image that shows the beginning of a set of images:

![alt text][TrainImagesSet]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a data preparation I decided to normalize images from uint8 type with range of [0..255] for float32 type with max range of [ -1.0..1.0 ) by linear transformation:
**I_norm = (  (float) I - 128 ) / 128**,
where 128 is a middle of source range (0..255).
This transformation could be applied in other way by centering each image around its own mean, which is individual to each image, to elimitate intensity differency, but this was enough to achieve an acceptable level of accuracy (>=0.93) after 24 iterations of training. If there was a lower level of accuracy, I could try to smooth out the data with some filter and / or shift the average of the input image closer to 0. In addition, if there is a gap between validation and testing results, it would be possible to expand the training set by the data augmentation.

Since the normalized image has a real data type, it is difficult to display it in the colored image, and the contrast of such an image remains unchanged.

Source image:

![alt text][ToNormalize]

This is a normalization result for image [21810] in training set:

````
Before normalization:
Image type: <uint8>
Image range: [6..130]

After normalization:
Image type: <float32>
Image range: [-0.953..0.016]
````


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
As a basis of my NN model I took a LeNet-like Lab model with 7 hidden levels (not counting dropout levels).
My final model consisted of the following layers:

| Layer         		        |  Description	        			| 
|:-----------------------------:|:---------------------------------------------:| 
| 1. Input        		        | 32x32x3 RGB image   				            |
| 2. Convolution 5x5, 6 kernels | 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout level 		        | rate = 1 - dropout_value, outputs 28x28x6	    |
| RELU				            |						                        |
| 3. Max pooling	   	        | 2x2 stride,  outputs 14x14x6	 		        |
| 4. Convolution 5x5, 16 kernels| 1x1 stride, valid padding, outputs 10x10x16 	|
| 5. Max pooling	      	    | 2x2 stride,  outputs 5x5x16	 		        |
| 6. Flattern			        | outputs 400x1 				                |
| 7. Fully connected		    | 400x120, outputs 120x1   			            |
| Dropout level 		        | rate = 1 - dropout_value, outputs 120x1	    |
| RELU				            |						                        |
| 8. Fully connected		    | 120x84, outputs 84x1   			            |
| RELU				            |						                        |
| 9. Fully connected		    | 84x43, outputs 43x1   			            |
| Softmax (for training&testing)| outputs 43x1   				                |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, which realized advanced SGD training method.
To do this, I created a TF graph with the following nodes:
**logits** - computes LeNet output w/out an activation function
**cross_entropy** - combines softmax & cross_entropy calculation
**loss_operation** - computes Loss function
**optimizer** - realizes an Adam optimizer with hand-given *learning_rate* parameter
**training_operation** - minimize loss function during the training

After that, I have found some hyperparameters, namely:
- *batch_size* (128, as proposed in the lectures)
- *start learning rate* (1e-3)
- *learning rate decreasing strategy* (multiply by 0.8 after each 8 iterations)
- *dropout value* for beginner layers (0.6)
- amount of training *epochs/iteration* (24)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.953
* test set accuracy of 0.932

I made several training attempts and got similar results (including the test set accuracy >= 0.93), which indicates the stability of the configuration - despite the different initial level of accuracy on each attempt due to the random distribution of weights. It like as a plot with several monotonic curves with different starting points but with asymptotic convergence.


````
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture,
  adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function.
  One common justification for adjusting an architecture would be due to overfitting or underfitting.
  A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem?
  How might a dropout layer help with creating a successful model?
````
*First I tried to work with light architecture model without Dropout levels and with fewer amount of parameters. But I have faced with overfitting effect. I also experimented with amount of nodes in different levels. Now I concluded that some more parameters with dropout levels are preferable to fewer parameters w/out dropout, because it provides a principal opportunity to reduce DoF in the learning process .*
````
If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 ````
*I choose architecture that was described in lectures and labs. It was developed for similar task, albeit a simpler one. The final model's accuracy on the training, validation and test set (0.998, 0.947 and 0.932 respectively) are quite good and can be applied for some recognition tasks (probably after some slight improving or  just additional  training, in particular on the extended data set).*

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][TrafficSign4]
![alt text][TrafficSign12]
![alt text][TrafficSign9]
![alt text][TrafficSign14]
![alt text][TrafficSign17]
![alt text][TrafficSign35]

I randomly choose 5 of them to test my NN. These are:

![alt text][TrafficSign4]
![alt text][TrafficSign12]
![alt text][TrafficSign9]
![alt text][TrafficSign14]
![alt text][TrafficSign35]

Only one image had wrong prediction. Traffic Sign 'Speed limit 70 kmph' ![alt text][TrafficSign4] is predicted as 'Speed limit 30 kmph' due to illegible digit shapes in the training set.

Note: With the set of images
![alt text][TrafficSign12]
![alt text][TrafficSign9]
![alt text][TrafficSign14]
![alt text][TrafficSign17]
![alt text][TrafficSign35]
there are no misshits



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction        | 
|:---------------------:|:---------------------:| 
| Speed limit 70 kmph	| Speed limit 30 kmph   | 
| Priority road   		| Priority road 		|
| No passing			| No passing			|
| Stop	      		    | Stop					|
| Ahead only			| Ahead only      		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is less than an accuracy of the test data set, which can be explained by a different clarity and a different background of the new picture.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image (class 4), the model is relatively sure that this is a '30 kmph' sign (class 1, probability of 0.48).
The top five soft max probabilities are shown here:

![alt text][Softmax4]

For other images, we can see an absolutely reliable softmax score close to 1.
![alt text][Softmax12]
![alt text][Softmax9]
![alt text][Softmax14]
![alt text][Softmax35]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


