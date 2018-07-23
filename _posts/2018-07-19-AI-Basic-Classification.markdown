---
layout: post
title:  "AI Basic Classification"
date:   2018-07-19 14:53:39 +0100
categories: jekyll update
---

in this tutorial I'm going to talk about some of the basics of Artificial intelligence



This guide trains a neural network model to classify images of clothing, like sneakers and shirts.

This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:
![alt text](http://127.0.0.1:4000/images/clothes.png "Logo Title Text 1")

## Import the Fashion MNIST dataset
We will use 60,000 images to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow, just import and load the data:

Loading the dataset returns four NumPy arrays:

- The train_images and train_labels arrays are the training set—the data the model uses to learn.
- The model is tested against the test set, the test_images, and test_labels arrays.

The images are 28x28 NumPy arrays, with pixel values ranging between 0 and 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
__(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()__

Label | Class | 
--- | ---  
0  |	T-shirt/top
1 |	Trouser	   
2 |	Pullover   	
3 |	Dress      
4 |	Coat       
5 |	Sandal     
6 |	Shirt      
7 |	Sneaker    
8 |	Bag        
9 |	Ankle boot 

Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
## Explore the data

Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:

the command __train_images.shape__ returns 60.000



Likewise, there are 60,000 labels in the training set:

we can also use this command : 
__len(train_labels)__

## Preprocess the data
The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

- __plt.figure()__
- __plt.imshow(train_images[1])__
- __plt.colorbar()__
- __plt.gca().grid(False)__

We scale these values to a range of 0 to 1 before feeding to the neural network model. For this, cast the datatype of the image components from an integer to a float, and divide by 255. Here's the function to preprocess the images:

It's important that the training set and the testing set are preprocessed in the same way:

- __rain_images = train_images / 255.0__

- __test_images = test_images / 255.0__

We can display the first 25 images from the training set and display the class name below each image.

__import matplotlib.pyplot as plt__

__%matplotlib inline__

__plt.figure(figsize=(10,10))__

__for i in range(25):__

__plt.subplot(5,5,i+1)__

__plt.xticks([])__

__plt.yticks([])__

__plt.grid('off')__

__plt.imshow(train_images[i], cmap=plt.cm.binary)__

__plt.xlabel(class_names[train_labels[i]])__

## Build the model
Building the neural network requires configuring the layers of the model, then compiling the model.

### Setup the layers
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.
Most of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.
	
__model = keras.Sequential([__

__keras.layers.Flatten(input_shape=(28, 28)),__

__keras.layers.Dense(128, activation=tf.nn.relu),__

__keras.layers.Dense(10, activation=tf.nn.softmax)__

__])__

The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels. Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.

After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely-connected, or fully-connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 digit classes.

### Compile the model

Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

### Train the model

Training the neural network model requires the following steps:

Feed the training data to the model—in this example, the train_images and train_labels arrays.
The model learns to associate images and labels.
We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array.

To start training, call the model.fit method—the model is "fit" to the training data:

__model.fit(train_images, train_labels, epochs=5)__

As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.

### Evaluate accuracy

Next, compare how the model performs on the test dataset:


__test_loss, test_acc = model.evaluate(test_images, test_labels)__

​

__print('Test accuracy:', test_acc)__

It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of overfitting. Overfitting is when a machine learning model performs worse on new data than on their training data.
