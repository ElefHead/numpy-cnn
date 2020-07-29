# Numpy CNN
A numpy based CNN implementation for classifying images.

**status: archived**

## Usage

Follow the steps listed below for using this repository after cloning it.  
For examples, you can look at the code in [fully_connected_network.py](https://github.com/ElefHead/numpy-cnn/blob/master/fully_connected_network.py) and [cnn.py](https://github.com/ElefHead/numpy-cnn/blob/master/cnn.py).  
I placed the data inside a folder called data within the project root folder (this code works by default with cifar10, for other datasets, the filereader in utilities can't be used). 

After placing data, the directory structure looks as follows 
- root
    * data\
        * data_batch_1
        * data_batch_2 
        * ..
    * layers\
    * loss\
    * utilities\
    * cnn.py
    * fully_connected_network.py
    
---  

1) Import the required layer classes from layers folder, for example
    ```python
    from layers.fully_connected import FullyConnected
    from layers.convolution import Convolution
    from layers.flatten import Flatten
    ```
2) Import the activations and losses in a similar way, for example
    ```python
    from layers.activation import Elu, Softmax
    from loss.losses import CategoricalCrossEntropy
    ```
3) Import the model class from utilities folder
    ```python
    from utilities.model import Model
    ```
4) Create a model using Model and layer classes
    ```python
    model = Model(
        Convolution(filters=5, padding='same'),
        Elu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=10),
        Softmax(),
        name='cnn-model'
    )
    ```
5) Set model loss
    ```python
    model.set_loss(CategoricalCrossEntropy)
    ```
6) Train the model using
    ```python
    model.train(data, labels)
    ```
    * set load_and_continue = True for loading trained weights and continue training
    * By default the model uses AdamOptimization with AMSgrad
    * It also saves the weights after each epoch to a models folder within the project
7) For prediction, use
    ```python
    prediction = model.predict(data)
    ```
8) For calculating accuracy, the model class provides its own function
    ```python
    accuracy = model.evaluate(data, labels)
    ```
9) To load model in a different place with the trained weights, follow till step 5 and then
    ```python
    model.load_weights()
    ```
    Note: You will have to have similar directory structure.


---
This was a fun project that started out as me trying to implement a CNN by myself for classifying cifar10 images. In process, I was able to implement a reusable (numpy based)
library-ish code for creating CNNs with adam optimization.

Anyone wanting to understand how backpropagation works in CNNs is welcome to try out this code, but for all practical usage there are better frameworks
with performances that this code cannot even come close to replicating.

The CNN implemented here is based on [Andrej Karpathy's notes](http://cs231n.github.io/convolutional-networks/)
