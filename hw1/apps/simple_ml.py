import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    ################################################################
    # first 4 bytes is a magic number
    image_f = gzip.open(image_filesname,'r')
    magic_number = int.from_bytes(image_f.read(4), 'big')
    # second 4 bytes is the number of images
    image_count = int.from_bytes(image_f.read(4), 'big')
    # third 4 bytes is the row count
    row_count = int.from_bytes(image_f.read(4), 'big')
    # fourth 4 bytes is the column count
    column_count = int.from_bytes(image_f.read(4), 'big')
    # rest is the image pixel data, each pixel is stored as an unsigned byte
    # pixel values are 0 to 255
    image_data = image_f.read()
    images = np.frombuffer(image_data, dtype=np.uint8)\
        .reshape((image_count, row_count*column_count))\
        .astype(np.float32)
    images = images / 255.0
    #################################################################
    label_f = gzip.open(label_filename, 'r')
    magic_number = int.from_bytes(label_f.read(4), 'big')
    # second 4 bytes is the number of labels
    label_count = int.from_bytes(label_f.read(4), 'big')
    # rest is the label data, each label is stored as unsigned byte
    # label values are 0 to 9
    label_data = label_f.read()
    labels = np.frombuffer(label_data, dtype=np.uint8)
    #################################################################
    # print(images.shape,labels.shape)
    return (images, labels)
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return (ndl.ops.log(ndl.ops.exp(Z).sum((1,))) - (y_one_hot * Z).sum((1,))).sum() / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION 
    # TODO: depend on numpy(np)
    batches = X.shape[0] // batch
    if X.shape[0] % batch != 0:
        batches += 1
    for batchid in range(batches):
        batchX = X[batchid*batch:min((batchid+1)*batch,X.shape[0]),:]
        batchy = y[batchid*batch:min((batchid+1)*batch,y.shape[0])]
        batch_size = batchX.shape[0]
        batchy_one_hot = np.eye(W2.shape[1])[batchy]
        batchX, batchy_one_hot = ndl.Tensor(batchX), ndl.Tensor(batchy_one_hot)
        Z = (ndl.ops.relu(batchX @ W1)) @ W2
        l = softmax_loss(Z,batchy_one_hot)
        l.backward()
        W1numpy = W1.numpy() - lr * W1.grad.numpy()
        W2numpy = W2.numpy() - lr * W2.grad.numpy()
        W1 = ndl.Tensor(W1numpy)
        W2 = ndl.Tensor(W2numpy)
    return W1, W2


    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
