import numpy as np


def load_data(path):
    """ 
    Loads the shuffled data from a numpy array.
    Returns 4 numpy arrays in the following order:
    training images, training labels / order, test images, and test labels/order.
    path: path to numpy array
    """
    data = np.load(path)

    x_tr = data['a']     # Loads the training images.
    y_tr = data['b']     # Loads the labels/order of the training images.
    x_te = data['c']     # Loads the test images.
    y_te = data['d']     # Loads the labels/order of the test images.

    return x_tr, y_tr, x_te, y_te
