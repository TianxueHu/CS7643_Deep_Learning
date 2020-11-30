"""TODO"""

import numpy as np

from gt_7643 import optim
from gt_7643.cola_utils import sample_minibatch


class ClassificationSolver(object):
    """
    A ClassificationSolver encapsulates all the logic necessary for training
    classification models. The ClassificationSolver performs stochastic gradient
    descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a ClassificationSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new ClassificationSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data
        """
        pass

    def train(self):
        """
        Run optimization to train the model.
        """
        pass