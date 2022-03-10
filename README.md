# PyTorch: Multilayer Perceptron
In this repo we implement a multilayer perceptron using PyTorch.

## Overview

Multilayer perceptrons (MLPs), also call feedforward neural networks, are basic but flexible and powerful machine learning models which can be used for many different kinds of problems. I used this class many times for surrogate modeling problems in laser-plasma physics.

Basically, as long as the underlying data set is not too high-dimensional, MLPs can be a good start (e.g. images). Otherwise, MLPs tend to overfit.

Furthermore, if we can, e.g. due to physical considerations, expect a smooth dependent variable (also called response surface, or target function), we can use a sigmoid activation function (e.g., tanh) which has shown in my data sets better performance than the more common ReLU activation function.

In case you search for a different MLP implementation, check out [scikit-learn.org][scikit].

## Dependencies

 1. numpy
 2. torch
    *  torch.optim
    *  torch.nn


[scikit]: <https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron>
