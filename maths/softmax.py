import numpy as np


def softmax(x, axis, keepdims):
    """
    softmax = lambda x, axis, keepdims: np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=keepdims)
    """
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=keepdims)


def nll(num_samples, y, softmax):
    """Negative log likelihood

    Args:
    -----
        num_samples (_type_): number of samples

        y (_type_): the true labels

        softmax (_type_): the predicted probabilities of the classes

    Returns:
    --------
        average_nll: the average deviation between the predicted probabilities of the model and the true labels of the samples.
    """
    nnl = -np.log(softmax[range(num_samples), y])
    average_nnl = np.mean(nnl)
    return average_nnl


def reg_loss(model, reg):
    """Following the L2 regularization, the regularization loss is defined as:

    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    Args:
    -----
        model (_type_): the model

        reg (_type_): the regularization parameter

    Returns:
    --------
        reg_loss: the regularization loss
    """
    reg_loss = 0
    for param in model.params:
        reg_loss += 0.5 * reg * np.sum(model.params[param] * model.params[param])
    return reg_loss
