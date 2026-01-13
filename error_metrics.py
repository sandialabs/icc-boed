import numpy as np

def rmse(y_true, y_pred, axis=0):
    """ Root mean squared error (normalized) """

    epsilon = 1e-20
    loss = np.sqrt(np.mean(np.square(\
        (y_true - y_pred)/(y_true + epsilon)\
        ), axis=axis))
    return loss

def rms_dev(y_true, y_pred, axis=0):
    """ root mean squared deviation (unnormalized) """
    loss = np.sqrt(np.mean(np.square(\
        (y_true - y_pred) ), axis=axis))
    return loss

def rae(y_true, y_pred, axis=1):
    """ Relative Absolute Error

    y_true and y_pred are nsamples x npoints
    in dimension """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)

    num = np.sum(np.abs(y_true - y_pred), axis=1)
    denom = np.sum(np.abs(y_true - y_true.mean(axis=1)), axis=1)
    loss = num / denom
    return loss

def sMAPE(y_true, y_pred, axis=0):
    """
    Symmetric Mean Absolute Percentage Error

    provides a result between 0% and 200%
    since a percentage error between 0% and 100% is
    easier to interpret, the division by two is often omitted
    in practice.
    """

    loss = np.abs(y_true - y_pred) /\
        ( (np.abs(y_true) + np.abs(y_pred))/2 )
    loss_2 = np.abs(y_true - y_pred) /\
        ( np.abs(y_true) + np.abs(y_pred) )
    mean_percent_loss = 100 * np.mean(loss_2, axis=axis)
    return mean_percent_loss


def mape(truth, pred, axis=0):
    """ Mean Absolute Percentage Error"""
    eps=1e-10
    return 100.0 * np.average(np.abs((pred - truth) / (truth+eps)), axis=axis)


def absolute_percentage_error(truth, pred):
    return 100.0 * np.abs((pred - truth) / truth)

