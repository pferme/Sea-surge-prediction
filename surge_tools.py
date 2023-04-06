#/usr/bin/env python3
# coding: utf8

"""General and common tools for surge prediction challenge."""

import numpy as np
import pandas as pd
import math

# Metric of the challenge
surge_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]

def surge_prediction_metric(dataframe_y_true, dataframe_y_pred):
    weights = np.linspace(1, 0.1, 10)[np.newaxis]
    surge1_columns = [
        'surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4',
        'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9' ]
    surge2_columns = [
        'surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4',
        'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9' ]
    surge1_score = (weights * (dataframe_y_true[surge1_columns].values - dataframe_y_pred[surge1_columns].values)**2).mean()
    surge2_score = (weights * (dataframe_y_true[surge2_columns].values - dataframe_y_pred[surge2_columns].values)**2).mean()

    return surge1_score + surge2_score

# Normalizing the weight wrt metric data
weights_in_sqrt = [np.sqrt(i) for i in np.linspace(1, 0.1, 10)]
weights_in_sqrt_inv = [1/np.sqrt(i) for i in np.linspace(1, 0.1, 10)]

def normalize_l2_y(y_sample):
    y_new = [0. for i in y_sample]
    for i in range(10):
        y_new[i] = weights_in_sqrt[i]*y_sample[i]
    return y_new

def denormalize_l2_y(y_sample):
    y_new = [0. for i in y_sample]
    for i in range(10):
        y_new[i] = weights_in_sqrt_inv[i]*y_sample[i]
    return y_new

# Loading data
X_train = np.load('X_train_surge_new.npz')
y_train = pd.read_csv('Y_train_surge.csv')
X_test = np.load('X_test_surge_new.npz')

