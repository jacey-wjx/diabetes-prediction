import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import minmax_scale
from tensorflow.keras.models import load_model
from numpy import dstack
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics


def makeSteps(dat, length, dist):
    width = dat.shape[1]
    numOfSteps = int(np.floor((width - length) / dist) + 1)

    # Initialize the output
    segments = np.zeros([dat.shape[0], numOfSteps, length],
                        dtype=dat.dtype)

    for l in range(numOfSteps):
        segments[:, l, :] = dat[:, (l * dist):(l * dist + length)]

    return segments

modelname='diabetes-resnetv1'
def load_resNet_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = './model/' + modelname +"_"+ str(i + 1)
        # load model from file
        model = load_model(filename)
        model.load_weights(filename+'.hdf5')
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

members = load_resNet_models(5)
print('Loaded %d models' % len(members))

def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        yhat = yhat[:,1]
        yhat = yhat.reshape(yhat.shape[0], 1)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

model = joblib.load('./model/stacked_GB.m')

