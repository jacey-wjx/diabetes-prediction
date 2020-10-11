import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import minmax_scale
from numpy import dstack
import joblib

# model_name = 'diabetes-resnetv1_1'
# model = load_model('./stacked_resnet/'+model_name)
# model.load_weights('./stacked_resnet/'+model_name+'.hdf5')
min_max = np.loadtxt('./stacked_resnet/min_max.txt')
_input=np.array([6,148,72,35,169.5,33.6,0.627,50])

_input[:] = (_input[:]-min_max[:,0])/(min_max[:,1]-min_max[:,0])
# Initialize the output
segments = np.zeros([1, 5, 4],dtype=_input.dtype)

for l in range(5):
    segments[0, l, :] = _input[(l * 1):(l * 1 + 4)]

# res = model.predict(segments)
# if round(res[0][0]) == 1:
#     print("Healthy")
# if round(res[0][1]) == 1:
#     print("Diabetes")

modelname = 'diabetes-resnet'
def load_resNet_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = './stacked_resnet/' + modelname + "_" + str(i + 1)
        # load model from file
        model = load_model(filename)
        model.load_weights(filename + '.hdf5')
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

n_members = 5
members = load_resNet_models(n_members)

def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        yhat = yhat[:, 1]
        yhat = yhat.reshape(yhat.shape[0], 1)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX

input = stacked_dataset(members, segments)
print(input)
gbr = joblib.load('./stacked_resnet/stacked_GB.m')
yhat = gbr.predict(input)
if yhat == 0:
    print("Healthy")
if yhat == 1:
    print("Diabetes")