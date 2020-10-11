import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import minmax_scale
import joblib
from numpy import dstack

class Model:
    def __init__(self, model_name):
        if model_name == 'stacked_resnet':
            self.name = model_name
            self.min_max = np.loadtxt('./model/stacked_resnet/min_max.txt')
            self.model_members = list()
            for i in range(5):
                # define filename for this ensemble
                filename = './model/stacked_resnet/diabetes-resnetv1' + "_" + str(i + 1)
                # load model from file
                model = load_model(filename)
                model.load_weights(filename + '.hdf5')
                # add to list of members
                self.model_members.append(model)
                print('>loaded %s' % filename)
            self.model = joblib.load('./model/stacked_resnet/stacked_GB.m')
            print('>loaded stacked model')
        else:
            self.name = model_name
            self.model = load_model('./model/' + model_name)

    def predict(self, _input):
        if self.name == 'stacked_resnet':
            _input[:] = (_input[:] - self.min_max[:, 0]) / (self.min_max[:, 1] - self.min_max[:, 0])
            segments = np.zeros([1, 5, 4], dtype=_input.dtype)
            for l in range(5):
                segments[0, l, :] = _input[(l * 1):(l * 1 + 4)]

            stack_input = None
            for model in self.model_members:
                # make prediction
                yhat = model.predict(segments, verbose=0)
                yhat = yhat[:, 1]
                yhat = yhat.reshape(yhat.shape[0], 1)
                # stack predictions into [rows, members, probabilities]
                if stack_input is None:
                    stack_input = yhat
                else:
                    stack_input = dstack((stack_input, yhat))
            # flatten predictions to [rows, members x probabilities]
            stack_input = stack_input.reshape((stack_input.shape[0], stack_input.shape[1] * stack_input.shape[2]))
            yhat = self.model.predict(stack_input)
            return yhat
        else:
            _input = minmax_scale(_input)
            segments = np.zeros([1, 5, 4], dtype=_input.dtype)

            for l in range(5):
                segments[0, l, :] = _input[(l * 1):(l * 1 + 4)]

            res = self.model.predict(segments)
            if round(res[0][0]) == 1:
                return 0
            if round(res[0][1]) == 1:
                return 1
