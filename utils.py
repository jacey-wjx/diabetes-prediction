import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import minmax_scale

class Model:
    def __init__(self, model_name):
        self.name = model_name
        self.model = load_model('./model/' + model_name)

    def predict(self, _input):
        _input = minmax_scale(_input)
        segments = np.zeros([1, 5, 4], dtype=_input.dtype)

        for l in range(5):
            segments[0, l, :] = _input[(l * 1):(l * 1 + 4)]

        res = self.model.predict(segments)
        if round(res[0][0]) == 1:
            return 0
        if round(res[0][1]) == 1:
            return 1
