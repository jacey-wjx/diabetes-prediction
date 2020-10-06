import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import minmax_scale

model_name = 'diabetes-resnetv1'
model = load_model('./model/'+model_name)

_input=np.array([0,80,80,20,80,23.1,0.52,34])
_input= minmax_scale(_input)
# Initialize the output
segments = np.zeros([1, 5, 4],dtype=_input.dtype)

for l in range(5):
    segments[0, l, :] = _input[(l * 1):(l * 1 + 4)]

res = model.predict(segments)
if round(res[0][0]) == 1:
    print("Healthy")
if round(res[0][1]) == 1:
    print("Diabetes")