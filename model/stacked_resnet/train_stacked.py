import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import minmax_scale
#Loading the dataset
diabetes_data = pd.read_csv('../diabetes.csv')

var_list = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
diabetes_data[var_list] = diabetes_data[var_list].replace(0,np.NaN)

def find_median(var):
    temp = diabetes_data[diabetes_data[var].notnull()][[var,'Outcome']]
    temp = temp.groupby('Outcome').median().reset_index()
    return temp[var].values

for var in var_list:
    diabetes_data.loc[(diabetes_data['Outcome'] == 0) & (diabetes_data[var].isnull()), var] = find_median(var)[0]
    diabetes_data.loc[(diabetes_data['Outcome'] == 1) & (diabetes_data[var].isnull()), var] = find_median(var)[1]

X = minmax_scale(diabetes_data.drop("Outcome",axis = 1))
y = diabetes_data.Outcome.values
y = y.reshape(y.shape[0],1)

total = diabetes_data.isnull().sum().sort_values(ascending=False)
percent = (diabetes_data.isnull().sum()/diabetes_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head())


def makeSteps(dat, length, dist):
    width = dat.shape[1]
    numOfSteps = int(np.floor((width - length) / dist) + 1)

    # Initialize the output
    segments = np.zeros([dat.shape[0], numOfSteps, length],
                        dtype=dat.dtype)

    for l in range(numOfSteps):
        segments[:, l, :] = dat[:, (l * dist):(l * dist + length)]

    return segments

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)

length = 4
dist = 1
trDat = makeSteps(X_train, length, dist)
tsDat = makeSteps(X_test, length, dist)

trLbl = y_train
tsLbl = y_test

print('The shape of trDat is', trDat.shape, "and the type is", trDat.dtype)
print('The shape of trLbl is', trLbl.shape, "and the type is", trLbl.dtype)
print('')
print('The shape of tsDat is', tsDat.shape, "and the type is", tsDat.dtype)
print('The shape of tsLbl is', tsLbl.shape, "and the type is", tsLbl.dtype)

trLbl = to_categorical(trLbl)
tsLbl = to_categorical(tsLbl)


modelname = 'diabetes-resnetv1'
def load_resNet_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = './' + modelname + "_" + str(i + 1)
        # load model from file
        model = load_model(filename)
        model.load_weights(filename + '.hdf5')
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


n_members = 5
members = load_resNet_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
testout = np.argmax(tsLbl,axis=1)
for model in members:
    predicts    = model.predict(tsDat)
    predout     = np.argmax(predicts,axis=1)
    testScores  = metrics.accuracy_score(testout,predout)
    print('Model Accuracy: %.3f' % testScores)


from numpy import dstack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# create stacked model input dataset as outputs from the ensemble
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


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = GradientBoostingClassifier()
    model.fit(stackedX, inputy)
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

# fit stacked model using the ensemble
model = fit_stacked_model(members, tsDat, testout)
# evaluate model on test set
yhat = stacked_prediction(members, model, tsDat)
acc = accuracy_score(testout, yhat)

print('Stacked Test Accuracy: %.3f' % acc)

import joblib
joblib.dump(model, 'stacked_GB.m')
# gbr = joblib.load('stacked_GB.m')
print('GB model saved...')


