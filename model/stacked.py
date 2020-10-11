import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import minmax_scale

import warnings

warnings.filterwarnings('ignore')

diabetes_data = pd.read_csv('./diabetes.csv')

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


total = diabetes_data.isnull().sum().sort_values(ascending=False)
percent = (diabetes_data.isnull().sum()/diabetes_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


def makeSteps(dat, length, dist):
    width = dat.shape[1]
    numOfSteps = int(np.floor((width - length) / dist) + 1)

    # Initialize the output
    segments = np.zeros([dat.shape[0], numOfSteps, length],
                        dtype=dat.dtype)

    for l in range(numOfSteps):
        segments[:, l, :] = dat[:, (l * dist):(l * dist + length)]

    return segments


outcome_ = y
y = np.zeros([outcome_.shape[0], 1], dtype=np.int64)

for l in range(y.shape[0]):
    y[l, 0] = outcome_[l]

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

num_classes = tsLbl.shape[1]

# fit model on dataset
from tensorflow.keras import optimizers

optmz = optimizers.Adam(lr=0.001)
modelname = 'diabetes-resnetv1'


def resLyr(inputs,
           numFilters=16,
           kernelSz=3,
           strides=1,
           activation='relu',
           batchNorm=True,
           convFirst=True,
           lyrName=None):
    convLyr = Conv1D(numFilters,
                     kernel_size=kernelSz,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4),
                     name=lyrName + '_conv' if lyrName else None)
    x = inputs

    if convFirst:
        x = convLyr(x)

        if batchNorm:
            x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

        if activation is not None:
            x = Activation(activation,
                           name=lyrName + '_' + activation if lyrName else None)(x)
    else:
        if batchNorm:
            x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

        if activation is not None:
            x = Activation(activation,
                           name=lyrName + '_' + activation if lyrName else None)(x)

        x = convLyr(x)
    return x

    # Step 4


def resBlkV1(inputs,
             numFilters=16,
             numBlocks=3,
             downsampleOnFirst=True,
             names=None):
    x = inputs

    for run in range(0, numBlocks):
        strides = 1
        blkStr = str(run + 1)

        if downsampleOnFirst and run == 0:
            strides = 2

        y = resLyr(inputs=x,
                   numFilters=numFilters,
                   strides=strides,
                   lyrName=names + '_Blk' + blkStr + '_Res1' if names else None)
        y = resLyr(inputs=y,
                   numFilters=numFilters,
                   activation=None,
                   lyrName=names + '_Blk' + blkStr + '_Res2' if names else None)

        if downsampleOnFirst and run == 0:
            x = resLyr(inputs=x,
                       numFilters=numFilters,
                       kernelSz=1,
                       strides=strides,
                       activation=None,
                       batchNorm=False,
                       lyrName=names + '_Blk' + blkStr + '_lin' if names else None)

        x = add([x, y],
                name=names + '_Blk' + blkStr + '_add' if names else None)
        x = Activation('relu',
                       name=names + '_Blk' + blkStr + '_relu' if names else None)(x)

    return x

    # Step 5


def createResNetV1(inputShape=(5, 4),
                   numClasses=2):
    inputs = Input(shape=inputShape)
    v = resLyr(inputs,numFilters=8)
    v = resBlkV1(inputs=v,
                 numFilters=8,
                 numBlocks=2,
                 downsampleOnFirst=True
                 )
    v = resBlkV1(inputs=v,
                 numFilters=8,
                 numBlocks=2,
                 downsampleOnFirst=False
                 )
    v = resBlkV1(inputs=v,
                 numFilters=8,
                 numBlocks=2,
                 downsampleOnFirst=True
                 )
    v = AveragePooling1D(pool_size=2)(v)
    v = Flatten()(v)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(v)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optmz,
                  metrics=['accuracy'])

    return model


def lrSchedule(epoch):
    lr = 1e-3

    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 140:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr


def CreateCallbacks(index):
    LRScheduler = LearningRateScheduler(lrSchedule)

    # Step 2
    folderpath = './'
    filename = modelname + "_" + str(index + 1)
    filepath = folderpath + filename + ".hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='max')

    csv_logger = CSVLogger(folderpath + filename + '.csv')  # Step 2
    callbacks_list = [checkpoint, csv_logger, LRScheduler]  # Step 3
    return callbacks_list

#pre-train
n_members=2
for i in range(n_members):
    model = createResNetV1()
    callbacks_list = CreateCallbacks(i)
    model.fit(trDat, trLbl, batch_size=16,
              validation_data=(tsDat, tsLbl),
              epochs=150,
              verbose=1,
              steps_per_epoch=len(trDat)/16,
              callbacks=callbacks_list)

    filename = './' + modelname +"_"+ str(i + 1)
    model.save(filename)
    print('>Saved %s' % filename)

#seperate stack
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from numpy import dstack


# load models from file
def load_resNet_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = './' + modelname + "_" + str(i + 1)
        # load model from file
        model = load_model(filename)
        model.load_weights(filename+'.hdf5')
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
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
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

# load all models
members = load_resNet_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
testout = np.argmax(tsLbl,axis=1)
for model in members:
    predicts = model.predict(tsDat)
    predout = np.argmax(predicts,axis=1)

    testScores  = metrics.accuracy_score(testout,predout)
    print('Model Accuracy: %.3f' % testScores)

# fit stacked model using the ensemble
model = fit_stacked_model(members, tsDat, testout)
# evaluate model on test set
yhat = stacked_prediction(members, model, tsDat)
acc = accuracy_score(testout, yhat)
print('Stacked Test Accuracy: %.3f' % acc)