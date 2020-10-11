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

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import minmax_scale
#Loading the dataset
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

num_classes = tsLbl.shape[1]

from tensorflow.keras import optimizers

optmz = optimizers.Adam(lr=0.001)  # Step 1
modelname = 'diabetes-resnetv1'  # Step 2


# Step 3
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
    v = resLyr(inputs, lyrName='Inpt')
    v = resBlkV1(inputs=v,
                 numFilters=16,
                 numBlocks=3,
                 downsampleOnFirst=False,
                 names='Stg1')
    v = resBlkV1(inputs=v,
                 numFilters=16,
                 numBlocks=3,
                 downsampleOnFirst=True,
                 names='stg2')
    v = resBlkV1(inputs=v,
                 numFilters=16,
                 numBlocks=3,
                 downsampleOnFirst=True,
                 names='stg3')
    v = AveragePooling1D(pool_size=2, name="AvgPool")(v)
    v = Flatten()(v)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(v)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optmz,
                  metrics=['accuracy'])

    return model

    # Step 6


model = createResNetV1()  # This is meant for training
modelGo = createResNetV1()  # This is used for final testing

print(model.summary())


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


LRScheduler = LearningRateScheduler(lrSchedule)

# Step 2
folderpath = './'
filepath = folderpath + modelname + ".hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=0,
                             save_best_only=True,
                             mode='max')

csv_logger = CSVLogger(folderpath + modelname + '.csv')  # Step 2
callbacks_list = [checkpoint, csv_logger, LRScheduler]  # Step 3

print("Callbacks created:")
print(callbacks_list[0])
print(callbacks_list[1])
print(callbacks_list[2])
print('')
print("Path to model:", filepath)
print("Path to log:  ", folderpath + modelname + '.csv')

model.fit(trDat, trLbl, batch_size=16,
                    validation_data=(tsDat, tsLbl),
                    epochs=100,
                    verbose=1,
                    steps_per_epoch=len(trDat)/16,
                    callbacks=callbacks_list)

modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy',
                optimizer=optmz,
                metrics=['accuracy'])

predicts    = modelGo.predict(tsDat)
print("Prediction completes.")


labelname   = ['health',          # The label for reporting metrics
               'diabetes',
               ]

predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(tsLbl,axis=1)

testScores  = metrics.accuracy_score(testout,predout)


print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,
                                    predout,
                                    target_names=labelname,
                                    digits=4))

confusion   = metrics.confusion_matrix(testout,predout)
print(confusion)
#
# records     = pd.read_csv(folderpath+modelname +'.csv')
# plt.figure()
# plt.subplot(211)
# plt.plot(records['val_loss'], label="validation")
# plt.plot(records['loss'],label="training")
# plt.yticks([0.00,0.50,1.00,1.50])
# plt.title('Loss value',fontsize=12)
#
# ax          = plt.gca()
# ax.set_xticklabels([])
#
# plt.subplot(212)
# plt.plot(records['val_accuracy'],label="validation")
# plt.plot(records['accuracy'],label="training")
# plt.yticks([0.5,0.6,0.7,0.8])
# plt.title('Accuracy',fontsize=12)
# ax.legend()
# plt.show()

