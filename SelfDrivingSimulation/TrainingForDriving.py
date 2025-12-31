import matplotlib.pyplot as plt

from support import *
from sklearn.model_selection import train_test_split
import tensorflow as tf

##### STEP 1 - INITIALIZE DATA
path = 'myDataMapDiff'
data = importDataInfo(path)

##### STEP 2 - VISUALIZATION AND BALANCE DATA
data = balanceData(data, display=True)

##### STEP 3 - PREPROCESSING
imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])

##### STEP 4 - SPLIT DATA TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

##### STEP 5 -

##### STEP 6 - PRE-PROCESSING

##### STEP 7 -

##### STEP 8 - NVIDIA'S MODEL
model = nvidiaModel()
model.summary()

##### STEP 9 - TRAIN
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              min_lr=0.00001)

history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                    steps_per_epoch=500,
                    epochs=30,
                    validation_data=batchGen(xVal, yVal, 100, 0),
                    validation_steps=125,
                    callbacks=[early_stopping, reduce_lr])

##### STEP 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
