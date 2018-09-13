import keras
import pandas as pd
import os
from keras.utils.np_utils import to_categorical
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Activation
from matplotlib import pyplot

# Loading the database
# Enter your folder path
Dataset_Path = 'C:\FeedVisor\Deep_Learning\FinalProject\data\dataset'
files = os.listdir(Dataset_Path)
dataset_csv = pd.DataFrame()
for file in files:
    if file.endswith('.csv'):
        cur_df = pd.read_csv(file, usecols=[1,6,8])
        frames = [cur_df, dataset_csv]
        data_set = pd.concat(frames)

data_set['Date'] = pd.to_datetime(data_set.Date)
data_set = data_set[['Date','Demand']]
# Arrange the data by dates - make an avg of all products demands on a specific date
data_set = data_set.groupby('Date').mean()

data_set_vec = data_set.values  # As a Numpy array
data_set_vec = data_set_vec.astype('float32')  # convert demands to float


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = MinMaxScaler(feature_range=(-1, 1))
data_set_vec = scaler.fit_transform(data_set_vec)

# split into train and test sets
train_size = int(len(data_set_vec) * 0.67)
test_size = len(data_set_vec) - train_size
train, test = data_set_vec[0:train_size,:], data_set_vec[train_size:len(data_set_vec),:]

# reshape into X=t and Y=t+1
# input: look_back = 5
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1, activation = 'sigmoid'))
# model.add(Dense(1, activation = 'tanh'))
model.compile(loss='mean_squared_error', optimizer='adam')

# input : num of total runs = 50
for i in range(50):
    print (i)
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
    i=i+1

# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")














