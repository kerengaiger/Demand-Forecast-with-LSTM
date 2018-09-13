import pandas as pd
import dask
import numpy
from keras.models import model_from_json

# Make Predictions:

batch_size = 1
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# predict
# Input : insert a sequence of specific product demands from the last 5 days (look_back = 5)
# Example :
obs = numpy.array([[1,4,5,0,0]])
obs_reshaped = numpy.reshape(obs, (obs.shape[0], obs.shape[1], 1))
obs_predict = loaded_model.predict(obs_reshaped, batch_size=batch_size)
loaded_model.reset_states()

print('The demand for the next day is:',obs_predict)
