# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import os


# %%
from main import CustomisedHighSpeedAIMDAlgorithm, Simulation

# Create a simulation for the LSTM approach
random.seed(100)

# Create a customisedAIMDAlgo
customisedAIMDAlgo = CustomisedHighSpeedAIMDAlgorithm(1, 0.5, 1)

# Run a simulation with the customisedAIMDAlgo and retrieve the maxBandwidthHistory and numOfSourcesHistory
customisedSimulation = Simulation(10, 4000, 200, customisedAIMDAlgo, True)
customisedSimulation.run()

maxBandwidthHistory = customisedSimulation.maxBandwidthHistory
numOfSourcesHistory = customisedSimulation.numOfSourcesHistory

if not os.path.exists('data'):
    os.makedirs('data')

# Export maxBandwidthHistory_df to a CSV file
maxBandwidthHistory_df = pd.DataFrame({'Index': range(len(maxBandwidthHistory)), 'MaxBandwidth': maxBandwidthHistory})
maxBandwidthHistory_df.to_csv('data/maxBandwidthHistory.csv', index=False)

# Export numOfSourcesHistory_df to a CSV file
numOfSourcesHistory_df = pd.DataFrame({'Index': range(len(numOfSourcesHistory)), 'NumOfSources': numOfSourcesHistory})
numOfSourcesHistory_df.to_csv('data/numOfSourcesHistory.csv', index=False)


# %%
maxBandwidthHistory_df.sample(10)

# %%
numOfSourcesHistory_df.sample(10)

# %%
numOfSourcesHistory_df['NumOfSources'].plot(figsize=(12,6))


# %%
# Normalise the maxBandwidthHistory and numOfSourcesHistory
bandWidth_scaler = MinMaxScaler()
numOfSources_scaler = MinMaxScaler()

# %%
bandWidth_scaler.fit(maxBandwidthHistory_df[['MaxBandwidth']])
maxBandwidthHistory_scaled_train = bandWidth_scaler.transform(maxBandwidthHistory_df[['MaxBandwidth']])

numOfSources_scaler.fit(numOfSourcesHistory_df[['NumOfSources']])
numOfSourcesHistory_scaled_train = numOfSources_scaler.transform(numOfSourcesHistory_df[['NumOfSources']])


# %%
maxBandwidthHistory_scaled_train[:10],numOfSourcesHistory_scaled_train[:10]

# %%
from keras.utils import timeseries_dataset_from_array

# %%
# Create a batched dataset for the maxBandwidthHistory and numOfSourcesHistory
seqLen = 64
maxBandwidthBatchedDataset = timeseries_dataset_from_array(
    maxBandwidthHistory_scaled_train[0:-seqLen], 
    maxBandwidthHistory_scaled_train[seqLen:], 
    sequence_length=seqLen, 
    sampling_rate=1, 
    batch_size=1)
numOfSourceBatchedDataset = timeseries_dataset_from_array(
    numOfSourcesHistory_scaled_train[0:-seqLen], 
    numOfSourcesHistory_scaled_train[seqLen:], 
    sequence_length=seqLen, 
    sampling_rate=1, 
    batch_size=1)

# %%
example = list(numOfSourceBatchedDataset.as_numpy_iterator())
example[100]

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# %%
# define model
bandWidthModel = Sequential()
bandWidthModel.add(LSTM(128, activation='relu', input_shape=(seqLen, 1)))
bandWidthModel.add(Dense(1))
bandWidthModel.compile(optimizer='adam', loss='mse')

numOfSourcesModel = Sequential()
numOfSourcesModel.add(LSTM(128, activation='relu', input_shape=(seqLen, 1)))
numOfSourcesModel.add(Dense(1))
numOfSourcesModel.compile(optimizer='adam', loss='mse')

# %%
# fit model
bandWidthModel.fit(maxBandwidthBatchedDataset,epochs=25)

# %%
if not os.path.exists('models'):
    os.makedirs('models')

# Save bandWidthModel
bandWidthModel.save('models/bandWidthModel.h5')

# %%
numOfSourcesModel.fit(numOfSourceBatchedDataset,epochs=25)

# %%
# Save bandWidthModel
numOfSourcesModel.save('models/numOfSourcesModel.h5')

# %%
test_predictions = []

first_eval_batch = bandWidth_scaler.transform([[300] for _ in range(32)] + [[400] for _ in range(32)])
current_batch = first_eval_batch.reshape((1, 64, 1))
for i in range(10):
    
    # get the prediction value for the first batch
    current_pred = bandWidthModel.predict(current_batch)[0]

    # append the prediction into the array
    test_predictions.append(current_pred) 

    # use the prediction to update the batch and remove the first value
    actual = first_eval_batch[i]
    formatActual = actual.reshape((1, 1, 1))
    current_batch = np.append(current_batch[:,1:,:],formatActual,axis=1)

# %%
results = []
for _ in range(10):
    results.append(bandWidth_scaler.inverse_transform(test_predictions))
results

# %%
test_predictions


