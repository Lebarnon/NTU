import random
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

random.seed(0)
class Source:
    def __init__(self, cwnd):
        self.cwnd = cwnd
        self.history = {0: cwnd}

    def updateCwnd(self, round, congestionEvent, alpha, beta):
        if congestionEvent:
            self.cwnd = self.cwnd * (1-beta)
        else:
            self.cwnd += alpha
        self.updateHistory(round)

    def updateHistory(self, round):
        self.history[round] = self.cwnd

class SourceManager:
    def __init__(self, numSources):
        self.numSources = numSources
        self.allSources = {id: Source(random.randint(1, 10)) for id in range(numSources)}
        self.sourcesInUse = {}
        self.addAllSources()
        
    def addAllSources(self):
        self.sourcesInUse = self.allSources
        self.allSources = {}

    def addRandomSource(self):
        if(len(self.allSources) == 0):
            return
        id = random.choice(list(self.allSources.keys()))
        self.sourcesInUse[id] = self.allSources.pop(id)
    
    def removeRandomSource(self):
        if(len(self.sourcesInUse) == 0):
            return
        id = random.choice(list(self.sourcesInUse.keys()))
        self.allSources[id] = self.sourcesInUse.pop(id)
    
    def updateSourcesInUse(self, round, congestionEvent, alpha, beta):
        for id, source in self.sourcesInUse.items():
            source.updateCwnd(round, congestionEvent, alpha, beta)

    def updateSourceInUser(self, id, round, congestionEvent, alpha, beta):
        self.sourcesInUse[id].updateCwnd(round, congestionEvent, alpha, beta)
    
    def totalBandwidth(self):
        if len(self.sourcesInUse) == 0:
            return 0
        return sum([source.cwnd for source in self.sourcesInUse.values()])

    def getAllSources(self):
        all = {}
        for i in range(self.numSources):
            if i in self.sourcesInUse:
                all[i] = self.sourcesInUse[i]
            else:
                all[i] = self.allSources[i]

        return all
            
class Simulation:
    """
    A class representing a simulation of a network congestion control algorithm.

    Attributes:
        numSources (int): The number of sources in the simulation.
        maxIterations (int): The maximum number of iterations for the simulation.
        maxBandwidth (int): The maximum bandwidth allowed in the simulation.
        aimdAlgorithm (AIMDAlgorithm): The AIMD algorithm used for congestion control.
        useRandomEvents (bool): Flag indicating whether to use random events in the simulation.
        utilisationHistory (list): List to store the utilisation rate history.
        congestionHistory (list): List to store the rounds where congestion events occurred.
        maxBandwidthHistory (list): List to store the maximum bandwidth history.
        numOfSourcesHistory (list): List to store the number of sources history.

    Methods:
        run(): Run the simulation.
        runWithRL(): Run the simulation with reinforcement learning.
        randomSourceEvent(): Generate a random source event.
        updateUtilisationHistory(utilisedBandwidth): Update the utilisation rate history.
        moving_average(data, window_size): Calculate the moving average of a data series.
        plot(): Plot the utilisation rate and source history.
    """

    def __init__(self, numSources, maxIterations, maxBandwidth, aimdAlgorithm, useRandomEvents=False):
        self.numSources = numSources
        self.maxIterations = maxIterations
        self.maxBandwidth = maxBandwidth
        self.sourceManager = SourceManager(self.numSources)
        self.aimdAlgorithm = aimdAlgorithm
        self.useRandomEvents = useRandomEvents
        self.utilisationHistory = []
        self.congestionHistory = []
        self.maxBandwidthHistory = []
        self.numOfSourcesHistory = []

    def run(self):
        """
        Run the simulation.

        This method iterates over the specified number of rounds and updates the congestion control parameters
        for each source based on the AIMD algorithm. It also tracks the utilisation rate, congestion events,
        maximum bandwidth, and number of sources history.
        """
        for round in range(self.maxIterations):
            isRandomDrop = self.randomSourceEvent()
            utilisedBandwidth = self.sourceManager.totalBandwidth()
            congestionEvent = utilisedBandwidth > self.maxBandwidth or isRandomDrop
            if congestionEvent:
                self.congestionHistory.append(round)

            for id, source in self.sourceManager.sourcesInUse.items():
                self.aimdAlgorithm.updateParams(
                    self.maxBandwidth,
                    len(self.sourceManager.sourcesInUse),
                    source.cwnd)
                alpha, beta = self.aimdAlgorithm.getParams()
                source.updateCwnd(round, congestionEvent, alpha, beta)

            utilisedBandwidth = self.sourceManager.totalBandwidth()
            # Update all histories
            self.updateUtilisationHistory(utilisedBandwidth)
            self.maxBandwidthHistory.append(self.maxBandwidth)
            self.numOfSourcesHistory.append(len(self.sourceManager.sourcesInUse))

    def runWithLSTM(self):
        """
        Run the simulation with LSTM to predict the max bandwidth and number of sources.

        This method is similar to the `run` method, but it uses reinforcement learning to update the congestion control
        parameters for each source.
        """
        self.lstmModel = LSTMModel()
        nextMaxBandwidth = self.maxBandwidth
        nextNumOfSources = len(self.sourceManager.sourcesInUse)
        for round in range(self.maxIterations):
            isRandomDrop = self.randomSourceEvent()
            utilisedBandwidth = self.sourceManager.totalBandwidth()
            congestionEvent = utilisedBandwidth > self.maxBandwidth or isRandomDrop
            if congestionEvent:
                self.congestionHistory.append(round)

            for id, source in self.sourceManager.sourcesInUse.items():
                self.aimdAlgorithm.updateParams(
                    nextMaxBandwidth,
                    nextNumOfSources,
                    source.cwnd)
                alpha, beta = self.aimdAlgorithm.getParams()
                source.updateCwnd(round, congestionEvent, alpha, beta)

            utilisedBandwidth = self.sourceManager.totalBandwidth()
            # Get next prediction
            nextMaxBandwidth = self.lstmModel.predictMaxBandwidth(self.maxBandwidth)
            nextNumOfSources = self.lstmModel.predictNumOfSources(len(self.sourceManager.sourcesInUse))

            # Update all histories
            self.updateUtilisationHistory(utilisedBandwidth)
            self.maxBandwidthHistory.append(self.maxBandwidth)
            self.numOfSourcesHistory.append(len(self.sourceManager.sourcesInUse))

    def randomSourceEvent(self):
        """
        Generate a random source event.

        This method generates random events such as adding or removing a source, changing the maximum bandwidth,
        or dropping a packet.

        Returns:
            bool: True if a random drop event occurred, False otherwise.
        """
        if not self.useRandomEvents:
            return False

        if random.random() < 0.05:
            self.sourceManager.addRandomSource()
        if random.random() < 0.05:
            self.sourceManager.removeRandomSource()
        # Random bandwidth change event
        if random.random() < 0.05:
            self.maxBandwidth = random.randint(1, 1000)
        # Random drop event
        if random.random() < 0.05:
            return True

    def updateUtilisationHistory(self, utilisedBandwidth):
        """
        Update the utilisation rate history.

        This method calculates the utilisation rate based on the utilised bandwidth and maximum bandwidth,
        and appends it to the utilisation history list.

        Parameters:
            utilisedBandwidth (int): The current utilised bandwidth.
        """
        utilisationRate = utilisedBandwidth / self.maxBandwidth * 100
        self.utilisationHistory.append(utilisationRate if utilisationRate <= 100 else 0)

    def moving_average(self, data, window_size):
        """
        Calculate the moving average of a data series.

        This method calculates the moving average of a data series using a specified window size.

        Parameters:
            data (list): The data series.
            window_size (int): The size of the moving average window.

        Returns:
            list: The smoothed data series.
        """
        weights = np.repeat(1.0, window_size) / window_size
        smoothed_data = np.convolve(data, weights, 'valid')
        return smoothed_data

    def plot(self):
        """
        Plot the utilisation rate and source history.

        This method plots the utilisation rate history and the history of each source in separate subplots.
        It also highlights the rounds where congestion events occurred.
        """
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot utilisation history
        ax1.plot(range(self.maxIterations), self.utilisationHistory, label="Utilisation Rate")
        smoothed_utilisation = self.moving_average(self.utilisationHistory, 30)
        ax1.plot(range(len(smoothed_utilisation)), smoothed_utilisation, label="Smoothed Utilisation Rate")
        ax1.set_xlabel("Round Trip")
        ax1.set_ylabel("Utilisation (%)")
        ax1.set_title("Utilisation History")
        ax1.legend()

        # Plot source history
        for id, source in self.sourceManager.getAllSources().items():
            x_values = []
            y_values = []
            for x in range(self.maxIterations):
                if x in source.history:
                    x_values.append(x)
                    y_values.append(source.history[x])
            ax2.plot(x_values, y_values, label=f"Source {id+1}")
        ax2.set_xlabel("Round Trip")
        ax2.set_ylabel("Source Value")
        ax2.set_title("Source History")
        ax2.legend()

        # Plot vertical lines for congestion events
        for congestion_round in self.congestionHistory:
            ax1.axvline(x=congestion_round, color='r', linestyle='--', label="Congestion Event")
            ax2.axvline(x=congestion_round, color='r', linestyle='--', label="Congestion Event")

        plt.tight_layout()
        plt.show()

    def plotLSTM(self):
        self.lstmModel.plot()

class AIMDAlgorithm:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def getParams(self):
        return self.alpha, self.beta
    def updateParams(self, *args):
        pass

class SimpleAIMDAlgorithm(AIMDAlgorithm):
    def __init__(self, alpha, beta):
        super().__init__(alpha, beta)

class CustomisedHighSpeedAIMDAlgorithm(AIMDAlgorithm):
    def __init__(self, alpha, beta, gapFactor=1, baseRate=1):
        super().__init__(alpha, beta)
        self.gapFactor = gapFactor
        self.baseRate = baseRate
    
    def updateParams(self, maxBandwidth, numSources, currentWindow, *args):
        self.alpha = self.baseRate * math.log(self.gapFactor * abs(maxBandwidth/numSources - currentWindow))
        self.beta = max(0,(currentWindow - maxBandwidth/numSources)) / currentWindow + 0.05

class LSTMModel:
    def __init__(self):
        # scalers
        bandWidth_scaler = MinMaxScaler()
        numOfSources_scaler = MinMaxScaler()

        maxbandwidthHistory_df = pd.read_csv('data/maxBandwidthHistory.csv')
        numOfSourcesHistory_df = pd.read_csv('data/numOfSourcesHistory.csv')

        bandWidth_scaler.fit(maxbandwidthHistory_df[['MaxBandwidth']].values)
        numOfSources_scaler.fit(numOfSourcesHistory_df[['NumOfSources']].values)

        self.bandWidth_scaler = bandWidth_scaler
        self.numOfSources_scaler = numOfSources_scaler

        # load models
        self.bandWidthModel = load_model('models/bandWidthModel.h5')
        self.numOfSourcesModel = load_model('models/numOfSourcesModel.h5')

        # track history
        self.actualMaxBandwidthHistory = []
        self.actualNumOfSourcesHistory = []
        self.maxBandwidthPredictionHistory = []
        self.numOfSourcesPredictionHistory = []

        self.maxBandwidthPredictionWindow = np.array([]).reshape(1, 0, 1)
        self.numOfSourcesPredictionWindow = np.array([]).reshape(1, 0, 1)


    def predictMaxBandwidth(self, curMaxBandWidth):
         # use the prediction to update the batch and remove the first value
        actual = np.array([[curMaxBandWidth]])
        actual = self.bandWidth_scaler.transform(actual)
        formatActual = actual.reshape((1, 1, 1))
        # if the history is not enough, return the current value
        if len(self.maxBandwidthPredictionWindow[0]) < 32:
            self.maxBandwidthPredictionWindow = np.append(
                self.maxBandwidthPredictionWindow,
                formatActual,axis=1)
            return curMaxBandWidth
            
        self.maxBandwidthPredictionWindow = np.append(
            self.maxBandwidthPredictionWindow[:,1:,:],
            formatActual,axis=1)
        # get the prediction value for the first batch
        current_pred = self.bandWidthModel.predict(
            self.maxBandwidthPredictionWindow[
                -min(len(self.maxBandwidthPredictionWindow), 32):
                ], verbose = 0)[0]
        
        current_pred = self.bandWidth_scaler.inverse_transform([current_pred])[0][0]

        self.actualMaxBandwidthHistory.append(curMaxBandWidth)
        self.maxBandwidthPredictionHistory.append(current_pred)

        return current_pred
    def predictNumOfSources(self, curNumOfSources):
         # use the prediction to update the batch and remove the first value
        actual = np.array([[curNumOfSources]])
        actual = self.numOfSources_scaler.transform(actual)
        formatActual = actual.reshape((1, 1, 1))
        
        # if the history is not enough, return the current value
        if len(self.numOfSourcesPredictionWindow[0]) < 32:
            self.numOfSourcesPredictionWindow = np.append(
                self.numOfSourcesPredictionWindow,
                formatActual,axis=1)
            return curNumOfSources
        
        self.numOfSourcesPredictionWindow = np.append(
            self.numOfSourcesPredictionWindow[:,1:,:],
            formatActual,axis=1)
        # get the prediction value for the first batch
        current_pred = self.numOfSourcesModel.predict(
            self.numOfSourcesPredictionWindow[
                -min(len(self.numOfSourcesPredictionWindow), 32):
                ], verbose = 0)[0]
        
        current_pred = self.numOfSources_scaler.inverse_transform([current_pred])[0][0]
        
        self.actualNumOfSourcesHistory.append(curNumOfSources)
        self.numOfSourcesPredictionHistory.append(current_pred)

        return current_pred
    def plot(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        # Plot for actual and predicted max bandwidth
        axs[0].plot(self.actualMaxBandwidthHistory, label='Actual Max Bandwidth')
        axs[0].plot(self.maxBandwidthPredictionHistory, label='Predicted Max Bandwidth')
        axs[0].legend()

        # Plot for actual and predicted number of sources
        axs[1].plot(self.actualNumOfSourcesHistory, label='Actual Number of Sources')
        axs[1].plot(self.numOfSourcesPredictionHistory, label='Predicted Number of Sources')
        axs[1].legend()

        plt.show()
    
# Create an instance of the AIMD algorithm
# simpleAIMDAlgo = SimpleAIMDAlgorithm(1, 0.5)
# simpleSimulation = Simulation(10, 400, 200, simpleAIMDAlgo, True)
# simpleSimulation.run()
# simpleSimulation.plot()
        
# Create an instance of the CustomisedHighSpeedAIMDAlgorithm
customisedAIMDAlgo = CustomisedHighSpeedAIMDAlgorithm(1, 0.5, 1, )
customisedSimulation = Simulation(10, 400, 200, customisedAIMDAlgo, True)
# customisedSimulation.run()
customisedSimulation.runWithLSTM()
customisedSimulation.plot()
customisedSimulation.plotLSTM()


# proActiveSimulation.runWithProActive()
