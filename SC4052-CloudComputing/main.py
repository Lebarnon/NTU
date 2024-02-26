import random
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

random.seed(10)
class Source:
    """
    Represents a source with a congestion window (cwnd) and a history of cwnd values.

    Attributes:
        cwnd (float): The current congestion window size.
        history (dict): A dictionary that stores the cwnd values for each round.

    Methods:
        updateCwnd(round, congestionEvent, alpha, beta): Updates the cwnd based on the congestion event and parameters.
        updateHistory(round): Updates the history dictionary with the current cwnd value.
    """

    def __init__(self, cwnd):
        self.cwnd = cwnd
        self.history = {0: cwnd}

    def updateCwnd(self, round, congestionEvent, alpha, beta):
        if congestionEvent:
            self.cwnd = self.cwnd * beta
        else:
            self.cwnd += alpha
        self.updateHistory(round)

    def updateHistory(self, round):
        self.history[round] = self.cwnd

class SourceManager:
    """
    A class that manages sources for a cloud computing system.

    Attributes:
        numSources (int): The total number of sources.
        allSources (dict): A dictionary containing all available sources.
        sourcesInUse (dict): A dictionary containing sources currently in use.

    Methods:
        __init__(self, numSources): Initializes the SourceManager object.
        addAllSources(self): Moves all sources to the sourcesInUse dictionary.
        addRandomSource(self): Moves a random source from allSources to sourcesInUse.
        removeRandomSource(self): Moves a random source from sourcesInUse to allSources.
        updateSourcesInUse(self, round, congestionEvent, alpha, beta): Updates the cwnd of all sources in sourcesInUse.
        updateSourceInUser(self, id, round, congestionEvent, alpha, beta): Updates the cwnd of a specific source in sourcesInUse.
        totalBandwidth(self): Calculates the total bandwidth of all sources in sourcesInUse.
        getAllSources(self): Returns a dictionary containing all sources, including those in sourcesInUse and allSources.
    """
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
    def __init__(self, numSources, maxIterations, maxBandwidth, aimdAlgorithm, useRandomEvents=False):
        """
        Initialize the Simulation class.

        Parameters:
            numSources (int): The number of sources in the simulation.
            maxIterations (int): The maximum number of iterations for the simulation.
            maxBandwidth (int): The maximum bandwidth for the simulation.
            aimdAlgorithm (AIMDAlgorithm): The AIMD algorithm used for congestion control.
            useRandomEvents (bool, optional): Flag indicating whether to use random events in the simulation. Defaults to False.
        """
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
        """
        Plot the LSTM model prediction results.

        This method plots the LSTM model used for predicting the max bandwidth and number of sources.
        """
        self.lstmModel.plot()

class AIMDAlgorithm:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def getParams(self):
        self.alpha = max(0, self.alpha)
        return self.alpha, self.beta
    def updateParams(self, *args):
        pass

class SimpleAIMDAlgorithm(AIMDAlgorithm):
    def __init__(self, alpha, beta):
        super().__init__(alpha, beta)

class CustomisedHighSpeedAIMDAlgorithm(AIMDAlgorithm):
    """
    A custom implementation of the High-Speed AIMD (Additive Increase Multiplicative Decrease) algorithm.
    
    Args:
        alpha (float): The additive increase factor.
        beta (float): The multiplicative decrease factor.
        gapFactor (float, optional): The gap factor used in the calculation of alpha. Defaults to 1.
        baseRate (float, optional): The base rate used in the calculation of alpha. Defaults to 1.
    """
    def __init__(self, alpha, beta, gapFactor=1, baseRate=1):
        super().__init__(alpha, beta)
        self.gapFactor = gapFactor
        self.baseRate = baseRate
    
    def updateParams(self, maxBandwidth, numSources, currentWindow, *args):
        """
        Updates the parameters alpha and beta based on the given inputs.
        
        Args:
            maxBandwidth (float): The maximum available bandwidth.
            numSources (int): The number of sources.
            currentWindow (float): The current congestion window size.
            *args: Additional arguments (not used in this method).
        """
        self.alpha = 0 if maxBandwidth/numSources <= currentWindow else self.baseRate * math.log(self.gapFactor * (maxBandwidth/numSources - currentWindow))
        self.beta = max(0,(currentWindow - maxBandwidth/numSources)) / currentWindow + 0.05

class LSTMModel:
    """
    LSTMModel class for predicting max bandwidth and number of sources using LSTM models.

    Attributes:
        bandWidth_scaler (MinMaxScaler): Scaler for max bandwidth values.
        numOfSources_scaler (MinMaxScaler): Scaler for number of sources values.
        bandWidthModel (keras.Model): LSTM model for predicting max bandwidth.
        numOfSourcesModel (keras.Model): LSTM model for predicting number of sources.
        actualMaxBandwidthHistory (list): List to track actual max bandwidth values.
        actualNumOfSourcesHistory (list): List to track actual number of sources values.
        maxBandwidthPredictionHistory (list): List to track predicted max bandwidth values.
        numOfSourcesPredictionHistory (list): List to track predicted number of sources values.
        maxBandwidthPredictionWindow (numpy.ndarray): Window of max bandwidth values for prediction.
        numOfSourcesPredictionWindow (numpy.ndarray): Window of number of sources values for prediction.
    """

    def __init__(self):
        """
        Initializes the class instance.

        This method performs the following tasks:
        1. Initializes the scalers for bandwidth and number of sources.
        2. Reads the maxBandwidthHistory and numOfSourcesHistory data from CSV files.
        3. Fits the scalers with the respective data.
        4. Loads the bandWidthModel and numOfSourcesModel from saved files.
        5. Initializes the history and prediction windows for bandwidth and number of sources.
        """
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
        """
        Predicts the max bandwidth based on the current max bandwidth value.

        Args:
            curMaxBandWidth (float): Current max bandwidth value.

        Returns:
            float: Predicted max bandwidth value.
        """
        # use the prediction to update the batch and remove the first value
        actual = np.array([[curMaxBandWidth]])
        actual = self.bandWidth_scaler.transform(actual)
        formatActual = actual.reshape((1, 1, 1))
        # if the history is not enough, return the current value
        if len(self.maxBandwidthPredictionWindow[0]) < 64:
            self.maxBandwidthPredictionWindow = np.append(
                self.maxBandwidthPredictionWindow,
                formatActual, axis=1)
            return curMaxBandWidth

        self.maxBandwidthPredictionWindow = np.append(
            self.maxBandwidthPredictionWindow[:, 1:, :],
            formatActual, axis=1)
        # get the prediction value for the first batch
        current_pred = self.bandWidthModel.predict(
            self.maxBandwidthPredictionWindow[
                -min(len(self.maxBandwidthPredictionWindow), 64):
            ], verbose=0)[0]

        current_pred = self.bandWidth_scaler.inverse_transform([current_pred])[0][0]

        self.actualMaxBandwidthHistory.append(curMaxBandWidth)
        self.maxBandwidthPredictionHistory.append(current_pred)

        return max(1,current_pred)

    def predictNumOfSources(self, curNumOfSources):
        """
        Predicts the number of sources based on the current number of sources value.

        Args:
            curNumOfSources (float): Current number of sources value.

        Returns:
            float: Predicted number of sources value.
        """
        # use the prediction to update the batch and remove the first value
        actual = np.array([[curNumOfSources]])
        actual = self.numOfSources_scaler.transform(actual)
        formatActual = actual.reshape((1, 1, 1))

        # if the history is not enough, return the current value
        if len(self.numOfSourcesPredictionWindow[0]) < 64:
            self.numOfSourcesPredictionWindow = np.append(
                self.numOfSourcesPredictionWindow,
                formatActual, axis=1)
            return curNumOfSources

        self.numOfSourcesPredictionWindow = np.append(
            self.numOfSourcesPredictionWindow[:, 1:, :],
            formatActual, axis=1)
        # get the prediction value for the first batch
        current_pred = self.numOfSourcesModel.predict(
            self.numOfSourcesPredictionWindow[
                -min(len(self.numOfSourcesPredictionWindow), 64):
            ], verbose=0)[0]

        current_pred = self.numOfSources_scaler.inverse_transform([current_pred])[0][0]

        self.actualNumOfSourcesHistory.append(curNumOfSources)
        self.numOfSourcesPredictionHistory.append(current_pred)

        return current_pred

    def plot(self):
        """
        Plots the actual and predicted max bandwidth and number of sources values.
        """
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
    
def mainMenu():
    print("Select an option:")
    print("1. Run Simulation with Simple AIMD")
    print("2. Run Simulation with customised AIMD")
    print("3. Run Simulation with customised AIMD enhanced with LSTM")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        runSimpleSimulation()
    elif choice == "2":
        runCustomisedSimulation()
    elif choice == "3":
        runCustomisedSimulationWithLSTM()
    elif choice == "4":
        exit()
    else:
        print("Invalid choice. Please try again.")
        mainMenu()

def runSimpleSimulation():
    # Create an instance of the AIMD algorithm
    simpleAIMDAlgo = SimpleAIMDAlgorithm(1, 0.5)
    simpleSimulation = Simulation(10, 400, 200, simpleAIMDAlgo, True)
    simpleSimulation.run()
    simpleSimulation.plot()
    mainMenu()

customisedAIMDAlgo = CustomisedHighSpeedAIMDAlgorithm(1, 0.5, 1)
def runCustomisedSimulation():
    # Create an instance of the CustomisedHighSpeedAIMDAlgorithm
    customisedSimulation = Simulation(10, 400, 200, customisedAIMDAlgo, True)
    customisedSimulation.run()
    customisedSimulation.plot()
    mainMenu()

def runCustomisedSimulationWithLSTM():
    customisedSimulation = Simulation(10, 400, 200, customisedAIMDAlgo, True)
    customisedSimulation.runWithLSTM()
    customisedSimulation.plot()
    customisedSimulation.plotLSTM()
    mainMenu()

mainMenu()
