import random
import numpy as np
import matplotlib.pyplot as plt
import math

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

    def runWithRL(self):
        for round in range(len(self.maxIterations)):
            isRandomDrop = self.randomSourceEvent()
            utilisedBandwidth = self.sourceManager.totalBandwidth()
            congestionEvent = utilisedBandwidth > self.maxBandwidth or isRandomDrop
            if congestionEvent:
                self.congestionHistory.append(round)

            for id, source in self.sourceManager.sourcesInUse.items():
                self.aimdAlgorithm.updateParams(
                    self.maxBandwidth, 
                    self.numSources, 
                    source.cwnd)
                alpha, beta = self.aimdAlgorithm.getParams()
                source.updateCwnd(round, congestionEvent, alpha, beta)
            
            utilisedBandwidth = self.sourceManager.totalBandwidth()
            # Update all histories
            self.updateUtilisationHistory(utilisedBandwidth)
            self.maxBandwidthHistory.append(self.maxBandwidth)
            self.numOfSourcesHistory.append(len(self.sourceManager.sourcesInUse))

    def randomSourceEvent(self):
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
        utilisationRate = utilisedBandwidth/self.maxBandwidth*100
        self.utilisationHistory.append(utilisationRate if utilisationRate <= 100 else 0)

    def moving_average(self, data, window_size):
        weights = np.repeat(1.0, window_size) / window_size
        smoothed_data = np.convolve(data, weights, 'valid')
        return smoothed_data

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
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


# Create an instance of the AIMD algorithm
# simpleAIMDAlgo = SimpleAIMDAlgorithm(1, 0.5)
# simpleSimulation = Simulation(10, 400, 200, simpleAIMDAlgo, True)
# simpleSimulation.run()
# simpleSimulation.plot()
        
# Create an instance of the CustomisedHighSpeedAIMDAlgorithm
# customisedAIMDAlgo = CustomisedHighSpeedAIMDAlgorithm(1, 0.5, 1, )
# customisedSimulation = Simulation(10, 400, 200, customisedAIMDAlgo, False)
# customisedSimulation.run()
# customisedSimulation.plot()
        
# Create a simulation for the pro-active approach
random.seed(100)
# create a customisedAIMDAlgo
customisedAIMDAlgo = CustomisedHighSpeedAIMDAlgorithm(1, 0.5, 1)

# run a simulation with the customisedAIMDAlgo and retrieve the maxBandwidthHistory and numOfSourcesHistory
customisedSimulation = Simulation(10, 400, 200, customisedAIMDAlgo, False)
customisedSimulation.run()
maxBandwidthHistory = customisedSimulation.maxBandwidthHistory
numOfSourcesHistory = customisedSimulation.numOfSourcesHistory

# create a machine learning model that predict the maxBandwidthHistory and numOfSourcesHistory given a turn where the turn is the index of the history
# (Code for creating the machine learning model is not provided)

# use the model to predict the maxBandwidth and maxSources for the next 400 turns
predictedMaxBandwidth = model.predict(maxBandwidthHistory[-1] + list(range(1, 401)))
predictedNumSources = model.predict(numOfSourcesHistory[-1] + list(range(1, 401)))

# create a new simulation with the customisedAIMDAlgo and the predicted maxBandwidth and maxSources
proActiveSimulation = Simulation(10, 400, predictedMaxBandwidth, customisedAIMDAlgo, True)
proActiveSimulation.numSources = predictedNumSources

# run the simulation with the pro-active approach
proActiveSimulation.runWithProActive()
