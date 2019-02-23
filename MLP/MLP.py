import MLPDataGen as dataGen
import MLPFunctions as mLPFunctions
import numpy as np
import MLPTraining as mLPTraining
import UserInput as userInput

data = dataGen.create()
networkConfig = userInput.Read()
mLPTraining.train(data["X"], data["y"], networkConfig["layerCount"], networkConfig["nodeCounts"])