class SingleGoalExperiment:
    def __init__(self, trial, writer, experimentValues, samplePosition):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.samplePosition = samplePosition

    def __call__(self, noiseDesignValues, conditionList):
        for trialIndex, condition in enumerate(conditionList):
            playerGrid, target = self.samplePosition(condition)
            results = self.trial(target, playerGrid, noiseDesignValues[trialIndex])
            results["playerGrid"] = str(playerGrid)
            results["target"] = str(target)
            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)


class ObstacleExperiment():
    def __init__(self, normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath):
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.samplePositionFromCondition = samplePositionFromCondition
        self.drawImage = drawImage
        self.resultsPath = resultsPath

    def __call__(self, noiseDesignValues, conditionList):
        for trialIndex, condition in enumerate(conditionList):
            playerGrid, target1, target2, obstacles, decisionSteps, targetDiff = self.samplePositionFromCondition(condition)
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex])
            else:
                results = self.specialTrial(target1, target2, playerGrid, obstacles)
            results["conditionName"] = condition.name
            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)
            results["obstacles"] = str(obstacles)
            results["targetDiff"] = targetDiff
            results["decisionSteps"] = decisionSteps
            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)


class ObstacleModelSimulation():
    def __init__(self, normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath, runVI):
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.samplePositionFromCondition = samplePositionFromCondition
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.runVI = runVI

    def __call__(self, noiseDesignValues, conditionList):
        for trialIndex, condition in enumerate(conditionList):
            playerGrid, target1, target2, obstacles, decisionSteps, targetDiff = self.samplePositionFromCondition(condition)
            QDict = self.runVI((target1, target2), obstacles)

            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex], QDict)
            else:
                results = self.specialTrial(target1, target2, playerGrid, obstacles, QDict)
            results["conditionName"] = condition.name
            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)
            results["obstacles"] = str(obstacles)
            results["targetDiff"] = targetDiff
            results["decisionSteps"] = decisionSteps
            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
