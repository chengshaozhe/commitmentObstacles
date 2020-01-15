
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
            playerGrid, target1, target2, obstacles, minSteps = self.samplePositionFromCondition(condition)

            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex])
            else:
                results = self.specialTrial(target1, target2, playerGrid)

            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)
            results["minSteps"] = str(minSteps)

            # results["areaType"] = chooseConditionDF['areaType']
            # results["distanceDiff"] = chooseConditionDF['distanceDiff']
            # results["minDis"] = chooseConditionDF['minDis']
            # results["intentionedDisToTargetMin"] = chooseConditionDF['intentionedDisToTargetMin']
            # results["avoidCommitmentZone"] = chooseConditionDF['avoidCommitmentZone']

            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
