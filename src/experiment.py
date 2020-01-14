
class Experiment():
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
            playerGrid, bean1Grid, bean2Grid, chooseConditionDF = self.samplePositionFromCondition(condition)
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(bean1Grid, bean2Grid, playerGrid, noiseDesignValues[trialIndex])
            else:
                results = self.specialTrial(bean1Grid, bean2Grid, playerGrid)

            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = chooseConditionDF['playerGrid']
            results["target1"] = chooseConditionDF['target1']
            results["target2"] = chooseConditionDF['target2']
            results["areaType"] = chooseConditionDF['areaType']
            results["distanceDiff"] = chooseConditionDF['distanceDiff']
            results["minDis"] = chooseConditionDF['minDis']
            results["intentionedDisToTargetMin"] = chooseConditionDF['intentionedDisToTargetMin']
            results["avoidCommitmentZone"] = chooseConditionDF['avoidCommitmentZone']

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
            playerGrid, bean1Grid, bean2Grid, chooseConditionDF = self.samplePositionFromCondition(condition)

            playerGrid, bean1Grid, bean2Grid = [(1, 1), (6, 11), (11, 6)]
            obstacles = ((4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4))
            # obstacles = random.choice(obstaclesStates)

            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(bean1Grid, bean2Grid, playerGrid, obstacles, noiseDesignValues[trialIndex])
            else:
                results = self.specialTrial(bean1Grid, bean2Grid, playerGrid)

            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = chooseConditionDF['playerGrid']
            results["target1"] = chooseConditionDF['target1']
            results["target2"] = chooseConditionDF['target2']
            results["areaType"] = chooseConditionDF['areaType']
            results["distanceDiff"] = chooseConditionDF['distanceDiff']
            results["minDis"] = chooseConditionDF['minDis']
            results["intentionedDisToTargetMin"] = chooseConditionDF['intentionedDisToTargetMin']
            results["avoidCommitmentZone"] = chooseConditionDF['avoidCommitmentZone']

            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
