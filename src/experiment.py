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
    def __init__(self, creatMap, normalTrial, specialTrial, writer, experimentValues, drawImage, restTrialIndex, restImage):
        self.creatMap = creatMap
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.drawImage = drawImage
        self.restTrialIndex = restTrialIndex
        self.restImage = restImage

    def __call__(self, noiseDesignValues, expDesignValues):
        for trialIndex, [condition, targetDiff] in enumerate(expDesignValues):
            playerGrid, target1, target2, obstacles, avoidCommitPoint, crossPoint = self.creatMap(condition, targetDiff)
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex], condition.decisionSteps)
            else:
                results = self.specialTrial(target1, target2, playerGrid, obstacles)

            results["conditionName"] = condition.name
            results["decisionSteps"] = str(condition.decisionSteps)
            results["targetDiff"] = targetDiff
            results["avoidCommitPoint"] = str(avoidCommitPoint)
            results["crossPoint"] = str(crossPoint)
            results["obstacles"] = str(obstacles)

            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)

            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
            if trialIndex in self.restTrialIndex:
                self.drawImage(self.restImage)


class ObstacleModelSimulation():
    def __init__(self, creatMap, normalTrial, specialTrial, writer, experimentValues, drawImage, resultsPath, runVI):
        self.creatMap = creatMap
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.runVI = runVI

    def __call__(self, noiseDesignValues, conditionList):
        for trialIndex, [condition, targetDiff] in enumerate(conditionList):
            playerGrid, target1, target2, obstacles, avoidCommitPoint, crossPoint = self.creatMap(condition, targetDiff)
            QDict = self.runVI((target1, target2), obstacles)
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex], condition.decisionSteps, QDict)
            else:
                results = self.specialTrial(target1, target2, playerGrid, obstacles, QDict)

            results["conditionName"] = condition.name
            results["decisionSteps"] = str(condition.decisionSteps)
            results["targetDiff"] = targetDiff
            results["avoidCommitPoint"] = str(avoidCommitPoint)
            results["crossPoint"] = str(crossPoint)
            results["obstacles"] = str(obstacles)

            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)

            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)


class IntentionModelSimulation():
    def __init__(self, creatMap, normalTrial, specialTrial, writer, experimentValues, drawImage, resultsPath, runModel):
        self.creatMap = creatMap
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.runModel = runModel

    def __call__(self, noiseDesignValues, expDesignValues):
        for trialIndex, [condition, targetDiff] in enumerate(expDesignValues):
            playerGrid, target1, target2, obstacles, avoidCommitPoint, crossPoint = self.creatMap(condition, targetDiff)

            policies = self.runModel(target1, target2, obstacles)

            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(policies, target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex], condition.decisionSteps)
            else:
                results = self.specialTrial(policies, target1, target2, playerGrid, obstacles)

            results["conditionName"] = condition.name
            results["decisionSteps"] = str(condition.decisionSteps)
            results["targetDiff"] = targetDiff
            results["avoidCommitPoint"] = str(avoidCommitPoint)
            results["crossPoint"] = str(crossPoint)
            results["obstacles"] = str(obstacles)
            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)

            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)


class OnlineIntentionModelSimulation():
    def __init__(self, creatMap, normalTrial, specialTrial, writer, experimentValues, resultsPath, getPolices):
        self.creatMap = creatMap
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.resultsPath = resultsPath
        self.getPolices = getPolices

    def __call__(self, noiseDesignValues, expDesignValues):
        for trialIndex, [condition, targetDiff] in enumerate(expDesignValues):
            playerGrid, target1, target2, obstacles, avoidCommitPoint, crossPoint = self.creatMap(condition, targetDiff)

            targets = tuple([target1, target2])
            # goalPolicies = [self.runVI(goal, obstacles)[-2] for goal in targets]

            RLPolicy, goalQDict, intentionPolicies = self.getPolices(target1, target2, obstacles)
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(RLPolicy, goalQDict, intentionPolicies, target1, target2, playerGrid, obstacles, noiseDesignValues[trialIndex], condition.decisionSteps)
            else:
                results = self.specialTrial(RLPolicy, goalQDict, intentionPolicies, target1, target2, playerGrid, obstacles)

            results["conditionName"] = condition.name
            results["decisionSteps"] = str(condition.decisionSteps)
            results["targetDiff"] = targetDiff
            results["avoidCommitPoint"] = str(avoidCommitPoint)
            results["crossPoint"] = str(crossPoint)
            results["obstacles"] = str(obstacles)
            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["playerGrid"] = str(playerGrid)
            results["target1"] = str(target1)
            results["target2"] = str(target2)

            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
