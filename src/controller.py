import numpy as np
import pygame as pg
import random
from pygame import time
from scipy.stats import entropy


class HumanController():
    def __init__(self, actionDict, responseTimeLimits):
        self.actionDict = actionDict
        self.responseTimeLimits = responseTimeLimits

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        action = [0, 0]
        isReactionTimely = True
        pause = True
        startTime = time.get_ticks()
        while pause:
            if time.get_ticks() - startTime < self.responseTimeLimits:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                    if event.type == pg.KEYDOWN:
                        if event.key in self.actionDict.keys():
                            action = self.actionDict[event.key]
                            aimePlayerGrid = tuple(np.add(playerGrid, action))
                            isReactionTimely = True
                            pause = False
                        if event.key == pg.K_ESCAPE:
                            pg.quit()
                            exit()
            else:
                isReactionTimely = False
                action = random.choice(list(self.actionDict.values()))
                aimePlayerGrid = tuple(np.add(playerGrid, action))
                pause = False
        return aimePlayerGrid, action, isReactionTimely


class HumanControllerWithTimePressure():
    def __init__(self, actionDict):
        self.actionDict = actionDict

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        action = [0, 0]
        pause = True
        while pause:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    exit()
                if event.type == pg.KEYDOWN:
                    if event.key in self.actionDict.keys():
                        action = self.actionDict[event.key]
                        aimePlayerGrid = tuple(np.add(playerGrid, action))
                        pause = False
        return aimePlayerGrid, action


class ModelController():
    def __init__(self, policy, gridSize, softmaxBeta):
        self.policy = policy
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        try:
            policyForCurrentStateDict = self.policy[(playerGrid, (targetGrid1, targetGrid2))]
        except KeyError as e:
            policyForCurrentStateDict = self.policy[(playerGrid, (targetGrid2, targetGrid1))]
        if self.softmaxBeta < 0:
            actionMaxList = [action for action in policyForCurrentStateDict.keys() if
                             policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
            action = random.choice(actionMaxList)
        else:

            actionValue = list(policyForCurrentStateDict.values())
            softmaxProbabilityList = calculateSoftmaxProbability(actionValue, self.softmaxBeta)
            action = list(policyForCurrentStateDict.keys())[
                list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
        aimePlayerGrid = tuple(np.add(playerGrid, action))
        # pg.time.delay(500)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    exit()
        return aimePlayerGrid, action


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


def inferGoal(originGrid, aimGrid, targetGridA, targetGridB):
    pacmanBean1aimDisplacement = calculateGridDis(targetGridA, aimGrid)
    pacmanBean2aimDisplacement = calculateGridDis(targetGridB, aimGrid)
    pacmanBean1LastStepDisplacement = calculateGridDis(targetGridA, originGrid)
    pacmanBean2LastStepDisplacement = calculateGridDis(targetGridB, originGrid)
    bean1Goal = pacmanBean1LastStepDisplacement - pacmanBean1aimDisplacement
    bean2Goal = pacmanBean2LastStepDisplacement - pacmanBean2aimDisplacement
    if bean1Goal > bean2Goal:
        goal = 1
    elif bean1Goal < bean2Goal:
        goal = 2
    else:
        goal = 0
    return goal


def calculateSoftmaxProbability(acionValues, beta):
    exponents = np.multiply(beta, acionValues)
    exponents = np.array([min(700, exponent) for exponent in exponents])
    newProbabilityList = list(np.divide(np.exp(exponents), np.sum(np.exp(exponents))))
    return newProbabilityList


class NormalNoise():
    def __init__(self, actionSpace, gridSize):
        self.actionSpace = actionSpace
        self.gridSize = gridSize

    def __call__(self, playerGrid, action, noiseStep, stepCount):
        if stepCount in noiseStep:
            realAction = random.choice(self.actionSpace)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, realAction


class AimActionWithNoise():
    def __init__(self, actionSpace, gridSize):
        self.actionSpace = actionSpace
        self.gridSize = gridSize

    def __call__(self, playerGrid, action, noiseStep, stepCount):
        if stepCount in noiseStep:
            actionSpace = self.actionSpace.copy()
            actionSpace.remove(action)
            actionList = [str(action) for action in actionSpace]
            actionStr = np.random.choice(actionList)
            realAction = eval(actionStr)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, realAction


# def calMidPoints(target1, target2, zone):
#     midpoints = list([(target1[0], target2[1]), (target2[0], target1[1])])
#     midPoint = list(set(zone).intersection(set(midpoints)))[0]
#     return midPoint

def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZoneAll(playerGrid, target1, target2):
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
    avoidCommitmentZone.remove(tuple(playerGrid))
    return avoidCommitmentZone


def calculateInsideArea(playerGrid, target1, target2):
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    insideArea = list(set(rect1).union(set(rect2)))
    return insideArea


def calMidPoints(initPlayerGrid, target1, target2):
    zone = calculateAvoidCommitmnetZoneAll(initPlayerGrid, target1, target2)
    if zone:
        playerDisToZoneGrid = [calculateGridDis(initPlayerGrid, zoneGrid) for zoneGrid in zone]
        midPointIndex = np.argmax(playerDisToZoneGrid)
        midPoint = zone[midPointIndex]
    else:
        midPoint = []
    return midPoint


def backToCrossPointNoise(trajectory, target1, target2, noiseStep, firstIntentionFlag):
    playerGrid = tuple(trajectory[-1])
    initPlayerGrid = tuple(trajectory[0])
    midpoint = calMidPoints(initPlayerGrid, target1, target2)
    realPlayerGrid = None
    zone = calculateAvoidCommitmnetZoneAll(initPlayerGrid, target1, target2)
    if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
        realPlayerGrid = midpoint
        noiseStep = len(trajectory)
        firstIntentionFlag = True
    return realPlayerGrid, noiseStep, firstIntentionFlag


def backToZoneNoise(playerGrid, trajectory, noiseStep, firstIntentionFlag):
    realPlayerGrid = None
    if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
        realPlayerGrid = trajectory[-3]
        noiseStep = len(trajectory)
        firstIntentionFlag = True
    return realPlayerGrid, noiseStep, firstIntentionFlag


class SampleToZoneNoise:
    def __init__(self, noiseActionSpace):
        self.noiseActionSpace = noiseActionSpace

    def __call__(self, playerGrid, trajectory, zone, noiseStep, firstIntentionFlag):
        realPlayerGrid = None
        if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
            possibleGrid = (tuple(np.add(playerGrid, action)) for action in self.noiseActionSpace)
            realPlayerGrids = list(filter(lambda x: x in zone, possibleGrid))
            realPlayerGrid = random.choice(realPlayerGrids)
            noiseStep = len(trajectory)
            firstIntentionFlag = True
        return realPlayerGrid, noiseStep, firstIntentionFlag


def isGridsNotALine(playerGrid, bean1Grid, bean2Grid):
    line = np.array((playerGrid, bean1Grid, bean2Grid)).T
    if len(set(line[0])) != len(line[0]) or len(set(line[1])) != len(line[1]):
        return False
    else:
        return True


class SampleToZoneNoiseNoLine:
    def __init__(self, noiseActionSpace):
        self.noiseActionSpace = noiseActionSpace

    def __call__(self, playerGrid, bean1Grid, bean2Grid, trajectory, zone, noiseStep, firstIntentionFlag):
        realPlayerGrid = None
        if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
            possibleGrid = (tuple(np.add(playerGrid, action)) for action in self.noiseActionSpace)
            realPlayerGrids = tuple(filter(lambda x: x in zone, possibleGrid))
            noLineGrids = list(filter(lambda x: isGridsNotALine(x, bean1Grid, bean2Grid), realPlayerGrids))
            if noLineGrids:
                realPlayerGrid = random.choice(noLineGrids)
            else:
                realPlayerGrid = random.choice(realPlayerGrids)
            noiseStep = len(trajectory)
            firstIntentionFlag = True
        return realPlayerGrid, noiseStep, firstIntentionFlag


def selectActionMinDistanceFromTarget(goal, playerGrid, bean1Grid, bean2Grid, actionSpace):
    allPosiibilePlayerGrid = [np.add(playerGrid, action) for action in actionSpace]
    allActionGoal = [inferGoal(playerGrid, possibleGrid, bean1Grid, bean2Grid) for possibleGrid in allPosiibilePlayerGrid]
    if goal == 1:
        realActionIndex = allActionGoal.index(2)
    else:
        realActionIndex = allActionGoal.index(1)
    realAction = actionSpace[realActionIndex]
    return realAction


class AwayFromTheGoalNoise():
    def __init__(self, actionSpace, gridSize):
        self.actionSpace = actionSpace
        self.gridSize = gridSize

    def __call__(self, playerGrid, bean1Grid, bean2Grid, action, goal, firstIntentionFlag, noiseStep, stepCount):
        if goal != 0 and not firstIntentionFlag:
            noiseStep.append(stepCount)
            firstIntentionFlag = True
            realAction = selectActionMinDistanceFromTarget(goal, playerGrid, bean1Grid, bean2Grid, self.actionSpace)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, firstIntentionFlag, noiseStep


class IsInBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        if position[0] > self.xMax or position[0] < self.xMin or position[1] > self.yMax or position[1] < self.yMin:
            return False
        else:
            return True


class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        adjustedX, adjustedY = position
        if position[0] >= self.xMax:
            adjustedX = self.xMax
        if position[0] <= self.xMin:
            adjustedX = self.xMin
        if position[1] >= self.yMax:
            adjustedY = self.yMax
        if position[1] <= self.yMin:
            adjustedY = self.yMin
        checkedPosition = (adjustedX, adjustedY)
        return checkedPosition


class SampleSoftmaxAction:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, Q_dict, playerGrid):
        actionDict = Q_dict[playerGrid]
        actionKeys = list(Q_dict[playerGrid].keys())
        actionValues = list(Q_dict[playerGrid].values())

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)
        aimPlayerGrid = tuple(np.add(playerGrid, action))
        return aimPlayerGrid, action


def chooseMaxAcion(actionDict):
    actionMaxList = [action for action in actionDict.keys() if
                     actionDict[action] == np.max(list(actionDict.values()))]
    action = random.choice(actionMaxList)
    return action


def chooseSoftMaxAction(actionDict, softmaxBeta):
    actionValue = list(actionDict.values())
    softmaxProbabilityList = calculateSoftmaxProbability(actionValue, softmaxBeta)
    action = list(actionDict.keys())[
        list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
    return action


def calBasePolicy(posteriorList, actionProbList):
    basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(posteriorList, actionProbList)]
    basePolicy = np.sum(basePolicyList, axis=0)
    return basePolicy


def sigmoidScale(x, intensity=30, threshold=0.2):
    return 1 / (1 + np.exp(- intensity * (x - threshold)))


class CalPerceivedIntentions:
    def __init__(self, intensity, threshold):
        self.intensity = intensity
        self.threshold = threshold

    def __call__(self, bayesIntentions):
        a, b = bayesIntentions
        diff = abs(a - b)
        perceivedDiff = sigmoidScale(diff, self.intensity, self.threshold)

        if a > b:
            aNew = (perceivedDiff + 1) / 2
        else:
            aNew = (1 - perceivedDiff) / 2

        perceivedIntentions = [aNew, 1 - aNew]
        return perceivedIntentions


def getSoftmaxGoalPolicy(Q_dict, playerGrid, target, softmaxBeta):
    actionDict = Q_dict[(playerGrid, target)]
    actionValues = list(actionDict.values())
    softmaxProbabilityList = calculateSoftmaxProbability(actionValues, softmaxBeta)
    softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
    return softMaxActionDict


class ActWithMonitorIntention:
    def __init__(self, softmaxBeta, calPerceivedIntentions):
        self.softmaxBeta = softmaxBeta
        self.calPerceivedIntentions = calPerceivedIntentions

    def __call__(self, intentionPolicies, playerGrid, target1, target2, priorList):
        targets = list([target1, target2])
        perceivedIntentions = self.calPerceivedIntentions(priorList)

        targetIndex = list(np.random.multinomial(1, perceivedIntentions)).index(1)
        goal = targets[targetIndex]

        actionDict = intentionPolicies[targetIndex][playerGrid, goal]
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)
        aimPlayerGrid = tuple(np.add(playerGrid, action))
        return aimPlayerGrid, action


class ActWithPerceviedIntention:
    def __init__(self, softmaxBeta, calPerceivedIntentions, intentionThreshold):
        self.softmaxBeta = softmaxBeta
        self.calPerceivedIntentions = calPerceivedIntentions
        self.intentionThreshold = intentionThreshold

    def __call__(self, RLPolicy, intentionPolicies, playerGrid, target1, target2, priorList):
        targets = list([target1, target2])

        if abs(priorList[0] - priorList[1]) < self.intentionThreshold:
            # perceivedIntentions = [0.5, 0.5]
            actionDict = RLPolicy[playerGrid]
        else:
            # perceivedIntentions = self.calPerceivedIntentions(priorList)
            perceivedIntentions = priorList
            targetIndex = list(np.random.multinomial(1, perceivedIntentions)).index(1)
            goal = targets[targetIndex]
            actionDict = intentionPolicies[targetIndex][playerGrid, goal]

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)
        aimPlayerGrid = tuple(np.add(playerGrid, action))
        return aimPlayerGrid, action


class ModelControllerWithGoal:
    def __init__(self, softmaxBeta, goalPolicy):
        self.softmaxBeta = softmaxBeta
        self.goalPolicy = goalPolicy

    def __call__(self, playerGrid, targetGrid1, targetGrid2, priorList):
        targets = list([targetGrid1, targetGrid2])
        actionProbList = [list(self.goalPolicy(playerGrid, goal).values()) for goal in targets]
        actionProbs = calBasePolicy(priorList, actionProbList)

        actionKeys = self.goalPolicy(playerGrid, targetGrid1).keys()
        actionDict = dict(actionKeys, actionProbs)

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


class ModelControllerOnlineReward:
    def __init__(self, softmaxBeta, goalPolicy):
        self.softmaxBeta = softmaxBeta
        self.goalPolicy = goalPolicy

    def __call__(self, playerGrid, targetGrid1, targetGrid2, goalRewardList):
        QDict = runVI((targetGrid1, targetGrid2), goalRewardList)
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


class ModelControllerOnline:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2, QDict):
        actionDict = QDict[(playerGrid, (targetGrid1, targetGrid2))]
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)
        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


class AvoidCommitModel:
    def __init__(self, softmaxBeta, actionSpace, checkBoundary):
        self.softmaxBeta = softmaxBeta
        self.actionSpace = actionSpace
        self.checkBoundary = checkBoundary

    def __call__(self, playerGrid, targetGrid1, targetGrid2, Q_dict):
        actionInformationList = []
        for aimAction in self.actionSpace:
            aimNextState = tuple(np.add(playerGrid, aimAction))

            nextState = self.checkBoundary(aimNextState)
            rlActionDict = Q_dict[(nextState, (targetGrid1, targetGrid2))]
            rlActionValues = list(rlActionDict.values())
            softmaxRLPolicy = calculateSoftmaxProbability(rlActionValues, self.softmaxBeta)

            actionInformation = entropy(softmaxRLPolicy)
            actionInformationList.append(actionInformation)

        action = self.actionSpace[np.argmax(actionInformationList)]
        aimePlayerGrid = self.checkBoundary((tuple(np.add(playerGrid, action))))
        return aimePlayerGrid, action


class InferGoalPosterior:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, action, target1, target2, priorList, goalQDicts):
        targets = [target1, target2]

        goalPolicies = [getSoftmaxGoalPolicy(Q_dict, playerGrid, goal, self.softmaxBeta) for Q_dict, goal in zip(goalQDicts, targets)]

        likelihoodList = [goalPolicies[goalIndex].get(action) for goalIndex, goal in enumerate(targets)]
# round
#         likelihoodList = [round(likelihood, 3) for likelihood in likelihoodList]
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
        evidence = sum(posteriorUnnormalized)
        posteriors = [posterior / evidence for posterior in posteriorUnnormalized]
        return posteriors


class ActWithMonitorIntentionThreshold:
    def __init__(self, softmaxBeta, intentionThreshold):
        self.softmaxBeta = softmaxBeta
        self.intentionThreshold = intentionThreshold

    def __call__(self, RLPolicy, intentionPolicies, playerGrid, target1, target2, priorList):
        targets = list([target1, target2])
        if abs(priorList[0] - priorList[1]) < self.intentionThreshold:
            actionDict = RLPolicy[playerGrid]
        else:
            targetIndex = list(np.random.multinomial(1, priorList)).index(1)
            goal = targets[targetIndex]
            actionDict = intentionPolicies[targetIndex][playerGrid, goal]
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)
        aimPlayerGrid = tuple(np.add(playerGrid, action))
        return aimPlayerGrid, action


if __name__ == '__main__':
    softmaxBeta = 2.5
    inferGoalPosterior = InferGoalPosterior(softmaxBeta)
    trajectory = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (1, 6), (1, 7), (1, 8), (2, 8), (1, 9), (2, 9), (3, 9), (4, 9)]
    aimAction = [(0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]

    posteriorList = []
    priorList = [0.5, 0.5]

    import os
    import sys
    sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
    from machinePolicy.showIntentionModel import RunVI, GetShowIntentionPolices

    noise = 0.067
    gamma = 0.9
    goalReward = 30
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    gridSize = 15

    threshold = 1
    infoScale = 2
    softmaxBetaInfer = 2.5
    softmaxBetaAct = 2.5

    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)
    getPolices = GetShowIntentionPolices(runVI, softmaxBetaAct, infoScale)
    target1 = (9, 4)
    target2 = (4, 9)
    obstacles = [(1, 1), (1, 3), (3, 1), (1, 4), (4, 1), (8, 0), (12, 0), (13, 10), (3, 0), (6, 7), (7, 10), (5, 0), (11, 10), (13, 1), (10, 12), (13, 12), (8, 9), (7, 13)]
    RLQDict, goalQDicts, intentionQDict = getPolices(target1, target2, obstacles)

    for playerGrid, action in zip(trajectory, aimAction):
        posteriors = inferGoalPosterior(playerGrid, action, target1, target2, priorList, goalQDicts)
        posteriorList.append(posteriors)
        priorList = posteriors
    posteriorList.insert(0, [0.5, 0.5])
    print(posteriorList)
