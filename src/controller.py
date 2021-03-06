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
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))

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
            softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
            action = actionKeys[
                list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]

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


class InferGoalPosterior:
    def __init__(self, runVI, commitBeta):
        self.runVI = runVI
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, action, target1, target2, priorList):
        targets = list([target1, target2])
        goalPolicy = [self.runVI(goal, obstacles) for goal in targets]
# todo
        actionValueList = [goalPolicy[playerGrid, goal].get(action) for goal in targets]
        likelihoodList = calculateSoftmaxProbability(actionValueList, self.softmaxBeta)

        posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
        evidence = sum(posteriorUnnormalized)
        posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]

        return posteriorList


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
