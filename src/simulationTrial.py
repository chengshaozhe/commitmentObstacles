import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
import random


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return int(gridDis)


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = calculateGridDis(playerGrid, target1)
    dis2 = calculateGridDis(playerGrid, target2)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []

    return avoidCommitmentZone


def calculateAvoidCommitmnetZoneAll(playerGrid, target1, target2):
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
    avoidCommitmentZone.remove(tuple(playerGrid))
    return avoidCommitmentZone


def calMidPoints(initPlayerGrid, target1, target2):
    zone = calculateAvoidCommitmnetZoneAll(initPlayerGrid, target1, target2)
    if zone:
        playerDisToZoneGrid = [calculateGridDis(initPlayerGrid, zoneGrid) for zoneGrid in zone]
        midPointIndex = np.argmax(playerDisToZoneGrid)
        midPoint = zone[midPointIndex]
    else:
        midPoint = initPlayerGrid
    return midPoint


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


def checkTerminationOfTrial(bean1Grid, bean2Grid, humanGrid):
    if calculateGridDis(humanGrid, bean1Grid) == 0 or calculateGridDis(humanGrid, bean2Grid) == 0:
        pause = False
    else:
        pause = True
    return pause


def chooseOptimalGoal(playerGrid, targetGridA, targetGridB):
    toGoalA = calculateGridDis(playerGrid, targetGridA)
    toGoalB = calculateGridDis(playerGrid, targetGridB)
    if toGoalA > toGoalB:
        goal = targetGridB
    elif toGoalA < toGoalB:
        goal = targetGridA
    else:
        goal = random.choice([targetGridA, targetGridB])
    return goal


class NormalTrial():
    def __init__(self, renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles, designValues, decisionSteps, QDict):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(decisionSteps + 1, leastStep)), designValues))
        stepCount = 0
        goalList = list()

        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
                pg.time.wait(50)
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, QDict)
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            if noisePlayerGrid in obstacles:
                noisePlayerGrid = tuple(trajectory[-1])
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrial():
    def __init__(self, renderOn, controller, drawNewState, drawText, specialNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.specialNoise = specialNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles, QDict):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()

        pause = True
        realPlayerGrid = initialPlayerGrid
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, QDict)
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1

            if len(trajectory) > 3:
                noisePlayerGrid, noiseStep, firstIntentionFlag = self.specialNoise(trajectory, bean1Grid, bean2Grid, noiseStep, firstIntentionFlag)
                if noisePlayerGrid:
                    realPlayerGrid = self.checkBoundary(noisePlayerGrid)
                else:
                    realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            else:
                realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            if realPlayerGrid in obstacles:
                realPlayerGrid = tuple(trajectory[-1])
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)

        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class NormalTrialOnline():
    def __init__(self, renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, Q_dictList, bean1Grid, bean2Grid, playerGrid, obstacles, designValues, decisionSteps):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(decisionSteps + 1, leastStep)), designValues))
        stepCount = 0
        goalList = list()

        midpoint = calMidPoints(initialPlayerGrid, bean1Grid, bean2Grid)
        disToMidPoint = calculateGridDis(initialPlayerGrid, midpoint)
        RLDict, avoidCommitQDicts, commitQDicts = Q_dictList
        target = chooseOptimalGoal(playerGrid, bean1Grid, bean2Grid)

        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
                pg.time.wait(100)

            # commited = isCommitted(realPlayerGrid, bean1Grid, bean2Grid)
            # re-planning ratio

            # commitStep = random.randint(int(disToMidPoint / 2), disToMidPoint)
            # commitStep = 3
            # commited = 1 if stepCount >= commitStep else 0
            commited = 1
            # QDict = avoidCommitQDicts[target]
            if commited:
                QDict = commitQDicts[target]
            else:
                # target = random.choice([bean1Grid, bean2Grid])
                # QDict = avoidCommitQDicts[target]
                QDict = RLDict

            aimPlayerGrid, aimAction = self.controller(QDict, realPlayerGrid)

            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            if noisePlayerGrid in obstacles:
                noisePlayerGrid = tuple(trajectory[-1])
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrialOnline():
    def __init__(self, renderOn, controller, drawNewState, drawText, specialNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.specialNoise = specialNoise
        self.checkBoundary = checkBoundary

    def __call__(self, Q_dictList, bean1Grid, bean2Grid, playerGrid, obstacles):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()

        midpoint = calMidPoints(initialPlayerGrid, bean1Grid, bean2Grid)
        disToMidPoint = calculateGridDis(initialPlayerGrid, midpoint)
        RLDict, avoidCommitQDicts, commitQDicts = Q_dictList
        target = chooseOptimalGoal(playerGrid, bean1Grid, bean2Grid)

        pause = True
        realPlayerGrid = initialPlayerGrid
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)

            # commited = isCommitted(realPlayerGrid, bean1Grid, bean2Grid)
            # commitStep = random.randint(int(disToMidPoint / 2), disToMidPoint)
            # commited = 1 if stepCount >= commitStep else 0

            commited = 1
            if commited:
                QDict = commitQDicts[target]
            else:
                # target = random.choice([bean1Grid, bean2Grid])
                # Q_dict = avoidCommitQDicts[target]
                QDict = RLDict

            aimPlayerGrid, aimAction = self.controller(QDict, realPlayerGrid)
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            if len(trajectory) > 2:
                noisePlayerGrid, noiseStep, firstIntentionFlag = self.specialNoise(trajectory, bean1Grid, bean2Grid, noiseStep, firstIntentionFlag)
                if noisePlayerGrid:
                    realPlayerGrid = self.checkBoundary(noisePlayerGrid)
                else:
                    realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            else:
                realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            if realPlayerGrid in obstacles:
                realPlayerGrid = tuple(trajectory[-1])
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)

        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class NormalTrialWithOnlineIntention():
    def __init__(self, renderOn, controller, normalNoise, checkBoundary, inferGoalPosterior, drawNewState=None):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary
        self.inferGoalPosterior = inferGoalPosterior
        self.initPrior = [0.5, 0.5]

    def __call__(self, RLQDict, goalQDict, intentionQDict, bean1Grid, bean2Grid, playerGrid, obstacles, designValues, decisionSteps):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(decisionSteps + 1, leastStep)), designValues))
        stepCount = 0
        goalList = list()

        realPlayerGrid = initialPlayerGrid
        pause = True
        priorList = self.initPrior
        posteriorData = [priorList]

        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
                pg.time.wait(100)

            aimPlayerGrid, aimAction = self.controller(RLQDict, intentionQDict, realPlayerGrid, bean1Grid, bean2Grid, priorList)

            posteriorList = self.inferGoalPosterior(realPlayerGrid, aimAction, bean1Grid, bean2Grid, priorList, goalQDict)
            posteriorData.append(posteriorList)

            priorList = posteriorList
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)

            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            if realPlayerGrid in obstacles:
                realPlayerGrid = tuple(trajectory[-1])

            stepCount = stepCount + 1
            goalList.append(goal)
            trajectory.append(tuple(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)

            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
            if len(trajectory) > 30:
                break

        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        results["posteriors"] = str(posteriorData)

        return results


class SpecialTrialWithOnlineIntention():
    def __init__(self, renderOn, controller, specialNoise, checkBoundary, inferGoalPosterior, drawNewState=None):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.specialNoise = specialNoise
        self.checkBoundary = checkBoundary
        self.inferGoalPosterior = inferGoalPosterior
        self.initPrior = [0.5, 0.5]

    def __call__(self, RLQDict, goalQDict, intentionQDict, bean1Grid, bean2Grid, playerGrid, obstacles):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()

        pause = True
        realPlayerGrid = initialPlayerGrid
        priorList = self.initPrior
        posteriorData = [priorList]

        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
                pg.time.wait(100)

            aimPlayerGrid, aimAction = self.controller(RLQDict, intentionQDict, realPlayerGrid, bean1Grid, bean2Grid, priorList)
            posteriorList = self.inferGoalPosterior(realPlayerGrid, aimAction, bean1Grid, bean2Grid, priorList, goalQDict)
            posteriorData.append(posteriorList)

            priorList = posteriorList
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)

            if len(trajectory) > 2:
                noisePlayerGrid, noiseStep, firstIntentionFlag = self.specialNoise(trajectory, bean1Grid, bean2Grid, noiseStep, firstIntentionFlag)
                if noisePlayerGrid:
                    realPlayerGrid = self.checkBoundary(noisePlayerGrid)
                else:
                    realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            else:
                realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            if realPlayerGrid in obstacles:
                realPlayerGrid = tuple(trajectory[-1])

            stepCount = stepCount + 1
            goalList.append(goal)
            trajectory.append(tuple(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)

            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
            if len(trajectory) > 30:
                break

        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        results["posteriors"] = str(posteriorData)

        return results
