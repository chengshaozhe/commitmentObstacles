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


class SingleGoalTrial():
    def __init__(self, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, beanGrid, playerGrid, designValues):
        obstacles = []
        initialPlayerGrid = playerGrid
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        totalStep = int(np.linalg.norm(np.array(playerGrid) - np.array(beanGrid), ord=1))
        noiseStep = random.sample(list(range(2, totalStep)), designValues)
        stepCount = 0
        goalList = list()
        self.drawText("+", [0, 0, 0], [7, 7])
        pg.time.wait(1300)
        self.drawNewState(beanGrid, beanGrid, initialPlayerGrid, obstacles)
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        realPlayerGrid = initialPlayerGrid
        pause = True
        initialTime = time.get_ticks()
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, beanGrid, beanGrid)
            reactionTime.append(time.get_ticks() - initialTime)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            self.drawNewState(beanGrid, beanGrid, realPlayerGrid, obstacles)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = checkTerminationOfTrial(beanGrid, beanGrid, realPlayerGrid)
        pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        return results


class NormalTrial():
    def __init__(self, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles, designValues):
        initialPlayerGrid = playerGrid
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(2, leastStep)), designValues))

        stepCount = 0
        goalList = list()
        self.drawText("+", [0, 0, 0], [7, 7])
        pg.time.wait(1300)
        self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid, obstacles)
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        realPlayerGrid = initialPlayerGrid
        pause = True
        initialTime = time.get_ticks()
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid)
            reactionTime.append(time.get_ticks() - initialTime)
            goal = inferGoal(trajectory[-1], aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            if noisePlayerGrid in obstacles:
                noisePlayerGrid = tuple(trajectory[-1])
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrial():
    def __init__(self, controller, drawNewState, drawText, specialNoise, checkBoundary):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.specialNoise = specialNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles):
        initialPlayerGrid = playerGrid
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()
        self.drawText("+", [0, 0, 0], [7, 7])
        pg.time.wait(1300)
        self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid, obstacles)
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        pause = True
        realPlayerGrid = initialPlayerGrid
        initialTime = time.get_ticks()
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid)
            reactionTime.append(time.get_ticks() - initialTime)
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

            self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)

        pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results
