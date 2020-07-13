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


class NormalTrial():
    def __init__(self, renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles, designValues):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(2, leastStep)), designValues))
        stepCount = 0
        goalList = list()

        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, obstacles)
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

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles):
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
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, obstacles)
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
