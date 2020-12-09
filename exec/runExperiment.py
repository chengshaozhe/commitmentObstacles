import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import pygame as pg
import collections as co
import numpy as np
import pickle
from itertools import permutations
from collections import namedtuple
import random
import pandas as pd
import math

from src.writer import WriteDataFrameToCSV
from src.visualization import InitializeScreen, DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import HumanController, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, backToZoneNoise, backToCrossPointNoise, SampleToZoneNoise, AimActionWithNoise, InferGoalPosterior, ModelControllerWithGoal, ModelControllerOnline
from src.trial import NormalTrial, SpecialTrial
from src.experiment import ObstacleExperiment, SingleGoalExperiment
from src.design import CreatRectMap, SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from src.design import *
from src.controller import *


def main():
    experimentValues = co.OrderedDict()
    experimentValues["name"] = 'test'
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    fullScreen = 0
    mouseVisible = 1

    screenWidth = 600
    screenHeight = 600
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    pg.mouse.set_visible(mouseVisible)

    gridSize = 15
    bounds = [0, 0, gridSize - 1, gridSize - 1]

    leaveEdgeSpace = 1  # 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))

    formalTrialImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))

    formalTrialImage = pg.transform.scale(formalTrialImage, (screenWidth, screenHeight))
    restImage = pg.transform.scale(restImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

# condition maps
    condition = namedtuple('condition', 'name decisionSteps initAgent avoidCommitPoint crossPoint targetDisToCrossPoint fixedObstacles')

    map1ObsStep0a = condition(name='expCondition1', decisionSteps=0, initAgent=(0, 2), avoidCommitPoint=(1, 2), crossPoint=(3, 3), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1)])

    map2ObsStep0a = condition(name='expCondition2', decisionSteps=0, initAgent=(0, 2), avoidCommitPoint=(1, 2), crossPoint=(4, 4), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1), (1, 4), (4, 1)])

    map1ObsStep0b = condition(name='expCondition1', decisionSteps=0, initAgent=(2, 0), avoidCommitPoint=(2, 1), crossPoint=(3, 3), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1)])

    map2ObsStep0b = condition(name='expCondition2', decisionSteps=0, initAgent=(2, 0), avoidCommitPoint=(2, 1), crossPoint=(4, 4), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1), (1, 4), (4, 1)])

    map1ObsStep1a = condition(name='expCondition1', decisionSteps=1, initAgent=(1, 0), avoidCommitPoint=(2, 1), crossPoint=(3, 3), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1)])

    map2ObsStep1a = condition(name='expCondition2', decisionSteps=1, initAgent=(1, 0), avoidCommitPoint=(2, 1), crossPoint=(4, 4), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1), (1, 4), (4, 1)])

    map1ObsStep1b = condition(name='expCondition1', decisionSteps=1, initAgent=(0, 1), avoidCommitPoint=(1, 2), crossPoint=(3, 3), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1)])

    map2ObsStep1b = condition(name='expCondition2', decisionSteps=1, initAgent=(0, 1), avoidCommitPoint=(1, 2), crossPoint=(4, 4), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(1, 1), (1, 3), (3, 1), (1, 4), (4, 1)])

    map1ObsStep2 = condition(name='expCondition1', decisionSteps=2, initAgent=(0, 0), avoidCommitPoint=(2, 2), crossPoint=(3, 3), targetDisToCrossPoint=[4, 5, 6], fixedObstacles=[(1, 1), (1, 3), (3, 1)])

    map2ObsStep2 = condition(name='expCondition2', decisionSteps=2, initAgent=(0, 0), avoidCommitPoint=(2, 2), crossPoint=(4, 4), targetDisToCrossPoint=[4, 5, 6], fixedObstacles=[(1, 1), (1, 3), (3, 1), (1, 4), (4, 1)])

    map1ObsStep4 = condition(name='expCondition1', decisionSteps=4, initAgent=(0, 0), avoidCommitPoint=(3, 3), crossPoint=(4, 4), targetDisToCrossPoint=[4, 5, 6], fixedObstacles=[(2, 2), (2, 0), (2, 4), (0, 2), (4, 2)])

    map2ObsStep4 = condition(name='expCondition2', decisionSteps=4, initAgent=(0, 0), avoidCommitPoint=(3, 3), crossPoint=(5, 5), targetDisToCrossPoint=[4, 5, 6], fixedObstacles=[(2, 2), (2, 0), (2, 4), (0, 2), (4, 2), (2, 5), (5, 2)])

    map1ObsStep6 = condition(name='expCondition1', decisionSteps=6, initAgent=(0, 0), avoidCommitPoint=(4, 4), crossPoint=(5, 5), targetDisToCrossPoint=[4, 5, 6], fixedObstacles=[(3, 3), (4, 0), (3, 1), (3, 5), (5, 3), (1, 3), (0, 4)])

    map2ObsStep6 = condition(name='expCondition2', decisionSteps=6, initAgent=(0, 0), avoidCommitPoint=(4, 4), crossPoint=(6, 6), targetDisToCrossPoint=[4, 5, 6], fixedObstacles=[(3, 3), (4, 0), (3, 1), (3, 5), (5, 3), (1, 3), (0, 4), (3, 6), (6, 3), ])

    specialCondition = condition(name='specialCondition', decisionSteps=0, initAgent=(0, 0), avoidCommitPoint=[-1, -1], crossPoint=(5, 5), targetDisToCrossPoint=[5], fixedObstacles=[(3, 0), (0, 3), (1, 4), (1, 6), (4, 1), (6, 1), (6, 2), (2, 6), (5, 3), (3, 5)])

    controlCondition1 = condition(name='controlCondition', decisionSteps=0, initAgent=(0, 0), avoidCommitPoint=[-1, -1], crossPoint=(5, 5), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(3, 0), (0, 3), (1, 4), (1, 6), (4, 1), (6, 1), (6, 2), (2, 6), (5, 3), (3, 5)])

    controlCondition2 = condition(name='controlCondition', decisionSteps=0, initAgent=(0, 0), avoidCommitPoint=[-1, -1], crossPoint=(5, 5), targetDisToCrossPoint=[5, 6, 7], fixedObstacles=[(3, 0), (0, 3), (1, 4), (5, 3), (4, 1), (3, 5), (2, 2), (6, 1), (6, 2), (1, 6), (2, 6), (5, 3)])

    numOfObstacles = 18
    controlDiffList = [0, 1, 2, 3, 4]
    minSteps = 10
    minDistanceBetweenTargets = 5

    minDistanceBetweenGrids = max(controlDiffList) + 1
    maxDistanceBetweenGrids = calculateMaxDistanceOfGrid(bounds) - minDistanceBetweenGrids
    randomWorld = RandomWorld(bounds, minDistanceBetweenGrids, maxDistanceBetweenGrids, numOfObstacles)
    randomCondition = namedtuple('condition', 'name creatMap decisionSteps minSteps minDistanceBetweenTargets controlDiffList')
    randomMaps = randomCondition(name='randomCondition', creatMap=randomWorld, decisionSteps=0, minSteps=minSteps, minDistanceBetweenTargets=minDistanceBetweenTargets, controlDiffList=controlDiffList)

    conditionList = [map1ObsStep0a, map1ObsStep0b, map1ObsStep1a, map1ObsStep1b] + [map1ObsStep2, map1ObsStep4, map1ObsStep6, controlCondition1] * 2 + [map2ObsStep0a, map2ObsStep0b, map2ObsStep1a, map2ObsStep1b, controlCondition2] + [map2ObsStep2, map2ObsStep4, map2ObsStep6] * 2 + [randomMaps] * 4

    # conditionList = [map1ObsStep0a] + [randomMaps] * 2
    targetDiffsList = [0, 1, 2, 'controlAvoid']
    # targetDiffsList = ['controlAvoid']
    # conditionList = [controlCondition2] * 10

    numBlocks = 3
    expDesignValues = [[condition, diff] for condition in conditionList for diff in targetDiffsList] * numBlocks
    numExpTrial = len(expDesignValues)

    # controlDesign = [[randomMaps, -1] * numExpTrial]
    # expDesignValues.append(controlDesign)

    numNormalTrial = len(expDesignValues)

    random.shuffle(expDesignValues)
    specialDesign = [specialCondition, 0]
    expDesignValues.append(specialDesign)

    numTrialsPerNoiseBlock = 3
    noiseCondition = list(permutations([1, 2, 0], numTrialsPerNoiseBlock)) + [(1, 1, 1)]
    blockNumber = int(numNormalTrial / numTrialsPerNoiseBlock)
    noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

    # noiseDesignValues = [0] * numNormalTrials

# deubg
    # expDesignValues = [specialDesign] * 2
    # noiseDesignValues = ['special'] * 2
# debugs

    print('trial:', len(expDesignValues))
    # if len(expDesignValues) != len(noiseDesignValues):
    #     print(len(noiseDesignValues))
    #     raise Exception("unmatch condition design")

    writerPath = os.path.join(resultsPath, experimentValues["name"] + '.csv')
    writer = WriteDataFrameToCSV(writerPath)

    rotatePoint = RotatePoint(gridSize)
    isInBoundary = IsInBoundary([0, gridSize - 1], [0, gridSize - 1])

    rotateAngles = [0, 90, 180, 270]
    creatMap = CreatMap(rotateAngles, gridSize, rotatePoint, numOfObstacles)

    pygameActionDict = {pg.K_UP: (0, -1), pg.K_DOWN: (0, 1), pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0)}
    humanController = HumanController(pygameActionDict)
    controller = humanController

    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    # noiseActionSpace = [(0, 0)]

    normalNoise = AimActionWithNoise(noiseActionSpace, gridSize)
    specialNoise = backToCrossPointNoise

    normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
    specialTrial = SpecialTrial(controller, drawNewState, drawText, specialNoise, checkBoundary)

    restTrialInterval = math.ceil(numExpTrial / 4)
    restTrialIndex = list(range(restTrialInterval, numExpTrial, restTrialInterval))

    experiment = ObstacleExperiment(creatMap, normalTrial, specialTrial, writer, experimentValues, drawImage, restTrialIndex, restImage)

# start exp
    # drawImage(formalTrialImage)
    experiment(noiseDesignValues, expDesignValues)
    drawImage(finishImage)


if __name__ == "__main__":
    main()
