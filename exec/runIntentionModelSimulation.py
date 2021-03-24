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

from src.writer import WriteDataFrameToCSV
from src.visualization import InitializeScreen, DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import AvoidCommitModel, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, backToZoneNoise, backToCrossPointNoise, SampleToZoneNoise, AimActionWithNoise, InferGoalPosterior, ModelControllerWithGoal, ModelControllerOnline
from src.simulationTrial import NormalTrialOnline, SpecialTrialOnline
from src.experiment import IntentionModelSimulation
from src.design import SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from machinePolicy.showIntentionModel import RunVI, RunIntentionModel
from src.design import *
from src.controller import *


def main():
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    dataPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'conditionData'))

    gridSize = 15
    bounds = [0, 0, gridSize - 1, gridSize - 1]

    screenWidth = 600
    screenHeight = 600
    fullScreen = False
    renderOn = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    pg.mouse.set_visible(False)

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
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
    controlDiffList = [0, 1, 2]
    minSteps = 8
    maxsteps = 13
    minDistanceBetweenTargets = 4

    minDistanceBetweenGrids = max(controlDiffList) + 1
    maxDistanceBetweenGrids = calculateMaxDistanceOfGrid(bounds) - minDistanceBetweenGrids
    randomWorld = RandomWorld(bounds, minDistanceBetweenGrids, maxDistanceBetweenGrids, numOfObstacles)
    randomCondition = namedtuple('condition', 'name creatMap decisionSteps minSteps maxsteps minDistanceBetweenTargets controlDiffList')
    randomMaps = randomCondition(name='randomCondition', creatMap=randomWorld, decisionSteps=0, minSteps=minSteps, maxsteps=maxsteps, minDistanceBetweenTargets=minDistanceBetweenTargets, controlDiffList=controlDiffList)

    # conditionList = [map1ObsStep0a, map1ObsStep0b, map1ObsStep1a, map1ObsStep1b, controlCondition1] + [map1ObsStep2, map1ObsStep4, map1ObsStep6] * 2 + [map2ObsStep0a, map2ObsStep0b, map2ObsStep1a, map2ObsStep1b, controlCondition2] + [map2ObsStep2, map2ObsStep4, map2ObsStep6] * 2 + [randomMaps] * 2
    # targetDiffsList = [0, 1, 2, 'controlAvoid']

    conditionList = [map1ObsStep0a, map1ObsStep0b, map1ObsStep1a, map1ObsStep1b, controlCondition1] + [map1ObsStep2, map1ObsStep4, map1ObsStep6] * 2 + [map2ObsStep0a, map2ObsStep0b, map2ObsStep1a, map2ObsStep1b, controlCondition2] + [map2ObsStep2, map2ObsStep4, map2ObsStep6] * 2
    targetDiffsList = [0]

    # conditionList = [map1ObsStep0a] + [randomMaps] * 2

    n = 1
    numBlocks = 3 * n
    expDesignValues = [[condition, diff] for condition in conditionList for diff in targetDiffsList] * numBlocks
    numExpTrial = len(expDesignValues)

    # conditionList = [condition2]
    specialDesign = [specialCondition, 0]

    numTrialsPerNoiseBlock = 3
    noiseCondition = list(permutations([1, 2, 0], numTrialsPerNoiseBlock)) + [(1, 1, 1)]
    blockNumber = int(numExpTrial / numTrialsPerNoiseBlock)
    noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

    rotatePoint = RotatePoint(gridSize)
    isInBoundary = IsInBoundary([0, gridSize - 1], [0, gridSize - 1])

    rotateAngles = [0, 90, 180, 270]
    creatMap = CreatMap(rotateAngles, gridSize, rotatePoint, numOfObstacles)

    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    normalNoise = AimActionWithNoise(noiseActionSpace, gridSize)
    specialNoise = backToCrossPointNoise

    # inferGoalPosterior = InferGoalPosterior(goalPolicy)

    softmaxBetaList = [2.5]
    noise = 0.067
    gamma = 0.9
    goalReward = [30, 30]
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)

    intentionInfoScale = [-0.5, 0.5]

    for softmaxBeta in softmaxBetaList:
        for i in range(20):
            print(i)

            expDesignValues = [[condition, diff] for condition in conditionList for diff in targetDiffsList] * numBlocks
            random.shuffle(expDesignValues)
            expDesignValues.append(specialDesign)

            runModel = RunIntentionModel(runVI, softmaxBeta, intentionInfoScale)

            # modelController = ModelControllerOnline(softmaxBeta)
            # modelController = AvoidCommitModel(softmaxBeta, actionSpace, checkBoundary)
            controller = SampleSoftmaxAction(softmaxBeta)

            renderOn = 1
            normalTrial = NormalTrialOnline(renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary)
            specialTrial = SpecialTrialOnline(renderOn, controller, drawNewState, drawText, specialNoise, checkBoundary)

            experimentValues = co.OrderedDict()
            experimentValues["name"] = "noise" + str(noise) + '_' + "intentionInfoScale" + str(intentionInfoScale[1]) + '_' + str(i)
            # resultsDirPath = os.path.join(resultsPath, "Intention-" + "noise" + str(noise) + '_' + "softmaxBeta" + str(softmaxBeta))

            resultsDirPath = os.path.join(resultsPath, "showIntention3")

            if not os.path.exists(resultsDirPath):
                os.mkdir(resultsDirPath)

            writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
            writer = WriteDataFrameToCSV(writerPath)
            experiment = IntentionModelSimulation(creatMap, normalTrial, specialTrial, writer, experimentValues, drawImage, resultsPath, runModel)
            experiment(noiseDesignValues, expDesignValues)


if __name__ == "__main__":
    main()
