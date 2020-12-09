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
from src.trial import NormalTrial, SingleGoalTrial
from src.experiment import ObstacleExperiment, SingleGoalExperiment
from src.design import CreatRectMap, SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from src.design import *
from src.controller import *


def main():
    experimentValues = co.OrderedDict()
    experimentValues["name"] = 'test'
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    fullScreen = 1
    mouseVisible = False

    screenWidth = 600
    screenHeight = 600
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    pg.mouse.set_visible(mouseVisible)

    gridSize = 15
    leaveEdgeSpace = 2
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

    formalTrialImage = pg.image.load(os.path.join(picturePath, 'formalTrial.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))

    formalTrialImage = pg.transform.scale(formalTrialImage, (screenWidth, screenHeight))
    restImage = pg.transform.scale(restImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

# condition
    width = [5]
    height = [5]
    intentionDis = [2, 3, 4]
    decisionSteps = [0, 1, 2, 4, 6]
    targetDiffs = [0, 0, 1, 2]
    rotateAngles = [0, 90, 180, 270]

    # obstaclesMapStep
    obstaclesMapStep0 = [[(3, 2), (4, 2), (5, 3), (6, 3), (7, 4), (1, 3), (2, 4), (2, 5), (3, 6)]]

    obstaclesMapStep1 = [[(1, 2), (2, 4), (2, 5), (3, 2), (4, 2), (3, 6), (4, 7), (5, 3), (6, 4)]]

    obstaclesMapStep2 = [[(2, 2), (2, 4), (2, 5), (3, 6), (4, 7), (4, 2), (5, 2), (6, 3), (7, 3)]]

    obstaclesMapStep4 = [[(3, 3), (4, 1), (1, 4), (5, 1), (1, 5), (5, 3), (6, 3), (3, 5), (3, 6)]]

    obstaclesMapStep6 = [[(4, 4), (4, 1), (4, 2), (6, 4), (7, 4), (4, 6), (4, 7), (1, 4), (2, 4)]]

    speicalObstacleMap = [[(4, 1), (4, 2), (6, 3), (6, 4), (1, 4), (2, 4), (3, 6), (4, 6)],
                          [(5, 1), (4, 2), (6, 3), (6, 4), (1, 5), (2, 4), (3, 6), (4, 6)],
                          [(3, 1), (4, 2), (6, 3), (6, 4), (1, 3), (2, 4), (3, 6), (4, 6)]]

    obstaclesCondition = [obstaclesMapStep0, obstaclesMapStep1, obstaclesMapStep2, obstaclesMapStep4, obstaclesMapStep6]

# debug
    # decisionSteps = [6]
    # obstaclesCondition = [obstaclesMapStep6]
# debug

    obstaclesMaps = dict(zip(decisionSteps, obstaclesCondition))

    numBlocks = 3
    expDesignValues = [[b, h, d, m, diff] for b in width for h in height for d in intentionDis for m in decisionSteps for diff in targetDiffs] * numBlocks

    random.shuffle(expDesignValues)
    numExpTrial = len(expDesignValues)

    specialDesign = [5, 5, 4, 10, 0]
    expDesignValues.append(specialDesign)

    condition = namedtuple('condition', 'name decisionSteps')
    expCondition = condition(name='expCondition', decisionSteps=decisionSteps)
    lineCondition = condition(name='lineCondition', decisionSteps=decisionSteps)
    # specialCondition = condition(name='specialCondition', decisionSteps=[10])

    conditionList = [expCondition] * numExpTrial  # + [lineCondition] * numExpTrial
    # conditionList = [lineCondition] * numExpTrial

    random.shuffle(conditionList)
    numNormalTrials = len(conditionList)

    # numTrialsPerBlock = 3
    # noiseCondition = list(permutations([1, 2, 0], numTrialsPerBlock)) + [(1, 1, 1)]
    # blockNumber = int(numNormalTrials / numTrialsPerBlock)
    # noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

    # conditionList.append(specialCondition)

    noiseDesignValues = [0] * numNormalTrials

# deubg
    # expDesignValues = [specialDesign] * 10
    # noiseDesignValues = ['special'] * 10
    # conditionList = [specialCondition] * 10
# debug

    print('trial:', len(conditionList))
    if len(conditionList) != len(noiseDesignValues):
        raise Exception("unmatch condition design")

    writerPath = os.path.join(resultsPath, experimentValues["name"] + '.csv')
    writer = WriteDataFrameToCSV(writerPath)

    rotatePoint = RotatePoint(gridSize)
    isInBoundary = IsInBoundary([0, gridSize - 1], [0, gridSize - 1])
    creatRectMap = CreatRectMap(rotateAngles, gridSize, obstaclesMaps, rotatePoint)
    creatLineMap = CreatLineMap(rotateAngles, gridSize, rotatePoint, isInBoundary)
    samplePositionFromCondition = SamplePositionFromCondition(creatRectMap, creatLineMap, expDesignValues)

    pygameActionDict = {pg.K_UP: (0, -1), pg.K_DOWN: (0, 1), pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0)}
    humanController = HumanController(pygameActionDict)
    controller = humanController

    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    # noiseActionSpace = [(0, 0)]

    normalNoise = AimActionWithNoise(noiseActionSpace, gridSize)
    specialNoise = backToCrossPointNoise

    normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)

    restTrialInterval = math.ceil(numNormalTrials / 4)
    restTrial = list(range(restTrialInterval, numNormalTrials, restTrialInterval))

    experiment = ObstacleExperiment(normalTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath, restTrial, restImage)

# start exp
    drawImage(formalTrialImage)
    experiment(noiseDesignValues, conditionList)
    drawImage(finishImage)


if __name__ == "__main__":
    main()
