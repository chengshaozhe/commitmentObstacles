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
from src.controller import HumanController, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, backToZoneNoise, backToCrossPointNoise, SampleToZoneNoise, AimActionWithNoise, InferGoalPosterior, ModelControllerWithGoal, ModelControllerOnline
from src.trial import NormalTrial, SpecialTrial, SingleGoalTrial
from src.experiment import ObstacleExperiment, SingleGoalExperiment
from src.design import CreatRectMap, SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from src.design import *
from src.controller import *


def main():
    gridSize = 15

    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    dataPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'conditionData'))
    df = pd.read_csv(os.path.join(dataPath, 'DesignConditionForAvoidCommitmentZone.csv'))
    df['intentionedDisToTargetMin'] = df.apply(lambda x: x['minDis'] - x['avoidCommitmentZone'], axis=1)

    screenWidth = 600
    screenHeight = 600
    fullScreen = False

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

    testTrialImage = pg.image.load(os.path.join(picturePath, 'testTrial.png'))
    formalTrialImage = pg.image.load(os.path.join(picturePath, 'formalTrial.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))

    testTrialImage = pg.transform.scale(testTrialImage, (screenWidth, screenHeight))
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
    intentionDis = [3, 4, 5]
    rotateAngles = [0, 90, 180, 270]
    decisionSteps = [2, 4, 6, 10]
    targetDiffs = [0, 2, 4]

    obstaclesMap1 = [(2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2)]
    obstaclesMap2 = [(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)]
    obstaclesMap3 = [(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)]

    speicalObstacleMap = [(4, 1), (4, 2), (6, 3), (6, 4), (1, 4), (2, 4), (3, 6), (4, 6)]
    obstaclesCondition = [obstaclesMap1, obstaclesMap2, obstaclesMap3, speicalObstacleMap]
    obstaclesMaps = dict(zip(decisionSteps, obstaclesCondition))

    numBlocks = 5
    expDesignValues = [[b, h, d, m, diff] for b in width for h in height for d in intentionDis for m in decisionSteps for diff in targetDiffs] * numBlocks

    random.shuffle(expDesignValues)
    numExpTrial = len(expDesignValues)

    specialDesign = [5, 5, 4, 10, 0]
    expDesignValues.append(specialDesign)

    condition = namedtuple('condition', 'name decisionSteps')
    expCondition = condition(name='expCondition', decisionSteps=decisionSteps[:-1])
    lineCondition = condition(name='lineCondition', decisionSteps=decisionSteps[:-1])
    specialCondition = condition(name='specialCondition', decisionSteps=[10])

    numControlTrial = int(numExpTrial / 2)
    conditionList = [expCondition] * numControlTrial + [lineCondition] * numControlTrial
    # conditionList = [lineCondition] * numControlTrial

    random.shuffle(conditionList)
    conditionList.append(specialCondition)

    numNormalTrials = len(conditionList)
    numTrialsPerBlock = 3
    noiseCondition = list(permutations([1, 2, 0], numTrialsPerBlock)) + [(1, 1, 1)]
    blockNumber = int(numNormalTrials / numTrialsPerBlock)
    noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

    noise = 0.067
    if noise == 0:
        noiseDesignValues = [0] * numNormalTrials

    if len(conditionList) != len(noiseDesignValues):
        raise Exception("unmatch condition design")


# deubg
    # expDesignValues = [specialDesign] * 10
    # noiseDesignValues = ['special'] * 10
    # conditionList = [specialCondition] * 10
# debug

    experimentValues = co.OrderedDict()
    # experimentValues["name"] = 'test'
    experimentValues["name"] = input("Please enter your name:").capitalize()

    writerPath = os.path.join(resultsPath, experimentValues["name"] + '.csv')
    writer = WriteDataFrameToCSV(writerPath)

    baseLineWriterPath = os.path.join(resultsPath, 'baseLine' + experimentValues["name"] + '.csv')
    baseLineWriter = WriteDataFrameToCSV(baseLineWriterPath)

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
    normalNoise = AimActionWithNoise(noiseActionSpace, gridSize)
    specialNoise = backToCrossPointNoise

    normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
    specialTrial = SpecialTrial(controller, drawNewState, drawText, specialNoise, checkBoundary)

    experiment = ObstacleExperiment(normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath)

    singleGoalTrial = SingleGoalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
    creatSingleGoalMap = CreatSingleGoalMap(gridSize)
    singleGoalExperiment = SingleGoalExperiment(singleGoalTrial, baseLineWriter, experimentValues, creatSingleGoalMap)

    drawImage(testTrialImage)
    baseLineTrialCondition = [6, 8, 10, 12, 14]
    numBaseLineTrialBlock = 5
    numBaseLineTrial = len(baseLineTrialCondition) * numBaseLineTrialBlock
    baseLineNoiseDesignValues = np.array([random.choice(noiseCondition) for _ in range(numBaseLineTrial)]).flatten().tolist()
    baseLineConditionList = baseLineTrialCondition * numBaseLineTrialBlock
    random.shuffle(baseLineConditionList)

    singleGoalExperiment(baseLineNoiseDesignValues, baseLineConditionList)
    drawImage(restImage)
    drawImage(formalTrialImage)
    experiment(noiseDesignValues, conditionList)
    drawImage(finishImage)


if __name__ == "__main__":
    main()
