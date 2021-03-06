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
from src.simulationTrial import NormalTrial, SpecialTrial
from src.experiment import IntentionModelSimulation
from src.design import SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from machinePolicy.onlineVIWithObstacle import RunVIMDP, RunIntentionModel
from src.design import *
from src.controller import *


def main():
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    dataPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'conditionData'))
    df = pd.read_csv(os.path.join(dataPath, 'DesignConditionForAvoidCommitmentZone.csv'))
    df['intentionedDisToTargetMin'] = df.apply(lambda x: x['minDis'] - x['avoidCommitmentZone'], axis=1)

    gridSize = 15

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

# condition
    width = [5]
    height = [5]
    intentionDis = [3, 4, 5]
    decisionSteps = [2, 4, 6, 10]
    targetDiffs = [0, 0, 1, 2]

    rotateAngles = [0, 90, 180, 270]
    obstaclesMap1 = [[(2, 2), (2, 4), (3, 5), (3, 6), (4, 2), (5, 3), (6, 3)],
                     [(2, 2), (2, 4), (2, 5), (3, 6), (4, 2), (5, 2), (6, 3)],
                     [(2, 2), (2, 4), (3, 5), (2, 6), (4, 2), (5, 3), (6, 2)]]

    obstaclesMap2 = [[(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)],
                     [(3, 3), (5, 1), (1, 5), (5, 3), (3, 5), (6, 3), (3, 6)],
                     [(3, 3), (3, 1), (1, 3), (5, 3), (3, 5), (6, 3), (3, 6)]]

    obstaclesMap3 = [[(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)],
                     [(4, 4), (5, 1), (4, 2), (6, 4), (4, 6), (1, 5), (2, 4)],
                     [(4, 4), (3, 1), (4, 2), (6, 4), (4, 6), (1, 3), (2, 4)]]

    speicalObstacleMap = [[(4, 1), (4, 2), (6, 3), (6, 4), (1, 4), (2, 4), (3, 6), (4, 6)],
                          [(5, 1), (4, 2), (6, 3), (6, 4), (1, 5), (2, 4), (3, 6), (4, 6)],
                          [(3, 1), (4, 2), (6, 3), (6, 4), (1, 3), (2, 4), (3, 6), (4, 6)]]

    obstaclesCondition = [obstaclesMap1, obstaclesMap2, obstaclesMap3, speicalObstacleMap]
    obstaclesMaps = dict(zip(decisionSteps, obstaclesCondition))

    rotatePoint = RotatePoint(gridSize)

    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    normalNoise = AimActionWithNoise(noiseActionSpace, gridSize)
    specialNoise = backToCrossPointNoise

    initPrior = [0.5, 0.5]
    # inferGoalPosterior = InferGoalPosterior(goalPolicy)

    softmaxBetaList = [4, 6, 8, 10]
    noiseList = [0.067]
    noise = 0.067
    for softmaxBeta in softmaxBetaList:
        # for noise in noiseList:
        for i in range(30):
            print(i)

            numBlocks = 3
            expDesignValues = [[b, h, d, m, diff] for b in width for h in height for d in intentionDis for m in decisionSteps for diff in targetDiffs] * numBlocks

            random.shuffle(expDesignValues)
            numExpTrial = len(expDesignValues)

            specialDesign = [5, 5, 4, 10, 0]
            expDesignValues.append(specialDesign)

            condition = namedtuple('condition', 'name decisionSteps')
            expCondition = condition(name='expCondition', decisionSteps=decisionSteps[:-1])
            lineCondition = condition(name='lineCondition', decisionSteps=decisionSteps[:-1])
            specialCondition = condition(name='specialCondition', decisionSteps=[10])

            conditionList = [expCondition] * numExpTrial
            random.shuffle(conditionList)
            numNormalTrials = len(conditionList)

            numTrialsPerBlock = 3
            noiseCondition = list(permutations([1, 2, 0], numTrialsPerBlock)) + [(1, 1, 1)]
            blockNumber = int(numNormalTrials / numTrialsPerBlock)
            noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

            if noise == 0:
                noiseDesignValues = [0] * numNormalTrials

            conditionList.append(specialCondition)

            if len(conditionList) != len(noiseDesignValues):
                raise Exception("unmatch condition design")

    # deubg
    #         expDesignValues = [specialDesign] * 10
    #         noiseDesignValues = ['special'] * 10
    #         conditionList = [expCondition] * 10
    # debug

            isInBoundary = IsInBoundary([0, gridSize - 1], [0, gridSize - 1])
            creatRectMap = CreatRectMap(rotateAngles, gridSize, obstaclesMaps, rotatePoint)
            creatLineMap = CreatLineMap(rotateAngles, gridSize, rotatePoint, isInBoundary)
            samplePositionFromCondition = SamplePositionFromCondition(creatRectMap, creatLineMap, expDesignValues)

            noise = 0.1
            gamma = 0.9
            goalReward = 10
            actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]

            runVI = RunVIMDP(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)

            intentionInfoScale = [-1, 1]
            runModel = RunIntentionModel(runVI, softmaxBeta, intentionInfoScale)

            # modelController = ModelControllerOnline(softmaxBeta)
            # modelController = AvoidCommitModel(softmaxBeta, actionSpace, checkBoundary)
            # controller = modelController
            controller = SampleSoftmaxAction(softmaxBeta)

            renderOn = 1
            normalTrial = NormalTrial(renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary)
            specialTrial = SpecialTrial(renderOn, controller, drawNewState, drawText, specialNoise, checkBoundary)

            experimentValues = co.OrderedDict()
            experimentValues["name"] = "noise" + str(noise) + '_' + "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
            resultsDirPath = os.path.join(resultsPath, "noise" + str(noise) + '_' + "softmaxBeta" + str(softmaxBeta))
            if not os.path.exists(resultsDirPath):
                os.mkdir(resultsDirPath)

            writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
            writer = WriteDataFrameToCSV(writerPath)
            experiment = IntentionModelSimulation(normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath, runModel)
            experiment(noiseDesignValues, conditionList)


if __name__ == "__main__":
    main()
