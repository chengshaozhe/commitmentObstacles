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
from src.experiment import ObstacleModelSimulation, IntentionModelSimulation
from src.design import SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from machinePolicy.onlineVIWithObstacle import RunVI
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
    # pg.mouse.set_visible(False)

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

# condition
    width = [5]
    height = [5]
    intentionDis = [2, 3, 4]
    decisionSteps = [0, 1, 2, 4, 6, 8]
    # decisionSteps = [1]
    targetDiffs = [0, 0, 1, 2]

    rotateAngles = [0, 90, 180, 270]

    obstaclesMapStep0 = [[(1, 4), (1, 5), (1, 6), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)]]

    obstaclesMapStep1 = [[(1, 2), (1, 4), (1, 5), (1, 6), (3, 2), (4, 2), (5, 2), (6, 2)]]

    obstaclesMapStep2 = [[(2, 2), (2, 4), (3, 5), (3, 6), (4, 2), (5, 3), (6, 3)],
                         [(2, 2), (2, 4), (2, 5), (3, 6), (4, 2), (5, 2), (6, 3)],
                         [(2, 2), (2, 4), (3, 5), (2, 6), (4, 2), (5, 3), (6, 2)]]

    obstaclesMapStep4 = [[(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)],
                         [(3, 3), (5, 1), (1, 5), (5, 3), (3, 5), (6, 3), (3, 6)],
                         [(3, 3), (3, 1), (1, 3), (5, 3), (3, 5), (6, 3), (3, 6)]]

    obstaclesMapStep6 = [[(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)],
                         [(4, 4), (5, 1), (4, 2), (6, 4), (4, 6), (1, 5), (2, 4)],
                         [(4, 4), (3, 1), (4, 2), (6, 4), (4, 6), (1, 3), (2, 4)]]

    obstaclesMapStep8 = [[(5, 5), (4, 2), (2, 4), (6, 1), (6, 2), (6, 3), (1, 6), (2, 6), (3, 6), (7, 5), (5, 7)]]

    obstaclesCondition = [obstaclesMapStep0, obstaclesMapStep1, obstaclesMapStep2, obstaclesMapStep4, obstaclesMapStep6, obstaclesMapStep8]
    obstaclesMaps = dict(zip(decisionSteps, obstaclesCondition))

    rotatePoint = RotatePoint(gridSize)
    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    normalNoise = AimActionWithNoise(noiseActionSpace, gridSize)
    specialNoise = backToCrossPointNoise

    initPrior = [0.5, 0.5]
    # inferGoalPosterior = InferGoalPosterior(goalPolicy)

    softmaxBetaList = [5]
    noise = 0
    gamma = 0.9
    goalReward = [50, 50]
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)

    for softmaxBeta in softmaxBetaList:
        # for noise in noiseList:
        for i in range(20):
            print(i)

            numBlocks = 3
            expDesignValues = [[b, h, d, m, diff] for b in width for h in height for d in intentionDis for m in decisionSteps for diff in targetDiffs] * numBlocks

            random.shuffle(expDesignValues)
            numExpTrial = len(expDesignValues)

            condition = namedtuple('condition', 'name decisionSteps')
            expCondition = condition(name='expCondition', decisionSteps=decisionSteps)
            lineCondition = condition(name='lineCondition', decisionSteps=decisionSteps)
            conditionList = [expCondition] * numExpTrial  # + [lineCondition] * numExpTrial

            random.shuffle(conditionList)

            numNormalTrials = len(conditionList)
            noiseDesignValues = [0] * numNormalTrials

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

            modelController = ModelControllerOnline(softmaxBeta)

            # modelController = AvoidCommitModel(softmaxBeta, actionSpace, checkBoundary)

            controller = modelController

            renderOn = 1
            normalTrial = NormalTrial(renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary)

            experimentValues = co.OrderedDict()
            experimentValues["name"] = "noise" + str(noise) + '_' + "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
            resultsDirPath = os.path.join(resultsPath, "noise" + str(noise) + '_' + "softmaxBeta" + str(softmaxBeta))
            if not os.path.exists(resultsDirPath):
                os.mkdir(resultsDirPath)

            writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
            writer = WriteDataFrameToCSV(writerPath)
            experiment = ObstacleModelSimulation(normalTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath, runVI)
            experiment(noiseDesignValues, conditionList)


if __name__ == "__main__":
    main()
