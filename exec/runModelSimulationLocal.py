import os
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
from src.controller import ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, backToZoneNoise, SampleToZoneNoise, AimActionWithNoise, InferGoalPosterior, ModelControllerWithGoal, ModelControllerOnline
from src.simulationTrial import NormalTrial, SpecialTrial, NormalTrialWithGoal, SpecialTrialWithGoal, NormalTrialRewardOnline, SpecialTrialRewardOnline
from src.experiment import ObstacleExperiment
from src.design import CreatExpCondition, SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue, RotatePoint
from machinePolicy.onlineVIWithObstacle import runVI


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

    width = [6]
    height = [6]
    intentionDis = [3, 4, 5, 6]
    rotateAngles = [0, 90, 180, 270]
    minSteps = [2, 4, 6]
    expDesignValues = [[b, h, d, m] for b in width for h in height for d in intentionDis for m in minSteps]

    obstaclesCondition = [[(2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2)], [(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)], [(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)]]
    obstaclesMaps = dict(zip(minSteps, obstaclesCondition))

    rotatePoint = RotatePoint(gridSize)

    # createExpCondition = CreatExpCondition(rotateAngles, gridSize, obstaclesMaps, rotatePoint)
    # samplePositionFromCondition = SamplePositionFromCondition(createExpCondition, expDesignValues)

    # distanceDiffList = [0, 2, 4]
    # minDisList = range(5, 15)
    # intentionedDisToTargetList = [4, 6]
    # rectAreaSize = [36]
    # lineAreaSize = [4, 5, 6, 7, 8, 9, 10]

    # condition = namedtuple('condition', 'name areaType distanceDiff minDis areaSize intentionedDisToTarget')

    # expCondition = condition(name='expCondition', areaType='rect', distanceDiff=[0], minDis=minDisList, areaSize=rectAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    # rectCondition = condition(name='controlRect', areaType='rect', distanceDiff=[2, 4], minDis=minDisList, areaSize=rectAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    # straightLineCondition = condition(name='straightLine', areaType='straightLine', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=lineAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    # midLineCondition = condition(name='MidLine', areaType='midLine', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=lineAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    # noAreaCondition = condition(name='noArea', areaType='none', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=[0], intentionedDisToTarget=intentionedDisToTargetList)

    # policy = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGird15_policy.pkl"), "rb"))

    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -2), (0, 2), (-2, 0), (2, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    normalNoise = NormalNoise(noiseActionSpace, gridSize)
    sampleToZoneNoise = SampleToZoneNoise(noiseActionSpace)

    softmaxBetaList = [2.5]

    # goalPolicy = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))

    initPrior = [0.5, 0.5]
    inferGoalPosterior = InferGoalPosterior(goalPolicy)
    priorBeta = 5
    softmaxBeta = 2.5
    rewardVarianceList = [50]
    # for softmaxBeta in softmaxBetaList:

    for i in range(10):
        print(i)
        # expDesignValues = [[b, h, d] for b in width for h in height for d in intentionDis]
        # numExpTrial = len(expDesignValues)
        # random.shuffle(expDesignValues)
        # expDesignValues.append(random.choice(expDesignValues))
        # createExpCondition = CreatExpCondition(direction, gridSize)
        # samplePositionFromCondition = SamplePositionFromCondition(df, createExpCondition, expDesignValues)
        # numExpTrial = len(expDesignValues) - 1
        # numControlTrial = int(numExpTrial * 2 / 3)
        # expTrials = [expCondition] * numExpTrial
        # conditionList = list(expTrials + [rectCondition] * numExpTrial + [straightLineCondition] * numControlTrial + [midLineCondition] * numControlTrial + [noAreaCondition] * numControlTrial)
        # numNormalTrials = len(conditionList)

        # random.shuffle(conditionList)
        # conditionList.append(expCondition)
        expCondition = 'exp'
        conditionList = [expCondition] * 27
        numNormalTrials = len(conditionList)

        numTrialsPerBlock = 3
        noiseCondition = list(permutations([1, 2, 0], numTrialsPerBlock))
        noiseCondition.append((1, 1, 1))
        blockNumber = int(numNormalTrials / numTrialsPerBlock)
        noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

        createExpCondition = CreatExpCondition(rotateAngles, gridSize, obstaclesMaps, rotatePoint)
        samplePositionFromCondition = SamplePositionFromCondition(createExpCondition, expDesignValues)
# deubg
        # conditionList = [expCondition] * 27
        # noiseDesignValues = ['special'] * 27
# debug
        # modelController = ModelController(policy, gridSize, softmaxBeta)
        # modelControllerWithGoal = ModelControllerWithGoal(gridSize, softmaxBeta, goalPolicy, priorBeta)

        modelController = ModelControllerOnline(softmaxBeta, runVI)
        controller = modelController

        normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
        specialTrial = SpecialTrial(controller, drawNewState, drawText, sampleToZoneNoise, checkBoundary)
        # normalTrial = NormalTrialWithGoal(controller, drawNewState, drawText, normalNoise, checkBoundary, initPrior, inferGoalPosterior)
        # specialTrial = SpecialTrialWithGoal(controller, drawNewState, drawText, sampleToZoneNoise, checkBoundary, initPrior, inferGoalPosterior)
        # normalTrial = NormalTrialRewardOnline(controller, drawNewState, drawText, normalNoise, checkBoundary, rewardVariance)
        # specialTrial = SpecialTrialRewardOnline(controller, drawNewState, drawText, sampleToZoneNoise, checkBoundary, rewardVariance)

        experimentValues = co.OrderedDict()
        experimentValues["name"] = "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
        writerPath = os.path.join(resultsPath, experimentValues["name"] + '.csv')
        writer = WriteDataFrameToCSV(writerPath)
        experiment = ObstacleExperiment(normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath)
        experiment(noiseDesignValues, conditionList)


if __name__ == "__main__":
    main()
