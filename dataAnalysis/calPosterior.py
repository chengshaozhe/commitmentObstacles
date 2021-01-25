import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy, mannwhitneyu, ranksums
from scipy.interpolate import interp1d
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZone
from machinePolicy.onlineVIWithObstacle import RunVI
from dataAnalysis import *


class SoftmaxPolicy:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, QDict, playerGrid, targetGrid, obstacles):
        actionDict = QDict[(playerGrid, targetGrid)]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class BasePolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1, target2):
        actionDict = self.Q_dict[(playerGrid, tuple(sorted((target1, target2))))]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class GoalInfernce:
    def __init__(self, initPrior, goalPolicy, runVI):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy
        self.runVI = runVI

    def __call__(self, trajectory, aimAction, target1, target2, obstacles):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]

        QDictGoal = self.runVI(goal, obstacles)
        QDictNoGoal = self.runVI(noGoal, obstacles)
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(QDictGoal, playerGrid, goal, obstacles).get(action)
            likelihoodB = self.goalPolicy(QDictNoGoal, playerGrid, noGoal, obstacles).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodB)
            goalPosteriorList.append(posteriorGoal)
            priorGoal = posteriorGoal
        goalPosteriorList.insert(0, initPrior[0])
        return goalPosteriorList


def calInformationGain(baseProb, conditionProb):

    return entropy(baseProb) - entropy(conditionProb)


# class CalculateActionInformation:
#     def __init__(self, initPrior, goalPolicy, basePolicy):
#         self.initPrior = initPrior
#         self.goalPolicy = goalPolicy
#         self.basePolicy = basePolicy

#     def __call__(self, trajectory, aimAction, target1, target2):
#         trajectory = list(map(tuple, trajectory))
#         targets = list([target1, target2])
#         expectedInfoList = []
#         cumulatedInfoList = []
#         priorList = self.initPrior
#         for index, (playerGrid, action) in enumerate(zip(trajectory, aimAction)):
#             likelihoodList = [self.goalPolicy(playerGrid, goal).get(action) for goal in targets]
#             posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
#             evidence = sum(posteriorUnnormalized)

#             posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]
#             prior = posteriorList

#             goalProbList = [list(self.goalPolicy(playerGrid, goal).values()) for goal in targets]
#             baseProb = list(self.basePolicy(playerGrid, target1, target2).values())

#             # expectedInfo = sum([goalPosterior * KL(goalProb, baseProb) for goalPosterior, goalProb in zip(posteriorList, goalProbList)])
#             expectedInfo = sum([goalPosterior * calInformationGain(baseProb, goalProb) for goalPosterior, goalProb in zip(posteriorList, goalProbList)])
#             expectedInfoList.append(expectedInfo)
#             cumulatedInfo = sum(expectedInfoList)
#             cumulatedInfoList.append(cumulatedInfo)

#         return cumulatedInfoList


class CalculateActionInformation:
    def __init__(self, initPrior, goalPolicy, basePolicy):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy
        self.basePolicy = basePolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]
        expectedInfoList = []
        cumulatedInfoList = []
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
            likelihoodNogoal = self.goalPolicy(playerGrid, noGoal).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodNogoal)
            priorGoal = posteriorGoal

            goalProb = list(self.goalPolicy(playerGrid, goal).values())
            baseProb = list(self.basePolicy(playerGrid, target1, target2).values())

            # expectedInfo = posteriorGoal * KL(goalProb, baseProb)
            expectedInfo = posteriorGoal * calInformationGain(baseProb, goalProb)
            expectedInfoList.append(expectedInfo)
            cumulatedInfo = sum(expectedInfoList)
            cumulatedInfoList.append(cumulatedInfo)

        return cumulatedInfoList


def calPosteriorByInterpolation(goalPosteriorList, xInterpolation):
    x = np.divide(np.arange(len(goalPosteriorList) + 1), len(goalPosteriorList))
    goalPosteriorList.append(1)
    y = np.array(goalPosteriorList)
    f = interp1d(x, y, kind='nearest')
    goalPosterior = f(xInterpolation)
    return goalPosterior


def calPosteriorByChosenSteps(goalPosteriorList, xnew):
    goalPosterior = np.array(goalPosteriorList)[xnew]
    return goalPosterior


def calInfo(expectedInfoList):
    x = np.divide(np.arange(len(expectedInfoList)), len(expectedInfoList) - 1)
    y = np.array(expectedInfoList)
    f = interp1d(x, y, kind='nearest')
    xnew = np.linspace(0., 1., 30)
    goalPosterior = f(xnew)
    return goalPosterior


class CalFirstIntentionStep:
    def __init__(self, inferThreshold):
        self.inferThreshold = inferThreshold

    def __call__(self, goalPosteriorList):
        for index, goalPosteriori in enumerate(goalPosteriorList):
            if goalPosteriori > self.inferThreshold:
                return index + 1
                break
        return len(goalPosteriorList)


class CalFirstIntentionStepRatio:
    def __init__(self, calFirstIntentionStep):
        self.calFirstIntentionStep = calFirstIntentionStep

    def __call__(self, goalPosteriorList):
        firstIntentionStep = self.calFirstIntentionStep(goalPosteriorList)
        firstIntentionStepRatio = firstIntentionStep / len(goalPosteriorList)
        return firstIntentionStepRatio


def isDecisionStepInZone(trajectory, target1, target2, decisionSteps):
    trajectory = list(map(tuple, trajectory))[:decisionSteps + 1]
    initPlayerGrid = trajectory[0]
    zone = calculateAvoidCommitmnetZone(initPlayerGrid, target1, target2)
    isStepInZone = [step in zone for step in trajectory[1:]]
    return np.all(isStepInZone)


if __name__ == '__main__':
    gridSize = 15
    noise = 0.067
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    gamma = 0.9
    goalReward = [30]
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)
    softmaxBeta = 2.5
    softmaxPolicy = SoftmaxPolicy(softmaxBeta)
    initPrior = [0.5, 0.5]
    goalInfernce = GoalInfernce(initPrior, softmaxPolicy, runVI)

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    statDFList = []

    participants = ['human', 'RL']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)

        df = df[(df['targetDiff'] == '0')]
        df = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]
        # df = df[(df['decisionSteps'] == 6)]
        # df = df[(df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

        # df = df[(df['decisionSteps'] == 2) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition') & (df['isDecisionStepInZone'] == 1)]

        df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)

        # df['expectedInfoList'] = df.apply(lambda x: calculateActionInformation(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

        # df['firstIntentionStep'] = df.apply(lambda x: calFirstIntentionStep(x['goalPosteriorList']), axis=1)

        # df['firstIntentionStepRatio'] = df.apply(lambda x: calFirstIntentionStepRatio(x['goalPosteriorList']), axis=1)

        xnew = np.linspace(0., 1., 15)
        df['goalPosterior'] = df.apply(lambda x: calPosteriorByInterpolation(x['goalPosteriorList'], xnew), axis=1)

        # xnew = np.array([2, 4, 6, 8])
        # df['goalPosterior'] = df.apply(lambda x: calPosteriorByChosenSteps(x['goalPosteriorList'], xnew), axis=1)

        # df['goalPosterior'] = df.apply(lambda x: calInfo(x['expectedInfoList']), axis=1)

        # df = df[(df['areaType'] == 'rect')]

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] == 'special')]

        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]

        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine') & (df['intentionedDisToTargetMin'] == 2)]

        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine') & (df['distanceDiff'] == 0)]
        # dfExpTrail = df[(df['areaType'] != 'none')]

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['areaType'] != 'rect')]

        # dfExpTrail = df[df['noiseNumber'] != 'special']
        # dfExpTrail = df

        statDF = pd.DataFrame()

        # statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        # statDF['firstIntentionStepRatio'] = dfExpTrail.groupby('name')["firstIntentionStepRatio"].mean()
        # statDF['firstIntentionStepRatio'] = dfExpTrail.groupby('name')["firstIntentionStepRatio"].mean()
        # statDF['goalPosterior'] = dfExpTrail.groupby('name')["goalPosterior"].mean()

        # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        stats = statDF.columns
        # statsList.append([np.mean(statDF[stat]) for stat in stats])
        # stdList.append([calculateSE(statDF[stat]) for stat in stats])

        goalPosterior = np.array(df['goalPosterior'].tolist())
        goalPosteriorMean = np.mean(goalPosterior, axis=0)
        goalPosteriorStd = np.divide(np.std(goalPosterior, axis=0, ddof=1), np.sqrt(len(goalPosterior) - 1))
        statsList.append(goalPosteriorMean)
        stdList.append(goalPosteriorStd)

        def arrMean(df, colnames):
            arr = np.array(df[colnames].tolist())
            return np.mean(arr, axis=0)
        grouped = pd.DataFrame(df.groupby('name').apply(arrMean, 'goalPosterior'))
        statArr = np.array(grouped.iloc[:, 0].tolist()).T

        statDFList.append(statArr)

    # testDataSet = abs(np.array(statDFList[0]) - np.array(statDFList[1]))
    # testData = testDataSet
    # print(testData)
    # print(ttest_ind(testData, np.zeros(len(testData))))

    pvalus = np.array([ttest_ind(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])
    # pvalus = np.array([mannwhitneyu(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])

    # sigArea = np.where(pvalus < 0.05)[0]
    # print(sigArea)

    # print(mannwhitneyu(statDFList[0], statDFList[1]))
    # print(ranksums(statDFList[0], statDFList[1]))

    # lables = participants
    lables = ['Human', 'RL']

    lineWidth = 1
    for i in range(len(statsList)):
        plt.plot(xnew, statsList[i], label=lables[i], linewidth=lineWidth)
    # plt.errorbar(xnew, statsList[i], yerr=stdList[i], label=lables[i])

 # sig area line
    # xnewSig = xnew[sigArea]
    # ySig = [stats[sigArea] for stats in statsList]
    # for sigLine in [xnewSig[0], xnewSig[-1]]:
    #     plt.plot([sigLine] * 10, np.linspace(0.5, 1., 10), color='black', linewidth=2, linestyle="--")

    # xlabels = ['firstIntentionStepRatio']
    # labels = participants
    # x = np.arange(len(xlabels))
    # totalWidth, n = 0.1, len(participants)
    # width = totalWidth / n
    # x = x - (totalWidth - width) / 2
    # for i in range(len(statsList)):
    #     plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    # plt.xticks(x, xlabels)
    # plt.ylim((0, 1.1))

    fontSize = 10
    plt.legend(loc='best', fontsize=fontSize)
    plt.xlabel('Time', fontsize=fontSize, color='black')
    plt.ylabel('Probability of Intention', fontsize=fontSize, color='black')

    plt.xticks(fontsize=fontSize, color='black')
    plt.yticks(fontsize=fontSize, color='black')

    plt.title('Inferred Intention Through Time', fontsize=fontSize, color='black')
    plt.show()
