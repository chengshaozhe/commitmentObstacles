import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy
from scipy.interpolate import interp1d
from sklearn.metrics import normalized_mutual_info_score as KL
from dataAnalysis import calculateSE


def calculateSoftmaxProbability(acionValues, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))
    return newProbabilityList


class SoftmaxPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1):
        actionDict = self.Q_dict[(playerGrid, target1)]
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
    def __init__(self, initPrior, goalPolicy):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
            likelihoodB = self.goalPolicy(playerGrid, noGoal).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodB)
            goalPosteriorList.append(posteriorGoal)
            priorGoal = posteriorGoal
        goalPosteriorList.insert(0, initPrior[0])
        return goalPosteriorList


def calInformationGain(baseProb, conditionProb):
    infoGain = entropy(baseProb) - entropy(conditionProb)
    return infoGain


def calBasePolicy(posteriorList, actionProbList):
    basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(posteriorList, actionProbList)]
    basePolicy = np.sum(basePolicyList, axis=0)
    return basePolicy


class CalculateActionInformation:
    def __init__(self, initPrior, goalPolicy, basePolicy):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy
        self.basePolicy = basePolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        targets = list([target1, target2])
        expectedInfoList = []
        cumulatedInfoList = []
        priorList = self.initPrior
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodList = [self.goalPolicy(playerGrid, goal).get(action) for goal in targets]
            posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
            evidence = sum(posteriorUnnormalized)

            posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]
            prior = posteriorList

            actionProbList = [list(self.goalPolicy(playerGrid, goal).values()) for goal in targets]
            baseProb = calBasePolicy(posteriorList, actionProbList)

            # baseProb = list(self.basePolicy(playerGrid, target1, target2).values())
            # baseProb = list(self.goalPolicy(playerGrid, trajectory[-1]).values())
            # baseProb = [0.25] * 4

            # expectedInfo = sum([goalPosterior * KL(goalProb, baseProb) for goalPosterior, goalProb in zip(posteriorList, actionProbList)])
            expectedInfo = sum([goalPosterior * calInformationGain(baseProb, goalProb) for goalPosterior, goalProb in zip(posteriorList, actionProbList)])
            expectedInfoList.append(expectedInfo)
            cumulatedInfo = sum(expectedInfoList)
            cumulatedInfoList.append(cumulatedInfo)

        cumulatedInfoList = [info / sum(cumulatedInfoList) for info in cumulatedInfoList]
        # cumulatedInfoList = [(info - np.mean(cumulatedInfoList)) / np.std(cumulatedInfoList) for info in cumulatedInfoList]
        return cumulatedInfoList[:-1]


# class CalculateActionInformation:
#     def __init__(self, initPrior, goalPolicy, basePolicy):
#         self.initPrior = initPrior
#         self.goalPolicy = goalPolicy
#         self.basePolicy = basePolicy

#     def __call__(self, trajectory, aimAction, target1, target2):
#         trajectory = list(map(tuple, trajectory))
#         goalPosteriorList = []
#         priorGoal = initPrior[0]

#         goal = trajectory[-1]
#         targets = list([target1, target2])
#         noGoal = [target for target in targets if target != goal][0]
#         expectedInfoList = []
#         cumulatedInfoList = []
#         for playerGrid, action in zip(trajectory, aimAction):
#             likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
#             likelihoodNogoal = self.goalPolicy(playerGrid, noGoal).get(action)
#             posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodNogoal)
#             priorGoal = posteriorGoal

#             goalProb = list(self.goalPolicy(playerGrid, goal).values())
#             baseProb = list(self.basePolicy(playerGrid, target1, target2).values())

#             # expectedInfo = posteriorGoal * KL(goalProb, baseProb)
#             expectedInfo = posteriorGoal * calInformationGain(baseProb, goalProb)
#             expectedInfoList.append(expectedInfo)
#             cumulatedInfo = sum(expectedInfoList)
#             cumulatedInfoList.append(cumulatedInfo)

#         cumulatedInfoList = [info / sum(cumulatedInfoList) for info in cumulatedInfoList]
#         return cumulatedInfoList


def calPosterior(goalPosteriorList):
    x = np.divide(np.arange(len(goalPosteriorList) + 1), len(goalPosteriorList))
    goalPosteriorList.append(1)
    y = np.array(goalPosteriorList)
    f = interp1d(x, y, kind='nearest')
    xnew = np.linspace(0., 1., 30)
    goalPosterior = f(xnew)
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


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))
    Q_dict_base = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGird15_policy.pkl"), "rb"))
    softmaxBeta = 1
    softmaxPolicy = SoftmaxPolicy(Q_dict, softmaxBeta)
    basePolicy = BasePolicy(Q_dict_base, softmaxBeta)
    initPrior = [0.5, 0.5]
    inferThreshold = 0.95
    goalInfernce = GoalInfernce(initPrior, softmaxPolicy)
    calFirstIntentionStep = CalFirstIntentionStep(inferThreshold)
    calFirstIntentionStepRatio = CalFirstIntentionStepRatio(calFirstIntentionStep)
    calculateActionInformation = CalculateActionInformation(initPrior, softmaxPolicy, basePolicy)

    # trajectory = [(1, 7), [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [8, 9], [9, 9], [10, 9], [8, 9], [9, 9], [10, 9], [11, 9], [12, 9], [12, 8], [12, 7]]
    # aimAction = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, -1), (0, -1)]
    # target1, target2 = (6, 13), (12, 7)

    # trajectory = [(9, 6), [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [8, 11], [7, 11], [6, 11], [5, 11], [6, 10], [6, 11], [6, 12], [6, 13]]
    # aimAction = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1), (0, 1)]
    # target1, target2 = (6, 13), (4, 11)

    # goalPosteriorList = goalInfernce(trajectory, aimAction, target1, target2)
    # print(len(goalPosteriorList))
    # firstIntentionStep = calFirstIntentionStep(goalPosteriorList)
    # firstIntentionStepRatio = calFirstIntentionStepRatio(goalPosteriorList)
    # goalPosteriorList.insert(0, initPrior[0])
    # goalPosteriori = np.array(goalPosteriorList).T
    # x = np.arange(len(goalPosteriorList))
    # lables = ['goalA']
    # for i in range(len(lables)):
    #     plt.plot(x, goalPosteriori, label=lables[i])
    # xlabels = np.arange(len(goalPosteriorList))
    # plt.xticks(x, xlabels)
    # plt.xlabel('step')
    # plt.legend(loc='best')
    # plt.show()

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'softmaxBeta1']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

        df['expectedInfoList'] = df.apply(lambda x: calculateActionInformation(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

        df['firstIntentionStep'] = df.apply(lambda x: calFirstIntentionStep(x['goalPosteriorList']), axis=1)

        df['firstIntentionStepRatio'] = df.apply(lambda x: calFirstIntentionStepRatio(x['goalPosteriorList']), axis=1)

        # df['goalPosterior'] = df.apply(lambda x: calPosterior(x['goalPosteriorList']), axis=1)

        df['goalPosterior'] = df.apply(lambda x: calInfo(x['expectedInfoList']), axis=1)

        dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        # dfExpTrail = df[(df['areaType'] == 'rect')]

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

        goalPosteriorList = dfExpTrail['goalPosterior'].tolist()
        goalPosterior = np.array(goalPosteriorList)
        goalPosteriorMean = np.mean(goalPosterior, axis=0)
        goalPosteriorStd = np.divide(np.std(goalPosterior, axis=0, ddof=1), np.sqrt(len(goalPosterior) - 1))
        # print(goalPosteriorMean)
        statsList.append(goalPosteriorMean)
        stdList.append(goalPosteriorStd)

    lables = participants
    # lables = ['Human', 'Agent']

    xnew = np.linspace(0., 1., 30)
    lineWidth = 1
    for i in range(len(statsList)):
        plt.plot(xnew, statsList[i], label=lables[i], linewidth=lineWidth)
        # plt.errorbar(xnew, statsList[i], yerr=stdList[i], label=lables[i])

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
    fontSize = 12
    plt.legend(loc='best', fontsize=fontSize)
    plt.xlabel('Time', fontsize=fontSize, color='black')
    plt.ylabel('Accumulated information gain', fontsize=fontSize, color='black')

    plt.xticks(fontsize=fontSize, color='black')
    plt.yticks(fontsize=fontSize, color='black')

    plt.title('Commitment Through Time', fontsize=fontSize, color='black')
    plt.show()
