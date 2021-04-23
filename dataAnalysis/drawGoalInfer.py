import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy, mannwhitneyu, ranksums
from scipy.interpolate import interp1d
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZone
from machinePolicy.onlineVIWithObstacle import RunVI
from dataAnalysis import *
import seaborn as sns


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


def calGoalPosteriorFromAll(posteriors, trajectory, target1, target2):
    trajectory = list(map(tuple, trajectory))
    goalIndex = None
    if trajectory[-1] == target1:
        goalIndex = 0
    elif trajectory[-1] == target2:
        goalIndex = 1
    else:
        print("trajectory no goal reach! ")
    goalPosteriorList = [posterior[goalIndex] for posterior in posteriors]
    return goalPosteriorList


def calParticipantType(name):
    if 'max' in name:
        participantsType = 'Desire Model'
    if 'intention' in name:
        participantsType = 'Intention Model'
    else:
        participantsType = 'Humans'

    return participantsType


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')

    # participants = ['human', 'RL']
    # participants = ['human', 'intentionModel/threshold0.5infoScale11']
    participants = ['intentionModel/threshold0.3infoScale11', 'intentionModel/threshold0.3infoScale8']
    participants = ['human', 'intentionModelChosen/threshold0.07infoScale8.5']
    participants = ['human', 'intentionModel/threshold0.07infoScale8.5']
    participants = ['intentionModel/threshold0.1infoScale7softmaxBetaInfer3']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    humanDataPath = os.path.join(os.path.join(DIRNAME, '..'), 'dataAnalysis/humanPosterior.csv')
    # humanDataPath = os.path.join(os.path.join(DIRNAME, '..'), 'dataAnalysis/dfWithOriginPosterior.csv')

    humanDf = pd.read_csv(humanDataPath)
    dfList.append(humanDf)

    df = pd.concat(dfList, sort=True)
    df['participantsType'] = df.apply(lambda x: calParticipantType(x['name']), axis=1)
    df["trajLength"] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

    df['isValidTraj'] = df.apply(lambda x: isValidTraj(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)
    df = df[df['isValidTraj'] == 1]

    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)

    df = df[(df['targetDiff'] == '0')]
    df = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]
    df = df.loc[df.decisionSteps == 6]

    chosenSteps = 16
    df = df[(df["trajLength"] > chosenSteps)]
    xnew = np.array(list(range(chosenSteps + 1)))

    df['goalPosteriorList'] = df.apply(lambda x: calGoalPosteriorFromAll(eval(x['posteriors']), eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)

    df["Length"] = df.apply(lambda x: len(x['goalPosteriorList']), axis=1)

    df['goalPosterior'] = df.apply(lambda x: calPosteriorByChosenSteps(x['goalPosteriorList'], xnew), axis=1)

    df = df.loc[:, ["participantsType", 'goalPosterior', 'decisionSteps']]

    subjectTypes = df['participantsType'].unique()
    dfLists = [df.loc[df.participantsType == subjectType] for subjectType in subjectTypes]

    statDf = pd.concat([pd.concat([pd.DataFrame(list(zip(xnew, row['goalPosterior'], [subjectType] * len(xnew))), columns=['timeStep', 'goalPosterior', 'participantsType']) for i, row in df.iterrows()]) for df, subjectType in zip(dfLists, subjectTypes)])

    ax = sns.lineplot('timeStep', 'goalPosterior', hue='participantsType', data=statDf, err_style="band", ci=95)

    # colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
    #              (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]
    # # colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)]
    # ax = sns.lineplot('timeStep', 'goalPosterior', hue='participantsType', data=statDf, err_style="band", ci=95, palette=colorList, alpha=0.3)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.legend(loc='best', fontsize=12)
    plt.xlabel("Agent's steps over time", fontsize=14, color='black')
    plt.ylabel('Posterior probability of goal-reached', fontsize=14, color='black')
    plt.ylim((0.47, 1))

    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    plt.rcParams['svg.fonttype'] = 'none'
    # plt.title('Inferred Goal Through Time', fontsize=fontSize, color='black')
    # plt.savefig('/Users/chengshaozhe/Downloads/exp2bStep{}.svg'.format(str(decisionStep)), dpi=600, format='svg')
    plt.show()
