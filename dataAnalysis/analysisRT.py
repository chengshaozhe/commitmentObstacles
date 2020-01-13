import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind

from dataAnalysis import calculateSE


def calculateSoftmaxProbability(acionValues, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))
    return newProbabilityList


class SoftmaxPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1):
        actionDict = Q_dict[(playerGrid, target1)]
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
        priorA = initPrior[0]
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodA = self.goalPolicy(playerGrid, tuple(target1)).get(action)
            likelihoodB = self.goalPolicy(playerGrid, tuple(target2)).get(action)
            posteriorA = priorA * likelihoodA / (priorA * likelihoodA + (1 - priorA) * likelihoodB)
            goalPosteriorList.append([posteriorA, 1 - posteriorA])
            priorA = posteriorA
        return goalPosteriorList


class CalFirstIntentionStep:
    def __init__(self, inferThreshold):
        self.inferThreshold = inferThreshold

    def __call__(self, goalPosteriorList):
        for index, goalPosteriori in enumerate(goalPosteriorList):
            if max(goalPosteriori) > self.inferThreshold:
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
def calculateTimeGap(timeList):
    time0=[timeList[0]-1300]
    timeGap = [timeList[i + 1] - timeList[i] for i in range(len(timeList) - 1)]
    return time0+timeGap


def calculateIsTimeMaxNextNoisePoint(timeList, noisePoint):
    noisePointNextStep = [i + 1 for i in noisePoint]
    timeGap = [timeList[i + 1] - timeList[i] for i in range(len(timeList) - 1)]
    maxReactTimeStep = [i + 2 for i, x in enumerate(timeGap) if x == max(timeGap)]
    if [i for i in maxReactTimeStep if i in noisePointNextStep] != []:
        isTimeMaxNextNoisePoint = 1
    else:
        isTimeMaxNextNoisePoint = 0
    return isTimeMaxNextNoisePoint

def calculateMeanTimeBeforeIntension(timeList,firstIntension):
    return np.mean(timeList[1:firstIntension])
def calculateMeanTimeAfterIntension(timeList,firstIntension):
    return np.mean(timeList[firstIntension+1:])    
if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))
    softmaxBeta = 2.5
    softmaxPolicy = SoftmaxPolicy(Q_dict, softmaxBeta)
    initPrior = [0.5, 0.5]
    inferThreshold = 0.95
    goalInfernce = GoalInfernce(initPrior, softmaxPolicy)
    calFirstIntentionStep = CalFirstIntentionStep(inferThreshold)
    calFirstIntentionStepRatio = CalFirstIntentionStepRatio(calFirstIntentionStep)

    # trajectory = [(1, 7), [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [8, 9], [9, 9], [10, 9], [8, 9], [9, 9], [10, 9], [11, 9], [12, 9], [12, 8], [12, 7]]
    # aimAction = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, -1), (0, -1)]
    # target1, target2 = (6, 13), (12, 7)
    # goalPosteriorList = goalInfernce(trajectory, aimAction, target1, target2)
    # print(len(goalPosteriorList))
    # firstIntentionStep = calFirstIntentionStep(goalPosteriorList)
    # firstIntentionStepRatio = calFirstIntentionStepRatio(goalPosteriorList)

    # goalPosteriori = np.array(goalPosteriorList).T
    # x = np.arange(len(aimAction))
    # lables = ['goalA', 'goalB']
    # for i in range(len(lables)):
    #     plt.plot(x, goalPosteriori[i], label=lables[i])
    # xlabels = np.arange(1, len(aimAction) + 1, 1)
    # plt.xticks(x, xlabels)
    # plt.xlabel('step')
    # plt.legend(loc='best')
    # plt.show()

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)
        # df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), [x['bean1GridX'], x['bean1GridY']],[x['bean2GridX'], x['bean2GridY']]), axis=1)

        df['firstIntentionStep'] = df.apply(lambda x: calFirstIntentionStep(x['goalPosteriorList']), axis=1)

          
        df["timeGap"] = df.apply(lambda x: calculateTimeGap(eval(x['reactionTime'])), axis=1)
        df["stepNum"] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)
        df['meanTimeBeforeIntention']=df.apply(lambda x: calculateMeanTimeBeforeIntension(x['timeGap'],x['firstIntentionStep']), axis=1)
        df['meanTimeAfterIntention']=df.apply(lambda x: calculateMeanTimeAfterIntension(x['timeGap'],x['firstIntentionStep']), axis=1)
        df['meanTimeStep0']=df.apply(lambda x: x['timeGap'][0], axis=1)

        # df.to_csv("humanResultReationTime(withStep0).csv")
        dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        dfExpTrail['RT1/3']=dfExpTrail.apply(lambda x: calculateMeanTimeBeforeIntension(x['timeGap'],x['noisePoint']), axis=1)
        dfExpTrail['RT2/3']=df.apply(lambda x: calculateMeanTimeAfterIntension(x['timeGap'],x['noisePoint']), axis=1)
        # dfExpTrail['RT3/3']=  
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

        statDF['meanTimeStep0'] = df.groupby('name')["meanTimeStep0"].mean()
        statDF['meanTimeBeforeIntention'] = df.groupby('name')["meanTimeBeforeIntention"].mean()
        statDF['meanTimeAfterIntention'] = df.groupby('name')["meanTimeAfterIntention"].mean()
        # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print(statDF)

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    # xlabels = ['meanTimeBeforeIntention','meanTimeAfterIntention']
    xlabels=['meanTimeStep0','meanTimeBeforeIntention','meanTimeAfterIntention']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.5, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 1000))
    plt.legend(loc='best')

    plt.title('WithoutStep0')
    plt.show()
