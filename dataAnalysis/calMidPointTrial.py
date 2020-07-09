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
from sklearn.metrics import mutual_info_score as KL
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone


def calAvoidPoints(playerGrid, decisionSteps):
    addSteps = decisionSteps / 2 + 1
    x, y = playerGrid
    if x < 7 and y < 7:
        avoidPoint = (x + addSteps, y + addSteps)
    if x < 7 and y > 7:
        avoidPoint = (x + addSteps, y - addSteps)
    if x > 7 and y < 7:
        avoidPoint = (x - addSteps, y + addSteps)
    elif x > 7 and y > 7:
        avoidPoint = (x - addSteps, y - addSteps)
    return avoidPoint


def isTrajHasAvoidPoints(trajectory, playerGrid, decisionSteps):
    trajectory = list(map(tuple, trajectory))
    avoidPoint = calAvoidPoints(playerGrid, decisionSteps)
    hasMidPoint = 1 if avoidPoint in trajectory else 0
    return hasMidPoint


def sliceTraj(trajectory, midPoint):
    trajectory = list(map(tuple, trajectory))
    index = trajectory.index(midPoint) + 1
    return trajectory[: index]


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'noise0.067_softmaxBeta']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)
        print(df.columns)
        # df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        # df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]

        df = df[(df['decisionSteps'] == 2) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

        # df = df[(df['decisionSteps'] == 2) & (df['conditionName'] == 'expCondition')]

        # print(len(df))
        df['hasAvoidPoint'] = df.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['playerGrid']), x['decisionSteps']), axis=1)

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

        statDF['avoidCommitPoint'] = df.groupby('name')["hasAvoidPoint"].sum() / (len(df) / len(df.groupby('name')["hasAvoidPoint"]))

        # statDF['midTriaPercent'] = df.groupby(['name','decisionSteps'])["hasAvoidPoint"].sum() / (len(df) / len(df.groupby(['name','decisionSteps'])["hasAvoidPoint"]))
        # stats = list(statDF.groupby('decisionSteps')['midTriaPercent'].mean())[:-1]
        # statsList.append(stats)
        # print(statDF)

        # print(df.groupby('name')["hasAvoidPoint"].head(6))

        # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        stats = statDF.columns

        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    print(statsList)
    print(stdList)

    # statsList = [[0.48124999999999996], [0.46562499999999996], [0.4270833333333333], [0.4875]]
    # stdList = [[0.02443813049769197], [0.01927505584370063], [0.027776388854164925], [0.013471506281091268]]

    # statsList = [[0.65625], [0.578125], [0.515625], [0.5538194444444444]]
    # stdList = [[0.02840909090909091], [0.022904141330393608], [0.032967604518047755], [0.015022732366528775]]

    labels = ['RL Agent', '4', '6', 'all']

    # labels = participants
    # labels = ['Human', 'Agent']

    xlabels = list(statDF.columns)

    # xlabels = ['avoidCommit']
    # labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.1, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('avoidCommit')  # Intention Consistency

    plt.show()
