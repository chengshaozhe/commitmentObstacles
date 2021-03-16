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
from dataAnalysis import calculateSE, calculateSD, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calMidPoints, calculateFirstIntentionConsistency
import researchpy
import seaborn as sns


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


def isGridsALine(playerGrid, targetGrid):
    if playerGrid[0] == targetGrid[0] or playerGrid[1] == targetGrid[1]:
        return True
    else:
        return False


def isTrajHasAvoidPoints(trajectory, aimAction, initPlayerGrid, target1, target2, decisionSteps, conditionName, obstacles):
    trajectory = list(map(tuple, trajectory))
    if conditionName == 'expCondition':
        avoidPoint = calAvoidPoints(initPlayerGrid, decisionSteps)
        hasMidPoint = 1 if avoidPoint in trajectory else 0
        if decisionSteps == 0:
            nextStep = trajectory[1]
            nextStepInLineWithObstacles = [isGridsALine(nextStep, targetGrid) for targetGrid in obstacles]
            hasMidPoint = 1 if sum(nextStepInLineWithObstacles) > 2 else 0
        if decisionSteps == 1:
            avoidPoint = calAvoidPoints(initPlayerGrid, decisionSteps - 1)
            hasMidPoint = 1 if avoidPoint in trajectory else 0

    if conditionName == 'lineCondition':
        avoidPoint = calMidPoints(initPlayerGrid, target1, target2)
        hasMidPoint = 1 if avoidPoint in trajectory else 0
        # hasMidPoint = 1 if aimAction[decisionSteps] == aimAction[decisionSteps - 1] else 0
    return hasMidPoint


def hasAvoidPoints(trajectory, avoidPoint):
    trajectory = list(map(tuple, trajectory))
    hasMidPoint = 1 if avoidPoint in trajectory else 0
    return hasMidPoint


def sliceTraj(trajectory, midPoint):
    trajectory = list(map(tuple, trajectory))
    index = trajectory.index(midPoint) + 1
    return trajectory[:index]


def isDecisionStepInZone(trajectory, target1, target2, decisionSteps):
    trajectory = list(map(tuple, trajectory))[:decisionSteps + 1]
    initPlayerGrid = trajectory[0]
    zone = calculateAvoidCommitmnetZone(initPlayerGrid, target1, target2)
    isStepInZone = [step in zone for step in trajectory[1:]]
    return np.all(isStepInZone)


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    participants = ['human', 'RL']
    # participants = ['noise0.0673_softmaxBeta2.5']
    # participants = ['human']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)
    df['participantsType'] = ['RL Agent' if 'max' in name else 'Human' for name in df['name']]

    #!!!!!!
    df['name'] = df.apply(lambda x: x['name'][:-1], axis=1)

    df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)
    df['totalTime'] = df.apply(lambda x: eval(x['reactionTime'])[-1], axis=1)

    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)

    # df = df[(df['noisePoint'] == '[]')]

    # df = df[(df['targetDiff'] == 0) & (df['isDecisionStepInZone'] == 1)]

    df = df[(df['targetDiff'] == '0')]
    # dfExpTrail = df[(df['conditionName'] == 'expCondition2')]

    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    statDF = pd.DataFrame()
    statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name', 'decisionSteps'])["hasAvoidPoint"].mean()

    # statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name'])["hasAvoidPoint"].mean()

    statDF['ShowCommitmentPercent'] = statDF.apply(lambda x: 1 - x['avoidCommitPercent'], axis=1)

    statDF = statDF.reset_index()
    statDF['participantsType'] = ['RL Agent fixed-future path' if 'max' in name else 'Human fixed-future path' for name in statDF['name']]

    # statDF['avoidCommitPercentSE'] = statDF["avoidCommitPercent"].apply(calculateSE)

    # statDF['meanReactionTime'] = [meanTime[name] for name in statDF['name']]

    # statDF['sem'] = df.groupby(['participantsType', 'decisionSteps'])["avoidCommitPercent"].apply(calculateSE)

    # statDF = statDF[statDF['participantsType'] == 'Human']
    # statDF = statDF[statDF['participantsType'] == 'RL Agent']

    # print(statDF)
    # dfExpTrail.to_csv('dfExpTrail.csv')

    stat2DF = pd.DataFrame()
    stat2DF['avoidCommitPercent'] = dfExpTrail.groupby(['name', 'decisionSteps'])["hasAvoidPoint"].mean()
    stat2DF['avoidCommitPercent'] = 1
    stat2DF = stat2DF.reset_index()
    stat2DF['participantsType'] = ['RL Agent open-ended path' if 'max' in name else 'Human open-ended path' for name in stat2DF['name']]

    # ax = sns.barplot(x="decisionSteps", y="ShowCommitmentPercent", hue="participantsType", data=statDF, ci=68)
    # bottom_plot = sns.barplot(x='decisionSteps', y='avoidCommitPercent', hue="participantsType", data=statDF, ci=68, palette="Set2")
    bottom_plot = sns.barplot(x='decisionSteps', y='avoidCommitPercent', hue="participantsType", data=stat2DF, ci=None, palette=["#FFC0CB", '#B0E0E6'])
    ax = sns.barplot(x="decisionSteps", y="ShowCommitmentPercent", hue="participantsType", data=statDF, ci=None)

    # ax = sns.boxplot(x="decisionSteps", y="avoidCommitPercent", hue="participantsType", data=statDF, palette="Set1", showmeans=True)

    # ax.set(xlabel='steps-to-crossroad', ylabel='Choosing ratio', title='Commitment with Deliberation', fontsize=20)
    ax.set_xlabel(xlabel='steps-to-crossroad', fontsize=20)
    ax.set_ylabel(ylabel='Choosing ratio', fontsize=20)
    ax.set_title(label='Commitment with Deliberation', fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='best', fontsize=16)
    plt.ylim((0, 1))
    plt.show()
