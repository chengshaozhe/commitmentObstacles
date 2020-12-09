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
from dataAnalysis import calculateSE, calculateSD, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calMidPoints
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols


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
    statsListAll = []
    stdListAll = []
    statDataAll = []
    # participants = ['humanTime', 'human', 'noise0.067_softmaxBeta8']
    participants = ['human', 'noise0_softmaxBeta5']
    conditionList = [0, 1, 2, 4, 6, 8]
    for participant in participants:
        statsList = []
        stdList = []
        statData = []

        for decisionStep in conditionList:
            dataPath = os.path.join(resultsPath, participant)
            df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
            nubOfSubj = len(df["name"].unique())
            print(participant, nubOfSubj)
            # print(df.columns)
            # df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
            # df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]

            df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)

            dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition') & (df['isDecisionStepInZone'] == 1)]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition') & (df['noisePoint'] == '[]') & (df['isDecisionStepInZone'] == 1)]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['conditionName'] == 'expCondition')]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition') & (df['noisePoint'] == '[]')]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition') & (df['isDecisionStepInZone'] == 1)]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'lineCondition')]

            # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'lineCondition') & (df['isDecisionStepInZone'] == 1)]

            dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName'], eval(x['obstacles'])), axis=1)

            # dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName']), axis=1)

            # df = df[(df['decisionSteps'] == 6) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

            # df = df[(df['decisionSteps'] == 2) & (df['conditionName'] == 'expCondition')]

            # print(len(df))

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

            # statDF['avoidCommitPoint'] = df.groupby('name')["hasAvoidPoint"].sum() / (len(df) / len(df.groupby('name')["hasAvoidPoint"]))

            # print(len(df.groupby('name')))

            # statDF['avoidCommitPoint'] = dfExpTrail.groupby('name')["hasAvoidPoint"].sum() / (len(dfExpTrail) / len(dfExpTrail.groupby('name')["hasAvoidPoint"]))

            # statDF['avoidCommitPoint'] = dfExpTrail.groupby('name')["hasAvoidPoint"].sum() / (len(dfExpTrail))

            # print(dfExpTrail.groupby('name')["hasAvoidPoint"].sum())
            # print(dfExpTrail.groupby('name')["hasAvoidPoint"].count())

            statDF['avoidCommitPoint'] = dfExpTrail.groupby('name')["hasAvoidPoint"].sum() / dfExpTrail.groupby('name')["hasAvoidPoint"].count()

            # print(dfExpTrail.groupby('name')["hasAvoidPoint"].count())
            # print(statDF['avoidCommitPoint'])

            # statDF['midTriaPercent'] = df.groupby(['name','decisionSteps'])["hasAvoidPoint"].sum() / (len(df) / len(df.groupby(['name','decisionSteps'])["hasAvoidPoint"]))
            # stats = list(statDF.groupby('decisionSteps')['midTriaPercent'].mean())[:-1]
            # statsList.append(stats)
            # print(statDF)

            # print(df.groupby('name')["hasAvoidPoint"].head(6))

            # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
            print('')

            stats = statDF.columns
            statData.append(statDF['avoidCommitPoint'])

            statsList.append([np.mean(statDF[stat]) for stat in stats][0])
            stdList.append([calculateSE(statDF[stat]) for stat in stats][0])
        statsListAll.append(statsList)
        stdListAll.append(stdList)
        statDataAll.append(statData)

    print(statsListAll)
    print(stdListAll)

    # print(statDataAll[0])
    print(ttest_ind(statDataAll[0][1], statDataAll[1][1]))

    # formula = "avoidCommitPoint~C(Model)+C(DecisionStep)+C(Model):C(DecisionStep)"

    # avoidCommitPointList = [np.concatenate(statData) for statData in statDataAll][0]
    # print(avoidCommitPointList)

    # numConditions = int(len(avoidCommitPointList) / 4)
    # ModelList = [['human'] * numConditions + ['RL'] * numConditions] * 2
    # ModelList = np.array(ModelList).flatten()
    # DecisionStepList = ['2'] * int(numConditions * 2) + ['4'] * int(numConditions * 2)
    # DecisionStepList = np.array(DecisionStepList).flatten()
    # statDict = {'Model': ModelList, 'DecisionStep': DecisionStepList, 'avoidCommitPoint': avoidCommitPointList}

    # print(len(avoidCommitPointList), len(ModelList), len(DecisionStepList))
    # statdf = pd.DataFrame(statDict)

    # print(statdf)
    # statdf.to_csv('statdf.csv')
    # anova_results = anova_lm(ols(formula, statdf).fit())
    # print(anova_results)

    labels = participants
    # labels = ['Human', 'Human No Time pressure ', 'RL Agent']

    # xlabels = list(statDF.columns)
    xlabels = conditionList

    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(xlabels)
    width = totalWidth / n
    x = x - (totalWidth - width) / 3
    for i in range(len(labels)):
        plt.bar(x + width * i, statsListAll[i], yerr=stdListAll[i], width=width, label=labels[i])
        # plt.boxplot(x + width * i, statsListAll[i], yerr=stdListAll[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.xlabel('Decision Step')
    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('Avoid Commitment Ratio')  # Intention Consistency

    plt.show()
