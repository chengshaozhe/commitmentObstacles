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


def isTrajHasAvoidPoints(trajectory, aimAction, initPlayerGrid, target1, target2, decisionSteps, conditionName):
    trajectory = list(map(tuple, trajectory))
    if conditionName == 'expCondition':
        avoidPoint = calAvoidPoints(initPlayerGrid, decisionSteps)
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

    participants = ['human', 'RL']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]

    df = pd.concat(dfList)
    df['participantsType'] = ['machine' if 'max' in name else 'human' for name in df['name']]
    df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)

    dfExpTrail = df[(df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition') & (df['decisionSteps'] != 10) & (df['decisionSteps'] != 6)]

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName']), axis=1)

    resultDf = df[df['participantsType'] == 'human']
    crosstab, res = researchpy.crosstab(resultDf['hasAvoidPoint'], resultDf['decisionSteps'], test="chi-square")
    print(crosstab)
    print(res)

    # human vs machine in diff decision steps[2,4,6]
    # decisionStep = 4
    # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

    # dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName']), axis=1)

    resultDf = dfExpTrail
    crosstab, res = researchpy.crosstab(resultDf['participantsType'], resultDf['isDecisionStepInZone'], test="chi-square")
    print(crosstab)
    print(res)

    df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

    dfNormailTrail = df[df['noiseNumber'] != 'special']
    dfSpecialTrail = df[df['noiseNumber'] == 'special']

    # statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
    # statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()

    # resultDf = dfSpecialTrail
    # crosstab, res = researchpy.crosstab(resultDf['participantsType'], resultDf['firstIntentionConsistFinalGoal'], test="chi-square")
    # print(crosstab)
    # print(res)

    res = stat()
    res.anova_stat(df=statDF, res_var='avoidCommitPercent', anova_model='avoidCommitPercent~C(participantsType)+C(decisionSteps)+C(participantsType):C(decisionSteps)')
