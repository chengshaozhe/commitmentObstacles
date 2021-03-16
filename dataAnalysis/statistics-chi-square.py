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


def isDecisionStepInZone(trajectory, target1, target2, decisionSteps):
    trajectory = list(map(tuple, trajectory))[:decisionSteps + 1]
    initPlayerGrid = trajectory[0]
    zone = calculateAvoidCommitmnetZone(initPlayerGrid, target1, target2)
    isStepInZone = [step in zone for step in trajectory[1:]]
    return np.all(isStepInZone)


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')

    participants = ['human', 'RL']
    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]

    df = pd.concat(dfList, sort=True)
    df['participantsType'] = ['RL' if 'max' in name else 'Humans' for name in df['name']]

    # controlDataPath = os.path.join(resultsPath, 'humanNoTime')
    # dfControl = pd.concat(map(pd.read_csv, glob.glob(os.path.join(controlDataPath, '*.csv'))), sort=False)
    # dfControl['participantsType'] = ['machine' if 'max' in name else 'Human No Time' for name in dfControl['name']]
    # df = pd.concat([df, dfControl], sort=True)

    df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)
    df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    # resultDf = dfExpTrail[dfExpTrail['participantsType'] == 'human']
    # crosstab, res = researchpy.crosstab(resultDf['hasAvoidPoint'], resultDf['decisionSteps'], test="chi-square")
    # print(crosstab)
    # print(res)

    # human vs machine in diff decision steps[2,4,6]
    # decisionStep = 4
    # dfExpTrail = df[(df['decisionSteps'] == decisionStep) & (df['targetDiff'] == 0) & (df['conditionName'] == 'expCondition')]

    # dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName']), axis=1)

    # resultDf = dfExpTrail
    # crosstab, res = researchpy.crosstab(resultDf['participantsType'], resultDf['isDecisionStepInZone'], test="chi-square")
    # print(crosstab)
    # print(res)

    dfNormailTrail = df[df['noiseNumber'] != 'special']
    dfSpecialTrail = df[df['noiseNumber'] == 'special']

    statDF = pd.DataFrame()
    statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
    statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()

# chi-squre
    resultDf = dfSpecialTrail
    crosstab, res = researchpy.crosstab(resultDf['participantsType'], resultDf['firstIntentionConsistFinalGoal'], test="chi-square")

    print(crosstab)
    print(res)
# χ2 (2) = 14.93, P < 0.001, VCramer = 0.50


# fisher
#     resultDf = resultDf[resultDf['participantsType'] != "machine"]
#     crosstab, res = researchpy.crosstab(resultDf['participantsType'], resultDf['firstIntentionConsistFinalGoal'], test='fisher')
#     print(crosstab)
#     print(res)

#
    df['trialType'] = ['Critical Disruption' if trial == "special" else 'Random Disruptions' for trial in df['noiseNumber']]

    statDF = pd.DataFrame()
    statDF['commitmentRatio'] = df.groupby(['name', 'trialType', 'participantsType'], sort=False)["firstIntentionConsistFinalGoal"].mean()
    statDF['commitmentRatio'] = statDF.apply(lambda x: int(x["commitmentRatio"] * 100), axis=1)

    statDF = statDF.reset_index()

# t-test
    humanDF = statDF[(statDF.participantsType == "Humans") & (statDF.trialType == 'Critical Disruption')]
    rLDF = statDF[(statDF.participantsType == "RL") & (statDF.trialType == 'Critical Disruption')]
    des, res = researchpy.ttest(humanDF['commitmentRatio'], rLDF['commitmentRatio'])
    print(des)
    print(res)
