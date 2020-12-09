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
    participants = ['humanEqualDisExp', 'noise0.067_softmaxBeta2.5']
    # participants = ['human', "noise0.067_softmaxBeta2.5"]

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)
    df['participantsType'] = ['machine' if 'max' in name else 'human' for name in df['name']]
    df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)

    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']),axis=1)

    # df = df[(df['noisePoint'] == '[]')]

    # df = df[(df['targetDiff'] == 0) & (df['isDecisionStepInZone'] == 1)]

    df = df[(df['targetDiff'] == '0')]
    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]

    # dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName'], eval(x['obstacles'])), axis=1)

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    statDF = pd.DataFrame()
    statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name', 'decisionSteps'])["hasAvoidPoint"].mean()
    statDF = statDF.reset_index()

    statDF['participantsType'] = ['machine' if 'max' in name else 'human' for name in statDF['name']]

    # statDF['avoidCommitPercentSE'] = statDF["avoidCommitPercent"].apply(calculateSE)

    df['totalTime'] = df.apply(lambda x: eval(x['reactionTime'])[-1], axis=1)
    # print(df.groupby(['name'])['totalTime'].mean())

    # print(statDF)
    # statDF['sem'] = df.groupby(['participantsType', 'decisionSteps'])["avoidCommitPercent"].apply(calculateSE)

    # statDF = statDF[statDF['participantsType'] == 'human']
    # statDF = statDF[statDF['participantsType'] == 'machine']

    # print(statDF)
    # dfExpTrail.to_csv('dfExpTrail.csv')

# Compute the two-way mixed-design ANOVA
    # import pingouin as pg
    # aov = pg.mixed_anova(dv='avoidCommitPercent', within='decisionSteps', between='participantsType', subject='name', data=statDF)
    # pg.print_table(aov)

    # posthocs = pg.pairwise_ttests(dv='avoidCommitPercent', within='decisionSteps', between='participantsType', subject='name', data=statDF, within_first=1)
    # pg.print_table(posthocs)

    VIZ = 1
    if VIZ:
        import seaborn as sns
        ax = sns.barplot(x="decisionSteps", y="avoidCommitPercent", hue="participantsType", data=statDF, ci=68)
        # ax = sns.boxplot(x="decisionSteps", y="avoidCommitPercent", hue="participantsType", data=statDF, palette="Set1", showmeans=True)
        ax.set(xlabel='Decision Step', ylabel='Avoid Commitment Ratio', title='Commitment with Deliberation')
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(loc='best')
        plt.show()

#     import statsmodels.api as sm
#     from statsmodels.formula.api import ols
#     from bioinfokit.analys import stat
#     model = ols('avoidCommitPercent ~ C(decisionSteps) + C(participantsType) + C(decisionSteps):C(participantsType)', data=statDF).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)

#     res = stat()
#     res.anova_stat(df=statDF, res_var='avoidCommitPercent', anova_model='avoidCommitPercent~C(participantsType)+C(decisionSteps)+C(participantsType):C(decisionSteps)')
#     print(res.anova_summary)


# # Post-hoc comparison
#     res.tukey_hsd(df=statDF, res_var='avoidCommitPercent', xfac_var=['decisionSteps', 'participantsType'], anova_model='avoidCommitPercent~C(participantsType)+C(decisionSteps)+C(participantsType):C(decisionSteps)')
#     print(res.tukey_summary)
