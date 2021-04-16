import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np
import math
import pickle
from scipy.stats import ttest_ind, entropy
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score as KL
from sklearn.metrics import r2_score
import researchpy
import seaborn as sns
import itertools as it

from dataAnalysis import calculateSE, calculateSD, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calMidPoints, calculatfeFirstIntentionConsistency


def hasAvoidPoints(trajectory, avoidPoint):
    trajectory = list(map(tuple, trajectory))
    hasMidPoint = 1 if avoidPoint in trajectory else 0
    return hasMidPoint


def calParticipantType(name):
    if 'max' in name:
        participantsType = 'Desire Model'
    if 'intention' in name:
        participantsType = 'Intention Model'
    else:
        participantsType = 'Humans'
    return participantsType


def calRMSE(x, y):
    return np.sqrt(np.power(np.array(x) - np.array(y), 2).mean())


def calRMSLE(x, y):
    return np.sqrt(np.power(np.log(np.array(x) + 1) - np.log(np.array(y) + 1), 2).mean())


def RSquared(x, y, ymean):
    error = np.power(np.array(x) - np.array(y), 2) / np.power(np.array(x) - np.array(ymean), 2)
    RSqured = 1 - error
    return RSqured


def calCorr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    sq = math.sqrt(sum([(x - a_avg)**2 for x in a]) * sum([(x - b_avg)**2 for x in b]))

    corrFactor = cov_ab / sq
    return corrFactor


if __name__ == '__main__':
    # a = [1, 2, 3, 4, 5]
    # b = [20, 30, 40, 50, 5]
    # c = [4, 3, 2, 1, 5]
    # print(calCorr(a, b), calCorr(a, c), r2_score(a, b), r2_score(a, c))
    # gg

    # modelName = 'intentionModelWithNaiveInfer2'
    # modelName = 'intentionModelWithSophistictedInfer2'

    modelName = 'intentionModelTest'
    # modelName = "intentionModelWithSpMonitor"

    # modelName = "intentionModelChosen"
    # modelName = 'intentionModelNaiveInferSearchBeta'

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results/' + modelName)
    # participants = ['human', 'RL']

    humansStats = np.array([42.50, 43.95, 63.80, 83.75, 77.60]) * 0.01
    # humansStats = np.array([50, 50, 63.80, 83.75, 77.60]) * 0.01

    # humanDataPath = os.path.join(os.path.join(DIRNAME, '..'), 'dataAnalysis/allhumansStats.csv')
    # humanDf = pd.read_csv(humanDataPath)

    # humanDf['ShowCommitmentPercent'] = np.round(humanDf['ShowCommitmentPercent'], 3)
    # humanStatDf = humanDf.groupby('name').ShowCommitmentPercent.apply(list).reset_index()

    # print(humanStatDf)
    # print(humanStatDf['ShowCommitmentPercent'])

    # participants = ['human', 'intentionModel/threshold0.08infoScale8.5']

    humanDataPaths = os.path.join(os.path.join(DIRNAME, '..'), 'results/' + 'human')
    humanDf = pd.concat(map(pd.read_csv, glob.glob(os.path.join(humanDataPaths, '*.csv'))), sort=False)
    humanDf['totalSteps'] = humanDf.apply(lambda x: len(eval(x['trajectory'])), axis=1)
    humanDf = humanDf[(humanDf['targetDiff'] == '0')]
    humanDf = humanDf[(humanDf['conditionName'] == 'expCondition1') | (humanDf['conditionName'] == 'expCondition2')]
    humanStatDf = humanDf.groupby('name')['totalSteps'].mean().reset_index()

    humansStepsStats = humanStatDf['totalSteps']
    # print()
    # gg

    dirs = os.listdir(resultsPath)[1:]
    participants = [d for d in dirs if not d[0] == '.']
    # print(participants)

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)

    df['participantsType'] = df.apply(lambda x: calParticipantType(x['name']), axis=1)
    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)
    df['totalSteps'] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

    df = df[(df['targetDiff'] == '0')]

    # dfExpTrail = df[(df['conditionName'] == 'expCondition1')]

    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    statDF = pd.DataFrame()
    statDF['avoidCommitPercent'] = dfExpTrail.groupby(['threshold', 'infoScale', "softmaxBeta", 'decisionSteps'])["hasAvoidPoint"].mean()
    # statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name'])["hasAvoidPoint"].mean()

    statDF['ShowCommitmentPercent'] = statDF.apply(lambda x: np.round(1 - x['avoidCommitPercent'], 3), axis=1)

    statDF = statDF.reset_index()
    pd.set_option("max_columns", 10)
    # print(statDF)

    # heatMapDf = statDF.groupby(['threshold', 'infoScale']).ShowCommitmentPercent.apply(list).reset_index()
    # heatMapDf = statDF.groupby(['threshold', "softmaxBeta"]).ShowCommitmentPercent.apply(list).reset_index()
    heatMapDf = statDF.groupby(['infoScale', "softmaxBeta"]).ShowCommitmentPercent.apply(list).reset_index()
    # print(heatMapDf)

# by human means
    heatMapDf['totalSteps'] = dfExpTrail.groupby(['softmaxBeta', 'infoScale'])['totalSteps'].mean().reset_index()['totalSteps']

    heatMapDf['stepsRSquared'] = heatMapDf.apply(lambda x: r2_score(humansStepsStats, [x['totalSteps']] * len(humansStepsStats)), axis=1)

    # heatMapDf['totalSteps'] = dfExpTrail.groupby(['threshold', 'infoScale'])['totalSteps'].mean().reset_index()['totalSteps']

    # heatMapDf['RMSE'] = heatMapDf.apply(lambda x: calRMSE(humansStats, x['ShowCommitmentPercent']), axis=1)
    # heatMapDf['RMSLE'] = heatMapDf.apply(lambda x: calRMSLE(humansStats, x['ShowCommitmentPercent']), axis=1)
    heatMapDf['showRSquared'] = heatMapDf.apply(lambda x: r2_score(humansStats, x['ShowCommitmentPercent']), axis=1)
    # heatMapDf['r'] = heatMapDf.apply(lambda x: calCorr(humansStats, x['ShowCommitmentPercent']), axis=1)

# by all humans
    # heatMapDf['r'] = heatMapDf.apply(lambda x: np.mean([calCorr(human, x['ShowCommitmentPercent']) for human in humanStatDf['ShowCommitmentPercent']]), axis=1)
    # heatMapDf['RSquared'] = heatMapDf.apply(lambda x: np.mean([r2_score(human, x['ShowCommitmentPercent']) for human in humanStatDf['ShowCommitmentPercent']]), axis=1)

    print(heatMapDf)

# all R2

    heatMapDf['RSquared'] = heatMapDf.apply(lambda x: np.mean([x['showRSquared'], x['stepsRSquared']]), axis=1)

    measureName = 'RSquared'
    heatMap = heatMapDf.pivot("infoScale", "softmaxBeta", measureName)
    # heatMap = heatMapDf.pivot("softmaxBeta", "infoScale", measureName)
    # ax = sns.heatmap(heatMap, annot=True, square=True, fmt='.3f', linewidths=.5)

    # plt.rcParams['figure.figsize'] = (8, 6)
    # plt.rcParams['figure.dpi'] = 200

    ax = sns.heatmap(heatMap, cmap="RdBu_r", annot=True, square=True, fmt='.2f', linewidths=.5).invert_yaxis()

    plt.title(modelName + '_' + measureName)

    # plt.savefig('/Users/chengshaozhe/Downloads/heatmapExp2{}.svg'.format(modelName))
    plt.show()

    heatMapDf["rank"] = heatMapDf[measureName].rank(method="min", ascending=False).astype(np.int64)
    print(heatMapDf[heatMapDf["rank"] <= 5])

    # sDF = pd.DataFrame()
    # sDF['RSquared'] = [calCorr(human, model) for human, model in it.product(humanStatDf['ShowCommitmentPercent'], heatMapDf['ShowCommitmentPercent'])]
    # t = researchpy.summary_cont(sDF['RSquared'])
    # print(sDF)
    # print(t)
    # print(t['Mean'])

    # # print(statDF)
    # # dfExpTrail.to_csv('dfExpTrail.csv')
    # sns.set_theme(style="white")
    # plt.rcParams['figure.figsize'] = (8, 6)
    # plt.rcParams['figure.dpi'] = 200

    # colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
    #              (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]  # blue
    # ax = sns.barplot(x="decisionSteps", y="ShowCommitmentPercent", hue="participantsType", data=statDF, ci=None, palette=colorList)

    # plt.axhline(y=50, color='k', linestyle='--', alpha=0.5)

    # def changeRectWidth(ax, new_value):
    #     xList = []
    #     yList = []
    #     for index, bar in enumerate(ax.patches):
    #         current_width = bar.get_width()
    #         diff = current_width - new_value
    #         bar.set_width(new_value)
    #         if index < len(ax.patches) / 2:
    #             bar.set_x(bar.get_x() + diff)
    #         xList.append(bar.get_x() + diff / 2.)
    #         yList.append(bar.get_height())
    #     return xList, yList

    # xList, yList = changeRectWidth(ax, 0.2)

    # # stats = statDF.groupby(['decisionSteps', 'participantsType'], sort=False)['ShowCommitmentPercent'].agg(['mean', 'count', 'std'])

    # stats = statDF.groupby(['threshold', 'infoScale'], sort=False)['ShowCommitmentPercent'].agg(['mean', 'count', 'std'])

    # # print(stats)
    # # print('-' * 50)

    # sem_hi = []
    # sem_lo = []

    # for i in stats.index:
    #     m, c, s = stats.loc[i]
    #     sem_hi.append(m + s / np.sqrt(c - 1))
    #     sem_lo.append(m - s / np.sqrt(c - 1))

    # stats['sem_hi'] = sem_hi
    # stats['sem_lo'] = sem_lo
    # pd.set_option('display.max_columns', None)
    # print(stats)

    # yerrList = [stats['mean'] - stats['sem_lo'], stats['sem_hi'] - stats['mean']]
    # plt.errorbar(x=xList, y=yList, yerr=yerrList, fmt='none', c='k', elinewidth=2, capsize=5)

    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')

    # # ax = sns.boxplot(x="decisionSteps", y="avoidCommitPercent", hue="participantsType", data=statDF, palette="Set1", showmeans=True)
    # # ax.set(xlabel='Steps-to-crossroad Condition', ylabel='Show Commitment Ratio', title='Commitment with Deliberation')
    # handles, labels = ax.get_legend_handles_labels()

    # plt.xticks(fontsize=14, color='black')
    # plt.yticks(np.arange(0, 101, 10), fontsize=14, color='black')

    # plt.xlabel('Steps-to-crossroad', fontsize=16, color='black')
    # plt.ylabel("% Choosing the fixed-future path", fontsize=20, color='black')

    # plt.legend(loc='best', fontsize=16)
    # plt.ylim((0, 101))
    # plt.rcParams['svg.fonttype'] = 'none'
    # # plt.savefig('/Users/chengshaozhe/Downloads/exp2a.svg', dpi=600, format='svg')

    # plt.show()
