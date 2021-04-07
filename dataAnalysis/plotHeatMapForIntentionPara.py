import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score as KL
from dataAnalysis import calculateSE, calculateSD, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calMidPoints, calculateFirstIntentionConsistency
import researchpy
import seaborn as sns


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


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results/intentionModelWithNaiveInfer')
    # participants = ['human', 'RL']

    humansStats = [42.50, 43.95, 63.80, 83.75, 77.60]

    # humansStats = [48.80, 46.35, 68.90, 83.10, 81.45] #obstacle 1

    # humansStats = [50, 50, 63.80, 83.75, 77.60]

    dirs = os.listdir(resultsPath)[1:]
    participants = [d for d in dirs if not d[0] == '.']
    print(participants)

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)

    df['participantsType'] = df.apply(lambda x: calParticipantType(x['name']), axis=1)
    df['totalTime'] = df.apply(lambda x: eval(x['reactionTime'])[-1], axis=1)

    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)

    # df = df[(df['noisePoint'] == '[]')]
    df = df[(df['targetDiff'] == '0')]

    # dfExpTrail = df[(df['conditionName'] == 'expCondition1')]

    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    statDF = pd.DataFrame()
    statDF['avoidCommitPercent'] = dfExpTrail.groupby(['threshold', 'infoScale', 'decisionSteps'])["hasAvoidPoint"].mean()
    # statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name'])["hasAvoidPoint"].mean()

    statDF['ShowCommitmentPercent'] = statDF.apply(lambda x: int((1 - x['avoidCommitPercent']) * 100), axis=1)

    statDF = statDF.reset_index()
    # statDF['participantsType'] = ['RL' if 'noise' in name else 'Humans' for name in statDF['name']]
    pd.set_option("max_columns", 10)
    # print(statDF)

    heatMapDf = pd.DataFrame()
    heatMapDf['avoidCommitPercent'] = dfExpTrail.groupby(['threshold', 'infoScale'])["hasAvoidPoint"].mean()
    heatMapDf['ShowCommitmentList'] = statDF.groupby(['threshold', 'infoScale'])['ShowCommitmentPercent']
    heatMapDf = heatMapDf.reset_index()

    def calRMSE(x, y):
        return np.sqrt(np.power(np.array(x) - np.array(y), 2).mean())
    print(heatMapDf)

    heatMapDf['RMSE'] = heatMapDf.apply(lambda x: calRMSE(x['ShowCommitmentList'][1], humansStats), axis=1)
    print(heatMapDf)

    heatMap = heatMapDf.pivot("threshold", "infoScale", "RMSE")
    ax = sns.heatmap(heatMap, cmap='RdBu', annot=True, linewidths=.5)
    plt.title("RMSE")
    plt.savefig('/Users/chengshaozhe/Downloads/heatmapExp2NaiveInfer.jpg', dpi=600)
    plt.show()

    # statDF['participantsType'] = statDF.apply(lambda x: calParticipantType(x['name']), axis=1)

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
