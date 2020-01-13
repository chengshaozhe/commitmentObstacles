import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind

from dataAnalysis import calculateSE


def calculateTimeGap(timeList):
    timeGap = [timeList[i + 1] - timeList[i] for i in range(len(timeList) - 1)]
    return timeGap


def calculateIsTimeMaxNextNoisePoint(timeList, noisePoint):
    noisePointNextStep = [i + 1 for i in noisePoint]
    timeGap = [timeList[i + 1] - timeList[i] for i in range(len(timeList) - 1)]
    maxReactTimeStep = [i + 2 for i, x in enumerate(timeGap) if x == max(timeGap)]
    if [i for i in maxReactTimeStep if i in noisePointNextStep] != []:
        isTimeMaxNextNoisePoint = 1
    else:
        isTimeMaxNextNoisePoint = 0
    return isTimeMaxNextNoisePoint


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'softmaxBeta2.5']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)

        df["timeGap"] = df.apply(lambda x: calculateTimeGap(eval(x['reactionTime'])), axis=1)

        dfNormailTrail = df[df['noiseNumber'] != 'special']
        dfSpecialTrail = df[df['noiseNumber'] == 'special']

        statDF = pd.DataFrame()
        statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
        statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
        # statDF.to_csv("statDF.csv")

        print('firstIntentionConsistFinalGoalNormal', np.mean(statDF['firstIntentionConsistFinalGoalNormal']))
        print('firstIntentionConsistFinalGoalSpecail', np.mean(statDF['firstIntentionConsistFinalGoalSpecail']))
        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    xlabels = ['normalTrial', 'specialTrial']
    lables = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 3
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=lables[i])
    plt.xticks(x, xlabels)

    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('commit to goal ratio')  # Intention Consistency
    plt.show()
