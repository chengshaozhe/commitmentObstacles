import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZone, calculateAvoidCommitmnetZoneAll, calculateFirstOutZoneRatio, calculateAvoidCommitmentRatio, calculateFirstIntentionStep, calculateFirstIntentionRatio


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []

    participants = ['human', 'softMaxBeta2.5', 'max']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)

        df['avoidCommitmentZone'] = df.apply(lambda x: calculateAvoidCommitmnetZone(eval(x['playerGrid']), eval(x['target1']), eval(x['target2'])), axis=1)
        df['avoidCommitmentRatio'] = df.apply(lambda x: calculateAvoidCommitmentRatio(eval(x['trajectory']), x['avoidCommitmentZone']), axis=1)
        df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
        df['firstIntentionRatio'] = df.apply(lambda x: calculateFirstIntentionRatio(eval(x['goal'])), axis=1)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]

        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine') & (df['intentionedDisToTargetMin'] > 2)]

        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]

        # dfExpTrail = df[((df['distanceDiff'] == 0) & df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine')]

        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['areaType'] == 'rect')]
        # dfExpTrail = df[(df['areaType'] != 'none')]
        # dfExpTrail = df[df['noiseNumber'] != 'special']
        # dfExpTrail = df

        statDF = pd.DataFrame()
        statDF['avoidCommitmentRatio'] = dfExpTrail.groupby('name')["avoidCommitmentRatio"].mean()
        statDF['firstIntentionRatio'] = dfExpTrail.groupby('name')["firstIntentionRatio"].mean()
        statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        # statDF.to_csv("statDF.csv")

        print('avoidCommitmentRatio', np.mean(statDF['avoidCommitmentRatio']))
        print('firstIntentionRatio', np.mean(statDF['firstIntentionRatio']))
        print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        # stats = statDF.columns
        stats = ['firstIntentionRatio', 'avoidCommitmentRatio']
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    xlabels = ['firstIntentionRatio', 'avoidCommitmentAreaRatio']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 1))
    plt.legend(loc='best')

    plt.title('avoidCommitment')
    plt.show()

    # a = [0.49711976902023564, 0.5810396187561783, 0.5878115167570421, 0.5912656253650558]
    # plt.plot(participants, a)
    # plt.ylim((0.45, 0.6))
    # plt.ylabel('firstIntentionRatio')
    # plt.title('firstIntentionRatio on different noise model')
    # plt.show()
