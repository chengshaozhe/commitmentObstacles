import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter

from dataAnalysis import calculateSE, calculateFirstIntentionStep


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'noise0.0673_softmaxBeta2.5', 'max']
    participants = ['human', 'intentionModelWithNaiveInfer/threshold0.4infoScale5']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)

        df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
        df['totalStep'] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        # dfExpTrail = df[(df['conditionName'] == 'expCondition2') & (df['decisionSteps'] == 6)]
        df = df[(df['targetDiff'] == '0') | (df['targetDiff'] == 0)]

        dfExpTrail = df
        # dfExpTrail = df[df['noiseNumber'] != 'special']

        statDF = pd.DataFrame()
        # statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        statDF['totalStep'] = dfExpTrail.groupby('name')["totalStep"].mean()

        print('totalStep', np.mean(statDF['totalStep']))
        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    xlabels = ['totalStep']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.1, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    # plt.ylim((0, 10))
    plt.legend(loc='best')

    plt.title('firstIntentionStep')
    plt.show()
