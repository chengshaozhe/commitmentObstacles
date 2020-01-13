import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind

from dataAnalysis import calculateFirstIntentionConsistency, calculateFirstIntention, calculateSE, calculateFirstIntention, calMidLineIntentionAfterNoise


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'max']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)

        dfExpTrail = df[(df['noiseNumber'] != 'special') & (len(df['noisePoint']) > 0) & (df['areaType'] == 'expRect')]

        dfExpTrail["goalChange"] = dfExpTrail.apply(lambda x: calMidLineIntentionAfterNoise(eval(x['trajectory']), eval(x['noisePoint']), eval(x['target1']), eval(x['target2']), eval(x['goal'])), axis=1)

        # dfExpTrail = dfExpTrail[dfExpTrail["goalChange"] == 0]
        # dfExpTrail.to_csv("gg.csv")

        statDF = pd.DataFrame()
        statDF['goalChange'] = dfExpTrail.groupby('name')["goalChange"].mean()
        print(statDF)
        # statDF.to_csv("statDF.csv")

        print('goalChange', np.mean(statDF['goalChange']))
        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    xlabels = ['normalTrial']
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
