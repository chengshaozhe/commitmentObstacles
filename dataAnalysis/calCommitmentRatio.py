import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind

from dataAnalysis import calculateFirstIntentionConsistency, calculateFirstIntention, calculateSE


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []

    statData = []
    # participants = ['human', 'softmaxBeta0.1', 'softmaxBeta0.5', 'softmaxBeta1', 'softmaxBeta2.5', 'softmaxBeta5']

    participants = ['human', 'softmaxBeta0.5']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)

        df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

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
        statData.append(statDF['firstIntentionConsistFinalGoalSpecail'])

    print(ttest_ind(statData[0], statData[1]))

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
