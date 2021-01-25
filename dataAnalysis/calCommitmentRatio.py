import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind

from dataAnalysis import *
from machinePolicy.onlineVIWithObstacle import RunVI
from dataAnalysis import *


gridSize = 15
noise = 0.067
noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
gamma = 0.9
goalReward = [10]
actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]

runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)
softmaxBeta = 5
softmaxPolicy = SoftmaxPolicy(softmaxBeta)
initPrior = [0.5, 0.5]
intentionInfernce = IntentionInfernce(initPrior, softmaxPolicy, runVI)
inferThreshold = 1
calFirstIntentionFromPosterior = CalFirstIntentionFromPosterior(inferThreshold)

if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []

    statData = []
    # participants = ['human', 'softmaxBeta0.1', 'softmaxBeta0.5', 'softmaxBeta1', 'softmaxBeta2.5', 'softmaxBeta5']
    # participants = ['human', 'noise0.0673_softmaxBeta2.5']
    participants = ['human', 'RL']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        # dataPath = resultsPath

        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)

        df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

        # df = df[df['noiseNumber'] == 'special']
        # df['posteriorList'] = df.apply(lambda x: intentionInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)

        # df['firstIntention'] = df.apply(lambda x: calFirstIntentionFromPosterior(x['posteriorList']), axis=1)

        # df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calIntentionCosistency(x['firstIntention'], x['posteriorList']), axis=1)

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

    # print(ttest_ind(statData[0], statData[1]))
    print(statsList, stdList)
    print(statsList)
    # statsList = [[0.98, 0.55]]
    # stdList = [[0.0032, 0.0527]]
    xlabels = ['normalTrial', 'specialTrial']
    lables = ['Human', 'Human No Time pressure ', 'RL Agent']
    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 3
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=lables[i])
    plt.xticks(x, xlabels)

    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('Commitment Ratio')  # Intention Consistency
    plt.show()
