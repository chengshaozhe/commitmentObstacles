import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
import researchpy

from dataAnalysis import *
from machinePolicy.onlineVIWithObstacle import RunVI
from dataAnalysis import *


gridSize = 15
noise = 0.067
noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
gamma = 0.9
goalReward = [30]
actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]

runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)
softmaxBeta = 2.5
softmaxPolicy = SoftmaxPolicy(softmaxBeta)
initPrior = [0.5, 0.5]
intentionInfernce = IntentionInfernce(initPrior, softmaxPolicy, runVI)
inferThreshold = 1
calFirstIntentionFromPosterior = CalFirstIntentionFromPosterior(inferThreshold)

if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    statDFList = []
    # participants = ['human', 'softmaxBeta0.1', 'softmaxBeta0.5', 'softmaxBeta1', 'softmaxBeta2.5', 'softmaxBeta5']

    participants = ['human', 'RL']
    participants = ['human', 'actWithMonitorIntentionShow-RLThreshold0.1']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        # dataPath = resultsPath

        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)

        df = df[df['noiseNumber'] == 'special']
        df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)


# bayesian
        # df['posteriorList'] = df.apply(lambda x: intentionInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)
        # df['firstIntention'] = df.apply(lambda x: calFirstIntentionFromPosterior(x['posteriorList']), axis=1)
        # df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calIntentionCosistency(x['firstIntention'], x['posteriorList']), axis=1)

        # df.to_csv("goalInfer.csv")

        dfNormailTrail = df[df['noiseNumber'] != 'special']
        dfSpecialTrail = df[df['noiseNumber'] == 'special']

        statDF = pd.DataFrame()
        # statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
        statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
        # statDF.to_csv("statDF.csv")

        statDF['firstIntentionConsistFinalGoalSpecail'] = statDF.apply(lambda x: int(x['firstIntentionConsistFinalGoalSpecail'] * 100), axis=1)

        # print('firstIntentionConsistFinalGoalNormal', np.mean(statDF['firstIntentionConsistFinalGoalNormal']))
        print('firstIntentionConsistFinalGoalSpecail', np.mean(statDF['firstIntentionConsistFinalGoalSpecail']))
        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats][0])
        stdList.append([calculateSE(statDF[stat]) for stat in stats][0])
        statDFList.append(statDF['firstIntentionConsistFinalGoalSpecail'].tolist())

    # print(ttest_ind(statDFList[0], statDFList[1]))
    # print(statsList, stdList)
    # print(statDFList[0], statDFList[1])
    # df = pd.DataFrame(np.array([statDFList[0], statDFList[1]]).T, columns=['human', 'model'])
    # print(researchpy.ttest(df['human'], df['model']))

    # dfchi = pd.DataFrame(np.array([statDFList[0], statDFList[1]]).T, columns=['human', 'model'])
    # crosstab, res = researchpy.crosstab(df['disease'], df['alive'], test="chi-square")
    # crosstab
    # statsList = [[0.98, 0.55]]
    # stdList = [[0.0032, 0.0527]]
    # xlabels = ['Normal Trial', 'Special Trial']
    # xlabels = ['Special Trial']
    # lables = ['Human Time Pressure', 'Human No Time Pressure ', 'RL Agent']
    # x = np.arange(len(xlabels))
    # totalWidth, n = 0.6, len(participants)
    # width = totalWidth / n
    # x = x - (totalWidth - width) / 3
    # for i in range(len(statsList)):
    #     plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=lables[i])
    # plt.xticks(x, xlabels)

    # plt.ylim((0, 1))
    # plt.legend(loc='best')
    # plt.title('Commitment Ratio')  # Intention Consistency
    # plt.show()

    labels = ['Humans', 'RL']
    x_pos = np.arange(len(labels))

    # import seaborn as sns
    # sns.set_theme(style="white")
    # plt.style.use('default')
    fig, ax = plt.subplots()

    colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
                 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]
    ax.bar(x_pos, statsList,
           yerr=stdList,
           align='center',
           # alpha=0.5,
           ecolor='black',
           capsize=16,
           color=colorList)

    # ax = sns.barplot(x='trialType', y="commitmentRatio", hue="participantsType", data=statDF, ci=68, palette=colorList)

    # plt.axhline(y=50, color='k', linestyle='--', alpha=0.5)
    plt.rcParams['figure.dpi'] = 200

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # ax.set_ylabel()
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    # ax.set_title('Commitment Ratio in Special Trial')
    # ax.yaxis.grid(True)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(np.arange(0, 101, 10), fontsize=12, color='black')

    plt.ylabel('% Choosing the Original Restaurant', fontsize=16, color='black')

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('/Users/chengshaozhe/Downloads/exp2c.jpg', dpi=200)
    plt.show()
