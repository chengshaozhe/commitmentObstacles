import os
DIRNAME = os.path.dirname(__file__)
import pandas as pd
import glob
import numpy as np
from scipy.stats import ttest_ind, chisquare

from dataAnalysis import calculateFirstIntentionConsistency, calculateFirstIntention
from dataAnalysis import calculateAvoidCommitmnetZone, calculateFirstOutZoneRatio, calculateAvoidCommitmentRatio, calculateFirstIntentionStep, calculateFirstIntentionRatio

if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    dirNames = os.listdir(resultsPath)
    # print(dirNames)
    dirNames = ['human']
    dataPaths = [os.path.join(resultsPath, filename) for filename in dirNames]

    filenames = list(filter(os.path.isdir, dataPaths))
    # print(filenames)
    df_all = pd.DataFrame()

    df = pd.concat([pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in filenames])

    # df.to_csv(os.path.join(resultsPath, 'allModel.csv'))

    df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

    df['avoidCommitmentZone'] = df.apply(lambda x: calculateAvoidCommitmnetZone(eval(x['playerGrid']), eval(x['target1']), eval(x['target2'])), axis=1)
    df['avoidCommitmentRatio'] = df.apply(lambda x: calculateAvoidCommitmentRatio(eval(x['trajectory']), x['avoidCommitmentZone']), axis=1)
    df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
    df['firstIntentionRatio'] = df.apply(lambda x: calculateFirstIntentionRatio(eval(x['goal'])), axis=1)

    dfNormailTrail = df[df['noiseNumber'] != 'special']
    dfSpecialTrail = df[df['noiseNumber'] == 'special']

    statDF = pd.DataFrame()
    statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
    statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()

    dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
    # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]
    # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
    # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]

    statDF['avoidCommitmentRatio'] = dfExpTrail.groupby('name')["avoidCommitmentRatio"].mean()

    statDF['firstIntentionRatio'] = dfExpTrail.groupby('name')["firstIntentionRatio"].mean()
    statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()

    # print(statDF)
    # statDF.to_csv(os.path.join(resultsPath, 'statDF.csv'))

    modelPath = os.path.join(resultsPath, 'softmaxBeta2.5')
    # modelPath = os.path.join(resultsPath, 'maxModelNoNoise')
    modelDf = pd.concat(map(pd.read_csv, glob.glob(os.path.join(modelPath, '*.csv'))), sort=False)

    modelDf["firstIntentionConsistFinalGoal"] = modelDf.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)
    modelDf['avoidCommitmentZone'] = modelDf.apply(lambda x: calculateAvoidCommitmnetZone(eval(x['playerGrid']), eval(x['target1']), eval(x['target2'])), axis=1)
    modelDf['avoidCommitmentRatio'] = modelDf.apply(lambda x: calculateAvoidCommitmentRatio(eval(x['trajectory']), x['avoidCommitmentZone']), axis=1)
    modelDf['firstIntentionStep'] = modelDf.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
    modelDf['firstIntentionRatio'] = modelDf.apply(lambda x: calculateFirstIntentionRatio(eval(x['goal'])), axis=1)

    modelDfNormailTrail = modelDf[modelDf['noiseNumber'] != 'special']
    modelDfSpecialTrail = modelDf[modelDf['noiseNumber'] == 'special']

    statModelDf = pd.DataFrame()
    statModelDf['firstIntentionConsistFinalGoalNormal'] = modelDfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
    statModelDf['firstIntentionConsistFinalGoalSpecail'] = modelDfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()

    dfExpTrailModel = modelDf[(modelDf['areaType'] == 'expRect') & (modelDf['noiseNumber'] != 'special')]
    # dfExpTrailModel = modelDf[(modelDf['distanceDiff'] == 0) & (modelDf['areaType'] != 'none')]
    # dfExpTrailModel = modelDf[(modelDf['distanceDiff'] == 0) & (modelDf['areaType'] == 'midLine')]
    # dfExpTrailModel = modelDf[(modelDf['distanceDiff'] == 0) & (modelDf['areaType'] == 'straightLine')]

    # dfExpTrailModel = modelDf

    statModelDf['avoidCommitmentRatio'] = dfExpTrailModel.groupby('name')["avoidCommitmentRatio"].mean()
    statModelDf['firstIntentionRatio'] = dfExpTrailModel.groupby('name')["firstIntentionRatio"].mean()
    statModelDf['firstIntentionStep'] = dfExpTrailModel.groupby('name')["firstIntentionStep"].mean()

    statModelDf.to_csv(os.path.join(resultsPath, 'modelStatDF.csv'))
    print(np.mean(statDF['firstIntentionRatio']), np.mean(statModelDf['firstIntentionRatio']))
    print(np.mean(statDF['firstIntentionStep']), np.mean(statModelDf['firstIntentionStep']))
    print(np.mean(statDF['avoidCommitmentRatio']), np.mean(statModelDf['avoidCommitmentRatio']))
    print(np.mean(statDF['firstIntentionConsistFinalGoalSpecail']), np.mean(statModelDf['firstIntentionConsistFinalGoalSpecail']))

    a = ttest_ind(statDF['firstIntentionRatio'], statModelDf['firstIntentionRatio'])
    c = ttest_ind(statDF['firstIntentionStep'], statModelDf['firstIntentionStep'])
    b = ttest_ind(statDF['avoidCommitmentRatio'], statModelDf['avoidCommitmentRatio'])
    d = ttest_ind(statDF['firstIntentionConsistFinalGoalSpecail'], statModelDf['firstIntentionConsistFinalGoalSpecail'])

    print(a, 'firstIntentionRatio')
    print(b, 'firstIntentionStep')
    print(c, 'avoidCommitmentRatio')
    print(d, 'firstIntentionConsistFinalGoalSpecail')

    # p = chisquare(statDF['firstIntentionConsistFinalGoalSpecail'].tolist(), statModelDf['firstIntentionConsistFinalGoalSpecail'].tolist()[:34])

    # if p[1] > 0.05 or p == "nan":
    #     print("H0 win,there is no difference")
    # else:
    #     print("H1 win,there is difference")
