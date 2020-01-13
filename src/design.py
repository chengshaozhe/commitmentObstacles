import numpy as np
import random
import itertools as it


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


class CreatStraightLineCondition():
    def __init__(self, direction, dimension):
        self.direction = direction
        self.dimension = dimension

    def __call__(self, width, height, distance, distanceDiff):
        direction = random.choice(self.direction)
        distanceDiff = random.choice([distanceDiff, -distanceDiff])
        if direction == 0:
            pacmanPosition = (random.randint(1, self.dimension - (distance + distanceDiff) - width - 2), random.randint(1, self.dimension - (distance + distanceDiff) - height - 2))
            bean1Position = (pacmanPosition[0] + width + (distance + distanceDiff), pacmanPosition[1] + height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] + height + distance)

        elif direction == 90:
            pacmanPosition = (random.randint(1 + (distance + distanceDiff) + width, self.dimension - 2), random.randint(1, self.dimension - 2 - (distance + distanceDiff) - height))
            bean1Position = (pacmanPosition[0] - width, pacmanPosition[1] + height + (distance + distanceDiff))
            bean2Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] + height)

        elif direction == 180:
            pacmanPosition = (random.randint(1 + (distance + distanceDiff) + width, self.dimension - 2), random.randint(1 + (distance + distanceDiff) + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] - width - (distance + distanceDiff), pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] - width, pacmanPosition[1] - height - distance)
        else:
            pacmanPosition = (random.randint(1, self.dimension - (distance + distanceDiff) - width - 2), random.randint(1 + (distance + distanceDiff) + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] + width + (distance + distanceDiff), pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] - height - distance)

        return bean1Position, bean2Position, pacmanPosition


class CreatExpCondition():
    def __init__(self, direction, dimension):
        self.direction = direction
        self.dimension = dimension

    def __call__(self, width, height, distance):
        direction = random.choice(self.direction)
        if direction == 45:
            pacmanPosition = (random.randint(1, self.dimension - distance - width - 2), random.randint(1, self.dimension - distance - height - 2))
            bean1Position = (pacmanPosition[0] + width + distance, pacmanPosition[1] + height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] + height + distance)

        elif direction == 135:
            pacmanPosition = (random.randint(1 + distance + width, self.dimension - 2), random.randint(1, self.dimension - 2 - distance - height))
            bean1Position = (pacmanPosition[0] - width, pacmanPosition[1] + height + distance)
            bean2Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] + height)

        elif direction == 225:
            pacmanPosition = (random.randint(1 + distance + width, self.dimension - 2), random.randint(1 + distance + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] - width, pacmanPosition[1] - height - distance)
        else:
            pacmanPosition = (random.randint(1, self.dimension - distance - width - 2), random.randint(1 + distance + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] + width + distance, pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] - height - distance)

        return pacmanPosition, bean1Position, bean2Position, direction


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = calculateGridDis(playerGrid, target1)
    dis2 = calculateGridDis(playerGrid, target2)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []

    return avoidCommitmentZone


def isZoneALine(zone):
    zoneArr = np.array(zone).T
    if len(set(zoneArr[0])) == 1 or len(set(zoneArr[1])) == 1:
        return True
    else:
        return False


def createNoiseDesignValue(noiseCondition, blockNumber):
    noiseDesignValues = np.array([random.choice(noiseCondition) for _ in range(blockNumber)]).flatten().tolist()
    noiseDesignValues.append('special')

    return noiseDesignValues


def createExpDesignValue(width, height, distance):
    expDesignValues = [[b, h, d] for b in width for h in height for d in distance]
    random.shuffle(expDesignValues)
    expDesignValues.append([random.choice(width), random.choice(height), random.choice(distance)])
    return expDesignValues


class SamplePositionFromCondition:
    def __init__(self, df, createExpCondition, expDesignValues):
        self.df = df
        self.createExpCondition = createExpCondition
        self.expDesignValues = expDesignValues
        self.index = 0

    def __call__(self, condition):
        if condition.name == 'expCondition':
            playerGrid, target1, target2, direction = self.createExpCondition(self.expDesignValues[self.index][0], self.expDesignValues[self.index][1], self.expDesignValues[self.index][2])
            dis1 = np.linalg.norm(np.array(playerGrid) - np.array(target1), ord=1)
            dis2 = np.linalg.norm(np.array(playerGrid) - np.array(target2), ord=1)
            minDis = min(dis1, dis2)
            avoidCommitmentZone = len(calculateAvoidCommitmnetZone(playerGrid, target1, target2))
            chooseConditionDF = {'areaType': 'expRect', 'playerGrid': str(playerGrid), 'target1': str(target1), 'target2': str(target2), 'minDis': minDis, 'distanceDiff': 0, 'avoidCommitmentZone': avoidCommitmentZone, 'intentionedDisToTargetMin': self.expDesignValues[self.index][2], 'distanceDiff': 0}
            self.index += 1
        else:
            conditionDf = self.df[(self.df['areaType'] == condition.areaType) & (self.df['avoidCommitmentZone'].isin(condition.areaSize)) & (self.df['distanceDiff'].isin(condition.distanceDiff)) & (self.df['minDis'].isin(condition.minDis)) & (self.df['intentionedDisToTargetMin'].isin(condition.intentionedDisToTarget))]
            choosenIndex = random.choice(conditionDf.index)
            chooseConditionDF = conditionDf.loc[choosenIndex]

            positionDf = conditionDf[['playerGrid', 'target1', 'target2']]
            playerGrid, target1, target2 = [eval(i) for i in positionDf.loc[choosenIndex]]
        return playerGrid, target1, target2, chooseConditionDF


if __name__ == '__main__':
    dimension = 15
    direction = [0, 90, 180, 270]
    import pygame
    from src.visualization import DrawBackground, DrawNewState, DrawImage, DrawText
    # pygame.init()
    screenWidth = 600
    screenHeight = 600
    screen = pygame.display.set_mode((screenWidth, screenHeight))

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

    creatStraightLineCondition = CreatStraightLineCondition(direction, dimension)
    width, height, distance, distanceDiff = [2, 2, 2, 0]

    target1, target2, playerGrid = creatStraightLineCondition(width, height, distance, distanceDiff)

    import pandas as pd
    df = pd.DataFrame(columns=('playerGrid', 'target1', 'target2'))
    coordinations = tuple(it.product(range(1, dimension - 1), range(1, dimension - 1)))
    stateAll = list(it.combinations(coordinations, 3))
    # print(len(stateAll))
    # condition = []
    # index = 0
    distanceDiffList = [0, 2, 4]
    intentionedDisToTargetList = [2, 4, 6]
    areaSize = [[3, 4, 5, 6], [3, 4, 5, 6]]

    # for diff in distanceDiffList:
    #     for state in stateAll:
    #         playerGrid, target1, target2 = state
    #         avoidCommitmentZone, distanceDiff = calculateAvoidCommitmnetZone(playerGrid, target1, target2)
    #         dis1 = np.linalg.norm(np.array(playerGrid) - np.array(target1), ord=1)
    #         dis2 = np.linalg.norm(np.array(playerGrid) - np.array(target2), ord=1)
    #         minDis = min(dis1, dis2)

    #         if len(avoidCommitmentZone) > 3 and distanceDiff == diff and minDis > 4 and isZoneALine(avoidCommitmentZone) == True:
    #             df = df.append(pd.DataFrame({'index': [index], 'distanceDiff': distanceDiff, 'playerGrid': [playerGrid], 'target1': [target1], 'target2': [target2]}))
    #             index += 1

    # df.to_csv('NoAvoidCommitmentZone.csv')

    # intentionedDisToTarget = minDis - areaSize
    distanceDiffList = [0, 2, 4]
    minDisList = range(5, 15)
    intentionedDisToTargetList = [2, 4, 6]
    rectAreaSize = [6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 36]
    lineAreaSize = [4, 5, 6, 7, 8, 9, 10]
    from collections import namedtuple
    condition = namedtuple('condition', 'name areaType distanceDiff minDis areaSize intentionedDisToTarget')

    expCondition = condition(name='expCondition', areaType='rect', distanceDiff=[0], minDis=minDisList, areaSize=rectAreaSize, intentionedDisToTarget=intentionedDisToTargetList)

    rectCondition = condition(name='controlRect', areaType='rect', distanceDiff=[2, 4], minDis=minDisList, areaSize=rectAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    straightLineCondition = condition(name='straightLine', areaType='straightLine', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=lineAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    midLineCondition = condition(name='MidLine', areaType='midLine', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=lineAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    noAreaCondition = condition(name='noArea', areaType='none', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=[0], intentionedDisToTarget=intentionedDisToTargetList)

    import os
    import pandas as pd
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'conditionData'))
    df = pd.read_csv(os.path.join(picturePath, 'DesignConditionForAvoidCommitmentZone.csv'))
    df['intentionedDisToTargetMin'] = df.apply(lambda x: x['minDis'] - x['avoidCommitmentZone'], axis=1)
    # print(df.head())
    # df.to_csv('condition.csv')

    width = [3, 4, 5]
    height = [3, 4, 5]
    intentionDis = [2, 4, 6]
    direction = [45, 135, 225, 315]
    gridSize = 15

    createExpCondition = CreatExpCondition(direction, gridSize)
    expDesignValues = [[b, h, d] for b in width for h in height for d in intentionDis]
    random.shuffle(expDesignValues)
    numExpTrial = len(expDesignValues)

    samplePositionFromCondition = SamplePositionFromCondition(df, createExpCondition, expDesignValues)

    numControlTrial = int(numExpTrial * 2 / 3)
    conditionList = list([expCondition] * numExpTrial + [rectCondition] * numExpTrial + [straightLineCondition] * numControlTrial + [midLineCondition] * numControlTrial + [noAreaCondition] * numControlTrial)

    # conditionList = list([expCondition] * numExpTrial)

    random.shuffle(conditionList)
    minDisList = []
    timeStep = 0
    for index, condition in enumerate(conditionList):
        print('index:', index + 1)
        playerGrid, target1, target2, chooseConditionDF = samplePositionFromCondition(condition)
        minDis = chooseConditionDF['minDis'] + chooseConditionDF['distanceDiff']
        minDisList.append(minDis)
        print(condition)
        print(chooseConditionDF)
        pause = True
        pygame.init()
        saveImage = False
        saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), 'gg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        while pause:
            screen = drawNewState(target1, target2, playerGrid)
            if saveImage == True:
                pygame.image.save(screen, saveImageDir + '/' + format(timeStep, '04') + ".png")
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    pause = False
                    timeStep += 1
                if event.type == pygame.QUIT:
                    pygame.quit()
    pygame.quit()
    print('minDis:', np.mean(minDisList))

    # for design in expDesignValues:
    #     playerGrid, target1, target2, direction = createExpCondition(design[0], design[1], design[2])
    #     pause = True
    #     while pause:
    #         drawNewState(target1, target2, playerGrid)
    #         for event in pygame.event.get():
    #             if event.type == pygame.KEYDOWN:
    #                 pause = False
    #             if event.type == pygame.QUIT:
    #                 pygame.quit()
    # pygame.quit()
