import numpy as np
import random
import itertools as it
import math


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


def rotatePoint(pointA, pointB, height, angle):
    x1, y1 = pointA
    x2, y2 = pointB
    y1 = height - y1
    y2 = height - y2
    radian = math.radians(angle)
    x = (x1 - x2) * math.cos(radian) - (y1 - y2) * math.sin(radian) + x2
    y = (x1 - x2) * math.sin(radian) + (y1 - y2) * math.cos(radian) + y2
    y = height - y
    return (int(x), int(y))


if __name__ == '__main__':
    dimension = 15
    direction = [0, 90, 180, 270]
    import pygame
    from visualization import DrawBackground, DrawNewState, DrawImage, DrawText
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

    pointList = [(1, 1), (6, 11), (11, 6)]
    center = (7, 7)
    angle = 0
    obstacles = ((2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2))
    # obstacles = ((3, 3), (4, 1), (1, 4),  (5, 3), (3, 5), (6, 3), (3, 6))
    # obstacles = ((4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4))

    t = [rotatePoint(point, center, dimension, angle) for point in pointList]
    print(t)

    playerGrid, target1, target2 = t
    pause = True
    pygame.init()
    while pause:
        screen = drawNewState(target1, target2, playerGrid, obstacles)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pause = False
            if event.type == pygame.QUIT:
                pygame.quit()
    pygame.quit()
