import numpy as np
import random
import itertools as it
import math


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


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


class CreatRectMap():
    def __init__(self, rotateAngles, dimension, obstaclesMaps, rotatePoint):
        self.rotateAngles = rotateAngles
        self.dimension = dimension
        self.obstaclesMaps = obstaclesMaps
        self.rotatePoint = rotatePoint

    def __call__(self, width, height, distance, decisionSteps, targetDiffs):
        targetDiff = targetDiffs / 2
        playerGrid = (random.randint(1, max(1, self.dimension - distance - width - targetDiff - 2)), random.randint(1, max(1, self.dimension - distance - height - targetDiff - 2)))

        target1 = (playerGrid[0] + width + distance, playerGrid[1] + height)
        target2 = (playerGrid[0] + width + targetDiff, playerGrid[1] + height + distance + targetDiff)

        obstaclesBase = self.obstaclesMaps[decisionSteps]
        transformBase = (playerGrid[0] - 1, playerGrid[1] - 1)
        obstacles = [tuple(map(sum, zip(obstacle, transformBase))) for obstacle in obstaclesBase]

        angle = random.choice(self.rotateAngles)
        playerGrid, target1, target2 = [self.rotatePoint(point, angle) for point in [playerGrid, target1, target2]]
        obstacles = [self.rotatePoint(point, angle) for point in obstacles]

        return playerGrid, target1, target2, obstacles


class CreatLineMap():
    def __init__(self, rotateAngles, dimension, rotatePoint, isInBoundary):
        self.rotateAngles = rotateAngles
        self.dimension = dimension
        self.rotatePoint = rotatePoint
        self.isInBoundary = isInBoundary

    def __call__(self, width, height, distance, decisionSteps, targetDiffs):
        targetDiff = random.choice([targetDiffs / 2, -targetDiffs / 2])
        playerGrid = (random.randint(math.floor(self.dimension / 2) - 1, math.floor(self.dimension / 2) + 1), random.randint(1, 2))
        target1 = (playerGrid[0] - distance + targetDiff, playerGrid[1] + width + height - 1)
        target2 = (playerGrid[0] + distance, playerGrid[1] + width + height - 1)

        while not self.isInBoundary(target1) or not self.isInBoundary(target1):
            playerGrid = (random.randint(math.floor(self.dimension / 2) - 1, math.floor(self.dimension / 2) + 1), random.randint(1, 2))
            target1 = (playerGrid[0] - distance + targetDiff, playerGrid[1] + width + height - 1)
            target2 = (playerGrid[0] + distance, playerGrid[1] + width + height - 1)

        obstacles = [(playerGrid[0] - 1, playerGrid[1]), (playerGrid[0] + 1, playerGrid[1])]
        for i in range(width + height - 1):
            obstacles.append((obstacles[0][0], obstacles[0][1] + i))
            obstacles.append((obstacles[1][0], obstacles[1][1] + i))

        if decisionSteps < 10:
            obstaclesAtdecisionSteps = [(playerGrid[0] - 1, playerGrid[1] + decisionSteps), (playerGrid[0] + 1, playerGrid[1] + decisionSteps)]
            [obstacles.remove(obstaclesAtdecisionStep) for obstaclesAtdecisionStep in obstaclesAtdecisionSteps]

        angle = random.choice(self.rotateAngles)
        playerGrid, target1, target2 = [self.rotatePoint(point, angle) for point in [playerGrid, target1, target2]]
        obstacles = [self.rotatePoint(point, angle) for point in obstacles]

        return playerGrid, target1, target2, obstacles


class CreatSingleGoalMap:
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, distance):
        playerGrid = (random.randint(1, max(1, self.dimension - 2)), random.randint(1, max(1, self.dimension - 2)))
        allGrids = tuple(it.product(range(1, self.dimension - 2), range(1, self.dimension - 2)))
        possibleGrids = list(filter(lambda x: calculateGridDis(playerGrid, x) == distance, allGrids))
        targetGrid = random.choice(possibleGrids)
        return playerGrid, targetGrid


class SamplePositionFromCondition:
    def __init__(self, creatRectMap, creatLineMap, expDesignValues):
        self.creatRectMap = creatRectMap
        self.creatLineMap = creatLineMap
        self.expDesignValues = expDesignValues
        self.index = 0

    def __call__(self, condition):
        width, height, distance, decisionSteps, targetDiff = self.expDesignValues[self.index]
        if condition.name == 'expCondition':
            playerGrid, target1, target2, obstacles = self.creatRectMap(width, height, distance, decisionSteps, targetDiff)
        if condition.name == 'lineCondition':
            playerGrid, target1, target2, obstacles = self.creatLineMap(width, height, distance, decisionSteps, targetDiff)
        self.index += 1
        return playerGrid, target1, target2, obstacles, decisionSteps, targetDiff


class RotatePoint:
    def __init__(self, dimension):
        self.dimension = dimension
        self.center = (int((self.dimension - 1) / 2), int((self.dimension - 1) / 2))

    def __call__(self, point, angle):
        x1, y1 = point
        x2, y2 = self.center
        y1 = self.dimension - y1
        y2 = self.dimension - y2
        radian = math.radians(angle)
        x = (x1 - x2) * int(math.cos(radian)) - (y1 - y2) * int(math.sin(radian)) + x2
        y = (x1 - x2) * int(math.sin(radian)) + (y1 - y2) * int(math.cos(radian)) + y2
        y = self.dimension - y
        return (int(x), int(y))


if __name__ == '__main__':
    dimension = 15
    import pygame
    from visualization import DrawBackground, DrawNewState, DrawImage, DrawText
    screenWidth = 600
    screenHeight = 600
    screen = pygame.display.set_mode((screenWidth, screenHeight))

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    textColorTuple = (255, 50, 50)
    targetRadius = 10
    playerRadius = 10

    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

# design
    width = [5]
    height = [5]
    intentionDis = [3, 4, 5, 6]
    decisionSteps = [2, 4, 6]
    targetDiffs = [0, 2, 4]
    rotateAngles = [0, 90, 180, 270]

    expDesignValues = [[b, h, d, m, diff] for b in width for h in height for d in intentionDis for m in decisionSteps for diff in targetDiffs]

    obstaclesMap1 = [(2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2)]
    obstaclesMap2 = [(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)]
    obstaclesMap3 = [(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)]

    # obstaclesMap4 =
    speicalObstacleMap = [(4, 1), (4, 2), (6, 3), (6, 4), (1, 4), (2, 4), (3, 6), (4, 6)]
    obstaclesCondition = [obstaclesMap1, obstaclesMap2, obstaclesMap3, speicalObstacleMap]
    obstaclesMaps = dict(zip(decisionSteps, obstaclesCondition))

    rotatePoint = RotatePoint(dimension)
    creatRectMap = CreatRectMap(rotateAngles, dimension, obstaclesMaps, rotatePoint)
    samplePositionFromCondition = SamplePositionFromCondition(creatRectMap, expDesignValues)


# sample
    condition = 'exp'
    for _ in range(30):
        playerGrid, target1, target2, obstacles, decisionSteps = samplePositionFromCondition(condition)
        # obstacles = creatObstacles(pointList)

        # playerGrid = (1, 1)
        # obstacles = [(2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2)]
        # obstacles = [(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)]
        # obstacles = [(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)]

        angle = random.choice(rotateAngles)
        playerGrid, target1, target2 = [rotatePoint(point, angle) for point in [playerGrid, target1, target2]]
        obstacles = [rotatePoint(point, angle) for point in obstacles]

        pause = True
        while pause:
            screen = drawNewState(target1, target2, playerGrid, obstacles)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    pause = False
                if event.type == pygame.QUIT:
                    pygame.quit()
    pygame.quit()
