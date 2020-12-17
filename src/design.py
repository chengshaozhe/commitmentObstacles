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
        playerGrid = (random.randint(2, max(1, self.dimension - distance - width - targetDiffs - 2)), random.randint(2, max(1, self.dimension - distance - height - targetDiffs - 2)))

        targetP1 = (playerGrid[0] + width + distance + targetDiffs, playerGrid[1] + height)
        targetP2 = (playerGrid[0] + width, playerGrid[1] + height + distance)
        targetList1 = [targetP1, targetP2]

        targetP3 = (playerGrid[0] + height, playerGrid[1] + width + distance + targetDiffs)
        targetP4 = (playerGrid[0] + height + distance, playerGrid[1] + width)
        targetList2 = [targetP3, targetP4]

        target1, target2 = random.choice([targetList1, targetList2])

        obstaclesBase = random.choice(self.obstaclesMaps[decisionSteps])
        initBase = (1, 1)
        if decisionSteps == 0:
            initBase = (2, 1)
        transformBase = (playerGrid[0] - initBase[0], playerGrid[1] - initBase[1])
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
        distance = distance + 1
        targetDiff = random.choice([targetDiffs / 2, -targetDiffs / 2])
        playerGrid = (math.floor(self.dimension / 2), random.randint(1, 2))
        target1 = (playerGrid[0] - distance - targetDiff, playerGrid[1] + width + height - 1)
        target2 = (playerGrid[0] + distance - targetDiff, playerGrid[1] + width + height - 1)

        obstacles = [(playerGrid[0] - 1, playerGrid[1]), (playerGrid[0] + 1, playerGrid[1])]
        for i in range(1, width + height - 1):
            obstacles.append((obstacles[0][0], obstacles[0][1] + i))
            obstacles.append((obstacles[1][0], obstacles[1][1] + i))

        obstaclesAtdecisionSteps = [(playerGrid[0] - 1, playerGrid[1] + decisionSteps), (playerGrid[0] + 1, playerGrid[1] + decisionSteps)]
        [obstacles.remove(obstaclesAtdecisionStep) for obstaclesAtdecisionStep in obstaclesAtdecisionSteps]

        angle = random.choice(self.rotateAngles)
        playerGrid, target1, target2 = [self.rotatePoint(point, angle) for point in [playerGrid, target1, target2]]
        obstacles = [self.rotatePoint(point, angle) for point in obstacles]

        print(decisionSteps)
        return playerGrid, target1, target2, obstacles


class CreatSingleGoalMap:
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, distance):
        playerGrid = (random.randint(1, max(1, self.dimension - 1)), random.randint(1, max(1, self.dimension - 1)))
        allGrids = tuple(it.product(range(self.dimension), range(self.dimension)))
        possibleGrids = list(filter(lambda x: calculateGridDis(playerGrid, x) == distance, allGrids))
        targetGrid = random.choice(possibleGrids)
        return playerGrid, targetGrid


class RandomWorld():
    def __init__(self, bounds, minDistanceBetweenGrids, maxDistanceBerweenGrids, numOfObstacles):
        self.bounds = bounds
        self.minDistanceBetweenGrids = minDistanceBetweenGrids
        self.maxDistanceBerweenGrids = maxDistanceBerweenGrids
        self.numOfObstacles = numOfObstacles

    def __call__(self):
        playerGrid = (random.randint(self.bounds[0], self.bounds[2]),
                      random.randint(self.bounds[1], self.bounds[3]))
        [meshGridX, meshGridY] = np.meshgrid(range(self.bounds[0], self.bounds[2] + 1, 1),
                                             range(self.bounds[1], self.bounds[3] + 1, 1))
        distanceOfPlayerGrid = abs(meshGridX - playerGrid[0]) + abs(meshGridY - playerGrid[1])
        target1GridArea = np.where(distanceOfPlayerGrid > self.minDistanceBetweenGrids)
        target1GridIndex = random.randint(0, len(target1GridArea[0]) - 1)
        target1Grid = tuple([meshGridX[target1GridArea[0][target1GridIndex]][target1GridArea[1][target1GridIndex]], meshGridY[target1GridArea[0][target1GridIndex]][target1GridArea[1][target1GridIndex]]])
        distanceOfTarget1Grid = abs(meshGridX - target1Grid[0]) + abs(meshGridY - target1Grid[1])
        target2GridArea = np.where((distanceOfPlayerGrid > self.minDistanceBetweenGrids) * (distanceOfTarget1Grid > self.minDistanceBetweenGrids) * (distanceOfTarget1Grid < self.maxDistanceBerweenGrids) == True)
        target2GridIndex = random.randint(0, len(target2GridArea[0]) - 1)
        target2Grid = tuple([meshGridX[target2GridArea[0][target2GridIndex]][target2GridArea[1][target2GridIndex]], meshGridY[target2GridArea[0][target2GridIndex]][target2GridArea[1][target2GridIndex]]])
        vectorBetweenTarget1AndPlayer = np.array(target2Grid) - np.array(playerGrid)
        vectorBetweenTarget2AndPlayer = np.array(target1Grid) - np.array(playerGrid)

        dimension = max(self.bounds)
        allGrids = tuple(it.product(range(dimension), range(dimension)))
        possibleGrids = list(filter(lambda x: x not in [playerGrid, target1Grid, target2Grid], allGrids))
        obstacles = random.sample(possibleGrids, self.numOfObstacles)
        return playerGrid, target1Grid, target2Grid, obstacles


def calculateMaxDistanceOfGrid(bounds):
    [meshGridX, meshGridY] = np.meshgrid(range(bounds[0], bounds[2] + 1, 1),
                                         range(bounds[1], bounds[3] + 1, 1))
    allDistance = np.array([abs(meshGridX - bounds[0]) + abs(meshGridY - bounds[1]),
                            abs(meshGridX - bounds[2]) + abs(meshGridY - bounds[1]),
                            abs(meshGridX - bounds[0]) + abs(meshGridY - bounds[3]),
                            abs(meshGridX - bounds[2]) + abs(meshGridY - bounds[3])])
    maxDistance = np.min(allDistance.max(0))
    return maxDistance


class SamplePositionFromCondition:
    def __init__(self, creatRectMap, creatLineMap, expDesignValues):
        self.creatRectMap = creatRectMap
        self.creatLineMap = creatLineMap
        self.expDesignValues = expDesignValues
        self.index = 0

    def __call__(self, condition):
        width, height, distance, decisionSteps, targetDiff = self.expDesignValues[self.index]
        if condition.name == 'expCondition' or condition.name == 'specialCondition':
            playerGrid, target1, target2, obstacles = self.creatRectMap(width, height, distance, decisionSteps, targetDiff)
        if condition.name == 'lineCondition':
            playerGrid, target1, target2, obstacles = self.creatLineMap(width, height, distance, decisionSteps, targetDiff)
        self.index += 1
        return playerGrid, target1, target2, obstacles, decisionSteps, targetDiff


def calTargetPosByTargetDiff(targetDiff, crossPoint, targetDisToCrossPointList):

    targetDiff = random.choice([targetDiff, -targetDiff])
    if targetDiff == 0:
        targetDisToCrossPoint1 = targetDisToCrossPoint2 = random.choice(targetDisToCrossPointList)
    if targetDiff == -1:
        targetDisToCrossPoints = [targetDisToCrossPointList[:2], targetDisToCrossPointList[1:]]
        targetDisToCrossPoint1, targetDisToCrossPoint2 = random.choice(targetDisToCrossPoints)
    if targetDiff == 1:
        targetDisToCrossPointListReveser = targetDisToCrossPointList[::-1]
        targetDisToCrossPoints = [targetDisToCrossPointList[:2], targetDisToCrossPointList[1:]]
        targetDisToCrossPoint1, targetDisToCrossPoint2 = random.choice(targetDisToCrossPoints)
    if targetDiff == -2:
        targetDisToCrossPoint1 = targetDisToCrossPointList[0]
        targetDisToCrossPoint2 = targetDisToCrossPointList[-1]
    if targetDiff == 2:
        targetDisToCrossPoint1 = targetDisToCrossPointList[-1]
        targetDisToCrossPoint2 = targetDisToCrossPointList[0]

    target1Pos = (crossPoint[0] + targetDisToCrossPoint1, crossPoint[1])
    target2Pos = (crossPoint[0], crossPoint[1] + targetDisToCrossPoint2)

    return target1Pos, target2Pos


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateInsideArea(playerGrid, target1, target2):
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    insideArea = list(set(rect1).union(set(rect2)))
    return insideArea


class CreatMap():
    def __init__(self, rotateAngles, dimension, rotatePoint, numOfObstacles):
        self.rotateAngles = rotateAngles
        self.dimension = dimension
        self.rotatePoint = rotatePoint
        self.numOfObstacles = numOfObstacles

    def __call__(self, condition, targetDiff):

        if condition.name == 'randomCondition':
            # playerGrid, target1, target2, obstacles = condition.creatMap()
            avoidCommitPoint = crossPoint = (0, 0)
            playerGrid = (random.randint(1, max(1, self.dimension - 2)), random.randint(1, max(1, self.dimension - 2)))
            allGrids = tuple(it.product(range(self.dimension), range(self.dimension)))
            possibleGrids = list(filter(lambda x: calculateGridDis(playerGrid, x) >= condition.minSteps and calculateGridDis(playerGrid, x) <= condition.maxsteps, allGrids))
            target1 = random.choice(possibleGrids)

            possibleGrids2 = []
            while possibleGrids2 == []:
                possibleGrids2 = list(filter(lambda x: calculateGridDis(playerGrid, x) - calculateGridDis(playerGrid, target1) == random.choice(condition.controlDiffList), possibleGrids))
                possibleGrids2 = list(filter(lambda x: calculateGridDis(target1, x) > condition.minDistanceBetweenTargets, possibleGrids2))
            target2 = random.choice(possibleGrids2)
            possibleObsGrids = list(filter(lambda x: x not in [playerGrid, target1, target2], allGrids))
            obstacles = random.sample(possibleObsGrids, self.numOfObstacles)
        else:

            if isinstance(targetDiff, int):
                # random init player
                # initBase = condition.initAgent
                # maxBound = max(1, self.dimension - max(condition.targetDisToCrossPoint) - 1 - max(condition.crossPoint) - 1)
                # playerGrid = (random.randint(1, maxBound), random.randint(1, maxBound))

                # transformBase = (playerGrid[0] - initBase[0], playerGrid[1] - initBase[1])
                # obstaclesBase = condition.obstacles
                # obstacles = [tuple(map(sum, zip(obstacle, transformBase))) for obstacle in obstaclesBase]
                # avoidCommitPoint, crossPoint = [tuple(map(sum, zip(point, transformBase))) for point in [condition.avoidCommitPoint, condition.crossPoint]]

                playerGrid = condition.initAgent
                avoidCommitPoint, crossPoint = [condition.avoidCommitPoint, condition.crossPoint]
                target1, target2 = calTargetPosByTargetDiff(targetDiff, crossPoint, condition.targetDisToCrossPoint)
                fixedObstacles = condition.fixedObstacles
                allGrids = tuple(it.product(range(self.dimension - 1), range(self.dimension - 1)))
                insideArea = calculateInsideArea(playerGrid, target1, target2)
                possibleObsGrids = list(filter(lambda x: x not in insideArea, allGrids))
                addObstacles = random.sample(possibleObsGrids, self.numOfObstacles - len(fixedObstacles))
                obstacles = fixedObstacles + addObstacles
            else:
                # need fix: avoid middle area control maps
                playerGrid = crossPoint = avoidCommitPoint = condition.initAgent
                targetDisToCrossPoint = [10, 11, 12]
                targetDiff = random.choice([0, 1, 2])
                target1, target2 = calTargetPosByTargetDiff(targetDiff, crossPoint, targetDisToCrossPoint)
                fixedObstacles = condition.fixedObstacles
                allGrids = tuple(it.product(range(self.dimension - 1), range(self.dimension - 1)))

                target1S, target2S = calTargetPosByTargetDiff(0, condition.crossPoint, condition.targetDisToCrossPoint)
                insideArea = calculateInsideArea(playerGrid, target1S, target2S)
                possibleObsGrids = list(filter(lambda x: x not in insideArea + [playerGrid, target1, target2], allGrids))
                addObstacles = random.sample(possibleObsGrids, self.numOfObstacles - len(fixedObstacles))
                obstacles = fixedObstacles + addObstacles

        angle = random.choice(self.rotateAngles)
        playerGrid, target1, target2, avoidCommitPoint, crossPoint = [self.rotatePoint(point, angle) for point in [playerGrid, target1, target2, avoidCommitPoint, crossPoint]]
        obstacles = [self.rotatePoint(point, angle) for point in obstacles]

        return playerGrid, target1, target2, obstacles, avoidCommitPoint, crossPoint


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


class IsInBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        if position[0] > self.xMax or position[0] < self.xMin or position[1] > self.yMax or position[1] < self.yMin:
            return False
        else:
            return True


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
    targetDiffs = [0, 0, 1, 2]
    rotateAngles = [0, 90, 180, 270]

    expDesignValues = [[b, h, d, m, diff] for b in width for h in height for d in intentionDis for m in decisionSteps for diff in targetDiffs] * 3

    print(len(expDesignValues))

    obstaclesMap1 = [[(2, 2), (2, 4), (3, 5), (3, 6), (4, 2), (5, 3), (6, 3)],
                     [(2, 2), (2, 4), (2, 5), (3, 6), (4, 2), (5, 2), (6, 3)],
                     [(2, 2), (2, 4), (3, 5), (2, 6), (4, 2), (5, 3), (6, 2)]]

    obstaclesMap2 = [[(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)],
                     [(3, 3), (5, 1), (1, 5), (5, 3), (3, 5), (6, 3), (3, 6)],
                     [(3, 3), (3, 1), (1, 3), (5, 3), (3, 5), (6, 3), (3, 6)]]

    obstaclesMap3 = [[(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)],
                     [(4, 4), (5, 1), (4, 2), (6, 4), (4, 6), (1, 5), (2, 4)],
                     [(4, 4), (3, 1), (4, 2), (6, 4), (4, 6), (1, 3), (2, 4)]]

    speicalObstacleMap = [[(4, 1), (4, 2), (6, 3), (6, 4), (1, 4), (2, 4), (3, 6), (4, 6)],
                          [(5, 1), (4, 2), (6, 3), (6, 4), (1, 5), (2, 4), (3, 6), (4, 6)],
                          [(3, 1), (4, 2), (6, 3), (6, 4), (1, 3), (2, 4), (3, 6), (4, 6)]]

    # obstaclesMap4 =
    speicalObstacleMap = [(4, 1), (4, 2), (6, 3), (6, 4), (1, 4), (2, 4), (3, 6), (4, 6)]
    obstaclesCondition = [obstaclesMap1, obstaclesMap2, obstaclesMap3, speicalObstacleMap]
    obstaclesMaps = dict(zip(decisionSteps, obstaclesCondition))

    gridSize = 15
    rotatePoint = RotatePoint(dimension)
    isInBoundary = IsInBoundary([0, gridSize - 1], [0, gridSize - 1])
    creatRectMap = CreatRectMap(rotateAngles, gridSize, obstaclesMaps, rotatePoint)
    creatLineMap = CreatLineMap(rotateAngles, gridSize, rotatePoint, isInBoundary)
    samplePositionFromCondition = SamplePositionFromCondition(creatRectMap, creatLineMap, expDesignValues)


# sample
    from collections import namedtuple
    condition = namedtuple('condition', 'name decisionSteps')
    expCondition = condition(name='expCondition', decisionSteps=decisionSteps[: -1])
    for _ in range(10):
        playerGrid, target1, target2, obstacles, decisionSteps, targetDiff = samplePositionFromCondition(expCondition)

        # obstacles = creatObstacles(pointList)
        print(decisionSteps, targetDiff)
        # playerGrid = (1, 1)
        # obstacles = [(2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2)]
        # obstacles = [(3, 3), (4, 1), (1, 4), (5, 3), (3, 5), (6, 3), (3, 6)]
        # obstacles = [(4, 4), (4, 1), (4, 2), (6, 4), (4, 6), (1, 4), (2, 4)]
        # obstacles = [(2, 2), (2, 4), (2, 5), (3, 6), (4, 2), (5, 2), (6, 3)]

        # angle = random.choice(rotateAngles)
        # playerGrid, target1, target2 = [rotatePoint(point, angle) for point in [playerGrid, target1, target2]]
        # obstacles = [rotatePoint(point, angle) for point in obstacles]

        pause = True
        while pause:
            screen = drawNewState(target1, target2, playerGrid, obstacles)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    pause = False
                if event.type == pygame.QUIT:
                    pygame.quit()
    pygame.quit()
