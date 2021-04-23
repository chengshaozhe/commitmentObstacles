import random
import numpy as np
from collections import defaultdict


def initQtable(actions):
    qTable = defaultdict(lambda: [0.0] * len(actions))
    return qTable


class UpdateQTable:
    def __init__(self, discountFactor, learningRate):
        self.discountFactor = discountFactor
        self.learningRate = learningRate

    def __call__(self, qTable, state, action, reward, nextState):
        currentQ = qTable[state][action]
        newQ = reward + self.discountFactor * max(qTable[nextState])
        qTable[state][action] += self.learningRate * (newQ - currentQ)
        return qTable


class GetAction:
    def __init__(self, epsilon, actions, sampleByValue):
        self.epsilon = epsilon
        self.actions = actions
        self.sampleByValue = sampleByValue

    def __call__(self, qTable, state):
        if np.random.rand() < self.epsilon:
            actionIndex = np.random.choice(range(len(self.actions)))
            action = self.actions[actionIndex]
        else:
            stateAction = qTable[state]
            actionIndex = self.sampleByValue(stateAction)
            action = self.actions[actionIndex]
        return actionIndex


def argMax(stateAction):
    maxIndexList = []
    maxValue = stateAction[0]
    for index, value in enumerate(stateAction):
        if value > maxValue:
            maxIndexList.clear()
            maxValue = value
            maxIndexList.append(index)
        elif value == maxValue:
            maxIndexList.append(index)
    return random.choice(maxIndexList)
