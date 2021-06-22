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


def chooseMaxAcion(actionValues):
    actionMaxList = [index for index, actionValue in enumerate(actionValues) if
                     actionValue == np.max(actionValues)]
    action = random.choice(actionMaxList)
    return action


def calculateSoftmaxProbability(acionValues, beta):
    exponents = np.multiply(beta, acionValues)
    exponents = np.array([min(700, exponent) for exponent in exponents])
    newProbabilityList = list(np.divide(np.exp(exponents), np.sum(np.exp(exponents))))
    return newProbabilityList


def chooseSoftMaxAction(actionValues, softmaxBeta=3):
    softmaxProbabilityList = calculateSoftmaxProbability(actionValues, softmaxBeta)
    action = list(np.random.multinomial(1, softmaxProbabilityList)).index(1)
    return action


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
