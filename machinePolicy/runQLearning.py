import numpy as np
import collections as co
import functools as ft
import itertools as it
import operator as op
import matplotlib.pyplot as plt
import os
import pickle
import sys
import time
import pandas as pd
import seaborn as sns
from random import randint

import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from machinePolicy.reward import *
from machinePolicy.viz import *
from machinePolicy.qLearning import *


class IsTerminal():
    def __init__(self, terminals):
        self.terminals = terminals

    def __call__(self, state):
        terminal = False
        if state in self.terminals:
            terminal = True
        return terminal


class Reset:
    def __init__(self, gridSize, isStateValid):
        self.gridX, self.gridY = gridSize
        self.isStateValid = isStateValid

    def __call__(self):
        validStart = False
        while not validStart:
            startState = (randint(0, self.gridX), randint(0, self.gridY))
            validStart = self.isStateValid(startState)
        return startState


class IsStateValid:
    def __init__(self, gridSize, obstacles):
        self.gridX, self.gridY = gridSize
        self.obstacles = obstacles

    def __call__(self, state):
        if state[0] not in range(self.gridX):
            return False
        if state[1] not in range(self.gridY):
            return False
        if state in self.obstacles:
            return False
        return True


def transition(state, action):
    nextState = tuple(np.array(state) + np.array(action))
    return nextState


class StochasticTransition:
    def __init__(self, noise, noiseActionSpace, terminals, isStateValid):
        self.noise = noise
        self.noiseActionSpace = noiseActionSpace
        self.terminals = terminals
        self.isStateValid = isStateValid

    def __call__(self, state, action):
        if state in self.terminals:
            return {state: 1}

        nextState = transition(state, action)
        if not self.isStateValid(nextState):
            return {state: 1}

        possibleNextStates = (transition(state, noiseAction) for noiseAction in self.noiseActionSpace)
        validNextStates = list(filter(self.isStateValid, possibleNextStates))

        noiseProbAverged = self.noise / (len(validNextStates) - 1)
        nextStateProb = {s: noiseProbAverged for s in validNextStates}
        nextStateProb.update({nextState: 1.0 - self.noise})

        return nextStateProb


class RewardFunction():
    def __init__(self, goalReward, stepCost, isTerminal):
        self.goalReward = goalReward
        self.stepCost = stepCost
        self.terminals = terminals

    def __call__(self, state, action, nextState):
        reward = self.stepCost
        if nextState in self.terminals:
            reward = self.goalReward
        return reward


def calculateSoftmaxProbability(acionValues, beta):
    exponents = np.multiply(beta, acionValues)
    exponents = np.array([min(700, exponent) for exponent in exponents])
    newProbabilityList = list(np.divide(np.exp(exponents), np.sum(np.exp(exponents))))
    return newProbabilityList


class SoftmaxPolicy:
    def __init__(self, softmaxBeta, QDict, actionSpace):
        self.QDict = QDict
        self.softmaxBeta = softmaxBeta
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionValues = self.QDict[state]
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(self.actionSpace, softmaxProbabilityList))
        return softMaxActionDict


def sampleFromStateProbDict(probDict):
    states = list(probDict.keys())
    probs = list(probDict.values())
    state = states[list(np.random.multinomial(1, probs)).index(1)]
    return state


def QLearning(episodes):
    qTable = initQtable(actionSpace)
    for episode in range(episodes):
        state = reset()
        for step in range(maxRunningSteps):
            actionIndex = getAction(qTable, state)
            action = actionSpace[actionIndex]
            nextStatesDict = stochasticTransition(state, action)
            nextState = sampleFromStateProbDict(nextStatesDict)
            reward = rewardFunction(state, action, nextState)
            done = isTerminal(nextState)

            qTable = updateQTable(qTable, state, actionIndex, reward, nextState)
            state = nextState
            if done:
                break
    return qTable


if __name__ == '__main__':
    gridSize = [15, 15]
    obstacles = [(3, 3), (4, 0), (3, 1), (3, 5), (5, 3), (1, 3), (0, 4)]
    isStateValid = IsStateValid(gridSize, obstacles)

    noise = 0.067
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    terminals = [(5, 9), (9, 5)]

    stochasticTransition = StochasticTransition(noise, noiseActionSpace, terminals, isStateValid)

    goalReward = 30
    stepCost = - goalReward / (gridSize[0] * 2)

    maxRunningSteps = 100
    # stepCost = - 1 / maxRunningSteps

    rewardFunction = RewardFunction(goalReward, stepCost, terminals)

    # state, action = [(3, 4), (0, 1)]
    # ns = stochasticTransition(state, action)
    # print(ns)
    isTerminal = IsTerminal(terminals)

    reset = Reset(gridSize, isStateValid)
# q-learing
    discountFactor = 0.9
    learningRate = 0.5
    updateQTable = UpdateQTable(discountFactor, learningRate)
    epsilon = 0.1
    getAction = GetAction(epsilon, actionSpace, argMax)

    episodes = 50000
    QDict = QLearning(episodes)
    VDict = {state: np.max(values) for state, values in QDict.items()}

    softmaxBeta = 3
    getPolicy = SoftmaxPolicy(softmaxBeta, QDict, actionSpace)

    S = tuple(it.product(range(gridSize[0]), range(gridSize[1])))
    excludedStates = set(obstacles)
    S = tuple(filter(lambda s: s not in excludedStates, S))
    policy = {state: getPolicy(state) for state in S}

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    draw_policy_4d_softmax(ax, policy, V=VDict, S=S, A=actionSpace)
    plt.show()
