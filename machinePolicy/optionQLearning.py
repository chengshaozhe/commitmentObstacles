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
from showIntentionModel import GetGoalPolices, RunVI, GridWorld, grid_reward


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


class OptionTransition:
    def __init__(self, actionTransition, actionSpace, terminalProb):
        self.actionTransition = actionTransition
        self.actionSpace = actionSpace
        self.terminalProb = terminalProb

    def __call__(self, state, option):
        actionProb = [option.policy[state][action] for action in self.actionSpace]
        # todo
        nextStateProb = [self.actionTransition(state, action) for action in self.actionSpace]

        return nextStateProb


class OptionTransitionStep:
    def __init__(self, gamma, step):
        self.gamma = gamma
        self.step = step
        self.options = options

    # todo
    def __call__(self, state, nextState):
        terminalProb = calTerminalProb(nextState, step)
        p = np.prod([terminalProb * self.gamma ** step in range(self.step)])
        return {nextState: p}


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


class OptionRewardFunction():
    def __init__(self, goalReward, stepCost):
        self.goalReward = goalReward
        self.stepCost = stepCost

    def __call__(self, state, option, nextState):
        reward = self.stepCost
        if nextState in option.terminals:
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


# def QLearning(episodes):
#     qTable = initQtable(actionSpace)
#     for episode in range(episodes):
#         state = reset()
#         for step in range(maxRunningSteps):
#             actionIndex = getAction(qTable, state)
#             action = actionSpace[actionIndex]
#             nextStatesDict = stochasticTransition(state, action)
#             nextState = sampleFromStateProbDict(nextStatesDict)
#             reward = rewardFunction(state, action, nextState)
#             done = isTerminal(nextState)

#             qTable = updateQTable(qTable, state, actionIndex, reward, nextState)
#             state = nextState
#             if done:
#                 break
#     return qTable


class Options:
    def __init__(self, name, policy, terminals):
        self.name = name
        self.policy = policy
        self.terminals = terminals

    def isTerminal(self, state):
        if state in self.terminals:
            return True
        else:
            return False


class CalTerminalProb:
    def __init__(self, terminalProbConst):
        self.terminalProbConst = terminalProbConst

    def __call__(self, state, options):
        if state in options.terminals:
            terminalProb = 1
        else:
            terminalProb = self.terminalProbConst
        return terminalProb


def calTerminalProb(state, options):
    if state in options.terminals:
        terminalProb = 1
    else:
        terminalProb = 0.05
    return terminalProb


def OptionQLearning(episodes):
    qTable = initQtable(optionSpace)
    for episode in range(episodes):
        state = reset()
        for step in range(maxRunningSteps):
            optionIndex = getAction(qTable, state)
            option = optionSpace[optionIndex]
            nextState, done, reward = runOption(state, option)

            qTable = updateQTable(qTable, state, optionIndex, reward, nextState)
            state = nextState
            if done:
                break
    return qTable


def runOption(state, option, gamma=0.9):
    done = False
    totalReward = 0
    totalSteps = 0

    terminated = False

    while not terminated:
        action = chooseSoftMaxActionDict(option.policy[state])
        terminated = option.isTerminal(state)
        nextStatesDict = stochasticTransition(state, action)
        nextState = sampleFromStateProbDict(nextStatesDict)
        reward = rewardFunction(state, action, nextState)
        done = isTerminal(nextState)
        totalReward += reward * (gamma ** totalSteps)

        totalSteps += 1
        if done:
            break

        state = nextState
    return nextState, done, totalReward


def chooseSoftMaxActionDict(actionDict, softmaxBeta=3):
    actionValue = list(actionDict.values())
    softmaxProbabilityList = calculateSoftmaxProbability(actionValue, softmaxBeta)
    action = list(actionDict.keys())[
        list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
    return action


if __name__ == '__main__':
    gridSize = [15, 15]
    obstacles = [(3, 3), (4, 0), (3, 1), (3, 5), (5, 3), (1, 3), (0, 4)]
    isStateValid = IsStateValid(gridSize, obstacles)

    noise = 0.067
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    terminals = [(5, 9), (9, 5)]
    isTerminal = IsTerminal(terminals)

    stochasticTransition = StochasticTransition(noise, noiseActionSpace, terminals, isStateValid)

    goalReward = 30
    stepCost = - goalReward / (gridSize[0] * 2)
    rewardFunction = RewardFunction(goalReward, stepCost, terminals)

    reset = Reset(gridSize, isStateValid)

    gamma = 0.9
    runVI = RunVI(gridSize[0], actionSpace, noiseActionSpace, noise, gamma, goalReward)

    softmaxBeta = 3
    getGetGoalPolices = GetGoalPolices(runVI, softmaxBeta)

    targetA, targetB = terminals
    policyA, policyB = getGetGoalPolices(targetA, targetB, obstacles)
    optionA = Options('a', policyA, targetA)
    optionB = Options('b', policyB, targetB)
    optionSpace = [optionA, optionB]
##
# q-learing
    discountFactor = 0.9
    learningRate = 0.5
    updateQTable = UpdateQTable(discountFactor, learningRate)
    epsilon = 0.1
    getAction = GetAction(epsilon, optionSpace, chooseSoftMaxAction)

    episodes = 30000
    maxRunningSteps = 100

    QDict = OptionQLearning(episodes)

    # state = (5, 6)
    state = (5, 5)
    state = (4, 2)

    # Q[state].name
    print(QDict[state])

    # print(QDict[state][optionA])
    # print(QDict[state][optionB])
    VDict = {state: np.max(values) for state, values in QDict.items()}

    y = dict_to_array(VDict)
    # y = np.round(y)  # round value
    y = y.reshape((gridSize[0], gridSize[1]))
    df = pd.DataFrame(y, columns=[x for x in range(gridSize[0])])
    sns.heatmap(df, annot=True, fmt='.3f')

    # softmaxBeta = 3
    # getPolicy = SoftmaxPolicy(softmaxBeta, QDict, optionSpace)

    # S = tuple(it.product(range(gridSize[0]), range(gridSize[1])))
    # excludedStates = set(obstacles)
    # S = tuple(filter(lambda s: s not in excludedStates, S))
    # policy = {state: getPolicy(state) for state in S}

    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # draw_policy_4d_softmax(ax, policy, V=VDict, S=S, A=optionSpace)
    plt.show()
