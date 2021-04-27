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


class Options:
    def __init__(self, name, policy, terminals):
        self.name = name
        self.policy = policy
        self.terminals = terminals


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
        terminalProb = 0.1
    return terminalProb


class OptionValueIteration():
    def __init__(self, calTerminalProb, gamma, epsilon=0.001, max_iter=100, terminals=[], obstacles=[]):
        self.calTerminalProb = calTerminalProb
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.terminals = terminals
        self.obstacles = obstacles
        self.calTerminalProb = calTerminalProb

    def __call__(self, S, O, A, T, R):
        gamma, epsilon, max_iter = self.gamma, self.epsilon, self.max_iter
        excludedState = (set(self.terminals))
        S_iter = tuple(filter(lambda s: s not in self.terminals, S))

        Q_init = {s: {o: 0.1 for o in O} for s in S_iter}
        Qterminals = {s: {o: 0 for o in O} for s in self.terminals}
        Q_init.update(Qterminals)

        delta = 0
        for i in range(max_iter):
            Q = Q_init.copy()
            for s in S_iter:
                for o in O:
                    optionValues = [o.policy[s][a] * sum([R[s][a][sn] + gamma * p * (1 - self.calTerminalProb(sn, o)) for (sn, p) in T[s][a].items()]) for a in A]
                    Q[s][o] = max(optionValues)

            delta = np.array([abs(Q[s][o] - Q_init[s][o]) for s in S_iter])
            if np.max(delta) < self.epsilon:
                break
        return Q


class RunOptionVI:
    def __init__(self, gridSize, optionSpace, actionSpace, noiseSpace, noise, gamma, goalReward):
        self.gridSize = gridSize
        self.optionSpace = optionSpace
        self.actionSpace = actionSpace
        self.noiseSpace = noiseSpace
        self.noise = noise
        self.gamma = gamma
        self.goalReward = goalReward

    def __call__(self, goalStates, obstacles):
        gridSize, O, A, noiseSpace, noise, gamma, goalReward = self.gridSize, self.optionSpace, self.actionSpace, self.noiseSpace, self.noise, self.gamma, self.goalReward

        env = GridWorld("test", nx=gridSize, ny=gridSize)

        obstacles = list(obstacles)  # list of tuples
        if isinstance(goalStates, tuple):
            goalStates = [goalStates]
        else:
            goalStates = list(goalStates)

        terminalValue = {s: goalReward for s in goalStates}

        env.add_feature_map("goal", terminalValue, default=0)
        env.add_terminals(goalStates)
        env.add_obstacles(obstacles)

        S = tuple(it.product(range(env.nx), range(env.ny)))

        excludedStates = set(obstacles)
        S = tuple(filter(lambda s: s not in excludedStates, S))

        transition_function = StochasticTransition(noise, noiseSpace, goalStates, env.is_state_valid)

        T = {s: {a: transition_function(s, a) for a in A} for s in S}
        T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S]
                             for a in A] for s in S])

        stepCost = - self.goalReward / (self.gridSize * 2)
        reward_func = ft.partial(
            grid_reward, env=env, const=stepCost, terminals=goalStates)

        R = {s: {a: {sn: reward_func(s, a, sn) for sn in S} for a in A} for s in S}
        R_arr = np.asarray([[[R[s][a].get(s_n, 0) for s_n in S]
                             for a in A] for s in S])

        terminalProbConst = 0.1
        calTerminalProb = CalTerminalProb(terminalProbConst)
        valueIteration = OptionValueIteration(calTerminalProb, gamma, epsilon=0.001, max_iter=100, terminals=goalStates, obstacles=obstacles)
        Q = valueIteration(S, O, A, T, R)
        return Q


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
    rewardFunction = RewardFunction(goalReward, stepCost, terminals)

    # isTerminal = IsTerminal(terminals)

    # valueIteration = ValueIteration(gamma, epsilon=0.001, max_iter=100, terminals=goalStates, obstacles=obstacles)
    # V = valueIteration(S, A, T, R)
    # V.update(terminalValue)

    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # draw_policy_4d_softmax(ax, policy, V=VDict, S=S, A=actionSpace)
    # plt.show()
    gamma = 0.9
    runVI = RunVI(gridSize[0], actionSpace, noiseActionSpace, noise, gamma, goalReward)
    softmaxBeta = 3
    getGetGoalPolices = GetGoalPolices(runVI, softmaxBeta)

    targetA, targetB = terminals
    policyA, policyB = getGetGoalPolices(targetA, targetB, obstacles)
    optionA = Options('a', policyA, targetA)
    optionB = Options('b', policyB, targetB)

    optionSpace = [optionA, optionB]
    runOptionVI = RunOptionVI(gridSize[0], optionSpace, actionSpace, noiseActionSpace, noise, gamma, goalReward)
    Q = runOptionVI(terminals, obstacles)
    state = (4, 9)
    # Q[state].name
    print(Q[state][optionA])
    print(Q[state][optionB])
