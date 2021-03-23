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
sys.setrecursionlimit(2**30)
import pandas as pd
import seaborn as sns

import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from machinePolicy.reward import *


class GridWorld():
    def __init__(self, name='', nx=None, ny=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.coordinates = tuple(it.product(range(self.nx), range(self.ny)))
        self.terminals = []
        self.obstacles = []
        self.features = co.OrderedDict()

    def add_terminals(self, terminals=[]):
        for t in terminals:
            self.terminals.append(t)

    def add_obstacles(self, obstacles=[]):
        for o in obstacles:
            self.obstacles.append(o)

    def add_feature_map(self, name, state_values, default=0):
        self.features[name] = {s: default for s in self.coordinates}
        self.features[name].update(state_values)

    def is_state_valid(self, state):
        if state[0] not in range(self.nx):
            return False
        if state[1] not in range(self.ny):
            return False
        if state in self.obstacles:
            return False
        return True

    def reward(self, s, a, s_n, W={}):
        if not W:
            return sum(map(lambda f: self.features[f][s_n], self.features))
        return sum(map(lambda f: self.features[f][s_n] * W[f], W.keys()))

    def draw(self, ax=None, ax_images={}, features=[], colors={},
             masked_values={}, default_masked=0, show=True):

        new_features = [f for f in features if f not in ax_images.keys()]
        old_features = [f for f in features if f in ax_images.keys()]
        ax, new_ax_images = self.draw_features_first_time(ax, new_features,
                                                          colors, masked_values, default_masked=0)
        old_ax_images = self.update_features_images(ax_images, old_features,
                                                    masked_values,
                                                    default_masked=0)
        ax_images.update(old_ax_images)
        ax_images.update(new_ax_images)

        return ax, ax_images


def T_dict(S=(), A=(), tran_func=None):
    return {s: {a: tran_func(s, a) for a in A} for s in S}


def R_dict(S=(), A=(), T={}, reward_func=None):
    return {s: {a: {s_n: reward_func(s, a, s_n) for s_n in T[s][a]} for a in A} for s in S}


def grid_transition(s, a, is_valid=None, terminals=()):
    if s in terminals:
        return {s: 1}
    s_n = tuple(map(sum, zip(s, a)))
    if is_valid(s_n):
        return {s_n: 1}
    return {s: 1}


def grid_transition_stochastic(s=(), a=(), noiseSpace=[], is_valid=None, terminals=(), mode=0.9):
    if s in terminals:
        return {s: 1}

    def apply_action(s, noise):
        return (s[0] + noise[0], s[1] + noise[1])

    s_n = (s[0] + a[0], s[1] + a[1])
    if not is_valid(s_n):
        return {s: 1}

    sn_iter = (apply_action(s, noise) for noise in noiseSpace)
    states = list(filter(is_valid, sn_iter))

    p_n = (1.0 - mode) / len(states)

    next_state_prob = {s: p_n for s in states}
    next_state_prob.update({s_n: mode})

    return next_state_prob


def grid_transition_noise(s=(), a=(), A=(), is_valid=None, terminals=(), noise=0.1):
    if s in terminals:
        return {s: 1}

    def apply_action(s, a):
        return (s[0] + a[0], s[1] + a[1])

    s_n = apply_action(s, a)
    if not is_valid(s_n):
        return {s: 1}

    noise_action = [i for i in A if i != a]
    sn_iter = (apply_action(s, noise) for noise in noise_action)
    states = list(filter(is_valid, sn_iter))

    p_n = noise / len(states)
    next_state_prob = {s: p_n for s in states}
    next_state_prob.update({s_n: 1 - noise})

    return next_state_prob


def grid_reward(s, a, sn, env=None, const=-1, terminals=None):
    if sn in terminals:
        return const + sum(map(lambda f: env.features[f][sn], env.features))
    else:
        return const + sum(map(lambda f: env.features[f][s], env.features))


class ValueIteration():
    def __init__(self, gamma, epsilon=0.001, max_iter=100, terminals=(), obstacles=()):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.terminals = terminals
        self.obstacles = obstacles

    def __call__(self, S, A, T, R):
        gamma, epsilon, max_iter = self.gamma, self.epsilon, self.max_iter
        excludedState = (set(self.terminals) | set(self.obstacles))
        S_iter = tuple(filter(lambda s: s not in excludedState, S))

        V_init = {s: 1 for s in S_iter}
        Vterminals = {s: 0 for s in excludedState}

        V_init.update(Vterminals)
        delta = 0
        for i in range(max_iter):
            V = V_init.copy()
            for s in S_iter:
                V_init[s] = max([sum([p * (R[s][a][s_n] + gamma * V[s_n]) for (s_n, p) in T[s][a].items()]) for a in A])
            delta = np.array([abs(V[s] - V_init[s]) for s in S_iter])
            # if np.all(delta < epsilon * (1 - gamma) / gamma):
            if np.all(delta < self.epsilon):
                # print(i)
                break
        return V


def dict_to_array(V):
    states, values = zip(*((s, v) for (s, v) in V.items()))
    row_index, col_index = zip(*states)
    num_row = max(row_index) + 1
    num_col = max(col_index) + 1
    I = np.empty((num_row, num_col))
    I[row_index, col_index] = values
    return I


def V_dict_to_array(V, S):
    V_lst = [V.get(s) for s in S]
    V_arr = np.asarray(V_lst)
    return V_arr


def T_dict_to_array(T):
    T_lst = [[[T[s][a].get(s_n, 0) for s_n in S] for a in A] for s in S]
    T_arr = np.asarray(T_lst)
    return T_arr


def R_dict_to_array(R):
    R_lst = [[[R[s][a].get(s_n, 0) for s_n in S] for a in A] for s in S]
    R_arr = np.asarray(R_lst, dtype=float)
    return R_arr


def V_to_Q(V, T=None, R=None, gamma=None):
    V_aug = V[np.newaxis, np.newaxis, :]
    return np.sum(T * (R + gamma * V_aug), axis=2)


def Q_from_V(s, a, T=None, R=None, V=None, gamma=None):
    return sum([p * (R[s][a] + gamma * V[s_n])
                for (s_n, p) in T[s][a].iteritems()])


def softmax_epislon_policy(Q, temperature=10, epsilon=0.1):
    na = Q.shape[-1]
    q_exp = np.exp(Q / temperature)
    norm = np.sum(q_exp, axis=1)
    prob = (q_exp / norm[:, np.newaxis]) * (1 - epsilon) + epsilon / na
    return prob


def pickle_dump_single_result(dirc="", prefix="result", name="", data=None):
    full_name = "_".join((prefix, name)) + ".pkl"
    path = os.path.join(dirc, full_name)
    pickle.dump(data, open(path, "wb"))
    print ("saving %s at %s" % (name, path))


class RunVI:
    def __init__(self, gridSize, actionSpace, noiseSpace, noise, gamma, goalReward):
        self.gridSize = gridSize
        self.actionSpace = actionSpace
        self.noiseSpace = noiseSpace
        self.noise = noise
        self.gamma = gamma
        self.goalReward = goalReward

    def __call__(self, goalStates, obstacles):
        gridSize, A, noiseSpace, noise, gamma, goalReward = self.gridSize, self.actionSpace, self.noiseSpace, self.noise, self.gamma, self.goalReward

        env = GridWorld("test", nx=gridSize, ny=gridSize)
        # terminalValue = {s: goalReward for s, goalReward in zip([goalStates], [goalReward] * len([goalStates]))}

        terminalValue = {s: goalReward for s, goalReward in zip([goalStates], [self.goalReward])}
        if isinstance(goalStates[0], int):
            terminalValue = {s: goalReward for s, goalReward in zip([goalStates], self.goalReward)}

        env.add_feature_map("goal", terminalValue, default=0)
        env.add_terminals(list(goalStates))
        env.add_obstacles(list(obstacles))

        S = tuple(it.product(range(env.nx), range(env.ny)))

        mode = 1 - noise
        transition_function = ft.partial(grid_transition_stochastic, noiseSpace=noiseSpace, terminals=goalStates, is_valid=env.is_state_valid, mode=mode)

        T = {s: {a: transition_function(s, a) for a in A} for s in S}
        T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S]
                             for a in A] for s in S])

        stepCost = - self.goalReward[0] / (self.gridSize * 2)
        reward_func = ft.partial(
            grid_reward, env=env, const=stepCost, terminals=goalStates)

        R = {s: {a: {sn: reward_func(s, a, sn) for sn in S} for a in A} for s in S}
        R_arr = np.asarray([[[R[s][a].get(s_n, 0) for s_n in S]
                             for a in A] for s in S])

        valueIteration = ValueIteration(gamma, epsilon=0.001, max_iter=100, terminals=goalStates)
        V = valueIteration(S, A, T, R)

        V_arr = V_dict_to_array(V, S)
        Q = V_to_Q(V=V_arr, T=T_arr, R=R_arr, gamma=gamma)
        Q_dict = {(s, goalStates): {a: Q[si, ai] for (ai, a) in enumerate(A)} for (si, s) in enumerate(S)}

        Q_dictDefault = {s: {a: Q[si, ai] for (ai, a) in enumerate(A)} for (si, s) in enumerate(S)}

        return S, A, T, R, gamma, Q_dict, Q_dictDefault


def calculateSoftmaxProbability(acionValues, beta):
    exponents = np.multiply(beta, acionValues)
    exponents = np.array([min(700, exponent) for exponent in exponents])
    newProbabilityList = list(np.divide(np.exp(exponents), np.sum(np.exp(exponents))))
    return newProbabilityList


class SoftmaxGoalPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target):
        actionDict = self.Q_dict[(playerGrid, target)]
        actions = list(actionDict.keys())
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        # softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        softMaxActionDict = {action: prob for action, prob in zip(actions, softmaxProbabilityList)}
        return softMaxActionDict


class GetLikelihoodRewardFunction:
    def __init__(self, transitionTable, goalPolicies, commitmentRatio):
        self.transitionTable = transitionTable
        self.goalPolicies = goalPolicies
        self.commitmentRatio = commitmentRatio

    def __call__(self, trueGoal, originalReward):
        likelihoodRewardFunction = self.createLikelihoodReward(trueGoal)
        newReward = self.mergeRewards(originalReward, likelihoodRewardFunction)
        return(newReward)

    def createLikelihoodReward(self, trueGoal):
        rewardFunction = {state: {action: {nextState: self.getLikelihoodRatio(state, nextState, trueGoal) for nextState in self.transitionTable[state][action].keys()}
                                  for action in self.transitionTable[state].keys()}
                          for state in self.transitionTable.keys()}
        return(rewardFunction)

    def mergeRewards(self, reward1, reward2):
        mergedReward = {state: {action: {nextState: reward1[state][action][nextState] + reward2[state][action][nextState] for nextState in reward2[state][action].keys()}
                                for action in reward2[state].keys()}
                        for state in reward2.keys()}
        return(mergedReward)

    def getLikelihoodRatio(self, state, nextState, goalTrue):
        goalLikelihood = self.getNextStateProbability(state, nextState, goalTrue)
        notGoalLikelihood = sum([self.getNextStateProbability(state, nextState, g)
                                 for g in self.goalPolicies.keys()])

        likelihoodRatio = self.commitmentRatio * goalLikelihood / notGoalLikelihood
        return(likelihoodRatio)

    def getNextStateProbability(self, state, nextState, goal):
        possibleActionsToNextState = [action for action in self.transitionTable[state]
                                      if nextState in self.transitionTable[state][action]]
        probNextState = sum([self.transitionTable[state][action][nextState] * self.goalPolicies[goal][state][action]
                             for action in possibleActionsToNextState])
        return(probNextState)


class RunIntentionModel:
    def __init__(self, runVI, softmaxBeta, intentionInfoScale):
        self.softmaxBeta = softmaxBeta
        self.runVI = runVI
        self.intentionInfoScale = intentionInfoScale

    def __call__(self, targetA, targetB, obstacles):
        _, _, transitionRL, rewardRL, _, _, RLDict = self.runVI(tuple((targetA, targetB)), obstacles)
        S, A, transitionTableA, rewardA, gamma, Q_dictA, _ = self.runVI(targetA, obstacles)
        S, A, transitionTableB, rewardB, gamma, Q_dictB, _ = self.runVI(targetB, obstacles)
        getPolicyA = SoftmaxGoalPolicy(Q_dictA, self.softmaxBeta)
        getPolicyB = SoftmaxGoalPolicy(Q_dictB, self.softmaxBeta)
        policyA = {state: getPolicyA(state, targetA) for state in transitionTableA.keys()}
        policyB = {state: getPolicyB(state, targetB) for state in transitionTableB.keys()}

        goalPoliciesDict = {'a': policyA, 'b': policyB}
        Q_dictList = []
        for intentionInfoScale in self.intentionInfoScale:
            runValueIterationA = ValueIteration(gamma, epsilon=0.001, max_iter=100, terminals=targetA, obstacles=obstacles)
            runValueIterationB = ValueIteration(gamma, epsilon=0.001, max_iter=100, terminals=targetB, obstacles=obstacles)

            getLikelihoodRewardFunctionA = GetLikelihoodRewardFunction(transitionTableA, goalPoliciesDict, intentionInfoScale)
            getLikelihoodRewardFunctionB = GetLikelihoodRewardFunction(transitionTableB, goalPoliciesDict, intentionInfoScale)

            infoRewardA = getLikelihoodRewardFunctionA('a', rewardA)
            infoRewardB = getLikelihoodRewardFunctionB('b', rewardB)

            V_A = runValueIterationA(S, A, transitionTableA, infoRewardA)
            V_B = runValueIterationB(S, A, transitionTableB, infoRewardB)

            print(rewardRL[(3, 6)][(0, 1)][(3, 7)], rewardA[(3, 6)][(0, 1)][(3, 7)], infoRewardA[(3, 6)][(0, 1)][(3, 7)], infoRewardB[(3, 6)][(0, 1)][(3, 7)])
            # runValueIteration = ValueIteration(gamma, epsilon=0.001, max_iter=100, terminals=tuple((targetA, targetB)), obstacles=obstacles)
            # getLikelihoodRewardFunction = GetLikelihoodRewardFunction(transitionRL, goalPoliciesDict, intentionInfoScale)
            # infoRewardA = getLikelihoodRewardFunction('a', rewardRL)
            # infoRewardB = getLikelihoodRewardFunction('b', rewardRL)

            # V_A = runValueIteration(S, A, transitionRL, infoRewardA)
            # V_B = runValueIteration(S, A, transitionRL, infoRewardB)

            visualMap = 0
            if visualMap:
                mapValue = 'V_A'
                heatMapValue = eval(mapValue)
                y = dict_to_array(heatMapValue)
                # y = np.round(y)  # round value
                y = y.reshape((gridSize, gridSize))
                df = pd.DataFrame(y, columns=[x for x in range(gridSize)])
                sns.heatmap(df, annot=True, fmt='.3f')
                # plt.title('{} for goal at {} noise={} goalReward={}'.format(mapValue, goalStates, self.noise, self.goalReward))
                plt.show()

            for V, R, T in zip([V_A, V_B], [infoRewardA, infoRewardB], [transitionTableA, transitionTableB]):
                V_arr = V_dict_to_array(V, S)
                T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S]
                                     for a in A] for s in S])
                R_arr = np.asarray([[[R[s][a].get(s_n, 0) for s_n in S]
                                     for a in A] for s in S])

                Q = V_to_Q(V=V_arr, T=T_arr, R=R_arr, gamma=gamma)
                Q_dict = {s: {a: Q[si, ai] for (ai, a) in enumerate(A)} for (si, s) in enumerate(S)}
                Q_dictList.append(Q_dict)

        avoidCommitQDicts = {targetA: Q_dictList[0], targetB: Q_dictList[1]}
        commitQDicts = {targetA: Q_dictList[2], targetB: Q_dictList[3]}
        return [RLDict, avoidCommitQDicts, commitQDicts]


if __name__ == '__main__':
    gridSize = 15
    noise = 0.067
    gamma = 0.9
    goalReward = [30, 30]
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)

    softmaxBeta = 2.5
    intentionInfoScale = [0.2]
    runModel = RunIntentionModel(runVI, softmaxBeta, intentionInfoScale)

    goalStates = ((4, 9), (9, 4))
    obstaclesMap1 = [[(2, 2), (2, 4), (2, 5), (4, 2), (5, 2), (0, 3), (0, 4), (0, 5), (3, 0), (4, 0), (5, 0)]]
    import random
    obstacles = random.choice(obstaclesMap1)
    target1, target2 = goalStates
    policies = runModel(target1, target2, obstacles)
