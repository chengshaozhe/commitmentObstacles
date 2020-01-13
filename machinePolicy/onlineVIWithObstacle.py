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

# from viz import *
# from reward import *


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
        # S_iter = tuple(filter(lambda s: s not in self.terminals, S))

        V_init = {s: 1 for s in S_iter}
        Vterminals = {s: 0 for s in excludedState}

        V_init.update(Vterminals)
        delta = 0
        for i in range(max_iter):
            V = V_init.copy()
            for s in S_iter:
                V_init[s] = max([sum([p * (R[s][a][s_n] + gamma * V[s_n]) for (s_n, p) in T[s][a].items()]) for a in A])
            delta = np.array([abs(V[s] - V_init[s]) for s in S_iter])
            if np.all(delta < epsilon * (1 - gamma) / gamma):
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


def runVI(sheep_states, obstacles_states, goalRewardList):
    gridSize = 15
    env = GridWorld("test", nx=gridSize, ny=gridSize)

    terminalValue = {s: goalReward for s, goalReward in zip(sheep_states, goalRewardList)}

    env.add_obstacles(list(obstacles_states))
    env.add_feature_map("goal", terminalValue, default=0)
    env.add_terminals(list(sheep_states))

    S = tuple(it.product(range(env.nx), range(env.ny)))

    A = ((1, 0), (0, 1), (-1, 0), (0, -1))
    # noiseSpace = [(0, -2), (0, 2), (-2, 0), (2, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    noiseSpace = [(0, -2), (0, 2), (-2, 0), (2, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    noise = 0.1
    mode = 1 - noise
    # transition_function = ft.partial(grid_transition_stochastic, noiseSpace=noiseSpace, terminals=sheep_states, is_valid=env.is_state_valid, mode=mode)

    transition_function = ft.partial(grid_transition_noise, A=A, terminals=sheep_states, is_valid=env.is_state_valid, noise=noise)

    T = {s: {a: transition_function(s, a) for a in A} for s in S}
    T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S]
                         for a in A] for s in S])

    reward_func = ft.partial(
        grid_reward, env=env, const=-1, terminals=sheep_states)

    R = {s: {a: {sn: reward_func(s, a, sn) for sn in S} for a in A} for s in S}
    R_arr = np.asarray([[[R[s][a].get(s_n, 0) for s_n in S]
                         for a in A] for s in S])

    gamma = 0.9
    value_iteration = ValueIteration(gamma, epsilon=0.0001, max_iter=100, terminals=sheep_states, obstacles=obstacles_states)
    V = value_iteration(S, A, T, R)

    V_arr = V_dict_to_array(V, S)
    Q = V_to_Q(V=V_arr, T=T_arr, R=R_arr, gamma=gamma)
    Q_dict = {(s, sheep_states): {a: Q[si, ai] for (ai, a) in enumerate(A)} for (si, s) in enumerate(S)}

    mapValue = 'V'
    heatMapValue = eval(mapValue)
    y = dict_to_array(heatMapValue)
    y = y.reshape((gridSize, gridSize))
    df = pd.DataFrame(y, columns=[x for x in range(gridSize)])
    sns.heatmap(df, annot=True, fmt='.3f')
    plt.title('{} for goal at {} noise={} goalReward={}'.format(mapValue, sheep_states, noise, goalRewardList))
    plt.show()

    return Q_dict[(3, 1), sheep_states]


if __name__ == '__main__':

    sheep_states = ((6, 11), (11, 6))
    # obstacles_states = ((3, 3), (4, 4), (4, 3), (3, 4))
    # obstacles_states = ((6, 3), (6, 4), (4, 6), (3, 6))
    obstacles_states = ((2, 2), (2, 4), (2, 5), (2, 6), (4, 2), (5, 2), (6, 2))
    # obstacles_states = ((3, 3), (3, 5), (3, 5), (3, 7), (7, 3), (6, 3), (5, 3))

    goalRewardList = [100, 100]

    Q_dict = runVI(sheep_states, obstacles_states, goalRewardList)
    print(Q_dict)
