# Non-Revealing and Revealing Value Functions
import copy
import numpy as np
from utils import get_av_val
from game_settings import NUM_STATES, VAL_L_T, VAL_R_T, STATES
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt


def non_revealing_value(timestep, game_dict, states, return_state=0):
    def get_game_dict(t, g, s, value_fun=non_revealing_value):
        return value_fun(t, g, s, return_state=1)

    if timestep > 1 and return_state == 0:
        temp_game = copy.deepcopy(game_dict)
        for i in range(1, timestep):
            average, _ = get_game_dict(i, temp_game, states)
            temp_game = dict(zip(states.flatten(), average.flatten()))

        game_dict = copy.deepcopy(temp_game)

    temp = np.full(states.shape, np.nan)
    action = np.full(states.shape, '%', dtype='U25')
    p1_amap = {'0': 'l', '1': 'L', '2': 'r', '3': 'R'}
    p2_amap = {'0': 'l', '1': 'r'}

    for i in range(NUM_STATES):
        for j in range(timestep, NUM_STATES - timestep):
            payoff = np.zeros((4, 2))
            if i - 1 < 0:
                new_L = i
                new_l = i
            elif i - 2 < 0:
                new_L = i - 1
                new_l = i - 1
            else:
                new_l = i - 1
                new_L = i - 2

            if i + 1 > NUM_STATES - 1:
                new_R = i
                new_r = i
            elif i + 2 > NUM_STATES - 1:
                new_r = i + 1
                new_R = i + 1
            else:
                new_r = i + 1
                new_R = i + 2

            payoff[0, 0] = game_dict[states[new_l, j - 1]]  # left left
            payoff[0, 1] = game_dict[states[new_l, j + 1]]  # left right
            payoff[1, 0] = game_dict[states[new_L, j - 1]]  # Left left
            payoff[1, 1] = game_dict[states[new_L, j + 1]]  # Left right
            payoff[2, 0] = game_dict[states[new_r, j - 1]]  # right left
            payoff[2, 1] = game_dict[states[new_r, j + 1]]  # right right
            payoff[3, 0] = game_dict[states[new_R, j - 1]]  # Right left
            payoff[3, 1] = game_dict[states[new_R, j + 1]]  # Right right

            temp[i, j] = np.max(np.min(payoff, 1))
            action_idx = np.where(np.min(payoff, 1) == temp[i, j])[0]  # check for same values
            if len(action_idx) == 1:
                action[i, j] = p1_amap[str(action_idx[0])]
            elif len(action_idx) == 4:
                action[i, j] = 'A'
            else:
                ac = ''
                for a in action_idx:
                    ac += p1_amap[str(a)]
                action[i, j] = ac

    return temp, action


def revealing_value(timestep, game_dict, states, p, return_state=0):
    def get_game_dict(t, g, s, ps, value_fun=revealing_value):
        return value_fun(t, g, s, ps, return_state=1)

    # redundant, do not need this during simulation
    if timestep == 1:
        cav_v, _ = non_revealing_value(1, game_dict, states)
        ps = np.linspace(0, 1, 100)
        cavs = np.zeros((5, 5, len(ps)))
        for i in range(len(ps)):
            av_game = get_av_val(VAL_R_T, VAL_L_T, p=ps[i])
            av_dict = dict(zip(STATES.flatten(), av_game.flatten()))
            temp, _ = non_revealing_value(1, av_dict, STATES)
            cavs[:, :, i] = temp

        for row in range(NUM_STATES):
            for col in range(timestep, NUM_STATES-timestep):
                vals = cavs[row, col, :]
                v = vals.reshape(-1, 1)
                x = ps.reshape(-1, 1)
                points = np.hstack((x, v))
                try:
                    hull = ConvexHull(points)
                    #                     convex_hull_plot_2d(hull)
                    p_min = points[hull.vertices[2:]][1, 0]
                    p_max = points[hull.vertices[2:]][0, 0]
                    if p_min <= p <= p_max:
                        cav_v[row, col] = points[hull.vertices[2:]][1, 1]
                except:
                    pass

    if timestep > 1 and return_state == 0:
        ps = np.linspace(0, 1, 100)
        cavs = np.zeros((NUM_STATES, NUM_STATES, len(ps)))
        for k in range(len(ps)):
            av_game = get_av_val(VAL_R_T, VAL_L_T, p=ps[k])
            av_dict = dict(zip(STATES.flatten(), av_game.flatten()))
            temp = get_game_dict(1, av_dict, STATES, ps=ps[k])
            maximin = np.full(STATES.shape, np.nan)
            for i in range(NUM_STATES):
                for j in range(timestep, NUM_STATES - timestep):
                    payoff = np.zeros((4, 2))
                    if i - 1 < 0:
                        new_L = i
                        new_l = i
                    elif i - 2 < 0:
                        new_L = i - 1
                        new_l = i - 1
                    else:
                        new_l = i - 1
                        new_L = i - 2

                    if i + 1 > NUM_STATES - 1:
                        new_R = i
                        new_r = i
                    elif i + 2 > NUM_STATES - 1:
                        new_r = i + 1
                        new_R = i + 1
                    else:
                        new_r = i + 1
                        new_R = i + 2

                    payoff[0, 0] = temp[new_l, j - 1]  # left left
                    payoff[0, 1] = temp[new_l, j + 1]  # left right
                    payoff[1, 0] = temp[new_L, j - 1]  # Left left
                    payoff[1, 1] = temp[new_L, j + 1]  # Left right
                    payoff[2, 0] = temp[new_r, j - 1]  # right left
                    payoff[2, 1] = temp[new_r, j + 1]  # right right
                    payoff[3, 0] = temp[new_R, j - 1]  # Right left
                    payoff[3, 1] = temp[new_R, j + 1]  # Right right

                    maximin[i, j] = np.max(np.min(payoff, 1))

            cavs[:, :, k] = maximin

        # get the cav_v for some particular p
        av_game = get_av_val(VAL_R_T, VAL_L_T, p=p)
        av_dict = dict(zip(STATES.flatten(), av_game.flatten()))
        cav_v, _ = non_revealing_value(timestep, av_dict, STATES)  # just a place holder
        temp = get_game_dict(1, av_dict, STATES, ps=p)
        for i in range(NUM_STATES):
            for j in range(timestep, NUM_STATES - timestep):
                payoff = np.zeros((4, 2))
                if i - 1 < 0:
                    new_L = i
                    new_l = i
                elif i - 2 < 0:
                    new_L = i - 1
                    new_l = i - 1
                else:
                    new_l = i - 1
                    new_L = i - 2

                if i + 1 > NUM_STATES - 1:
                    new_R = i
                    new_r = i
                elif i + 2 > NUM_STATES - 1:
                    new_r = i + 1
                    new_R = i + 1
                else:
                    new_r = i + 1
                    new_R = i + 2

                payoff[0, 0] = temp[new_l, j - 1]  # left left
                payoff[0, 1] = temp[new_l, j + 1]  # left right
                payoff[1, 0] = temp[new_L, j - 1]  # Left left
                payoff[1, 1] = temp[new_L, j + 1]  # Left right
                payoff[2, 0] = temp[new_r, j - 1]  # right left
                payoff[2, 1] = temp[new_r, j + 1]  # right right
                payoff[3, 0] = temp[new_R, j - 1]  # Right left
                payoff[3, 1] = temp[new_R, j + 1]  # Right right

                cav_v[i, j] = np.max(np.min(payoff, 1))

        for row in range(NUM_STATES):
            for col in range(timestep, NUM_STATES - timestep):
                vals = cavs[row, col, :]
                v = vals.reshape(-1, 1)
                x = ps.reshape(-1, 1)
                points = np.hstack((x, v))
                try:
                    hull = ConvexHull(points)
                    convex_hull_plot_2d(hull)
                    plt.show()# plots the cav of values
                    p_min = points[hull.vertices[2:]][1, 0]
                    p_max = points[hull.vertices[2:]][0, 0]
                    if p_min <= p <= p_max:
                        cav_v[row, col] = points[hull.vertices[2:]][1, 1]
                except:
                    pass

    return cav_v


