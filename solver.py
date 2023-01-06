# solve the optimization problem
import numpy as np
import multiprocess as mp
from utils import get_av_val
from itertools import product
from value_functions import revealing_value as get_cav_v
from game_settings import VAL_L_T, VAL_R_T, STATES, NUM_STATES

def optimization(p, v_curr, curr_x, game_dict):

    def constraint(var):
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        p_2 = (p - lam_1 * p_1)/lam_2

        lam_j = np.array([[lam_1], [lam_2]])
        p_j = np.array([[p_1], [p_2]])

        return 0 <= p_2 <= 1

    def objective(var):
        # here, V_curr is the value at the current state
        # V_next is the max min value at the previous state leading to current state
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        p_2 = (p - lam_1 * p_1) / (lam_2)

        lam_j = np.array([[lam_1], [lam_2]])
        v_next = np.zeros((2, 1))
        if curr_x == [2, 2]:
            game_dict_1 = get_av_val(VAL_R_T, VAL_L_T, p_1)
            game_dict_1 = dict(zip(STATES.flatten(), game_dict_1.flatten()))
            game_dict_2 = get_av_val(VAL_R_T, VAL_L_T, p_2)
            game_dict_2 = dict(zip(STATES.flatten(), game_dict_2.flatten()))
            v_next_0 = get_cav_v(1, game_dict_1, STATES, p_1)
            v_next_1 = get_cav_v(1, game_dict_2, STATES, p_2)

            pay_0 = np.zeros((4, 2))
            pay_1 = np.zeros((4, 2))

            pay_0[0, 0] = v_next_0[1, 1]
            pay_0[0, 1] = v_next_0[1, 3]
            pay_0[1, 0] = v_next_0[0, 1]
            pay_0[1, 1] = v_next_0[0, 3]
            pay_0[2, 0] = v_next_0[3, 1]
            pay_0[2, 1] = v_next_0[3, 3]
            pay_0[3, 0] = v_next_0[4, 1]
            pay_0[3, 1] = v_next_0[4, 3]

            pay_1[0, 0] = v_next_1[1, 1]
            pay_1[0, 1] = v_next_1[1, 3]
            pay_1[1, 0] = v_next_1[0, 1]
            pay_1[1, 1] = v_next_1[0, 3]
            pay_1[2, 0] = v_next_1[3, 1]
            pay_1[2, 1] = v_next_1[3, 3]
            pay_1[3, 0] = v_next_1[4, 1]
            pay_1[3, 1] = v_next_1[4, 3]

            # do min max on v_next
            v_next[0] = np.max(np.min(pay_0, 1))
            v_next[1] = np.max(np.min(pay_1, 1))

        else:  # that is, some other state in the second time-step

            i = curr_x[0]
            j = curr_x[1]

            v_next_0 = get_av_val(VAL_R_T, VAL_L_T, p_1)
            v_next_1 = get_av_val(VAL_R_T, VAL_L_T, p_2)

            pay_0 = np.zeros((4, 2))
            pay_1 = np.zeros((4, 2))

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

            pay_0[0, 0] = v_next_0[new_l, j - 1]  # left left
            pay_0[0, 1] = v_next_0[new_l, j + 1]  # left right
            pay_0[1, 0] = v_next_0[new_L, j - 1]  # Left left
            pay_0[1, 1] = v_next_0[new_L, j + 1]  # Left right
            pay_0[2, 0] = v_next_0[new_r, j - 1]  # right left
            pay_0[2, 1] = v_next_0[new_r, j + 1]  # right right
            pay_0[3, 0] = v_next_0[new_R, j - 1]  # Right left
            pay_0[3, 1] = v_next_0[new_R, j + 1]  # Right right

            pay_1[0, 0] = v_next_1[new_l, j - 1]  # left left
            pay_1[0, 1] = v_next_1[new_l, j + 1]  # left right
            pay_1[1, 0] = v_next_1[new_L, j - 1]  # Left left
            pay_1[1, 1] = v_next_1[new_L, j + 1]  # Left right
            pay_1[2, 0] = v_next_1[new_r, j - 1]  # right left
            pay_1[2, 1] = v_next_1[new_r, j + 1]  # right right
            pay_1[3, 0] = v_next_1[new_R, j - 1]  # Right left
            pay_1[3, 1] = v_next_1[new_R, j + 1]  # Right right

            # do min max on v_next
            v_next[0] = np.max(np.min(pay_0, 1))
            v_next[1] = np.max(np.min(pay_1, 1))

        return lam_1, p_1, abs((v_curr - np.matmul(lam_j.T, v_next)).item())

    lam = np.linspace(1e-6, 0.999999, 200)
    grid = product(lam, repeat=2)
    reduced = filter(constraint, grid)
    l_1 = float('inf')
    p_1 = float('inf')
    curr_min = float('inf')
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.imap_unordered(objective, reduced)

        for lam_1, P_1, val in res:
            if val < curr_min:
                curr_min = val
                l_1 = lam_1
                p_1 = P_1

    res = (l_1, p_1)

    return res


