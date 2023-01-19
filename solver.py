# solve the optimization problem
import numpy as np
import multiprocess as mp
import time
import concurrent.futures
from utils import get_av_val
from itertools import product, zip_longest, repeat
from value_functions import revealing_value as get_cav_v
from game_settings import VAL_L_T, VAL_R_T, STATES, NUM_STATES


def optimization(p, v_curr, curr_x, game_dict, stay=False):
    def constraint(var):
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        p_2 = (p - lam_1 * p_1) / lam_2

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

        if stay:
            pay_size = (5, 3)
        else:
            pay_size = (4, 2)

        if curr_x == [2, 2]:  # at the fist-time step, which is t=2, in the code
            game_dict_1 = get_av_val(VAL_R_T, VAL_L_T, p_1)
            game_dict_1 = dict(zip(STATES.flatten(), game_dict_1.flatten()))
            game_dict_2 = get_av_val(VAL_R_T, VAL_L_T, p_2)
            game_dict_2 = dict(zip(STATES.flatten(), game_dict_2.flatten()))
            v_next_0 = get_cav_v(1, game_dict_1, STATES, p_1)
            v_next_1 = get_cav_v(1, game_dict_2, STATES, p_2)

            pay_0 = np.zeros(pay_size)  # payoff table corresponding to p_1
            pay_1 = np.zeros(pay_size)  # payoff table corresponding to p_2

            if stay:
                pay_0[0, 0] = v_next_0[1, 1]  # l l
                pay_0[0, 1] = v_next_0[1, 2]  # l s
                pay_0[0, 2] = v_next_0[1, 3]  # l r
                pay_0[1, 0] = v_next_0[0, 1]  # L l
                pay_0[1, 1] = v_next_0[0, 2]  # L s
                pay_0[1, 2] = v_next_0[0, 3]  # L r
                pay_0[2, 0] = v_next_0[2, 1]  # s l
                pay_0[2, 1] = v_next_0[2, 2]  # s s
                pay_0[2, 2] = v_next_0[2, 3]  # s r
                pay_0[3, 0] = v_next_0[3, 1]  # r l
                pay_0[3, 1] = v_next_0[3, 2]  # r s
                pay_0[3, 2] = v_next_0[3, 3]  # r r
                pay_0[4, 0] = v_next_0[4, 1]  # R l
                pay_0[4, 1] = v_next_0[4, 2]  # R s
                pay_0[4, 2] = v_next_0[4, 3]  # R r

                pay_1[0, 0] = v_next_1[1, 1]  # l l
                pay_1[0, 1] = v_next_1[1, 2]  # l s
                pay_1[0, 2] = v_next_1[1, 3]  # l r
                pay_1[1, 0] = v_next_1[0, 1]  # L l
                pay_1[1, 1] = v_next_1[0, 2]  # L s
                pay_1[1, 2] = v_next_1[0, 3]  # L r
                pay_1[2, 0] = v_next_1[2, 1]  # s l
                pay_1[2, 1] = v_next_1[2, 2]  # s s
                pay_1[2, 2] = v_next_1[2, 3]  # s r
                pay_1[3, 0] = v_next_1[3, 1]  # r l
                pay_1[3, 1] = v_next_1[3, 2]  # r s
                pay_1[3, 2] = v_next_1[3, 3]  # r r
                pay_1[4, 0] = v_next_1[4, 1]  # R l
                pay_1[4, 1] = v_next_1[4, 2]  # R s
                pay_1[4, 2] = v_next_1[4, 3]  # R r
            else:
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

            # do maximin on v_next
            v_next[0] = np.max(np.min(pay_0, 1))
            v_next[1] = np.max(np.min(pay_1, 1))

        else:  # that is, some other state in the second time-step, which is t=1, in the code

            i = curr_x[0]
            j = curr_x[1]

            v_next_0 = get_av_val(VAL_R_T, VAL_L_T, p_1)
            v_next_1 = get_av_val(VAL_R_T, VAL_L_T, p_2)

            pay_0 = np.zeros(pay_size)
            pay_1 = np.zeros(pay_size)

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

            if stay:
                pay_0[0, 0] = v_next_0[new_l, j - 1]  # left left
                pay_0[0, 1] = v_next_0[new_l, j]  # left stay
                pay_0[0, 2] = v_next_0[new_l, j + 1]  # left right
                pay_0[1, 0] = v_next_0[new_L, j - 1]  # Left left
                pay_0[1, 1] = v_next_0[new_L, j]  # Left stay
                pay_0[1, 2] = v_next_0[new_L, j + 1]  # Left right
                pay_0[2, 0] = v_next_0[i, j - 1]  # stay left
                pay_0[2, 1] = v_next_0[i, j]  # stay stay
                pay_0[2, 2] = v_next_0[i, j + 1]  # stay right
                pay_0[3, 0] = v_next_0[new_r, j - 1]  # right left
                pay_0[3, 1] = v_next_0[new_r, j]  # right stay
                pay_0[3, 2] = v_next_0[new_r, j + 1]  # right right
                pay_0[4, 0] = v_next_0[new_R, j - 1]  # Right left
                pay_0[4, 1] = v_next_0[new_R, j]  # Right stay
                pay_0[4, 2] = v_next_0[new_R, j + 1]  # Right right

                pay_1[0, 0] = v_next_1[new_l, j - 1]  # left left
                pay_1[0, 1] = v_next_1[new_l, j]  # left stay
                pay_1[0, 2] = v_next_1[new_l, j + 1]  # left right
                pay_1[1, 0] = v_next_1[new_L, j - 1]  # Left left
                pay_1[1, 1] = v_next_1[new_L, j]  # Left stay
                pay_1[1, 2] = v_next_1[new_L, j + 1]  # Left right
                pay_1[2, 0] = v_next_1[i, j - 1]  # stay left
                pay_1[2, 1] = v_next_1[i, j]  # stay stay
                pay_1[2, 2] = v_next_1[i, j + 1]  # stay right
                pay_1[3, 0] = v_next_1[new_r, j - 1]  # right left
                pay_1[3, 1] = v_next_1[new_r, j]  # right stay
                pay_1[3, 2] = v_next_1[new_r, j + 1]  # right right
                pay_1[4, 0] = v_next_1[new_R, j - 1]  # Right left
                pay_1[4, 1] = v_next_1[new_R, j]  # Right stay
                pay_1[4, 2] = v_next_1[new_R, j + 1]  # Right right
            else:
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

            # do maximin on v_next
            v_next[0] = np.max(np.min(pay_0, 1))
            v_next[1] = np.max(np.min(pay_1, 1))

        return lam_1, p_1, abs((v_curr - np.matmul(lam_j.T, v_next)).item())  # \sum_j \lambda_j v(t=k+1, x', p_j)

    lam = np.linspace(1e-6, 0.999999, 100)
    grid = product(lam, repeat=2)
    reduced = filter(constraint, grid)
    l_1 = float('inf')
    p_1 = float('inf')
    curr_min = float('inf')
    # ini = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.imap_unordered(objective, reduced)

        for lam_1, P_1, val in res:
            if val < curr_min:
                curr_min = val
                l_1 = lam_1
                p_1 = P_1
    # out = time.time()
    # print(out-ini)
    res = (l_1, p_1)

    # # chat gpt's solution
    # # Generate a grid of values for lambda1 and lambda2
    # lam1, lam2 = np.meshgrid(np.linspace(1e-6, 0.999999, 200), np.linspace(1e-6, 0.999999, 200))
    #
    # # Flatten the grid of values into a single array
    # grid = np.stack((lam1, lam2), axis=-1).reshape(-1, 2)
    #
    # # Filter the grid to only include values that satisfy the constraint
    # reduced = filter(constraint, grid)
    #
    # # Initialize variables to store the minimum value and corresponding lambda1 and lambda2 values
    # l_1 = float('inf')
    # p_1 = float('inf')
    # curr_min = float('inf')
    #
    # # Use the multiprocessing.Pool to process the objective function in parallel
    # with mp.Pool(mp.cpu_count()) as pool:
    #     res = pool.imap_unordered(objective, reduced)
    #
    #     for lam_1, P_1, val in res:
    #         if val < curr_min:
    #             curr_min = val
    #             l_1 = lam_1
    #             p_1 = P_1
    #
    # res = (l_1, p_1)


    return res
