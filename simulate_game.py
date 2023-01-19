# simulate the game
import numpy as np
import random
from game_settings import NUM_STATES, VAL_R_T, VAL_L_T, STATES
from utils import get_av_val, process_actions
from value_functions import revealing_value as get_cav_v
from solver import optimization

# set nature distribution
p = 0.4

# set Player 1's type: 0 for right, 1 for left
p1_type = 1
type_map = {0: 'Right', 1: 'Left'}

stay = bool(int(input('stay in action set (1 or 0)? ')))

average_game = get_av_val(VAL_R_T, VAL_L_T, p=p)
average_dict = dict(zip(STATES.flatten(), average_game.flatten()))

if __name__ == "__main__":
    print(f'Player 1s Type is: {type_map[p1_type]}')
    print(f'Current position is center (2, 2) and the belief is {p}\n')
    p_t = p
    if stay:
        # add action 'stay'
        a_map = {'0': 'l', '1': 'L', '2': 's', '3': 'r', '4': 'R'}
        pay_size = (5, 3)
    else:
        a_map = {'0': 'l', '1': 'L', '2': 'r', '3': 'R'}
        pay_size = (4, 2)

    # first get strategy for initial state (2, 2)
    start_x = [2, 2]
    v_curr = get_cav_v(2, average_dict, STATES, p_t)[2, 2]
    lam_j, p_1 = optimization(p_t, v_curr, start_x, average_dict, stay)

    # for debugging
    # lam_j = 0.8
    # p_1 = 0.33

    p_2 = (p_t - lam_j * p_1) / (1 - lam_j)
    print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')

    # action selection for first splitting point (lam_1)
    game_dict_1 = get_av_val(VAL_R_T, VAL_L_T, p=p_1)
    game_dict_1 = dict(zip(STATES.flatten(), game_dict_1.flatten()))
    game_dict_2 = get_av_val(VAL_R_T, VAL_L_T, p=p_2)
    game_dict_2 = dict(zip(STATES.flatten(), game_dict_2.flatten()))

    v_next_0 = get_cav_v(1, game_dict_1, STATES, p_1, stay=stay)
    v_next_1 = get_cav_v(1, game_dict_2, STATES, p_2, stay=stay)

    pay_0 = np.zeros(pay_size)
    pay_1 = np.zeros(pay_size)

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
    else:
        pay_0[0, 0] = v_next_0[1, 1]
        pay_0[0, 1] = v_next_0[1, 3]
        pay_0[1, 0] = v_next_0[0, 1]
        pay_0[1, 1] = v_next_0[0, 3]
        pay_0[2, 0] = v_next_0[3, 1]
        pay_0[2, 1] = v_next_0[3, 3]
        pay_0[3, 0] = v_next_0[4, 1]
        pay_0[3, 1] = v_next_0[4, 3]

    # debug
    # print(pay_0)
    # input("Press return to continue!")

    v_0 = np.max(np.min(pay_0, 1))
    a_0 = None
    a_0_idx = np.where(np.min(pay_0, 1) == v_0)[0]  # check for same values

    if len(a_0_idx) == 1:
        a_0 = a_map[str(a_0_idx[0])]
    elif len(a_0_idx) == 5:
        a_0 = 'A'
    else:
        ac = ''
        for a in a_0_idx:
            ac += a_map[str(a)]
        a_0 = ac

    if stay:
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
        pay_1[0, 0] = v_next_1[1, 1]
        pay_1[0, 1] = v_next_1[1, 3]
        pay_1[1, 0] = v_next_1[0, 1]
        pay_1[1, 1] = v_next_1[0, 3]
        pay_1[2, 0] = v_next_1[3, 1]
        pay_1[2, 1] = v_next_1[3, 3]
        pay_1[3, 0] = v_next_1[4, 1]
        pay_1[3, 1] = v_next_1[4, 3]

    v_1 = np.max(np.min(pay_1, 1))
    a_1 = None
    a_1_idx = np.where(np.min(pay_1, 1) == v_1)[0]  # check for same values
    if len(a_1_idx) == 1:
        a_1 = a_map[str(a_1_idx[0])]
    elif len(a_1_idx) == len(a_map):
        a_1 = 'A'
    else:
        ac = ''
        for a in a_1_idx:
            ac += a_map[str(a)]
        a_1 = ac

    # calculate probability of each action
    if p1_type == 0:
        p_i = 1 - p_t
        p_1j = 1 - p_1
        p_2j = 1 - p_2
    else:
        p_i = p_t
        p_1j = p_1
        p_2j = p_2

    a0_p = (lam_j * p_1j) / p_i
    a1_p = ((1 - lam_j) * p_2j) / p_i

    print(f'At initial time, P1 with type {type_map[p1_type]} has the following options: \n')
    print(f'P1 could take action {a_0} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
    print(f'P1 could take action {a_1} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')

    dist = [a0_p, a1_p]
    a_idx = [0, 1]
    action_idx = random.choices(a_idx, dist)[0]
    # set to calculate strategy
    # action_idx = 0
    if action_idx == 0:
        action_1 = a_0
        p_t = p_1
    else:
        action_1 = a_1
        p_t = p_2

    # for simulation purpose select p2's action randomly
    if stay:
        p2_a = random.choices([-1, 0, 1], [0.33, 0.33, 0.33])[0]  # left or right
    else:
        p2_a = random.choices([-1, 1], [0.5, 0.5])[0]
    # to get the strategy pick one at a time
    # p2_a = -1  # -1 is l
    p2_amap = {'-1': 'l', '0': 's', '1': 'r'}
    p2_action = p2_amap[str(p2_a)]

    print(f'P1 chooses action: {action_1} and moves the belief to p_t = {p_t:.2f}')
    print(f'P2 chooses action: {p2_action} at random\n')

    ##########################################################################################
    #################################### NEXT TIME-STEP ######################################
    ##########################################################################################

    if action_1 == 'l':
        step = -1
    elif action_1 == 'r':
        step = 1
    elif action_1 == 'L':
        step = -2
    elif action_1 == 'R':
        step = 2
    else:
        step = 0

    curr_x = start_x
    curr_x = np.array(curr_x) + np.array([step, p2_a])  # get current position

    print(list(curr_x))
    print(f'The current position is: {curr_x}\n')

    ################### ADD SPLITTING POINTS FOR NEXT TIME-STEP #################################
    i = curr_x[0]
    j = curr_x[1]

    average_game = get_av_val(VAL_R_T, VAL_L_T, p=p_t)
    average_dict = dict(zip(STATES.flatten(), average_game.flatten()))
    v_curr = get_cav_v(1, average_dict, STATES, p=p_t)[i, j]  # current value

    # tests = np.linspace(0, 1, 50)
    # v = []
    # for test in tests:
    #     a_g = get_av_val(VAL_R_T, VAL_L_T, p=test)
    #     a_d = dict(zip(STATES.flatten(), a_g.flatten()))
    #     v.append(get_cav_v(1, a_d, STATES, p=test)[i, j])
    #
    # import matplotlib.pyplot as plt
    # plt.plot(tests, v)
    # plt.show()

    # lam_j, p_1 = optimization(p_t, v_curr, list(curr_x), average_dict)
    # since the value is concave, and assuming the beliefs converge,
    p_1 = 0
    p_2 = 1
    lam_j = 1 - p_t  # lam_j = p_j


    # p_2 = (p_t - lam_j * p_1) / (1 - lam_j)
    print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')

    # action selection for first splitting point (lam_1)
    v_next_0 = get_av_val(VAL_R_T, VAL_L_T, p=p_1)
    v_next_1 = get_av_val(VAL_R_T, VAL_L_T, p=p_2)


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
        pay_0[0, 1] = v_next_0[new_l, j]      # left stay
        pay_0[0, 2] = v_next_0[new_l, j + 1]  # left right
        pay_0[1, 0] = v_next_0[new_L, j - 1]  # Left left
        pay_0[1, 1] = v_next_0[new_L, j]      # Left stay
        pay_0[1, 2] = v_next_0[new_L, j + 1]  # Left right
        pay_0[2, 0] = v_next_0[i, j - 1]      # stay left
        pay_0[2, 1] = v_next_0[i, j]          # stay stay
        pay_0[2, 2] = v_next_0[i, j + 1]      # stay right
        pay_0[3, 0] = v_next_0[new_r, j - 1]  # right left
        pay_0[3, 1] = v_next_0[new_r, j]      # right stay
        pay_0[3, 2] = v_next_0[new_r, j + 1]  # right right
        pay_0[4, 0] = v_next_0[new_R, j - 1]  # Right left
        pay_0[4, 1] = v_next_0[new_R, j]      # Right stay
        pay_0[4, 2] = v_next_0[new_R, j + 1]  # Right right
    else:
        pay_0[0, 0] = v_next_0[new_l, j - 1]  # left left
        pay_0[0, 1] = v_next_0[new_l, j + 1]  # left right
        pay_0[1, 0] = v_next_0[new_L, j - 1]  # Left left
        pay_0[1, 1] = v_next_0[new_L, j + 1]  # Left right
        pay_0[2, 0] = v_next_0[new_r, j - 1]  # right left
        pay_0[2, 1] = v_next_0[new_r, j + 1]  # right right
        pay_0[3, 0] = v_next_0[new_R, j - 1]  # Right left
        pay_0[3, 1] = v_next_0[new_R, j + 1]  # Right right

    # debug
    # print(pay_0)
    # input("Press return to continue!")

    v_0 = np.max(np.min(pay_0, 1))
    a_0 = None
    a_0_idx = np.where(np.min(pay_0, 1) == v_0)[0]  # check for same values

    if len(a_0_idx) == 1:
        a_0 = a_map[str(a_0_idx[0])]
    elif len(a_0_idx) == 4:
        a_0 = 'A'
    else:
        ac = ''
        for a in a_0_idx:
            ac += a_map[str(a)]
        a_0 = ac
    if stay:
        pay_1[0, 0] = v_next_1[new_l, j - 1]  # left left
        pay_1[0, 1] = v_next_1[new_l, j]      # left stay
        pay_1[0, 2] = v_next_1[new_l, j + 1]  # left right
        pay_1[1, 0] = v_next_1[new_L, j - 1]  # Left left
        pay_1[1, 1] = v_next_1[new_L, j]      # Left stay
        pay_1[1, 2] = v_next_1[new_L, j + 1]  # Left right
        pay_1[2, 0] = v_next_1[i, j - 1]      # stay left
        pay_1[2, 1] = v_next_1[i, j]          # stay stay
        pay_1[2, 2] = v_next_1[i, j + 1]      # stay right
        pay_1[3, 0] = v_next_1[new_r, j - 1]  # right left
        pay_1[3, 1] = v_next_1[new_r, j]      # right stay
        pay_1[3, 2] = v_next_1[new_r, j + 1]  # right right
        pay_1[4, 0] = v_next_1[new_R, j - 1]  # Right left
        pay_1[4, 1] = v_next_1[new_R, j]      # Right stay
        pay_1[4, 2] = v_next_1[new_R, j + 1]  # Right right
    else:
        pay_1[0, 0] = v_next_1[new_l, j - 1]  # left left
        pay_1[0, 1] = v_next_1[new_l, j + 1]  # left right
        pay_1[1, 0] = v_next_1[new_L, j - 1]  # Left left
        pay_1[1, 1] = v_next_1[new_L, j + 1]  # Left right
        pay_1[2, 0] = v_next_1[new_r, j - 1]  # right left
        pay_1[2, 1] = v_next_1[new_r, j + 1]  # right right
        pay_1[3, 0] = v_next_1[new_R, j - 1]  # Right left
        pay_1[3, 1] = v_next_1[new_R, j + 1]  # Right right

    v_1 = np.max(np.min(pay_1, 1))
    a_1 = None
    a_1_idx = np.where(np.min(pay_1, 1) == v_1)[0]  # check for same values
    if len(a_1_idx) == 1:
        a_1 = a_map[str(a_1_idx[0])]
    elif len(a_1_idx) == 4:
        a_1 = 'A'
    else:
        ac = ''
        for a in a_1_idx:
            ac += a_map[str(a)]
        a_1 = ac

    # process actions
    a_00, a_11 = process_actions(a_0, a_1)

    # calculate probability of each action
    if p1_type == 0:
        p_i = 1 - p_t
        p_1j = 1 - p_1
        p_2j = 1 - p_2
    else:
        p_i = p_t
        p_1j = p_1
        p_2j = p_2

    a0_p = (lam_j * p_1j) / p_i
    a1_p = ((1 - lam_j) * p_2j) / p_i

    print(a_0, a_1)
    print(f'At second time-step, P1 with type {type_map[p1_type]} has the following options: \n')
    print(f'P1 could take action {"".join(a_00)} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
    print(f'P1 could take action {"".join(a_11)} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')

    dist = [a0_p, a1_p]
    a_idx = [0, 1]
    action_idx = random.choices(a_idx, dist)[0]
    if action_idx == 0:
        action_1 = a_00
        p_t = p_1
    else:
        action_1 = a_11
        p_t = p_2

    # for simulation purpose select p2's action randomly
    if stay:
        p2_a = random.choices([-1, 0, 1], [0.33, 0.33, 0.33])[0]  # left, stay, or right
    else:
        p2_a = random.choices([-1, 1], [0.5, 0.5])[0]
    # to get the strategy pick one at a time
    p2_amap = {'-1': 'l', '0': 's', '1': 'r'}
    p2_action = p2_amap[str(p2_a)]

    # force set to calculate strategy
    # action_1 = 'r'
    # p_t = 0.66

    print(f'P1 chooses action: {action_1} and moves the belief to p_t = {p_t:.2f}')
    print(f'P2 chooses action: {p2_action} at random\n')

################################# SECOND TIME-STEP WITHOUT SPLITTING POINTS #######################
# # test with p
# game = get_av_val(VAL_R_T, VAL_L_T, p=p_t)
# pay_0 = np.zeros((4, 2))
#
# i = curr_x[0]
# j = curr_x[1]
#
# if i - 1 < 0:
#     new_L = i
#     new_l = i
# elif i - 2 < 0:
#     new_L = i - 1
#     new_l = i - 1
# else:
#     new_l = i - 1
#     new_L = i - 2
#
# if i + 1 > NUM_STATES - 1:
#     new_R = i
#     new_r = i
# elif i + 2 > NUM_STATES - 1:
#     new_r = i + 1
#     new_R = i + 1
# else:
#     new_r = i + 1
#     new_R = i + 2
#
# #     print(new_l, new_L, new_r, new_R)
#
# pay_0[0, 0] = game[new_l, j - 1]  # left left
# pay_0[0, 1] = game[new_l, j + 1]  # left right
# pay_0[1, 0] = game[new_L, j - 1]  # Left left
# pay_0[1, 1] = game[new_L, j + 1]  # Left right
# pay_0[2, 0] = game[new_r, j - 1]  # right left
# pay_0[2, 1] = game[new_r, j + 1]  # right right
# pay_0[3, 0] = game[new_R, j - 1]  # Right left
# pay_0[3, 1] = game[new_R, j + 1]  # Right right
#
# v_0 = np.max(np.min(pay_0, 1))
# a_0 = None
# #     printMatrix(pay_0)
# a_0_idx = np.where(np.min(pay_0, 1) == v_0)[0]  # check for same values
# #     print(a_0_idx)
# if len(a_0_idx) == 1:
#     a_0 = a_map[str(a_0_idx[0])]
# elif len(a_0_idx) == 4:
#     a_0 = 'A'
# else:
#     ac = ''
#     for a in a_0_idx:
#         ac += a_map[str(a)]
#     a_0 = ac
# #     print(a_0)
# print(f'At second time-step P1 takes action {a_0} and maintains the belief at {p_t:.2f}')
