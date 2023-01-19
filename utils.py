# utility functions as necessary
import numpy as np


def printMatrix(s):
    # Do heading
    print("     ", end="")
    for j in range(len(s[0])):
        print("%5s " % j, end="")
    print()
    print("     ", end="")
    for j in range(len(s[0])):
        print("------", end="")
    print()
    # Matrix contents
    for i in range(len(s)):
        print("%3s |" % (i), end="")  # Row nums
        for j in range(len(s[0])):
            if type(s[i][j]) == np.float64:
                temp = round(s[i][j], 2)
            else:
                temp = s[i][j]
            print("%5s " % temp, end="")
        print()


def get_av_val(val_r, val_l, p):
    return p * val_l + (1 - p) * val_r


def process_actions(action1, action2):
    actions = ['l', 'L', 'r', 'R']
    if (action1 != 'A') and (action2 != 'A'):
        return action1, action2
    else:
        if action1 == 'A':
            action2 = list(action2)
            for each in action2:
                actions.remove(each)

            action1 = ''.join(actions)
        elif action2 == 'A':
            action1 = list(action1)
            for each in action1:
                actions.remove(each)

            action2 = ''.join(actions)

    return action1, action2
