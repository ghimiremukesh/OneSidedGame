# game settings and constants
import itertools

import numpy as np

NUM_STATES = 5
s = np.linspace(0, NUM_STATES-1, NUM_STATES)
states = np.array(list(itertools.product(s, repeat=2)))

val_l = np.zeros((len(s), len(s)))
for i in range(NUM_STATES):
    val_l[np.triu_indices(NUM_STATES, k=i)] = i

VAL_L_T = val_l.reshape(NUM_STATES, NUM_STATES)
VAL_R_T = VAL_L_T.T

s_pairs = np.array([[''.join(str(states[i, :])) for i in range(len(states))]])
STATES = s_pairs.reshape(NUM_STATES, NUM_STATES)

