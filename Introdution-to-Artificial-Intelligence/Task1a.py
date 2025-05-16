## Task 1

import random
import heapq
from copy import deepcopy
import numpy as np

def state_init():
    board = np.arange(16).reshape(4,4)
    for i in range(len(board)):
        random.shuffle(board[i])
    print(board)
   


state_init()
