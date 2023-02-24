import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

clientINFO = pd.read_excel('./clientINFO.ods')
clientINFO = clientINFO.values.tolist()
num = []
for i in range(3383):
    num.append(i)

ROUND = 300
for r in range(ROUND):
    random.shuffle(num)
    fo = open("clientINFO_random.txt", "a")
    for i in range(10):
        print(clientINFO[num[i]][0], ' ', end='')
        print(clientINFO[num[i]][1], ' ', end='')

        # print(clientINFO[num[i]][3], ' ', end='')
        # print(clientINFO[num[i]][4], ' ', end='')
        # print(clientINFO[num[i]][5], ' ', end='')
        print(clientINFO[num[i]][6], ' ', end='')
        print(clientINFO[num[i]][7], ' ', end='')
        print(clientINFO[num[i]][2], )


        for x in range(8):
            fo.write(str(clientINFO[num[i]][x]) + ' ')
        fo.write('\n')

    fo.close()
