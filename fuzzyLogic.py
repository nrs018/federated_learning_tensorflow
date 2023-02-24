import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from matplotlib import rcParams
rcParams['font.family']='DejaVu Serif'
rcParams['font.size']=18

# New Antecedent/Consequent objects hold universe variables and membership
# functions
# numSample 的平均值为102

numSample = ctrl.Antecedent(np.arange(0, 801, 1), 'numSample')
# Bandwidth 的平均值为8.8，fuzzyLogic中取值为10， 最大值为36.5， fuzzyLogic中可以取值为38
# 他的步长 设置为 0.01
#Bandwidth = ctrl.Antecedent(np.arange(0, 164, 0.1), 'Bandwidth')
Bandwidth = ctrl.Antecedent(np.arange(0, 1300, 2), 'Bandwidth')

ComputingPower = ctrl.Antecedent(np.arange(1, 21, 0.1), 'ComputingPower')

numUse = ctrl.Antecedent(np.arange(0, 300, 1), 'numUse')

Score = ctrl.Consequent(np.arange(0, 81, 1), 'Score')

# Auto-membership function population is possible with .automf(3, 5, or 7)
numSample.automf(3)
Bandwidth.automf(3)
ComputingPower.automf(3)
numUse.automf(3)

# 将numSample分为四个段：
#          0  -  176
#        177  -  204
#        205  -  222
#        223  -  282
# numSample['poor'] = fuzz.trimf(numSample.universe, [0, 0, 176])
# numSample['mediocre'] = fuzz.trimf(numSample.universe, [0, 176, 204])
# numSample['average'] = fuzz.trimf(numSample.universe, [176, 204, 222])
# numSample['decent'] = fuzz.trimf(numSample.universe, [204, 222, 282])
# numSample['good'] = fuzz.trimf(numSample.universe, [222, 282, 282])

# numSample['poor'] = fuzz.trimf(numSample.universe, [0, 176, 222])
# numSample['average'] = fuzz.trimf(numSample.universe, [176, 199, 222])
# numSample['good'] = fuzz.trimf(numSample.universe, [176, 222, 222])
numSample['poor'] = fuzz.gaussmf(numSample.universe, 20, 80)
numSample['average'] = fuzz.gaussmf(numSample.universe, 200, 80)
numSample['good'] = fuzz.gaussmf(numSample.universe, 800, 80)

# BandWidth 中最大值为164，平均值为39
Bandwidth['poor'] = fuzz.gaussmf(Bandwidth.universe, 0, 100)
Bandwidth['average'] = fuzz.gaussmf(Bandwidth.universe, 303, 100)
Bandwidth['good'] = fuzz.gaussmf(Bandwidth.universe, 1300, 500)

ComputingPower['good'] = fuzz.gaussmf(ComputingPower.universe, 1, 5)
ComputingPower['average'] = fuzz.gaussmf(ComputingPower.universe, 11, 5)
ComputingPower['poor'] = fuzz.gaussmf(ComputingPower.universe, 20, 5)

numUse['poor'] = fuzz.gaussmf(numUse.universe, 100, 50)
numUse['average'] = fuzz.gaussmf(numUse.universe, 200, 50)
numUse['good'] = fuzz.gaussmf(numUse.universe, 300, 50)

Score['L0'] = fuzz.gaussmf(Score.universe, 0, 5)
Score['L1'] = fuzz.gaussmf(Score.universe, 10, 5)
Score['L2'] = fuzz.gaussmf(Score.universe, 20, 5)
Score['L3'] = fuzz.gaussmf(Score.universe, 30, 5)
Score['L4'] = fuzz.gaussmf(Score.universe, 40, 5)
Score['L5'] = fuzz.gaussmf(Score.universe, 50, 5)
Score['L6'] = fuzz.gaussmf(Score.universe, 60, 5)
Score['L7'] = fuzz.gaussmf(Score.universe, 70, 5)
Score['L8'] = fuzz.gaussmf(Score.universe, 80, 5)
# Score['L9'] = fuzz.trimf(Score.universe, [80, 90, 90])
#  Score['L9'] = fuzz.trimf(Score.universe, [80, 90, 100])
# Score['L10'] = fuzz.trimf(Score.universe, [90, 100, 100])
# You can see how these look with .view()

#numSample['good'].view()
#Bandwidth['good'].view()
#ComputingPower['good'].view()
#numUse['good'].view()
#Score.view()
#plt.tight_layout()
#plt.show()
#time.sleep(100)
#sys.exit()


###################  RULE  #######################
# rule1 = ctrl.Rule(numSample['poor'] & Bandwidth['poor'], Score['poor'])
rule1 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['good']&numUse['good'],Score['L8'])
rule2 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['good']&numUse['good'],Score['L7'])
rule3 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['good']&numUse['good'],Score['L6'])
rule4 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['good']&numUse['good'],Score['L7'])
rule5 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['good']&numUse['good'],Score['L6'])
rule6 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['good']&numUse['good'],Score['L5'])
rule7 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['good']&numUse['good'],Score['L6'])
rule8 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['good']&numUse['good'],Score['L5'])
rule9 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['good']&numUse['good'],Score['L4'])
rule10 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['average']&numUse['good'],Score['L7'])
rule11 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['average']&numUse['good'],Score['L6'])
rule12 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['average']&numUse['good'],Score['L5'])
rule13 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['average']&numUse['good'],Score['L6'])
rule14 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['average']&numUse['good'],Score['L5'])
rule15 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['average']&numUse['good'],Score['L4'])
rule16 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['average']&numUse['good'],Score['L5'])
rule17 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['average']&numUse['good'],Score['L4'])
rule18 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['average']&numUse['good'],Score['L3'])
rule19 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['poor']&numUse['good'],Score['L6'])
rule20 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['poor']&numUse['good'],Score['L5'])
rule21 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['poor']&numUse['good'],Score['L4'])
rule22 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['poor']&numUse['good'],Score['L5'])
rule23 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['poor']&numUse['good'],Score['L4'])
rule24 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['poor']&numUse['good'],Score['L3'])
rule25 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['poor']&numUse['good'],Score['L4'])
rule26 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['poor']&numUse['good'],Score['L3'])
rule27 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['poor']&numUse['good'],Score['L2'])
rule28 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['good']&numUse['average'],Score['L6'])
rule29 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['good']&numUse['average'],Score['L5'])
rule30 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['good']&numUse['average'],Score['L4'])
rule31 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['good']&numUse['average'],Score['L5'])
rule32 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['good']&numUse['average'],Score['L4'])
rule33 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['good']&numUse['average'],Score['L3'])
rule34 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['good']&numUse['average'],Score['L4'])
rule35 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['good']&numUse['average'],Score['L3'])
rule36 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['good']&numUse['average'],Score['L2'])
rule37 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['average']&numUse['average'],Score['L5'])
rule38 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['average']&numUse['average'],Score['L4'])
rule39 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['average']&numUse['average'],Score['L3'])
rule40 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['average']&numUse['average'],Score['L4'])
rule41 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['average']&numUse['average'],Score['L3'])
rule42 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['average']&numUse['average'],Score['L2'])
rule43 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['average']&numUse['average'],Score['L3'])
rule44 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['average']&numUse['average'],Score['L2'])
rule45 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['average']&numUse['average'],Score['L1'])
rule46 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['poor']&numUse['average'],Score['L4'])
rule47 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['poor']&numUse['average'],Score['L3'])
rule48 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['poor']&numUse['average'],Score['L2'])
rule49 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['poor']&numUse['average'],Score['L3'])
rule50 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['poor']&numUse['average'],Score['L2'])
rule51 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['poor']&numUse['average'],Score['L1'])
rule52 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['poor']&numUse['average'],Score['L2'])
rule53 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['poor']&numUse['average'],Score['L1'])
rule54 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['poor']&numUse['average'],Score['L0'])
rule55 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule56 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule57 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule58 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule59 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule60 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule61 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule62 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule63 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['good']&numUse['poor'],Score['L0'])
rule64 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule65 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule66 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule67 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule68 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule69 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule70 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule71 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule72 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['average']&numUse['poor'],Score['L0'])
rule73 = ctrl.Rule(numSample['good']&ComputingPower['good']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule74 = ctrl.Rule(numSample['average']&ComputingPower['good']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule75 = ctrl.Rule(numSample['poor']&ComputingPower['good']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule76 = ctrl.Rule(numSample['good']&ComputingPower['average']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule77 = ctrl.Rule(numSample['average']&ComputingPower['average']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule78 = ctrl.Rule(numSample['poor']&ComputingPower['average']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule79 = ctrl.Rule(numSample['good']&ComputingPower['poor']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule80 = ctrl.Rule(numSample['average']&ComputingPower['poor']&Bandwidth['poor']&numUse['poor'],Score['L0'])
rule81 = ctrl.Rule(numSample['poor']&ComputingPower['poor']&Bandwidth['poor']&numUse['poor'],Score['L0'])

# rule2 = ctrl.rule(service['average'], tip['medium'])
# rule3 = ctrl.rule(service['good'] | quality['good'], tip['high'])

# rule1.view()

tipping_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
     rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
     rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28,
     rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37,
     rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46,
     rule47, rule48, rule49, rule50, rule51, rule52, rule53, rule54, rule55,
     rule56, rule57, rule58, rule59, rule60, rule61, rule62, rule63, rule64,
     rule65, rule66, rule67, rule68, rule69, rule70, rule71, rule72, rule73,
     rule74, rule75, rule76, rule77, rule78, rule79, rule80, rule81])

tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

clientINFO = pd.read_excel('./cifar10_clientINFO_2620.ods', sheet_name='extremeScenario')
clientINFO = clientINFO.values.tolist()
# print(clientINFO)
ROUND = 40
for r in range(ROUND):
    for i in range(len(clientINFO)):
        # pass inputs to the controlsystem using antecedent labels with pythonic api
        # note: if you like passing many inputs all at once, use .inputs(dict_of_data)
        tipping.input['numSample'] = clientINFO[i][1]
        tipping.input['ComputingPower'] = clientINFO[i][2]
        tipping.input['Bandwidth'] = clientINFO[i][3]
        tipping.input['numUse'] = clientINFO[i][4]
        #print(clientINFO[i][1], ' ', clientINFO[i][2], ' ', clientINFO[i][3], ' ', clientINFO[i][4])
        # crunch the numbers
        tipping.compute()
        #print('----------------------')
        Score.view(sim=tipping)
        ax = plt.subplot(111)
        plt.tight_layout()
        print(tipping.output['Score'])
        plt.legend(labels=[])
        plt.text(-1, 1.02, '${L0}$')
        plt.text(9, 1.02, '${L1}$')
        plt.text(19, 1.02, '${L2}$')
        plt.text(29, 1.02, '${L3}$')
        plt.text(39, 1.02, '${L4}$')
        plt.text(49, 1.02, '${L5}$')
        plt.text(59, 1.02, '${L6}$')
        plt.text(69, 1.02, '${L7}$')
        plt.text(79, 1.02, '${L8}$')
        #plt.text(50, -0.05, '${53.03}$')
        plt.text(19, 0.52, 'The value of \n centroid: 58.09')
        plt.arrow(31, 0.48, 25, -0.42)
        plt.xlabel('Score', fontdict={'size': 18, 'family': 'DejaVu Serif'})
        plt.ylabel('Membership Function', fontdict={'size': 18, 'family': 'DejaVu Serif'})
        plt.xticks([0, 20, 40, 60, 80], fontsize=18)
        plt.yticks([0, 0.5, 1], fontsize=18)
        ax.legend_.remove()
        plt.show()
        print('=====================')
        time.sleep(1000)
        exit(0)
        clientINFO[i][5] = tipping.output['Score']
        # print(clientinfo[i][0], ' ', tipping.output['score'])
        # score.view(sim=tipping)
    # print(len(clientINFO))
    # sys.exit()
    for i in range(60):

        mina = 1000
        pos = 0
        for j in range(60 - i):
            # print(j)
            if clientINFO[j][5] < mina:
                mina = clientINFO[j][5]
                pos = j
        tmp = clientINFO[pos]
        # print(pos,' cccc ', 3383 - i - 1)
        clientINFO[pos] = clientINFO[60 - i - 1]
        clientINFO[60 - i - 1] = tmp
    # print(clientINFO)
    print('=====================  round ', r, '  ==========================')
    fo = open("fuzzyLogicClientSelection.txt", "a")
    # for i in range(30):
    i = 0
    count = 0
    while count < 8:
        #tt = clientINFO[i][1]/20*30*0.06*clientINFO[i][2] + 4920/clientINFO[i][3]
        #if tt < 100:
        print(clientINFO[i][0], ' ', end='')
        print(clientINFO[i][1], ' ', end='')
        print(clientINFO[i][2], ' ', end='')
        print(clientINFO[i][3], ' ', end='')
        print(clientINFO[i][4], ' ', end='')
        print(clientINFO[i][5], ' ', end='')
        print(clientINFO[i][6], ' ', end='')
        print(clientINFO[i][7])

        for x in range(8):
            fo.write(str(clientINFO[i][x]) + ' ')
        fo.write('\n')
        clientINFO[i][4] = clientINFO[i][4] - 1
        count = count + 1
        # print(i, ' ', clientINFO[i][0], ' ', tt)
        i = i + 1
    fo.close()
