 #%%
from elm import elm
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

import argparse

is_py = True


stdsc = StandardScaler()

# regression ~probelm

print("stiff regression problem>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# generate dataset
x = np.arange(0.25, 20, 0.1).reshape(-1,1)
# x = np.random.uniform(0.25, 20, int((20-0.25)/0.1)).reshape(-1,1)
x = np.sort(x,axis=0)
# print(x)
y = (x<5)*0.01*np.ones_like(x)+(x>= 5)*(x<12)*(50*np.sin(x))+(x>=12)*(10*np.ones_like(x))#x*np.cos(x) + 0.5*np.sqrt(x)*np.random.randn(x.shape[0]).reshape(-1,1)
# y = (x<5)*0.01*np.ones_like(x)+(x>= 5)*(x<12)*(50*x)+(x>=12)*(10*np.ones_like(x))#x*np.cos(x) + 0.5*np.sqrt(x)*np.random.randn(x.shape[0]).reshape(-1,1)
xtoy, ytoy = stdsc.fit_transform(x), stdsc.fit_transform(y)
# xtoy, ytoy = x,y#stdsc.fit_transform(x), stdsc.fit_transform(y)

# build model and train

options = {
    0:{'C':1., 'alg':'no_re'},
    1:{'C':1e10, 'alg':'solution1'},
    2:{'C':1e10, 'alg':'solution2'}
    }



if is_py:
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_num',help='Option number',default=0)
    parser.add_argument('-act_func',help='Activation function',default='relu')
    args = parser.parse_args()
    opt_num = int(args.opt_num)
    act_func = args.act_func

    

else:
    opt_num = 0
    act_func = 'relu'

model = elm(x= xtoy, y=ytoy, C = options[opt_num]['C'], hidden_units=512, activation_function=act_func, random_type='normal', elm_type='reg')

sys.stdout = open(f"logs/1d_stiff_reg_result_method_{opt_num}_{model.activation_function}.txt",'w')
beta, train_score, running_time = model.fit(options[opt_num]['alg'])#'no_re','solution1'
print("regression beta:\n", beta)
print("regression train score:\n", train_score)
print("regression running time:\n", running_time)

# test
prediction = model.predict(xtoy)
print("regression result: ", prediction.reshape(-1,))
print("regression score: ",model.score(xtoy,ytoy))
if is_py:
    sys.stdout.close()

# plot
plt.figure(figsize=(5,4),facecolor='white')
plt.plot(xtoy, ytoy, 'bo',label = 'data')
plt.plot(xtoy, prediction, 'r--', label = 'prediction')
plt.title('regression result')
plt.legend()
plt.savefig("figure/1d_stiff_reg_"+model.elm_type+f"_result_method_{opt_num}_{model.activation_function}.pdf")

# %%
