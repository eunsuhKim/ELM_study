#%%
from elm_nonlin_reg import elm
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import os 
os.environ['CUDA_DEVICE_0_RDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
#%%
is_py = True



stdsc = StandardScaler()


# Linear Bivariate PDE 

print("Nonlinear Regression >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load test dataset
tl = 0.25
tr = 20
x = np.arange(tl,tr, 0.1).reshape(-1,1)
# x = np.random.uniform(0.25, 20, int((20-0.25)/0.1)).reshape(-1,1)
x = np.sort(x,axis=0)
# print(x)
y = (x<5)*0.01*np.ones_like(x)+(x>= 5)*(x<12)*(50*np.sin(x))+(x>=12)*(10*np.ones_like(x))#x*np.cos(x) + 0.5*np.sqrt(x)*np.random.randn(x.shape[0]).reshape(-1,1)
# y = (x<5)*0.01*np.ones_like(x)+(x>= 5)*(x<12)*(50*x)+(x>=12)*(10*np.ones_like(x))#x*np.cos(x) + 0.5*np.sqrt(x)*np.random.randn(x.shape[0]).reshape(-1,1)
xtoy, ytoy = stdsc.fit_transform(x), stdsc.fit_transform(y)
# xtoy, ytoy = x,y#stdsc.fit_transform(x), stdsc.fit_transform(y)

# build model and train

#%%
# from scipy.io import savemat
# saving_dict = {}
# saving_dict['t'] = X_test
# saving_dict['u'] = U_test
# savemat(f"dataset/dde_logistic_{a}_{tau}_.mat",saving_dict)

#%%
# from scipy.io import loadmat
# file = loadmat(f"dataset/dde_logistic_{a}_{tau}_.mat")
# X_test = file['t']
# U_test = file['u']
plt.plot(xtoy, ytoy, color='red', linewidth=1,label='ddeint')
# plt.plot(t, u, color='blue',linestyle='--' ,linewidth=1,label='ddeint')
plt.grid()
plt.legend()
plt.show()
#%%



#%%
# build model and train

options = {
    0:{'C':1., 'alg':'no_re'},
    1:{'C':1e16, 'alg':'solution1'},
    2:{'C':1e16, 'alg':'solution2'}
    }
if is_py:
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_num',help='Option number',default=0)
    parser.add_argument('-act_func',help='Activation function',default='tanh')
    args = parser.parse_args()
    opt_num = int(args.opt_num)
    act_func = args.act_func

    
else:
    opt_num = 0
    act_func = 'tanh'
model = elm(x= xtoy, y=ytoy, C = options[opt_num]['C'],
            hidden_units=500, activation_function=act_func,
            random_type='uniform', elm_type='nonlin_reg',
            random_seed = int(time.time()),Wscale=10, bscale=0.01)
if is_py:
    sys.stdout = open("logs/"+model.elm_type+f"_result_method_{opt_num}_act_func_{model.activation_function}.txt",'w')
#%%
print("model options: ",model.option_dict)
beta, train_score, running_time = model.fit(algorithm=options[opt_num]['alg'],
                                            num_iter = 5)#'no_re','solution1'
print("learned beta mean:\n", beta.mean())
print("learned beta shape:\n", beta.shape)
print("test score:\n", train_score)
print("running time:\n", running_time)
#%%
# test
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

# plt.show()

plt.savefig("figure/"+model.elm_type+f"_result_method_{opt_num}_act_func_{model.activation_function}_(snapshot).pdf")


#%%
if is_py:
    sys.stdout.close()

    

