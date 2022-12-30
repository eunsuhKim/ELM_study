#%%
from elm_direct_physics import elm
import numpy as np

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
is_py = False



stdsc = StandardScaler()


# Linear Bivariate PDE 

print("Nonlinear DDE Logistic >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load test dataset


# Make collocation points
a =1.4 #0.3 
tau = 1.5#0.0

tl = 0.0
tr = 50.0
num_test_pts = 1000
y0 = 0.1
from ddeint import ddeint

from pylab import linspace
import numpy as np
# def equation(Y, t, a, tau):
#     return np.array([a*Y(t)*(1-Y(t - tau))])

# def initial_history_func(t):
#     return np.array([y0*t**0])
equation = lambda Y, t,a , tau: np.array([a*Y(t)*(1-Y(t-tau))])
initial_history_func = lambda t: y0



ts = np.linspace(tl, tr, num_test_pts)#.astype(np.float64)

ys_ = ddeint(equation, initial_history_func, ts,fargs=(a,tau,))
ys = [ys_[0]]
for i in range(1,len(ys_)):
    ys.append(ys_[i][0])


plt.rcParams['font.size'] = 15
fig = plt.figure(figsize=(10,8),facecolor='white')
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)


plt.title(f"$y'(t)=y(t)(1-y(t-{tau}))$ solved by ddeint")



plt.plot(ts, ys, color='red', linewidth=1,label='odeint')
# plt.plot(t, u, color='blue',linestyle='--' ,linewidth=1,label='ddeint')
plt.grid()
plt.legend()
plt.show()

X_test = ts.reshape(-1,1)
U_test = ys_.reshape(-1,1)

#%%

N_colloc = 200

ts_ = np.random.uniform(tl,tr,N_colloc).reshape(-1,1)
X_colloc = ts_

U_colloc = np.zeros_like(X_colloc)

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
model = elm(x= X_colloc, y=U_colloc, C = options[opt_num]['C'],
            hidden_units=128, activation_function=act_func,
            random_type='normal', elm_type='de',de_name='dde_logistic',
            history_func=initial_history_func,
            physic_param = [a])
if is_py:
    sys.stdout = open("logs/"+model.elm_type+"_"+model.de_name+f"_result_method_{opt_num}_act_func_{model.activation_function}.txt",'w')
print("model options: ",model.option_dict)
beta, train_score, running_time = model.fit(options[opt_num]['alg'])#'no_re','solution1'
print("learned beta:\n", beta)
print("learned beta shape:\n", beta.shape)
print("test score:\n", train_score)
print("running time:\n", running_time)
#%%
# test
U_pred = model.predict(X_test)
print("predicted result: ", U_pred.shape)


err = np.abs(U_pred-U_test)
err = np.linalg.norm(err)/np.linalg.norm(U_test)
print("Relative L2-error norm: {}".format(err))


plt.figure(figsize=(10,8), facecolor = 'white')
plt.title(f"ELM for Logistic DDE SIR with delay={tau}")
plt.plot(X_test.cpu(),U_test.cpu()[:,0:1],color='coral',label='exact')
# plt.plot(X_test.cpu(),U_test.cpu()[:,1:2],color = 'lightgreen',label='exact Infected')
# plt.plot(X_test.cpu(),U_test.cpu()[:,2:3],'black',label='exact Recovered')
plt.plot(X_test.cpu(),U_pred.cpu()[:,0:1],color='b',ls='dashdot',label='pred')
# plt.plot(X_test.cpu(),U_pred.cpu()[:,1:2],'r',ls = 'dotted',label='predicted Infected')
# plt.plot(X_test.cpu(),U_pred.cpu()[:,2:3],'orange',ls='--',label='predicted Recovered')
plt.legend(loc=2)
plt.grid()
plt.xlabel('t')
plt.show()

# plt.savefig("figure/"+model.elm_type+"_"+model.de_name+f"_result_method_{opt_num}_act_func_{model.activation_function}_(snapshot).pdf")

#%%
if is_py:
    sys.stdout.close()

    

# %%