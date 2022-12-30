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
num_test_pts = 10000
y0 = 0.1
from ddeint import ddeint


def equation(Y, t):
    return a*Y(t)*(1-Y(t - tau))

def initial_history_func(t):
    return y0*np.ones_like(t)


plt.rcParams['font.size'] = 15
fig = plt.figure(figsize=(10,8),facecolor='white')
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
plt.title(f"$y'(t)=y(t)(1-y(t-{tau}))$ solved by ddeint")

ts = np.linspace(tl, tr, num_test_pts)

ys_ = ddeint(equation, initial_history_func, ts)
ys = [ys_[0]]
for i in range(1,len(ys_)):
    ys.append(ys_[i][0])


plt.plot(ts, ys_, color='blue',linestyle='--' ,linewidth=1,label='ddeint')
plt.grid()
plt.legend()
plt.show()
#%%

N_colloc = 200

ts_ = np.random.uniform(tl,tr,N_colloc).reshape(-1,1)
X_colloc = ts_

U_colloc = ~~~~~

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
    sys.stdout = open("logs/2d_"+model.elm_type+"_"+model.de_name+f"_result_method_{opt_num}_act_func_{model.activation_function}.txt",'w')
print("model options: ",model.option_dict)
beta, train_score, running_time = model.fit(options[opt_num]['alg'])#'no_re','solution1'
print("learned beta:\n", beta)
print("learned beta shape:\n", beta.shape)
print("test score:\n", train_score)
print("running time:\n", running_time)
#%%
# test
prediction = model.predict(X_test)
print("predicted result: ", prediction.shape)
#%%
if is_py:
    sys.stdout.close()
nx = xs.shape[1]
nt = ts.shape[1]
Xs, Ts = np.meshgrid(xs.flatten(), ts.flatten())
# plot
plt.figure(figsize=(10,3),facecolor='white',tight_layout='True')
plt.subplot(1,3,1)
plt.title("Prediction")
plt.contourf(Xs, Ts, prediction.reshape(nt,nx), 100, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplot(1,3,2)
plt.title("Exact")
plt.contourf(Xs, Ts, Us, 100, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplot(1,3,3)
plt.title("Absolute Error")
plt.contourf(Xs, Ts, np.abs(prediction.reshape(nt,nx)-Us), 100, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
plt.savefig("figure/2d_"+model.elm_type+"_"+model.de_name+f"_result_method_{opt_num}_act_func_{model.activation_function}.pdf")




# %%
t_snap = [0.0,0.20,0.40,0.60,0.80,1.00] # even number
plt.figure(figsize=(10,12),tight_layout=True)
for i in range(0,len(t_snap)):
    t_val = t_snap[i]
    plt.subplot(3,2, i+1)
    plt.title(f"t={t_val}",fontsize=20)
    plt.plot(xs.flatten(), Us[int((nt-1)*(t_val/(tr-tl))),:],lw=3,color='r')
    plt.plot(xs.flatten(), prediction.reshape(nt,nx)[int((nt-1)*(t_val/(tr-tl))),:],lw=3,color='b',ls='--')
    plt.xlabel('x',fontsize=15)
    plt.ylabel('u',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    
    plt.ylim([0,1.05])
plt.show()
plt.savefig("figure/2d_"+model.elm_type+"_"+model.de_name+f"_result_method_{opt_num}_act_func_{model.activation_function}_(snapshot).pdf")

# %%
