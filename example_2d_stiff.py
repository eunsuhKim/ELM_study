#%%
from elm import elm
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
is_py = True



stdsc = StandardScaler()


# Linear Bivariate PDE 

print("Regression 2d problem (sol of heat-diffusion)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load test dataset

# file = loadmat("heat_diffusion_kappa_3.0.mat")
# xs = file['x']
# ts = file['t']
# Us = file['usol']
nx = 50
nt = 51
xl = 0.0
xr = 1.0
tl = 0.0
tr = 1.0
xs = np.linspace(xl,xr,nx)
ts = np.linspace(tl,tr,nt)
Xs, Ts = np.meshgrid(xs,ts)
X_test = np.concatenate([Xs.reshape(-1,1),Ts.reshape(-1,1)], axis = 1)

def U(X):
    x = X[:,0:1]
    t = X[:,1:2]
    return ((x**2+t**2)<0.25)*np.ones_like(x)+\
         ((x**2+t**2)>0.25)*((x-t)>0.1)*0.01*np.ones_like(x)+\
          ((x**2+t**2)>0.25)*((x-t)<=0.1)*(-1.0)*np.ones_like(x)
U_test =U(X_test)
Us = U_test.reshape(nt,nx)
# U_test = Us.reshape(-1,1)

# Make collocation points
xl = 0.0
xr = 1.0
tl = 0.0
tr = 1.0

N_colloc = 10

xs_ = np.random.uniform(xl,xr,N_colloc).reshape(-1,1)
ts_ = np.random.uniform(tl,tr,N_colloc).reshape(-1,1)
X_colloc = np.concatenate([xs_,ts_], axis = 1)

U_colloc = np.zeros((N_colloc,1))

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
    parser.add_argument('-act_func',help='Activation function',default='relu')
    args = parser.parse_args()
    opt_num = int(args.opt_num)
    act_func = args.act_func

    

else:
    opt_num = 0
    act_func = 'relu'

model = elm(x= X_test, y=U_test, C = options[opt_num]['C'],
                hidden_units=2048, activation_function=act_func,
                random_type='normal', elm_type='reg')
sys.stdout = open(f"logs/2d_stiff_reg_result_method_{opt_num}_{model.activation_function}.txt",'w')
print("model option: ",model.option_dict)
beta, train_score, running_time = model.fit(options[opt_num]['alg'])#'no_re','solution1'
print("regression beta:\n", beta)
print("regression beta:\n", beta.shape)
print("regression train score:\n", train_score)
print("regression running time:\n", running_time)
#%%
# test
prediction = model.predict(X_test)
print("regression result: ", prediction.shape)
#%%
print("regression score: ",)
if is_py:
    sys.stdout.close()

Xs, Ts = np.meshgrid(xs.flatten(), ts.flatten())
# plot
plt.figure(figsize=(10,3),facecolor='white',tight_layout='True')
plt.subplot(1,3,1)
plt.title("Prediction")
plt.contourf(Xs, Ts, prediction.reshape(nt,nx), vmin=np.min(U_test), vmax=np.max(U_test))
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplot(1,3,2)
plt.title("Exact")
plt.contourf(Xs, Ts, Us, 100, vmin=np.min(U_test), vmax=np.max(U_test))
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplot(1,3,3)
plt.title("Absolute Error")
plt.contourf(Xs, Ts, np.abs(prediction.reshape(nt,nx)-Us), 100, vmin=np.min(U_test), vmax=np.max(U_test))
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()
plt.savefig("figure/2d_stiff_"+model.elm_type+f"_result_method_{opt_num}_{model.activation_function}.pdf")




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
    
    plt.ylim([np.min(U_test)-0.5,np.max(U_test)+0.5])
# plt.show()
plt.savefig("figure/2d_stiff_"+model.elm_type+f"_result_method_{opt_num}_{model.activation_function}_(snapshot).pdf")

# %%
