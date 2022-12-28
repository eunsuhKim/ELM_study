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
is_py = True



stdsc = StandardScaler()


# Linear Bivariate PDE 

print("Linear PDE problem (heat-diffusion)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load test dataset

kappa = 3.0

file = loadmat(f"heat_diffusion_kappa_{kappa}.mat")
xs = file['x']
ts = file['t']
Us = file['usol']
Xs, Ts = np.meshgrid(xs,ts)
X_test = np.concatenate([Xs.reshape(-1,1),Ts.reshape(-1,1)], axis = 1)
U_test = Us.reshape(-1,1)

# Make collocation points
xl = 0.0
xr = 1.0
tl = 0.0
tr = 1.0

N_colloc = 200

xs_ = np.random.uniform(xl,xr,N_colloc).reshape(-1,1)
ts_ = np.random.uniform(tl,tr,N_colloc).reshape(-1,1)
X_colloc = np.concatenate([xs_,ts_], axis = 1)

U_colloc = np.sin(np.pi*xs_)*np.exp(-np.pi**2/kappa* ts_)#np.zeros((N_colloc,1))

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
                random_type='normal', elm_type='pde',
                physic_param = [kappa])
if is_py:
    sys.stdout = open(f"logs/2d_pde_result_method_{opt_num}_act_func_{model.activation_function}.txt",'w')
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
plt.savefig("figure/2d_"+model.elm_type+f"_result_method_{opt_num}_act_func_{model.activation_function}.pdf")




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
plt.savefig("figure/2d_"+model.elm_type+f"_result_method_{opt_num}_act_func_{model.activation_function}_(snapshot).pdf")

# %%
