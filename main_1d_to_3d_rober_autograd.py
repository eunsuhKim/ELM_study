#%%
from elm_autograd_physics_1d_to_3d import elm
import numpy as onp
import time
import jax.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import os 
import jax
jax.config.update("jax_enable_x64", True)
os.environ['CUDA_DEVICE_0_RDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from scipy.special import roots_legendre, eval_legendre
import argparse
#%%
is_py = False
is_save = True




# Linear Bivariate PDE 

print("ROBER Problem >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load test dataset


# Make collocation points
k1 = 0.04
k2 = 3*1e7
k3 = 1e4


from scipy.io import loadmat 
file = loadmat("dataset/rober.mat")

upto_here = 51#12 #among 51
X_test =file['t'].reshape(-1,1)[:upto_here,:]
U_test =file['y'].reshape(-1,3)[:upto_here,:]


tl = np.min(X_test)
tr = np.max(X_test)

#%%
num_test_pts = U_test.shape[0]

x0 = 1.0
y0 = 0.0
z0 = 0.0

def ROBER_PDE(u,t):
    x, y, z = u
    dxdt = - k1*x + k3 * y * z
    dydt =   k1*x - k3 * y * z - k2 * y**2
    dzdt =                       k2 * y**2
    return dxdt, dydt, dzdt

u0 = x0, y0, z0


plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']=3
plt.rcParams['figure.figsize']=(10,8)

#%%
plt.figure()
plt.plot(X_test, U_test[:,0:1],label='x')
# plt.plot(X_test, U_test[:,1:2],label='y')
plt.plot(X_test, U_test[:,2:3],label='z',color='green')
# plt.plot(t, u, color='blue',linestyle='--' ,linewidth=1,label='ddeint')
plt.grid()
plt.legend()
plt.xscale("log")
# plt.xlim([1e-5,1e5])
if is_save:
    plt.savefig("figure/rober_goal_xz"+f"[{tl:.5f},{tr:.5f}].pdf",bbox_inches='tight')
else:
    plt.show()
#%%
plt.figure()
plt.plot(X_test, U_test[:,1:2],label='y',color='coral')
plt.grid()
plt.legend()
plt.xscale("log")
# plt.xlim([1e-5,1e5])
if is_save:
    plt.savefig("figure/rober_goal_y"+f"[{tl:.5f},{tr:.5f}].pdf",bbox_inches='tight')
else:
    plt.show()

#%%
seed = int(time.time())
print('Colloc random seed:',seed)
onp.random.seed(seed)
N_colloc =15

roots= roots_legendre(N_colloc)[0].reshape(-1,1)
ts_ = (roots+1)/2*(tr-tl)
# ts_ = np.random.uniform(tl,tr,N_colloc).reshape(-1,1)
# ts_ = np.linspace(tl,tr,N_colloc).reshape(-1,1)
X_colloc = ts_

U_colloc = np.zeros_like(X_colloc).repeat(3,1)


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
    parser.add_argument('-act_func',help='Activation function',default='sin')
    args = parser.parse_args()
    opt_num = int(args.opt_num)
    act_func = args.act_func

    
else:
    opt_num = 0
    act_func = 'sin'
model = elm(x= X_colloc, y=U_colloc, C = options[opt_num]['C'],
            hidden_units=10, activation_function=act_func,
            random_type='uniform', elm_type='de',de_name='rober',
            quadrature =True,
            physic_param = [k1,k2,k3], initial_val = u0,
            random_seed = seed,Wscale=None, bscale=None,fourier_embedding=False)
if is_save:
    sys.stdout = open("logs/"+model.de_name+f"[{tl:.5f},{tr:.5f}](using_autograd)_result_method_{opt_num}_act_func_{model.activation_function}.txt",'w')
#%%
print("model options: ",model.option_dict)
print('N_colloc: ',N_colloc)

beta, train_score, running_time = model.fit(
    algorithm=options[opt_num]['alg'],
    num_iter =100)#'no_re','solution1'
print("learned beta:\n", beta)
print("learned beta shape:\n", beta.shape)
print("test score:\n", train_score)
print("running time:\n", running_time)
plt.figure()
plt.semilogy(model.res_hist)

if is_save:
    plt.savefig("figure/"+model.de_name+f"[{tl:.5f},{tr:.5f}](using_autograd)_residual_history_method_{opt_num}_act_func_{model.activation_function}.pdf")
else:
    plt.show()
#%%
# test


U_pred = model.predict(X_test)
print("predicted result: ", U_pred.shape)


err = np.abs(U_pred-U_test)
err = np.linalg.norm(err)/np.linalg.norm(U_test)
print("Relative L2-error norm: {}".format(err))
#%%
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']=3
plt.figure(figsize=(10,8), facecolor = 'white')
plt.title(f"ELM for ROBER problem")
plt.plot(X_test,U_test[:,0:1],color='coral',label='exact x')
# plt.plot(X_test,U_test[:,1:2],color = 'lightgreen',label='exact y')
plt.plot(X_test,U_test[:,2:3],'black',label='exact z')
plt.plot(X_test,U_pred[:,0:1],color='b',ls='dashdot',label='pred x')
# plt.plot(X_test,U_pred[:,1:2],'r',ls = 'dotted',label='pred y')
plt.plot(X_test,U_pred[:,2:3],'orange',ls='--',label='pred z')
plt.legend(loc=2)
plt.xscale("log")
plt.grid()
plt.ylim([-0.1,1.5])
plt.xlabel('t')


if is_save:
    plt.savefig("figure/"+model.de_name+f"[{tl:.5f},{tr:.5f}](using_autograd)xz_result_method_{opt_num}_act_func_{model.activation_function}.pdf")
else:
    plt.show()
#%%
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']=3
plt.figure(figsize=(10,8), facecolor = 'white')
plt.title(f"ELM for ROBER problem")
# plt.plot(X_test,U_test[:,0:1],color='coral',label='exact x')
plt.plot(X_test,U_test[:,1:2],color = 'lightgreen',label='exact y')
# plt.plot(X_test,U_test[:,2:3],'black',label='exact z')
# plt.plot(X_test,U_pred[:,0:1],color='b',ls='dashdot',label='pred x')
plt.plot(X_test,U_pred[:,1:2],'r',ls = 'dotted',label='pred y')
# plt.plot(X_test,U_pred[:,2:3],'orange',ls='--',label='pred z')
plt.legend(loc=2)
plt.xscale("log")
plt.ylim([-1e-5,1e-4])
plt.grid()
plt.xlabel('t')


if is_save:
    plt.savefig("figure/"+model.de_name+f"[{tl:.5f},{tr:.5f}](using_autograd)y_result_method_{opt_num}_act_func_{model.activation_function}.pdf")
else:
    plt.show()
#%%
if is_save:
    sys.stdout.close()

    

# %%
