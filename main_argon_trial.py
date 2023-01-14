#%%
from elm_argon_class import elm
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


from scipy.linalg import inv

from jax import jacfwd, vmap, grad, jvp, vjp

jax.config.update("jax_enable_x64", True)
os.environ['CUDA_DEVICE_0_RDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import scipy
from scipy.special import roots_legendre, eval_legendre
import argparse
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']=3
# plt.rcParams['figure.figsize']=(3,15)

#%%

is_py = False
is_save = False
#%%
# Load test data

from scipy.io import loadmat 
# nx = 1e2
# nt = 1e9
# file = loadmat(f"dataset/argon_nx_{nx}_nt_{nt}.mat")

# X_test =file['t'].reshape(-1,2)
# U_test =file['y'].reshape(-1,5)


# xl = np.min(X_test[:,0])
# xr = np.max(X_test[:,0])
# tl = np.min(X_test[:,1])
# tr = np.max(X_test[:,1])

# L = xr-xl

# num_test_pts = U_test.shape[0]
#%%
random_seed = int(time.time())
print('Colloc random seed:',random_seed)
onp.random.seed(random_seed)

N_colloc =10

#roots= roots_legendre(N_colloc-2)[0].reshape(-1,1)
#ts_ = (roots+1)/2*(tr-tl)+tl
xl= 0.0
xr= 2e-2
tl = 0.0
tr =  4e-4
L = xl-xr
xs = onp.random.uniform(xl,xr,N_colloc)
ts = onp.random.uniform(tl,tr,N_colloc)
# Xs, Ts = onp.meshgrid(xs,ts)
X_colloc = np.concatenate([xs.reshape(1,-1),ts.reshape(1,-1)],axis=0)

#%%
# build model and train


act_func_name = 'sin'

def random_generating_func_W(size):
    return onp.random.uniform(-1,1,size)
def random_generating_func_b(size):
    return onp.random.uniform(-1,1,size)

p= 1.0
physics_param = {}
physics_param['L'] = L
physics_param['mu_e'] = 20/p
physics_param['gamma'] = 0.01
physics_param['D_e'] = 50/p
physics_param['D_i'] = 1e-2/p
physics_param['eps_0'] = scipy.constants.epsilon_0 # vacuume permittivity
physics_param['qe'] = scipy.constants.elementary_charge # elementary charge
physics_param['p'] = 0

def mu_i(E):
    # return (0.5740)/(1+0.66*np.sqrt(np.abs(E)*1e-2))
    return (0.5740)/(1+0.66*np.sqrt(np.sqrt(np.square(E)*1e-2)))
def alpha_iz(self,E):
    qe = self.physics_param['qe']
    p = self.physics_param['p']
    # return 2922*p*qe**(-26.62*np.sqrt(p/np.abs(E/100)))
    return 2922*p*qe**(-26.62*np.sqrt(p/np.sqrt(np.square(1e-2*E))))

physics_param['mu_i']=mu_i
physics_param['alpha_iz']=alpha_iz


model = elm(X=X_colloc,random_generating_func_W=random_generating_func_W,
                     random_generating_func_b=random_generating_func_b,act_func_name=act_func_name,
                     hidden_units=32, physics_param=physics_param,random_seed=random_seed,
                     quadrature=False)
if is_save:
    sys.stdout = open(f"logs/argon_act_func_{model.act_func}.txt",'w')

#%%
print("model options: ",model.option_dict)
print('N_colloc: ',N_colloc)

model.fit(num_iter =10)
print("learned beta:\n", model.beta)
print("learned beta shape:\n", model.beta.shape)
print("test score:\n", model.train_score)

plt.figure(figsize=(10,8))
plt.semilogy(model.res_hist)

if is_save:
    plt.savefig(f"figure/argon_res_hist_act_func_{model.act_func}.pdf")
else:
    plt.show()

#%%


# The following prediciton functions have arguments x, t.
ni, ne,V,Gamma_i,Gamma_e = model.prediction_functions()
E = grad(V,argnum=0)

ni_pred = ni(X_test[:,0].reshape(1,-1),X_test[:,1].reshape(1,-1))
ne_pred = ne(X_test[:,0].reshape(1,-1),X_test[:,1].reshape(1,-1))
V_pred = V(X_test[:,0].reshape(1,-1),X_test[:,1].reshape(1,-1))
E_pred = E(X_test[:,0].reshape(1,-1),X_test[:,1].reshape(1,-1))
Gamma_i_pred = Gamma_i(X_test[:,0].reshape(1,-1),X_test[:,1].reshape(1,-1))
Gamma_e_pred = Gamma_e(X_test[:,0].reshape(1,-1),X_test[:,1].reshape(1,-1))

U_pred = np.concatenate([ni_pred,ne_pred,E_pred,Gamma_i_pred,Gamma_e_pred],axis=0).T
# err = np.abs(U_pred-U_test)
# err = np.linalg.norm(err)/np.linalg.norm(U_test)
# print("Relative L2-error norm: {}".format(err))
plt.figure(figsize=(3,15))

for i in range(5):
    plt.subplot(5,1,i)
    plt.contourf(X_test[:,1],X_test[:,0],U_pred[:,i].reshape(nt,nx),100)
    plt.colorbar()
plt.show()

#%%
plt.figure(figsize=(8,10))

for i in range(0,15,3):
    plt.subplot(5,3,i)
    plt.contourf(X_test[:,1],X_test[:,0],U_test[:,i].reshape(nt,nx),100)
    plt.colorbar()
    
    plt.subplot(5,3,i+1)
    plt.contourf(X_test[:,1],X_test[:,0],U_pred[:,i].reshape(nt,nx),100)
    plt.colorbar()
    
    plt.subplot(5,3,i+2)
    plt.contourf(X_test[:,1],X_test[:,0],np.abs(U_pred[:,i]-U_test[:,i]).reshape(nt,nx),100)
    plt.colorbar()
    
plt.show()

if is_save:
    plt.savefig(f"figure/argon_result_act_func_{model.act_func}.pdf")
else:
    plt.show()

#%%
from scipy.io import savemat
saving_dict = {}
# saving_dict['t_final']=X_test[-1,0]
# saving_dict['x0']=U_pred[-1,0].item()
# saving_dict['y0']=U_pred[-1,1].item()
# saving_dict['z0']=U_pred[-1,2].item()
# saving_dict['tl']=tl
# saving_dict['tr']=tr
# saving_dict['start_idx']=start_idx
# saving_dict['end_idx']=end_idx
saving_dict['X_test']=onp.array(X_test)
saving_dict['X_colloc']=onp.array(X_colloc)
saving_dict['U_test']=onp.array(U_test)
saving_dict['U_pred']=onp.array(U_pred)
saving_dict['W']= onp.array(model.W)
saving_dict['b']= onp.array(model.b)
saving_dict['beta']= onp.array(model.beta)
savemat(f"each_time_interval/argon_result_act_func_{model.act_func}.mat",saving_dict)