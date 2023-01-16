#%%
from elm_argon_class_scaled_neural_net import elm
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import scipy
from scipy.special import roots_legendre, eval_legendre
import argparse
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']=3
# plt.rcParams['figure.figsize']=(3,15)

#%%

is_py = False
is_save_figure = True
is_save_txt = False
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

N_colloc =100

#roots= roots_legendre(N_colloc-2)[0].reshape(-1,1)
#ts_ = (roots+1)/2*(tr-tl)+tl
xl= 0.0
xr= 2e-2
tl = 0.0
tr =  2e-10#4e-4
L = xr-xl
xs = onp.random.uniform(xl,xr,N_colloc)
# xs = onp.zeros(N_colloc)
# xs = L*onp.ones(N_colloc)
# xs = onp.linspace(xl, xr, N_colloc)
ts = onp.random.uniform(tl,tr,N_colloc)
# ts = onp.linspace(tl,tr,N_colloc)
# ts = onp.zeros(N_colloc)
# Xs, Ts = onp.meshgrid(xs,ts)
X_colloc = np.concatenate([xs.reshape(1,-1),ts.reshape(1,-1)],axis=0)

#%%
# build model and train


act_func_name = 'sigmoid' #sigmoid,sin

def random_generating_func_W(size):
    # return 1e5*onp.random.uniform(-1,1,size)
    return 1*onp.random.randn(*size)
def random_generating_func_b(size):
    # return onp.random.uniform(-1,1,size)
    return 1*onp.random.randn(*size)
def random_initializing_func_betaT(size):
    # return 1*onp.random.uniform(-1,1,size)
    return 1e5*onp.random.randn(*size)
p= 1.0
physics_param = {}
physics_param['L'] = L
physics_param['mu_e'] = 20/p
physics_param['gamma'] = 0.01
physics_param['D_e'] = 50/p
physics_param['D_i'] = 1e-2/p
physics_param['eps_0'] = scipy.constants.epsilon_0 # vacuume permittivity
physics_param['qe'] = scipy.constants.elementary_charge # elementary charge
physics_param['p'] = p

def mu_i(E):
    return (0.5740)*(1+0.66*np.sqrt(np.abs(E)*1e-2))**(-1)
    # return (0.5740)/(1+0.66*np.sqrt(np.sqrt(np.square(E)*1e-2)))
def alpha_iz(self,E):
    qe = self.physics_param['qe']
    p = self.physics_param['p']
    return 2922*p*qe**(-26.62*np.sqrt(p*np.abs(1e-2*E)**(-1)))
    # return 2922*p*qe**(-26.62*np.sqrt(p/np.sqrt(np.square(1e-2*E))))

physics_param['mu_i']=mu_i
physics_param['alpha_iz']=alpha_iz


model = elm(X=X_colloc,random_generating_func_W=random_generating_func_W,
                     random_generating_func_b=random_generating_func_b,act_func_name=act_func_name,
                     hidden_units=10, physics_param=physics_param,random_seed=random_seed,
                     quadrature=False,random_initializing_func_betaT=random_initializing_func_betaT)
if is_save_txt:
    sys.stdout = open(f"logs/argon_scaled_nn_act_func_{model.act_func_name}_N_colloc_{N_colloc}.txt",'w')

#%%
print("model options: ",model.option_dict)
print('N_colloc: ',N_colloc)

model.fit(num_iter =50)
#%%
print("learned beta:\n", model.betaT['ne'].sum())
print("learned beta:\n", model.betaT['ni'].sum())
print("learned beta:\n", model.betaT['V'].sum())
print("learned beta:\n", model.betaT['Gamma_i'].sum())
print("learned beta:\n", model.betaT['Gamma_e'].sum())
# print("learned beta shape:\n", model.betaT.shape)
print("test score:\n", model.train_score)


#%%

plt.figure(figsize=(10,8))
plt.semilogy(model.res_hist)

if is_save_figure:
    plt.savefig(f"figure/argon_scaled_nn_res_hist_act_func_{model.act_func_name}_N_colloc_{N_colloc}.pdf",bbox_inches='tight')
else:
    plt.show()


#%%
nx = 100
nt = 40
xs_test = np.linspace(xl,xr,nx)
ts_test = np.linspace(tl,tr,nt)
Xs_test,Ts_test = np.meshgrid(xs_test,ts_test)
X_test = np.concatenate([Xs_test.flatten().reshape(-1,1),Ts_test.flatten().reshape(-1,1)],axis=1)
# The following prediciton functions have arguments x, t.
ni, ne,V,Gamma_i,Gamma_e = model.prediction_functions()
def V_s(x,t):
    return V(x,t)[0,0]
E_s = grad(V_s,argnums=0)
E = vmap(E_s,in_axes=1,out_axes=1)

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
plt.figure(figsize=(30,6))
titles = ['$n_i$','$n_e$','E','$\\Gamma_i$','$\\Gamma_e$']
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.title(titles[i])
    plt.contourf(X_test[:,1].reshape(nt,nx),
    X_test[:,0].reshape(nt,nx),U_pred[:,i].reshape(nt,nx),100)
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
if is_save_figure:
    plt.savefig(f"figure/argon_scaled_nn_prediction_ni_ne_E_act_func_{model.act_func_name}_N_colloc_{N_colloc}.pdf",bbox_inches='tight')
else:
    plt.show()
plt.figure(figsize=(20,8))
for i in range(3,5):
    plt.subplot(1,2,i-2)
    plt.title(titles[i])
    plt.contourf(X_test[:,1].reshape(nt,nx),
    X_test[:,0].reshape(nt,nx),U_pred[:,i].reshape(nt,nx),100)
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
if is_save_figure:
    plt.savefig(f"figure/argon_scaled_nn_prediction_Gamma_i_Gamma_e_act_func_{model.act_func_name}_N_colloc_{N_colloc}.pdf",bbox_inches='tight')
else:
    plt.show()

#%%
#%%
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
    plt.savefig(f"figure/argon_scaled_result_act_func_{model.act_func_name}.pdf")
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
savemat(f"each_time_interval/argon_scaled_result_act_func_{model.act_func}_N_colloc_{N_colloc}.mat",saving_dict)
