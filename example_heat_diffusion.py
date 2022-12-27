#%%
from elm import elm
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat

import argparse
#%%
is_py = False

if is_py:
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_num',help='Option number',default=0)
    args = parser.parse_args()
    opt_num = int(args.opt_num)

    sys.stdout = open('logs/result_txts.txt','w')

else:
    opt_num = 0

stdsc = StandardScaler()


# Linear Bivariate PDE 

print("Linear PDE problem (heat-diffusion)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load test dataset

file = loadmat("heat_diffusion_kappa_3.0.mat")
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

N_colloc = 100

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

model = elm(x= X_colloc, y=U_colloc, C = options[opt_num]['C'],
                hidden_units=32, activation_function='sigmoid',
                random_type='normal', elm_type='pde')
beta, train_score, running_time = model.fit(options[opt_num]['alg'])#'no_re','solution1'
print("regression beta:\n", beta)
print("regression beta:\n", beta.shape)
print("regression train score:\n", train_score)
print("regression running time:\n", running_time)
#%%
# test
prediction = model.predict(xtoy)
print("regression result: ", prediction.reshape(-1,))
#%%
print("regression score: ",)

sys.stdout.close()

# plot
plt.figure(figsize=(5,4),facecolor='white')
plt.plot(xtoy, ytoy, 'bo',label = 'data')
plt.plot(xtoy, prediction, 'r--', label = 'prediction')
plt.title('regression result')
plt.legend()
plt.savefig("figure/"+model.elm_type+"_result.pdf")





# regression ~probelm


print("regression problem>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# generate dataset
x = np.arange(0.25, 20, 0.1).reshape(-1,1)
y = x*np.cos(x) + 0.5*np.sqrt(x)*np.random.randn(x.shape[0]).reshape(-1,1)
xtoy, ytoy = stdsc.fit_transform(x), stdsc.fit_transform(y)

# build model and train

options = {
    0:{'C':1., 'alg':'no_re'},
    1:{'C':1e16, 'alg':'solution1'},
    2:{'C':1e16, 'alg':'solution2'}
    }

model = elm(x= xtoy, y=ytoy, C = options[opt_num]['C'], hidden_units=32, activation_function='sigmoid', random_type='normal', elm_type='reg')
beta, train_score, running_time = model.fit(options[opt_num]['alg'])#'no_re','solution1'
print("regression beta:\n", beta)
print("regression train score:\n", train_score)
print("regression running time:\n", running_time)

# test
prediction = model.predict(xtoy)
print("regression result: ", prediction.reshape(-1,))
print("regression score: ",model.score(xtoy,ytoy))

sys.stdout.close()

# plot
plt.figure(figsize=(5,4),facecolor='white')
plt.plot(xtoy, ytoy, 'bo',label = 'data')
plt.plot(xtoy, prediction, 'r--', label = 'prediction')
plt.title('regression result')
plt.legend()
plt.savefig("figure/"+model.elm_type+"_result.pdf")

# %%
