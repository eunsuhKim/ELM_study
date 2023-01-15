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

import scipy
from scipy.special import roots_legendre, eval_legendre

class elm():
    def __init__(self,X=None,random_generating_func_W=None,
                     random_generating_func_b=None,act_func_name='sin',
                     hidden_units=32, physics_param=None,random_seed=None,
                     quadrature=True,random_initializing_func_betaT=None):
        # save and print options
        option_dict = {}
        option_dict['random_generating_func_W'] = random_generating_func_W
        option_dict['random_generating_func_b'] = random_generating_func_b
        option_dict['random_initializing_func_betaT'] = random_initializing_func_betaT

        option_dict['act_func_name'] = act_func_name
        option_dict['hidden_units'] = hidden_units
        option_dict['physics_param']=physics_param
        option_dict['random_seed'] = random_seed
        option_dict['quadrature']=quadrature
        option_dict['input_dim']=X.shape[0]
        option_dict['output_dim']=5
        self.option_dict = option_dict
        # print(self.option_dict)
        
        # save options in model
        self.random_generating_func_W = random_generating_func_W
        self.random_generating_func_b = random_generating_func_b
        self.random_initializing_func_betaT = random_initializing_func_betaT
        self.act_func_name = act_func_name
        self.hidden_units = hidden_units
        self.input_dim = X.shape[0]
        self.output_dim = 5
        self.sample_size = X.shape[1]
        self.random_seed = random_seed
        self.quadrature = quadrature
        self.physics_param = physics_param
        onp.random.seed(random_seed)
        print('Random seed: ',random_seed)
        
        # set and initialize model parameters
        self.X = X
        self.set_W_b()
        self.set_act_func()
        self.init_betaT()
        

    def set_W_b(self):
        self.W = {}
        self.b = {}
        self.W['ni'] = self.random_generating_func_W(size=(self.hidden_units,self.input_dim))
        self.b['ni'] = self.random_generating_func_b(size=(self.hidden_units,1))
        self.W['ne'] = self.random_generating_func_W(size=(self.hidden_units,self.input_dim))
        self.b['ne'] = self.random_generating_func_b(size=(self.hidden_units,1))
        self.W['V'] = self.random_generating_func_W(size=(self.hidden_units,self.input_dim))
        self.b['V'] = self.random_generating_func_b(size=(self.hidden_units,1))
        self.W['Gamma_i'] = self.random_generating_func_W(size=(self.hidden_units,self.input_dim))
        self.b['Gamma_i'] = self.random_generating_func_b(size=(self.hidden_units,1))
        self.W['Gamma_e'] = self.random_generating_func_W(size=(self.hidden_units,self.input_dim))
        self.b['Gamma_e'] = self.random_generating_func_b(size=(self.hidden_units,1))
        
    def set_act_func(self):
        if self.act_func_name == 'sigmoid':
            self.act_func = lambda x: 1/(1+np.exp(-x))
        if self.act_func_name =='relu':
            self.act_func = lambda x: x * (x>0)
        if self.act_func_name == 'tanh':
            self.act_func = lambda x: np.tanh(x)
        if self.act_func_name == 'leaky_relu':
            self.act_func = lambda x: x*(x>0)+0.1*x*(x<0)
        if self.act_func_name == 'sin':
            self.act_func = lambda x: np.sin(x)
            
    def init_betaT(self):
        self.betaT = {}
        # self.betaT['ni'] = np.array(onp.random.randn(1,self.hidden_units))
        # self.betaT['ne'] = np.array(onp.random.randn(1,self.hidden_units))
        # self.betaT['V'] = np.array(onp.random.randn(1,self.hidden_units))
        # self.betaT['Gamma_i'] = np.array(onp.random.randn(1,self.hidden_units))
        # self.betaT['Gamma_e'] = np.array(onp.random.randn(1,self.hidden_units))
        self.betaT['ni'] = self.random_initializing_func_betaT((1,self.hidden_units))
        self.betaT['ne'] = self.random_initializing_func_betaT((1,self.hidden_units))
        self.betaT['V'] = self.random_initializing_func_betaT((1,self.hidden_units))
        self.betaT['Gamma_i'] = self.random_initializing_func_betaT((1,self.hidden_units))
        self.betaT['Gamma_e'] = self.random_initializing_func_betaT((1,self.hidden_units))
        
    def make_beta(self,num_iter = 10):
        x, t = self.X
        x = x.reshape(1,-1)
        t = t.reshape(1,-1)
#         if self.quadrature == True:
#             weights = roots_legendre(self.sample_size)[1].reshape(1,-1)
#         else: 
#             weights = np.ones((1,self.sample_size))
        def N(betaTs):
            beta_ni, beta_ne, beta_V, beta_Gamma_i, beta_Gamma_e = betaTs
            self.betaT['ni'] = beta_ni.reshape(1,-1)
            self.betaT['ne'] = beta_ne.reshape(1,-1)
            self.betaT['V'] = beta_V.reshape(1,-1)
            self.betaT['Gamma_i'] = beta_Gamma_i.reshape(1,-1)
            self.betaT['Gamma_e'] = beta_Gamma_e.reshape(1,-1)
            # CE_ni_s,CE_ne_s,CE_V_s,CE_Gamma_i_s,CE_Gamma_e_s = self.prediction_functions_scalar()
            CE_ni,CE_ne,CE_V,CE_Gamma_i,CE_Gamma_e = self.prediction_functions()
            def CE_ni_s(X,T):
                return CE_ni(X,T)[0,0]
            def CE_ne_s(X,T):
                return CE_ne(X,T)[0,0]
            def CE_V_s(X,T):
                return CE_V(X,T)[0,0]
            def CE_Gamma_i_s(X,T):
                return CE_Gamma_i(X,T)[0,0]
            def CE_Gamma_e_s(X,T):
                return CE_Gamma_e(X,T)[0,0]
            # CE_ni_s = lambda x,t: CE_ni(x,t)[0,0]
            # CE_ne_s = lambda x,t: CE_ne(x,t)[0,0]
            # CE_V_s = lambda x,t: CE_V(x,t)[0,0]
            # CE_Gamma_i_s = lambda x,t: CE_Gamma_i(x,t)[0,0]
            # CE_Gamma_e_s = lambda x,t: CE_Gamma_e(x,t)[0,0]

            ni = CE_ni(x,t)
            ne = CE_ne(x,t)
            ni_x_s = grad(CE_ni_s, argnums=0)
            ni_t_s = grad(CE_ni_s, argnums=1)
            ne_x_s = grad(CE_ne_s, argnums=0)
            ne_t_s= grad(CE_ne_s, argnums=1)
            ni_x = vmap(ni_x_s,in_axes=1,out_axes=1)
            ni_t = vmap(ni_t_s,in_axes=1,out_axes=1)
            ne_x = vmap(ne_x_s,in_axes=1,out_axes=1)
            ne_t = vmap(ne_t_s,in_axes=1,out_axes=1)
            # V = CE_V(x,t)
            Gamma_i = CE_Gamma_i(x,t)
            Gamma_e = CE_Gamma_e(x,t)
            Gamma_i_x_s = grad(CE_Gamma_i_s,argnums=0)
            Gamma_e_x_s = grad(CE_Gamma_e_s,argnums=0)
            Gamma_i_x = vmap(Gamma_i_x_s,in_axes=1,out_axes=1)
            Gamma_e_x = vmap(Gamma_e_x_s,in_axes=1,out_axes=1)
            
            mE_s_ = grad(CE_V_s, argnums=0)
            
            
            mE_ = vmap(mE_s_)
            mE = vmap(mE_s_,in_axes=1,out_axes=1)
            def mE_real_scalar(X,T):
                return mE_(X,T)[0]
            mE_x_s = grad(mE_real_scalar,argnums=0)
            mE_x = vmap(mE_x_s,in_axes=1,out_axes=1)
            
            # alpha_iz val and mu_i funciton were problematic.
            # Some vales of Gamma_e and Gamma_i are nan
            # ni_t,ne_t,V are zero
            res_1 = ni_t(x,t) + Gamma_i_x(x,t) - self.physics_param['alpha_iz'](self,-mE(x,t))*Gamma_e
            res_2 = ne_t(x,t) + Gamma_e_x(x,t) - self.physics_param['alpha_iz'](self,-mE(x,t))*Gamma_e
            res_3 = Gamma_i - self.physics_param['mu_i'](-mE(x,t))*(-mE(x,t)) + self.physics_param['D_i']*ni_x(x,t)
            # res_4 and res__5 only not NAN
            res_4 = Gamma_e + self.physics_param['mu_e']*(-mE(x,t))*ne + self.physics_param['D_e']*ne_x(x,t)
            res_5 = -mE_x(x,t) - self.physics_param['qe']/self.physics_param['eps_0'] *(ni-ne)
            res_mat = np.concatenate([res_1,res_2,res_3,res_4,res_5],axis=0)
            return res_mat
        self.N = N

        J = jacfwd(N) 
        # J originally has shape (self.output_dim,self.sample_size,self.output_dim,self.hidden_units)
        # real beta T with shape (self.output_dim,self.hidden_units)
         
        # for solving matrix equation (J.T@J)betaT_ =J.T@deltay
        betaTs = np.concatenate([self.betaT['ni'],self.betaT['ne'],self.betaT['V'],self.betaT['Gamma_i'],self.betaT['Gamma_e']],axis=0)
        betaTs_ = betaTs.reshape(self.output_dim*self.hidden_units,1)
        self.res_hist = []
        start = time.time()
        
        for i in range(num_iter):
            J_ = J(betaTs).reshape(self.output_dim*self.sample_size,
                                self.output_dim*self.hidden_units)
            deltay_ = -N(betaTs).reshape(self.output_dim*self.sample_size,1)
            delta_beta_ = np.linalg.solve(J_.T@J_+1e-6*np.eye(self.hidden_units*self.output_dim),J_.T@deltay_)
            betaTs = betaTs + delta_beta_.reshape(self.output_dim,self.hidden_units)
            betaTs_ = betaTs_ + delta_beta_
            train_score = np.mean(np.abs(self.N(betaTs)))
            self.res_hist.append(train_score)
            self.betaT['ni'],self.betaT['ne'],self.betaT['V'],self.betaT['Gamma_i'],self.betaT['Gamma_e'] = betaTs
            self.betaT['ni'] = self.betaT['ni'].reshape(1,-1)
            self.betaT['ne'] = self.betaT['ne'].reshape(1,-1)
            self.betaT['V'] = self.betaT['V'].reshape(1,-1)
            self.betaT['Gamma_i'] = self.betaT['Gamma_i'].reshape(1,-1)
            self.betaT['Gamma_e'] = self.betaT['Gamma_e'].reshape(1,-1)
            if i%1 == 0:
                print(f'Train_score when iter={i}: {train_score}')
        
        print(time.time()-start,' seconds cost for nonlinear least square.')
        
    def fit(self, num_iter = 10):
        # make beta using nonlinear least square
        self.make_beta(num_iter = num_iter)
        
        # calculate train score
        betaTs = np.concatenate([self.betaT['ni'],self.betaT['ne'],self.betaT['V'],self.betaT['Gamma_i'],self.betaT['Gamma_e']],axis=0)
        
        self.train_score = np.mean(np.abs(self.N(betaTs)))
        
        
    def sigma(self,X,token=None):
        return self.act_func(self.W[token] @ X + self.b[token])
    
    def prediction_functions(self):#, betaT_ni,betaT_ne, betaT_V, betaT_Gamma_i, betaT_Gamma_e):
        NN_ni = lambda x,t: self.betaT['ni'] @ self.sigma(np.concatenate([x.reshape(1,-1),t.reshape(1,-1)],axis=0),token='ni')
        NN_ne = lambda x,t: self.betaT['ne'] @ self.sigma(np.concatenate([x.reshape(1,-1),t.reshape(1,-1)],axis=0),token='ne')
        NN_V = lambda x,t: self.betaT['V'] @ self.sigma(np.concatenate([x.reshape(1,-1),t.reshape(1,-1)],axis=0),token='V')
        NN_Gamma_i = lambda x,t: self.betaT['Gamma_i'] @ self.sigma(np.concatenate([x.reshape(1,-1),t.reshape(1,-1)],axis=0),token='Gamma_i')
        NN_Gamma_e = lambda x,t: self.betaT['Gamma_e'] @ self.sigma(np.concatenate([x.reshape(1,-1),t.reshape(1,-1)],axis=0),token='Gamma_e')
        CE_ni = lambda x,t: self.constrained_expression(NN_ni=NN_ni,token='ni')(x,t)
        CE_ne = lambda x,t: self.constrained_expression(NN_ne=NN_ne,token='ne')(x,t)
        CE_V = lambda x,t: self.constrained_expression(NN_V=NN_V,token='V')(x,t)
        CE_V_s = lambda x,t: self.constrained_expression(NN_V=NN_V,token='V')(x,t)
        CE_Gamma_i = lambda x,t: self.constrained_expression(NN_Gamma_i=NN_Gamma_i,CE_ni=CE_ni,CE_V=CE_V,token='Gamma_i')(x,t)
        CE_Gamma_e = lambda x,t: self.constrained_expression(NN_Gamma_e=NN_Gamma_e,CE_ni=CE_ni,CE_ne=CE_ne,CE_V=CE_V,CE_Gamma_i = CE_Gamma_i,token='Gamma_e')(x,t)
        return CE_ni,CE_ne,CE_V,CE_Gamma_i,CE_Gamma_e

    def constrained_expression(self,NN_ni=None,NN_ne=None,NN_V = None,NN_Gamma_i = None,NN_Gamma_e = None,
                               CE_ni=None,CE_ne=None,CE_V = None,CE_Gamma_i = None,CE_Gamma_e = None,token=None,
                              ):
        if token == 'ni':
            def ni(x,t):
                return 1e16 + (t-0.0)* NN_ni(x,t)
            return ni
        if token == 'ne':
            def ne(x,t):
                return 1e16 + (t-0.0)* NN_ne(x,t)
            return ne
        if token == 'V':
            def V(x,t):
                return NN_V(x,t) - NN_V(np.zeros_like(x),t) - NN_V(x,np.zeros_like(t)) + NN_V(np.zeros_like(x),np.zeros_like(t)) + 5e4 * x -1e3
            return V
        if token == 'Gamma_i':
            def Gamma_i(x,t):
                def CE_V_s(X,T):
                    return CE_V(X,T)[0,0]
                mE = grad(CE_V_s,argnums=0)
                if len(x.shape) == 2:
                    mE_ = vmap(mE,in_axes=1,out_axes=1)
                    mE = mE_
                L = self.physics_param['L']
                return NN_Gamma_i(x,t) + (L-x)/L * (-self.physics_param['mu_i'](-mE(np.zeros_like(x),t))*CE_ni(np.zeros_like(x),t)*mE(np.zeros_like(x),t)\
                                - NN_Gamma_i(np.zeros_like(x),t)) - (x/L) * NN_Gamma_i(L*np.ones_like(x),t)
            return Gamma_i
        if token == 'Gamma_e':
            def Gamma_e(x,t):
                def CE_V_s(X,T):
                    return CE_V(X,T)[0,0]
                mE = grad(CE_V_s,argnums=0)
                if len(x.shape) == 2:
                    mE_ = vmap(mE,in_axes=1,out_axes=1)
                    mE = mE_
                L = self.physics_param['L']
                return NN_Gamma_e(x,t) + (L-x)/L * (-self.physics_param['gamma'] * CE_Gamma_i(np.zeros_like(x),t) - NN_Gamma_e(np.zeros_like(x),t)) \
                        + (x/L)*(self.physics_param['mu_e'] * CE_ne(L*np.ones_like(x),t) * mE(L*np.ones_like(x),t)-NN_Gamma_e(L*np.ones_like(x),t))
            return Gamma_e