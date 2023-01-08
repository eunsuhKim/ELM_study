import numpy as onp
from scipy.linalg import inv
import time
import jax.numpy as np
import jax
from jax import jacfwd, vmap, grad, jvp, vjp
from scipy.special import roots_legendre
jax.config.update("jax_enable_x64", True)

class elm():
    
    def __init__(self, x,y,C,physic_param = None,elm_type='reg',
                one_hot = False,hidden_units=32, activation_function='sin',
                random_type='normal',de_name=None,quadrature=False, tau = None,history_func = None,
                random_seed=None,Wscale=None,bscale=None,fourier_embedding=False):
        '''
        Function: elm class init
        -------------------------
        Parameters:
        shape: list, shape[hidden units, output units]
            numbers of hidden units and output units
        activation_function: str, 'sigmoid', 'relu', 'sin', 'tanh', or 'leaky_relu'
            Activation function of neurons
        x: array, shape[samples, features]
            training inputs
        y: array, shape[samples,]
            training targets
        C: float
            regularization parameter
        elm_type: str, 'clf' or 'reg'
            'clf' means ELM solve classification problems, 
            'reg' means ELM solve regression problems.
        one_hot: bool, True or False, default False
            The parameter is useful only when elm_type == 'clf'.
            If the labels need to be transformed to one_hot, this parameter is set to be True.
        random_type: str, 'uniform' or 'normal', default: 'normal'
            Weight initialization method
        '''
        option_dict = {}
        option_dict['elm_type'] = elm_type
        option_dict['random_type'] = random_type
        option_dict['activation_function'] = activation_function
        option_dict['hidden_units'] = hidden_units
        option_dict['C'] = C
        option_dict['one_hot']=one_hot
        option_dict['physics_param']=physic_param
        option_dict['de_name']=de_name
        option_dict['tau']=tau
        option_dict['random_seed'] = random_seed
        option_dict['fourier_embedding'] =fourier_embedding
        option_dict['quadrature']=quadrature
        self.random_seed = random_seed
        onp.random.seed(random_seed)
        print('Random seed: ',random_seed)
        self.option_dict = option_dict
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.physics_param = physic_param
        self.de_name=de_name
        self.history_func = history_func
        self.input_dim = x.shape[1]
        self.sample_size = x.shape[0]
        self.quadrature = quadrature
        self.x = x
        self.y = y
        self.C = C
        if elm_type == 'clf':
            self.output_dim = np.unique(self.y).shape[0]
        elif elm_type == 'reg':
            self.output_dim = self.y.shape[1]
        elif elm_type == 'de':
            self.output_dim = self.y.shape[1]
        # weight matrix beta is initialized as the zero matrix.
        self.beta = np.zeros((self.hidden_units, self.output_dim))
        self.one_hot = one_hot
        self.elm_type = elm_type
        if de_name == 'dde_logistic':
            self.tau = tau

        # if classification problem and one_hot == True
        if elm_type =='clf' and self.one_hot:
            self.one_hot_label = np.zeros((self.sample_size, self.output_dim))
            for i in range(self.y.shape[0]):
                self.one_hot_label[i,int(self.y[i])] = 1
        # Randomly generate the weight matrix and bias vector from input to hidden layer.
        # 'uniform': uniform distribution U(0,1)
        # 'normal': normal distribution N(0,0.5)
        if self.random_type == 'uniform':
            self.W = onp.random.uniform(low=-1,high = 1, size=(self.hidden_units, self.input_dim))
            self.b = onp.random.uniform(low = -1, high = 1, size = (self.hidden_units, 1))
        if self.random_type =='normal':

            print("W scale:",Wscale)
            print("b scale:",bscale)
            self.option_dict['Wscale']=Wscale
            self.option_dict['bscale']=bscale
            self.W = onp.random.normal(loc=0, scale=Wscale, size=(self.hidden_units, self.input_dim))
            self.b = onp.random.normal(loc=0, scale=bscale, size=(self.hidden_units, 1))
        if self.activation_function == 'sigmoid':
            self.act_func = lambda x: 1/(1+np.exp(-x))
            # self.act_func_p = lambda x: 0.5*(np.tanh(0.5*x)+1)*(1-0.5*(np.tanh(0.5*x)+1))
            # self.act_func_pp = lambda x:0.5*(np.tanh(0.5*x)+1)*(1-0.5*(np.tanh(0.5*x)+1))*(1-(np.tanh(0.5*x)+1))
        if self.activation_function =='relu':
            self.act_func = lambda x: x * (x>0)
            # self.act_func_p = lambda x: 1.0* (x>0)
        if self.activation_function == 'tanh':
            self.act_func = lambda x: np.tanh(x)#(np.exp(x)- np.exp(-x))/(np.exp(x)+np.exp(-x))
            # self.act_func_p = lambda x: 1/(np.cos(x)**2)#4/(np.exp(x)+np.exp(-x))**2
            # self.act_func_pp = lambda x:2*np.tanh(x)/(np.cosh(x)**2)#-8*(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))**3
        if self.activation_function == 'leaky_relu':
            self.act_func = lambda x: x*(x>0)+0.1*x*(x<0)#np.max(0, x) + 0.1* np.min(0, x)
            # self.act_func_p = lambda x: 0.1*(x<=0)+1.0*(x>0)   
        if self.activation_function == 'sin':
            self.act_func = lambda x: np.sin(x)         
            # self.act_func_p = lambda x: np.cos(x)         
            # self.act_func_pp = lambda x: -np.sin(x)         
    # This function computes the output of hidden layer
    # according to different activation function.
    def __input2hidden(self, x):
        self.temH = np.dot(self.W, x.T) + self.b    
        H = self.act_func(self.temH)
        return H
         
    
    # This function compute the output.
    def __hidden2output(self, H):
        self.output= np.dot(H.T, self.beta)
        return self.output

    def __hidden2residualscore(self,H_physics):
        self.output = np.dot(self.beta.T,H_physics)
        return np.mean(np.abs(self.output-self.y_temp))

    def __sigma(self,X,T=None):
        if self.input_dim == 2:
            x = np.concatenate([X,T],axis=0) # input_dim x sample_size
        elif self.input_dim == 1:
            x = X
            x = x/self.x.std()
        self.temH = np.matmul(self.W, x.reshape(1,-1)) + self.b    
        H = self.act_func(self.temH)
        return H
    # def __sigma_p(self,X,T=None):
    #     if self.input_dim == 2:
    #         x = np.concatenate([X,T],axis=0) # input_dim x sample_size
    #     elif self.input_dim == 1:
    #         x = X
    #         x = x/self.x.std()
    #     self.temH = np.dot(self.W, x) + self.b    
    #     H = self.act_func_p(self.temH)
    #     return H
    # def __sigma_pp(self,X,T=None):
    #     if self.input_dim == 2:
    #         x = np.concatenate([X,T],axis=0) # input_dim x sample_size
    #     elif self.input_dim == 1:
    #         x = X
    #         x = x/self.x.std()
    #     self.temH = np.dot(self.W, x) + self.b    
    #     H = self.act_func_pp(self.temH)
    #     return H
    def history_func(self,t):
        if self.de_name == 'dde_logistic':
            return 0.1*np.ones_like(t) 
    def __input2physics(self,x): # x has size of [sample_size x input_dim]
        if self.de_name == 'heat_diff':
            kappa = self.physics_param[0]
            X = x[:,0:1].reshape(1,-1) # 1 x sample_size
            T = x[:,1:2].reshape(1,-1) # 1 x sample_size
            Wx = self.W[:,0:1] # hidden_units x 1
            Wt = self.W[:,1:2] # hidden_units x 1
            sig_xx = (Wx**2)*self.__sigma_pp(X,T)
            sig_xx_t0 = (Wx**2)*self.__sigma_pp(X,np.zeros_like(T))
            sig_t = Wt*self.__sigma_p(X,T)
            sig_t_x0 = Wt*self.__sigma_p(np.zeros_like(X),T)
            sig_t_x1 = Wt*self.__sigma_p(np.ones_like(X),T)

            f_xx = sig_xx  - sig_xx_t0
            f_t = sig_t +(X-1)*sig_t_x0 - (X)*sig_t_x1

            expr_physics = f_xx - kappa*f_t
            self.y_temp = (np.pi**2)*np.sin(np.pi*X)
            return expr_physics

    def __make_beta(self,num_iter = 10):

        if self.quadrature == True:
            weights = roots_legendre(self.sample_size)[1].reshape(1,-1)
        else: 
            weights = np.ones((1,self.sample_size))
        
        betaT0 = onp.random.randn(self.input_dim,self.hidden_units)
        betaT0 = np.array(betaT0)
        
        a = self.physics_param[0]
        T = self.x[:,0:1].reshape(1,-1) # 1 x sample_size
        # sig = self.__sigma(X=T) # hidden_units x sample_size
        # sig_delay = self.__sigma(X=T-self.tau) # hidden_units x sample_size
        # sig_t = self.W*self.__sigma_p(X=T) # hidden_units x sample_size
        # sig_t = sig_t/(self.x.std())
        
        def N(betaT): # betaT has shape (output_dim x hidden_units)
            
            if self.de_name == 'dde_logistic':
                # Nonlinear w.r.t. beta
                def warpper_func(T):
                    return ((T<=0)*self.history_func(T)+(T>0)*(T*betaT@(self.__sigma(X = T))+self.history_func(0).item()))[0,0]

                u = (T<=0)*self.history_func(T)+(T>0)*(T*(betaT @ self.__sigma(X = T))+self.history_func(0).item())
                u_delay= ((T-self.tau)<=0)*self.history_func(T-self.tau)+\
                    ((T-self.tau)>0)*((T-self.tau)*(betaT @ self.__sigma(X=(T-self.tau)))+self.history_func(0).item())
                u_p = grad(warpper_func)

                u_p_vmap = vmap(u_p,in_axes=1,out_axes=1)
                u_t = u_p_vmap(T)
                expr_physics = u_t - a*u*(1-u_delay)
                return weights*expr_physics
        self.N = N

        J = jacfwd(N) 
        # J originally has shape (self.output_dim,self.sample_size,
        #                           self.output_dim,self.hidden_units)
        # real beta T with shape (self.output_dim,self.hidden_units)
        betaT = betaT0 
        # for solving matrix equation (J.T@J)betaT_ =J.T@deltay
        betaT_ = betaT0.reshape(self.output_dim*self.hidden_units,1)
        self.res_hist = []
        start = time.time()
        
        for i in range(num_iter):
            J_ = J(betaT).reshape(self.output_dim*self.sample_size,
                                self.output_dim*self.hidden_units)
            deltay_ = -N(betaT).reshape(self.output_dim*self.sample_size,1)
            delta_beta_ = np.linalg.solve(J_.T@J_+1e-4*np.eye(self.hidden_units),J_.T@deltay_)
            betaT = betaT + delta_beta_.reshape(self.output_dim,self.hidden_units)
            betaT_ = betaT_ + delta_beta_
            train_score = np.mean(np.abs(self.N(betaT)))
            self.res_hist.append(train_score)
            self.beta = betaT.T
            if i%500 == 0:
                print(f'Train_score when iter={i}: {train_score}')
        
        print(time.time()-start,' seconds cost for nonlinear least square.')
        return betaT.T

    def __constrained_expression(self, x):
        if self.de_name == 'heat_diff':
            X = x[:,0:1].reshape(1,-1) # 1 x sample_size
            T = x[:,1:2].reshape(1,-1) # 1 x sample_size
            X0 = np.zeros_like(X)
            T0 = np.zeros_like(T)
            X1 = np.ones_like(X)
            def g(X,T):
                sig_xt = self.__sigma(X,T)
                return np.matmul(self.beta.T,sig_xt) # beta has shape [hidden_units x output_dim]
            expr_result = g(X,T)+(X-1)*g(X0,T)- X*g(X1,T)-X*g(X0,T0)+X*g(X1,T0)\
                        -g(X,T0) + g(X0,T0) +np.sin(np.pi*X)
            return expr_result.T
        if self.de_name == 'dde_logistic':
            T = x[:,0:1].reshape(1,-1)
            
            expr_result = (T<=0)*self.history_func(T)+\
                (T>0)*(T*self.__hidden2output(self.__sigma(X=T)).T+self.history_func(0).item())
            return expr_result.T

    def fit(self, algorithm = 0, num_iter = 100):
        '''
        Function: Triain the model, compute beta matrix, 
        , the weight matrix from hidden layer to output layer.
        --------------------------------
        Parameter:
        algorithm: str, 'no_re', 'solution1' or 'solution2'
            The algorithm to comput beta matrix.
        -----------------------------------
        Return:
        beta: array
            the weight matrix from hidden layer to output layer
        train_score: float
            the accuracy or RMSE
        train_time: str
            time of computing beta
        '''
        self.time1 = time.time()
        self.H = self.__input2hidden(self.x)
        self.H_physics = self.__input2physics(self.x)
        if self.elm_type == 'clf':
            if self.one_hot:
                self.y_temp = self.one_hot_label
            else:
                self.y_temp = self.y
        
        if self.elm_type=='reg':
            self.y_temp = self.y
        # no regularization
        if algorithm == 'no_re':
            if self.elm_type == 'de':
                if self.de_name == 'heat_diff':
                    self.beta = np.dot(np.linalg.pinv(self.H_physics.T),self.y_temp.T)
                if self.de_name == 'dde_logistic':
                    self.beta = self.__make_beta(num_iter = num_iter)

            else:
                # self.beta = np.dot(pinv2(self.H.T), self.y_temp)
                self.beta = np.dot(np.linalg.pinv(self.H.T), self.y_temp)
        # faster algorithm 1
        if algorithm == 'solution1':
            if self.elm_type == 'de':
                self.tmp1 = inv(np.eye(self.H_physics.shape[0])/self.C +np.dot(self.H_physics, self.H_physics.T))
                self.tmp2 = np.dot(self.tmp1, self.H_physics)
                self.beta = np.dot(self.tmp2, self.y_temp.T)
            else:
                self.tmp1 = inv(np.eye(self.H.shape[0])/self.C +np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.tmp1, self.H)
                self.beta = np.dot(self.tmp2, self.y_temp)

        # faster algorithm 2
        if algorithm == 'solution2':
            if self.elm_type == 'de':
                self.tmp1 = inv(np.eye(self.H_physics.shape[0])/self.C + np.dot(self.H_physics, self.H_physics.T))
                self.tmp2 = np.dot(self.H_physics.T, self.tmp1)
                self.beta = np.dot(self.tmp2.T, self.y_temp.T)
            else:
                self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.H.T, self.tmp1)
                self.beta = np.dot(self.tmp2.T, self.y_temp)
        self.time2 = time.time()

        # compute the results
        if self.elm_type =='de':
            self.result = self.predict(self.x)
        else:
            self.result = self.__hidden2output(self.H)
        # If the problem is classification problem, 
        # the output is warpped by softmax.
        if self.elm_type == 'clf':
            self.result = np.exp(self.result)/ np.sum(np.exp(self.result), axis = 1).reshape(-1,1)

        # Evaluate training results
        # If the problem is classification, compute the accuracy.
        # If the problem is regressiont, compute the RMSE.
        if self.elm_type == 'clf':
            self.y_ = np.where(self.result==np.max(self.result, axis = 1).reshape(-1,1))[1]
            self.correct = 0
            for i in range(self.y.shape[0]):
                if self.y_[i] == self.y[i]:
                    self.correct  = self.correct + 1
            self.train_score = self.correct/ self.y.shape[0]
        if self.elm_type == 'reg':
            self.train_score = np.sqrt(
                np.sum(
                    (self.result - self.y)*(self.result-self.y)/self.y.shape[0])
            )
        if self.elm_type == 'de':
            if self.de_name == 'heat_diff':
                self.train_score = self.__hidden2residualscore(self.H_physics)
            if self.de_name == 'dde_logistic':
                
                self.train_score = np.mean(np.abs(self.N(self.beta.T)))
        train_time = str(self.time2- self.time1)
        return self.beta, self.train_score, train_time

    
    def predict(self, x): 
        '''
        Function: computes the result given new data.
        --------------------
        Parameters: 
        x: array, shape[samples, features]
        --------------------
        Return: 
        y_ : array
            predicted results
        '''
        if self.elm_type =='de':
            
            f_val = self.__constrained_expression(x)
            return f_val 
        else:
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1,1))[1]

            return self.y_

    

    
    def score(self, x, y):
        '''
        Function: computes accuracy or RMSE given data and labels
        -----------------
        Parameters: 
        x: array, shape[samples, features]
        y: array, shape[samples,]
        -------------------
        Return: 
        test_score: flaot, accuracy or RMSE
        '''
        self.prediction = self.predict(x)
        if self.elm_type == 'clf':
            self.correct = 0
            for i in range(y.shape[0]):
                if self.prediction[i] == y[i]:
                    self.correct = self.correct + 1
            self.test_score = self.correct/ y.shape[0]
        if (self.elm_type == 'reg') or (self.elm_type == 'de'):
            print("here")
            self.test_score = np.sqrt(
                np.sum(
                    (self.result-self.y)*(self.result-self.y)
                    )/self.y.shape[0]
                    )
        return self.test_score   