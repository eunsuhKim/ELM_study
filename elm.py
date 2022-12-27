import numpy as np
from scipy.linalg import pinv2, inv
import time
from autograd import grad
from autograd import elementwise_grad as egrad
import autograd.numpy as np
from autograd import jacobian
class elm():
    
    def __init__(self, x,y,C,elm_type='reg',one_hot = False,hidden_units=[50,1], activation_function='sin',random_type='normal'):
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
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.C = C
        if elm_type == 'clf':
            self.output_dim = np.unique(self.y).shape[0]
        elif elm_type == 'reg':
            self.output_dim = self.y.shape[1]
        elif elm_type == 'pde':
            self.output_dim = self.y.shape[1]
        # weight matrix beta is initialized as the zero matrix.
        self.beta = np.zeros((self.hidden_units, self.output_dim))
        self.one_hot = one_hot
        self.elm_type = elm_type

        # if classification problem and one_hot == True
        if elm_type =='clf' and self.one_hot:
            self.one_hot_label = np.zeros((self.y.shape[0], self.output_dim))
            for i in range(self.y.shape[0]):
                self.one_hot_label[i,int(self.y[i])] = 1
        # Randomly generate the weight matrix and bias vector from input to hidden layer.
        # 'uniform': uniform distribution U(0,1)
        # 'normal': normal distribution N(0,0.5)
        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=0,high = 1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low = 0, high = 1, size = (self.hidden_units, 1))
        if self.random_type =='normal':
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))
        if self.activation_function == 'sigmoid':
            self.act_func = lambda x: 1/(1+np.exp(-x))
        if self.activation_function =='relu':
            self.act_func = lambda x: x * (x>0)
        if self.activation_function == 'sin':
            self.act_func = lambda x: (np.exp(x)- np.exp(-x))/(np.exp(x)+np.exp(-x))
        if self.activation_function == 'leaky_relu':
            self.act_func = lambda x: np.max(0, x) + 0.1* np.min(0, x)
            
    # This function computes the output of hidden layer
    # according to different activation function.
    def __input2hidden(self, x):
        if (self.elm_type =='clf') or (self.elm_type =='reg'):
            self.temH = np.dot(self.W, x.T) + self.b    
            H = self.act_func(self.temH)
            return H
        elif self.elm_type == 'pde':
            def g(x,t):
                
                XT = np.concatenate([x,t],axis=0)
                res = np.dot(self.W, XT) + self.b   
                print('g_res:',res.shape)
                return self.act_func(res) 
            
            def f(X,T):
                
                X_ones = np.ones_like(X)
                X_zeros = np.zeros_like(X)
                T_zeros = np.zeros_like(T)
                res = g(X,T)+(X-1)*g(X_zeros,T)-X*g(X_ones,T)- X*g(X_zeros,T_zeros)\
                        + X*g(X_ones,T_zeros)-g(X,T_zeros)+ g(X_zeros,T_zeros)+np.sin(np.pi*X)
                print('f_res:',res.shape)
                return res
            def residual_(x,t):
                print('x.shape',x.shape)
                print('t.shape',t.shape)
                print('f(x,t).shape',f(x,t).shape)
                F_xx = egrad(egrad(f,0),0)(x,t)
                print("F_xx.shape",F_xx.shape)
                F_t = egrad(f,1)(x,t)
                print("F_t.shape",F_t.shape)

                res = F_xx -F_t
                print("residual shape",res.shape)
                return res
            # residual = np.vectorize(residual_,signature='(n,1),(n,1)->(32,n)')
            X = x[:,0][None,:]
            T = x[:,1][None,:]
            
            return residual_(X,T)
    
    # This function compute the output.
    def __hidden2output(self, H):
        self.output= np.dot(H.T, self.beta)
        return self.output

    
    def fit(self, algorithm):
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
        if self.elm_type == 'clf':
            if self.one_hot:
                self.y_temp = self.one_hot_label
            else:
                self.y_temp = self.y
        else:
            self.y_temp = self.y
        # no regularization
        if algorithm == 'no_re':
            if self.elm_type == 'pde':
                self.beta = np.dot(pinv2(self.H.T),self.y_temp)

            else:
                self.beta = np.dot(pinv2(self.H.T), self.y_temp)
        # faster algorithm 1
        if algorithm == 'solution1':
            self.tmp1 = inv(np.eye(self.H.shape[0])/self.C +np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.tmp1, self.H)
            self.beta = np.dot(self.tmp2, self.y_temp)

        # faster algorithm 2
        if algorithm == 'solution2':
            self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.H.T, self.tmp1)
            self.beta = np.dot(self.tmp2.T, self.y_temp)
        self.time2 = time.time()

        # comput the results
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
        if self.elm_type == 'pde':
            self.train_score = None
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
        if self.elm_type == 'reg':
            self.test_score = np.sqrt(
                np.sum(
                    (self.result-self.y)*(self.result-self.y)
                    )/self.y.shape[0]
                    )
        return self.test_score   