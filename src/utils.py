import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve
from functools import partial

class AMM:
    """
    Model parameters for the environment.
    """
    def __init__(self, lambda_sell, lambda_buy, kappa, s, depth, ymin, ymax, y0,  T =1., runn = 0.):
        # intensity
        self.lambda_sell = lambda_sell
        self.lambda_buy = lambda_buy
        
        # liquidity in the pool
        self.depth = depth
        self.ymin = int(ymin)
        self.ymax = int(ymax)
        self.y0 = y0
        self.x0 = self.depth/y0
        self.z0 = self.x0/self.y0
        
        # sensitivity of arrivals to differences in price
        self.kappa = kappa

        # running penalty to square deviations
        self.runn = runn
        
        # external venue price
        self.s = s
        
        # time horizon
        self.T = T
        self.yvector = [i + self.ymin for i in range(self.ymax - self.ymin + 1)]
    
    def level_fct(self, y): # we work in CPMM
        return self.depth / y
    
    
    def derivative_level_fct(self, y): # returns instant price
        return - self.depth/(y**2)
        
    
    def _calculate_omega_t(self, t):
        A_matrix = np.zeros((self.ymax - self.ymin + 1, self.ymax - self.ymin + 1))
        vector = np.ones((self.ymax - self.ymin + 1 , 1))  
        for i in range(self.ymax - self.ymin + 1):
            quantity = i + self.ymin
            A_matrix[i,i] = -self.runn* (-self.derivative_level_fct(quantity) - self.s)**2
            if i < self.ymax - self.ymin:
                A_matrix[i,i+1] = self.lambda_buy * np.exp(- self.kappa*(self.level_fct(quantity-1) - (self.level_fct(quantity))) + self.kappa * self.s - 1)
            if i > 0:
                A_matrix[i,i-1] = self.lambda_sell * np.exp(self.kappa*(self.level_fct(quantity) - self.level_fct(quantity+1)) - self.kappa * self.s - 1)
        return np.matmul(expm(A_matrix*(self.T-t) ), vector)
    
    def _calculate_gt(self, t): # Compute the function g
        omega_function = self._calculate_omega_t(t)
        return (1. / self.kappa) * np.log(omega_function)
    
    def calculate_fees(self, t): # Compute the best fees
        g_qs = self._calculate_gt(t)
        alpha = np.ones((self.ymax - self.ymin + 1))
        beta = np.ones((self.ymax - self.ymin + 1))
        for i in range(self.ymax - self.ymin + 1):
            quantity = i + self.ymin
            if i < self.ymax - self.ymin:
                alpha[i] = 1/(self.kappa*(self.level_fct(quantity) - self.level_fct(quantity+1))) - (g_qs[i+1,0] - g_qs[i,0])/(self.level_fct(quantity) - self.level_fct(quantity+1))
            if i > 0:
                beta[i] = 1/(self.kappa*(self.level_fct(quantity-1) - self.level_fct(quantity))) + (g_qs[i,0] - g_qs[i-1,0])/(self.level_fct(quantity-1) - self.level_fct(quantity))
        alpha[-1] = np.NaN
        beta[0] = np.NaN
        return alpha, beta
    
    
    
    
    
    
    
class AMM_proportional:
    """
    Model parameters for the environment.
    """
    def __init__(self, lambda_sell, lambda_buy, kappa, prop_down, prop_up, s, depth, max_shift, y0,  T =1., runn = 0.):
        # intensity
        self.lambda_sell = lambda_sell
        self.lambda_buy = lambda_buy
        
        # liquidity in the pool
        self.depth = depth
        self.max_shift = int(max_shift)
        self.y0 = y0
        self.x0 = self.depth/y0
        self.z0 = self.x0/self.y0
        
        self.prop_up = prop_up
        self.prop_down = prop_down

        self.ymin = y0 * (1. - prop_down)**(max_shift)
        self.ymax = y0 * (1. + prop_up)**(max_shift)
        
        
        # sensitivity of arrivals to differences in price
        self.kappa = kappa
        
        # trade size = q^Y * prop (instead of 1)

        # running penalty to square deviations
        self.runn = runn
        
        # external venue price
        self.s = s
        
        # time horizon
        self.T = T
        self.yvector = self.ymin *(1.+self.prop_up)**np.arange(0, 2*self.max_shift + 1, 1) 
    
    def level_fct(self, y): # we work in CPMM
        return self.depth / y
    
    
    def derivative_level_fct(self, y): # returns instant price
        return - self.depth/(y**2)
        
    
    def _calculate_omega_t(self, t):
        A_matrix = np.zeros((2*self.max_shift + 1, 2*self.max_shift + 1))
        vector = np.ones((2*self.max_shift + 1 , 1))  
        for i in range(2*self.max_shift + 1):
            quantity = self.yvector[i]
            A_matrix[i,i] = -self.runn* (-self.derivative_level_fct(quantity) - self.s)**2
            if i < 2*self.max_shift:
                A_matrix[i,i+1] = self.lambda_buy * np.exp(- self.kappa*(  1./(quantity*self.prop_up) * ( self.level_fct(quantity) - self.level_fct(quantity*(1.+self.prop_up))) - self.s ) - 1)
            if i > 0:
                A_matrix[i,i-1] = self.lambda_sell * np.exp(self.kappa*( 1./(quantity*self.prop_down) * (self.level_fct(quantity*(1.-self.prop_down)) - (self.level_fct(quantity)))  - self.s ) - 1)
        return np.matmul(expm(A_matrix*(self.T-t) ), vector)
    
    def _calculate_gt(self, t): # Compute the function g
        omega_function = self._calculate_omega_t(t)
        return (1. / self.kappa) * np.log(omega_function)
    
    def calculate_fees(self, t): # Compute the best fees
        g_qs = self._calculate_gt(t)
        alpha = np.ones((2*self.max_shift+1))
        beta = np.ones((2*self.max_shift+1))
        for i in range(2*self.max_shift+1):
            quantity = self.yvector[i]
            if i < 2*self.max_shift:
                alpha[i] = 1/(self.kappa*( 1./(quantity*self.prop_up) * ( self.level_fct(quantity) - self.level_fct(quantity*(1.+self.prop_up))))) - (g_qs[i+1,0] - g_qs[i,0])/( 1./(quantity*self.prop_up) * ( self.level_fct(quantity) - self.level_fct(quantity*(1.+self.prop_up))))
            if i>0:
                beta[i] = 1/(self.kappa*( 1./(quantity*self.prop_down) * (self.level_fct(quantity*(1.-self.prop_down)) - (self.level_fct(quantity))) )) + (g_qs[i,0] - g_qs[i-1,0])/( 1./(quantity*self.prop_down) * (self.level_fct(quantity*(1.-self.prop_down)) - (self.level_fct(quantity))) )
        alpha[-1] = np.NaN
        beta[0] = np.NaN
        return alpha, beta
    
    

    
    
    
class AMM_symmetric:
    """
    Model parameters for the environment.
    """
    def __init__(self, lambda_sell, lambda_buy, kappa, delta_z_tilde, s, depth, max_shift, y0,  T =1., runn = 0.):
        # intensity
        self.lambda_sell = lambda_sell
        self.lambda_buy = lambda_buy
        
        # liquidity in the pool
        self.depth = depth
        self.max_shift = int(max_shift)
        self.y0 = y0
        self.x0 = self.depth/y0
        self.z0 = self.x0/self.y0
        
        self.delta_z_tilde = delta_z_tilde
        
        self.yvector = self.compute_yvector()
        #self.ymin = y0 * (1. - prop_down)**(max_shift)
        #self.ymax = y0 * (1. + prop_up)**(max_shift)
        #self.yvector = self.ymin *(1.+self.prop_up)**np.arange(0, 2*self.max_shift + 1, 1) 
        
        
        # sensitivity of arrivals to differences in price
        self.kappa = kappa
        
        # trade size = q^Y * prop (instead of 1)

        # running penalty to square deviations
        self.runn = runn
        
        # external venue price
        self.s = s
        
        # time horizon
        self.T = T
    
    def level_fct(self, y): # we work in CPMM
        return self.depth / y   
    
    def derivative_level_fct(self, y): # returns instant price
        return - self.depth/(y**2)
    
    def find_root_auxiliary_buy(self, to_optimise, y, i):
        return self.depth/(y*to_optimise) - self.z0 - (i+1) *self.delta_z_tilde
    def find_root_auxiliary_sell(self, to_optimise, y, i):
        return self.z0 - self.depth/(y*to_optimise) - (i+1) * self.delta_z_tilde
    
    def compute_yvector(self):
        yvector = np.zeros(2*self.max_shift+1)
        yvector[self.max_shift] = self.y0
        for i in range(self.max_shift):
            gbuy = partial(self.find_root_auxiliary_buy, y = yvector[self.max_shift-i], i = i)
            res = fsolve(gbuy, yvector[self.max_shift-i])
            yvector[self.max_shift-i-1] = res[0]

            gsell = partial(self.find_root_auxiliary_sell, y = yvector[self.max_shift+i], i = i)
            res = fsolve(gsell, yvector[self.max_shift+i])
            yvector[self.max_shift+i+1] = res[0]
        return yvector
        
    def _calculate_omega_t(self, t):
        A_matrix = np.zeros((2*self.max_shift + 1, 2*self.max_shift + 1))
        vector = np.ones((2*self.max_shift + 1 , 1))  
        for i in range(2*self.max_shift+ 1):
            quantity = self.yvector[i]
            A_matrix[i,i] = -self.runn* (-self.derivative_level_fct(quantity) - self.s)**2
            if i > 0:
                A_matrix[i-1,i] = self.lambda_buy * np.exp(- self.kappa*( 1./(self.yvector[i] - self.yvector[i-1]) * ( self.level_fct(self.yvector[i-1]) - self.level_fct(self.yvector[i]) )  -  self.s ) - 1)
            if i < 2*self.max_shift:
                A_matrix[i+1,i] = self.lambda_sell * np.exp(self.kappa*( 1./(self.yvector[i+1] - self.yvector[i])*( self.level_fct(self.yvector[i]) - self.level_fct(self.yvector[i+1]) ) -  self.s) - 1)
        return np.matmul(expm(A_matrix*(self.T-t) ), vector)
    
    def _calculate_gt(self, t): # Compute the function g
        omega_function = self._calculate_omega_t(t)
        return (1. / self.kappa) * np.log(omega_function)
    
    def calculate_fees(self, t): # Compute the best fees
        g_qs = self._calculate_gt(t)
        alpha = np.ones((2*self.max_shift + 1))
        beta = np.ones((2*self.max_shift + 1))
        for i in range(2*self.max_shift + 1):
            quantity = self.yvector[i]
            
            if i  < 2*self.max_shift:
                quantity_P1 = self.yvector[i+1]
                delta_Q = quantity_P1 - quantity
                alpha[i] = 1/(self.kappa*( 1./delta_Q * (self.level_fct(quantity) - self.level_fct(quantity_P1)))) - (g_qs[i+1,0] - g_qs[i,0])/( 1./delta_Q * (self.level_fct(quantity) - self.level_fct(quantity_P1) ))
            if i > 0:
                quantity_M1 = self.yvector[i-1]
                delta_Q = quantity - quantity_M1
                beta[i] = 1/(self.kappa*( 1./delta_Q * ( self.level_fct(quantity_M1) - self.level_fct(quantity) ))) + (g_qs[i,0] - g_qs[i-1,0])/( 1./delta_Q * (self.level_fct(quantity_M1) - self.level_fct(quantity)))
        alpha[-1] = np.NaN
        beta[0] = np.NaN
        return alpha, beta
    