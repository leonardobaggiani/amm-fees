import numpy as np
from scipy.linalg import expm


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
        self.yvector = [i + self.ymin for i in range(self.ymax - self.ymin)]
    
    def level_fct(self, x): # x is cash; we work in CPMM
        return self.depth / x
    
    
    def derivative_level_fct(self, x): # x is cash; returns instant price
        return - self.depth/(x**2)
        
    
    def _calculate_omega_t(self, t):
        A_matrix = np.zeros((self.ymax - self.ymin + 1, self.ymax - self.ymin + 1))
        vector = np.ones((self.ymax - self.ymin + 1 , 1))  
        for i in range(self.ymax - self.ymin + 1):
            quantity = i + self.ymin
            A_matrix[i,i] = -self.runn* (-self.derivative_level_fct(quantity) - self.s)**2
            if i + 1 < self.ymax - self.ymin:
                A_matrix[i,i+1] = self.lambda_buy * np.exp(- self.kappa*(self.level_fct(quantity-1) - (self.level_fct(quantity))) + self.kappa * self.s - 1)
            if i > 0:
                A_matrix[i,i-1] = self.lambda_sell * np.exp(self.kappa*(self.level_fct(quantity) - self.level_fct(quantity+1)) - self.kappa * self.s - 1)
        return np.matmul(expm(A_matrix*(self.T-t) ), vector)
    
    def _calculate_gt(self, t): # Compute the function g
        omega_function = self._calculate_omega_t(t)
        return (1. / self.kappa) * np.log(omega_function)
    
    def calculate_fees(self, t): # Compute the best fees
        g_qs = self._calculate_gt(t)
        alpha = np.ones((self.ymax - self.ymin))
        beta = np.ones((self.ymax - self.ymin))
        for i in range(self.ymax - self.ymin):
            quantity = i + self.ymin
            alpha[i] = 1/(self.kappa*(self.level_fct(quantity) - self.level_fct(quantity+1))) - (g_qs[i+1,0] - g_qs[i,0])/(self.level_fct(quantity) - self.level_fct(quantity+1))
            beta[i] = 1/(self.kappa*(self.level_fct(quantity-1) - self.level_fct(quantity))) + (g_qs[i,0] - g_qs[i-1,0])/(self.level_fct(quantity-1) - self.level_fct(quantity))
        alpha[-1] = np.NaN
        beta[0] = np.NaN
        return alpha, beta