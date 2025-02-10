import numpy as np
from scipy.linalg import expm

class AMM: #This name should change as it does not express fully the fact that this is the case with equidistributed grid for the asset y
    def __init__(self,int_sell, int_buy, kappa, oracleprice, depth, y_grid, y_0, T =1., pen_const=0.):

        # Intensities
        self.int_sell = int_sell
        self.int_buy = int_buy

        # Initial values in the pool
        self.depth = depth
        self.y_0 = int(y_0)

        # Sensitivity of order arrivals
        self.kappa = kappa

        # Price outside the pool
        self.oracleprice = oracleprice

        # Time horizon
        self.T = T

        # Penalty constant
        self.pen_const = pen_const

        # Grid for asset risky asset
        self.y_grid = y_grid

        # Dimension of the grid
        self.dim = len(y_grid) 

    def level_fct(self,y): # We assume CPMM
        return self.depth / y
    
    def der_level_fct(self,y): # Returns the derivative of the level function
        return - self.depth/(y**2)
    
    def delta_buy(self, y, i): #Compute the trading size for buying
        return y[i] - y[i-1]
    
    def delta_sell(self, y, i): #Compute the trading size for selling
        return y[i+1] - y[i]
    
    def _calculate_matrix_t(self,t): # Compute the function omega 
        A_matrix = np.zeros((self.dim,self.dim))
        for i in range(self.dim): # Define the matrix A
            quantity = self.y_grid[i]  
            A_matrix[i,i] = - self.kappa*self.pen_const*(-self.der_level_fct(quantity) - self.oracleprice)**2
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i-1,i] = self.int_buy * np.exp(self.kappa * self.oracleprice*self.delta_buy(self.y_grid,i) -1 - self.kappa*(self.level_fct(quantity_M1) - self.level_fct(quantity)))
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+ 1]
                A_matrix[i+1,i] = self.int_sell * np.exp(-self.kappa * self.oracleprice*self.delta_sell(self.y_grid,i) -1 + self.kappa*(self.level_fct(quantity) - self.level_fct(quantity_P1)))
        return A_matrix
    
    
    def _calculate_omega_t(self,t): # Compute the function omega 
        A_matrix = np.zeros((self.dim,self.dim))
        vector = np.ones((self.dim,1))
        for i in range(self.dim): # Define the matrix A
            quantity = self.y_grid[i]  
            A_matrix[i,i] = - self.kappa*self.pen_const*(-self.der_level_fct(quantity) - self.oracleprice)**2
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i-1,i] = self.int_buy * np.exp(self.kappa * self.oracleprice*self.delta_buy(self.y_grid,i) -1 - self.kappa*(self.level_fct(quantity_M1) - self.level_fct(quantity)))
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+ 1]
                A_matrix[i+1,i] = self.int_sell * np.exp(-self.kappa * self.oracleprice*self.delta_sell(self.y_grid,i) -1 + self.kappa*(self.level_fct(quantity) - self.level_fct(quantity_P1)))
        return np.matmul(expm(A_matrix*(self.T-t) ), vector)

    
    def _calculate_v_t(self,t):
        omega_function = self._calculate_omega_t(t)
        return ( 1 / self.kappa) * np.log(omega_function)
    
    def _calculate_fees_t(self, t): # Compute the optimal fees
        v_qs = self._calculate_v_t(t)
        alpha = np.ones((self.dim))
        beta = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                alpha[i] = (1./self.kappa + v_qs[i,0] - v_qs[i+1,0])/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                beta[i] = (1./self.kappa + v_qs[i,0] - v_qs[i-1,0])/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        alpha[-1] = np.NaN
        beta[0] = np.NaN
        return alpha, beta
