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
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+1]
                A_matrix[i,i+1] = self.int_sell * np.exp(-self.kappa * self.oracleprice*self.delta_sell(self.y_grid,i) -1 + self.kappa*(self.level_fct(quantity) - self.level_fct(quantity_P1)))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i,i-1] = self.int_buy * np.exp(self.kappa * self.oracleprice*self.delta_buy(self.y_grid,i) -1 - self.kappa*(self.level_fct(quantity_M1) - self.level_fct(quantity)))
        return A_matrix
    
    def _calculate_omega_t(self,t): # Compute the function omega 
        A_matrix = np.zeros((self.dim,self.dim))
        vector = np.ones((self.dim,1))
        A_matrix = self._calculate_matrix_t(t)
        return np.matmul(expm(A_matrix*(self.T-t)), vector)

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
    
    def get_arrival(self,stoc_intensity_sell,stoc_intensity_buy,num_simulations,dt):
        unif_s = np.random.uniform(size=num_simulations)
        unif_b = np.random.uniform(size=num_simulations)
        return unif_s < 1. - np.exp(-stoc_intensity_sell * dt), unif_b < 1. - np.exp(-stoc_intensity_buy * dt)
    
    def get_linear_fees(self,t):
        alpha, beta = self._calculate_fees_t(t)
        lin_alpha = np.zeros((self.dim))
        lin_beta = np.zeros((self.dim))
        min_idx = self.dim//2 - 3
        max_idx = self.dim//2 + 3
        slope_alpha = (alpha[max_idx] - alpha[min_idx]) / (self.y_grid[max_idx] - self.y_grid[min_idx])
        slope_beta = (beta[max_idx] - beta[min_idx]) / (self.y_grid[max_idx] - self.y_grid[min_idx])
        for i,q in enumerate(self.y_grid):
            lin_alpha[i] = q * slope_alpha + alpha[max_idx] - slope_alpha* self.y_grid[max_idx]
            lin_beta[i] = q * slope_beta + beta[max_idx] - slope_beta* self.y_grid[max_idx]
        lin_alpha[-1] = "NaN"
        lin_beta[0] = "NaN"
        return lin_alpha,lin_beta
    
    def compute_intensities(self, alpha, beta, idx_quantity):
        stoch_int_sell = self.int_sell * np.exp( self.kappa * ((1 - alpha[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1])) - self.oracleprice * self.delta_sell(self.y_grid, idx_quantity)) )
        stoch_int_buy = self.int_buy * np.exp( -self.kappa * ((1 + beta[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity-1]) - self.level_fct(self.y_grid[idx_quantity])) - self.oracleprice * self.delta_buy(self.y_grid, idx_quantity) ) )
        return stoch_int_sell, stoch_int_buy
    
    
    def compute_cash_step(self, alpha, beta, idx_quantity, sell_order, buy_order):
        cash_step = alpha[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1])) * sell_order.astype(int)  \
                        + beta[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity-1]) - self.level_fct(self.y_grid[idx_quantity])) * buy_order.astype(int) 
        return cash_step
        
    def simulate_PnL(self, nsims, Nt, strategy, c = 0.01, seed = 123, return_trajectory = False):
        rng = np.random.default_rng(seed=seed)
        dt = self.T/Nt
        timesteps = np.linspace(0, self.T, Nt+1)
        cash = np.zeros((nsims))
        n_sell_order = np.zeros((nsims))
        n_buy_order = np.zeros((nsims))
        idx_quantity = (np.ones((nsims))*[self.dim // 2]).astype(int)
        if return_trajectory:
            traj_quantity = np.zeros((nsims, Nt+1))
            traj_quantity[:,0] = self.y_grid[[self.dim // 2]] 
        stoch_int_sell = np.zeros((nsims))
        stoch_int_buy = np.zeros((nsims))
        min_inventory = self.y_0
        max_inventory = self.y_0
        if strategy == "Constant":
            alpha = c*np.ones((self.dim))
            beta = c*np.ones((self.dim))
            alpha[-1] = "NaN"
            beta[0] = "NaN"
        for it, t in enumerate(timesteps[:-1]):
            if strategy == "Optimal":
                alpha, beta = self._calculate_fees_t(t)
            if strategy == "Linear":
                alpha, beta = self.get_linear_fees(t)
            stoch_int_sell, stoch_int_buy = self.compute_intensities(alpha, beta, idx_quantity)
            sell_order, buy_order = self.get_arrival(stoch_int_sell, stoch_int_buy, nsims, dt)
            indicator_buy = (idx_quantity - 1 >=0)
            indicator_sell = (idx_quantity + 1 <self.dim)
            
            sell_order = sell_order.astype(int) * indicator_sell
            buy_order = buy_order.astype(int) * indicator_buy
            
            cash += self.compute_cash_step(alpha, beta, idx_quantity, sell_order, buy_order) 
            
            idx_quantity += sell_order.astype(int) - buy_order.astype(int)
            n_sell_order += sell_order
            n_buy_order += buy_order

            min_inventory = np.minimum(min_inventory,self.y_grid[idx_quantity])
            max_inventory = np.maximum(max_inventory,self.y_grid[idx_quantity])
            
            if return_trajectory:
                traj_quantity[:,it+1] = self.y_grid[idx_quantity]
        if return_trajectory:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order, min_inventory, max_inventory, traj_quantity)
        else:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order, min_inventory, max_inventory)
        
        
    