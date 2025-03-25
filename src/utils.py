import numpy as np
from scipy.linalg import expm
import math
from scipy.integrate import solve_ivp


class AMM: 
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

        # volatility
        self.sigma = 0.2

    def level_fct(self,y): # We assume CPMM
        return self.depth / y
    
    def der_level_fct(self,y): # Returns the derivative of the level function
        return - self.depth/(y**2)
    
    def delta_buy(self, y, i): #Compute the trading size for buying
        indicator_buy = np.where(i - 1 >= 0, 1, 0)
        return y[i] - y[i-1*indicator_buy]
    
    def delta_sell(self, y, i): #Compute the trading size for selling
        indicator_sell = np.where(i + 1 < self.dim, 1, 0)
        return y[i+1*indicator_sell] - y[i]
    
    def _calculate_matrix_t(self,t): # Compute the matrix A
        A_matrix = np.zeros((self.dim,self.dim))
        for i in range(self.dim): 
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

    def _calculate_v_t(self,t):  # Compute the function v 
        omega_function = self._calculate_omega_t(t)
        return ( 1 / self.kappa) * np.log(omega_function)
    
    def _calculate_fees_t(self, t): # Compute the optimal fees
        v_qs = self._calculate_v_t(t)
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1./self.kappa + v_qs[i,0] - v_qs[i+1,0])/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1./self.kappa + v_qs[i,0] - v_qs[i-1,0])/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        #p[-1] = np.NaN
        #m[0] = np.NaN
        return p, m
    
    def get_arrival(self,stoc_intensity_sell,stoc_intensity_buy,num_simulations,dt): # Given intensity compute if there is a jump or not
        unif_s = np.random.uniform(size=num_simulations)
        unif_b = np.random.uniform(size=num_simulations)
        return unif_s < 1. - np.exp(-stoc_intensity_sell * dt), unif_b < 1. - np.exp(-stoc_intensity_buy * dt)
    
    def get_linear_fees(self,t): # Compute the linear fees 
        p, m = self._calculate_fees_t(t)
        lin_p = np.zeros((self.dim))
        lin_m = np.zeros((self.dim))
        min_idx = self.dim//2 - 1
        max_idx = self.dim//2 + 1
        slope_p = (p[max_idx] - p[min_idx]) / (self.y_grid[max_idx] - self.y_grid[min_idx])
        slope_m = (m[max_idx] - m[min_idx]) / (self.y_grid[max_idx] - self.y_grid[min_idx])
        for i,q in enumerate(self.y_grid):
            lin_p[i] = q * slope_p + p[max_idx] - slope_p* self.y_grid[max_idx]
            lin_m[i] = q * slope_m + m[max_idx] - slope_m* self.y_grid[max_idx]
        #lin_p[-1] = "NaN"
        #lin_m[0] = "NaN"
        return lin_p,lin_m,slope_p, slope_m
    
    def compute_intensities(self, p, m, idx_quantity): # Compute the intensities
        indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
        indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        stoch_int_sell = self.int_sell * np.exp( self.kappa * ((1. - p[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1*indicator_sell])) - self.oracleprice * self.delta_sell(self.y_grid, idx_quantity)) )
        stoch_int_buy = self.int_buy * np.exp( -self.kappa * ((1. + m[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity-1*indicator_buy]) - self.level_fct(self.y_grid[idx_quantity])) - self.oracleprice * self.delta_buy(self.y_grid, idx_quantity) ) )
        return stoch_int_sell, stoch_int_buy
     
    def compute_cash_step(self, p, m, idx_quantity, sell_order, buy_order):
        indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
        indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        cash_step = p[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1*indicator_sell])) * sell_order.astype(int)  \
                        + m[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity-1*indicator_buy]) - self.level_fct(self.y_grid[idx_quantity])) * buy_order.astype(int) 
        return cash_step
        
    def simulate_PnL(self, nsims, Nt, strategy, c=0.01, seed = 123, return_trajectory = False):
        np.random.seed(seed=seed)
        dt = self.T/Nt
        timesteps = np.linspace(0, self.T, Nt+1)
        cash = np.zeros((nsims))
        n_sell_order = np.zeros((nsims))
        n_buy_order = np.zeros((nsims))
        idx_quantity = np.full(nsims, self.dim // 2, dtype=int)
        if return_trajectory:
            traj_quantity = np.zeros((nsims, Nt+1))
            traj_quantity[:,0] = self.y_grid[[self.dim // 2]] 
        stoch_int_sell = np.zeros((nsims))
        stoch_int_buy = np.zeros((nsims))
        #min_inventory = self.y_0
        #max_inventory = self.y_0
        if strategy == "Constant":
            p = c*np.ones((self.dim))
            m = c*np.ones((self.dim))
            #p[-1] = "NaN"
            #m[0] = "NaN"
        for it, t in enumerate(timesteps[:-1]):
            if strategy == "Optimal":
                p, m = self._calculate_fees_t(t)
            if strategy == "Linear":
                p, m,_,_ = self.get_linear_fees(t)
            indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
            indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        
            stoch_int_sell, stoch_int_buy = self.compute_intensities(p, m, idx_quantity)
            sell_order, buy_order = self.get_arrival(stoch_int_sell, stoch_int_buy, nsims, dt)
            
            
            sell_order = (sell_order & indicator_sell).astype(int)
            buy_order = (buy_order & indicator_buy).astype(int)
            
            cash += self.compute_cash_step(p, m, idx_quantity, sell_order, buy_order)
            
            idx_quantity += sell_order.astype(int) - buy_order.astype(int)
            n_sell_order += sell_order.astype(int)
            n_buy_order += buy_order.astype(int)
            
            if return_trajectory:
                traj_quantity[:,it+1] = self.y_grid[idx_quantity]
        if return_trajectory:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order, traj_quantity)
        else:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order)
        
    def _calculate_fees_k_0_t(self, t): # Compute the optimal fees
        A_matrix = np.zeros((self.dim,self.dim))
        vector = np.ones((self.dim,1))
        for i in range(self.dim): # Define the matrix A
            quantity = self.y_grid[i]  
            A_matrix[i,i] = 0
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+1]
                A_matrix[i,i+1] = self.int_sell * np.exp(-1)
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i,i-1] = self.int_buy * np.exp(-1)
        omega = np.matmul(expm(A_matrix*(self.T-t)), vector)
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1. + np.log(omega[i,0]) - np.log(omega[i+1,0]))/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1. + np.log(omega[i,0]) - np.log(omega[i-1,0]))/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        p[-1] = np.NaN
        m[0] = np.NaN
        return p, m
    
class lin_quad_ansatz:
    def __init__(self,int_sell, int_buy, kappa, oracleprice, depth, y_grid, y_0, T =1., pen_const=0., sigma = 10, delta_minus = 1, delta_plus = 1):
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

        # volatility
        self.sigma = 0.2

        self.delta_minus = delta_minus
        self.delta_plus = delta_plus
        
        self.e = math.e  # Base of natural logarithm
        self.p4 = self.depth**2  # p^4 = (p^2)^2 = depth^2
    
    def level_fct(self,y): # We assume CPMM
        return self.depth / y
    
    def der_level_fct(self,y): # Returns the derivative of the level function
        return - self.depth/(y**2)

    def compute_psi_0(self):
        """
        psi_0 =
        [ 4 p^4 y_0^2 * lambda^- + p^4 (delta^-)^2 * lambda^- - 4 p^4 y_0 delta^- * lambda^- ]
        / [ e kappa (2 y_0^2 - y_0 delta^-)^4 ]
        + [ 4 p^4 y_0^2 * lambda^+ + p^4 (delta^+)^2 * lambda^+ + 4 p^4 y_0 delta^+ * lambda^+ ]
        / [ e kappa (y_0 delta^+ + 2 y_0^2)^4 ]
        - 4 p^4 pen_const / y_0^6"
        """
        
        y = self.y_0
        # Buy side: lambda^- is self.int_buy and delta^- is self.delta_minus
        psi_buy = (self.p4 * self.int_buy * (4 * y**2 + self.delta_minus**2 - 4 * y * self.delta_minus)) \
                / (self.e * self.kappa * (2 * y**2 - y * self.delta_minus)**4)
        
        # Sell side: lambda^+ is self.int_sell and delta^+ is self.delta_plus
        psi_sell = (self.p4 * self.int_sell * (4 * y**2 + self.delta_plus**2 + 4 * y * self.delta_plus)) \
                / (self.e * self.kappa * (y * self.delta_plus + 2 * y**2)**4)
        
        # Penalty term remains the same (with phi being represented by pen_const)
        psi_pen = (4 * self.p4 * self.pen_const) / (y**6)
        
        return psi_buy + psi_sell - psi_pen

    def compute_psi_1(self):
        """
        psi_1 = [4 p^2 (delta^-)^2 lambda^- - 8 p^2 y_0 delta^- lambda^-] / [e kappa (2 y_0^2 - y_0 delta^-)^2]
                - [4 p^2 (delta^+)^2 lambda^+ + 8 p^2 y_0 delta^+ lambda^+] / [e kappa (y_0 delta^+ + 2 y_0^2)^2]
        """
        y = self.y_0
        # Buy side (λ⁻, δ⁻)
        psi1_buy = (self.depth * self.int_buy * (4 * self.delta_minus**2 - 8 * y * self.delta_minus)) \
                / (self.e * self.kappa * (2 * y**2 - y * self.delta_minus)**2)
        
        # Sell side (λ⁺, δ⁺)
        psi1_sell = - (self.depth * self.int_sell * (4 * self.delta_plus**2 + 8 * y * self.delta_plus)) \
                    / (self.e * self.kappa * (y * self.delta_plus + 2 * y**2)**2)
        
        return psi1_buy + psi1_sell

    def compute_psi_2(self):
        """
        psi_2 = 4 int_buy (delta^-)^2 / (e kappa) + 4 int_sell (delta^+)^2 / (e kappa)
        """
        return (4 * self.delta_minus**2 * self.int_buy + 4 * self.delta_plus**2 * self.int_sell) / (self.e * self.kappa)

    def compute_psi_3(self):
        """
        Computes psi_3 given by:
        
        psi_3 = [2 p^4 (delta^-)^2 lambda^-]      / [e k (y_0^2 - y_0 delta^-)(2 y_0^2 - y_0 delta^-)^2]
                + [8 p^4 y_0^2 delta^- lambda^-]      / [e k (2 y_0^2 - y_0 delta^-)^4]
                - [4 p^4 y_0 delta^- lambda^-]        / [e k (y_0^2 - y_0 delta^-)(2 y_0^2 - y_0 delta^-)^2]
                - [8 p^4 y_0^3 lambda^-]              / [e k (2 y_0^2 - y_0 delta^-)^4]
                - [2 p^4 y_0 (delta^-)^2 lambda^-]     / [e k (2 y_0^2 - y_0 delta^-)^4]
                + [2 p^2 y_0 lambda^-]                / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [p^2 delta^- lambda^-]              / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [2 p^4 (delta^+)^2 lambda^+]         / [e k (y_0 delta^+ + y_0^2)(y_0 delta^+ + 2 y_0^2)^2]
                - [4 p^4 y_0 delta^+ lambda^+]         / [e k (y_0 delta^+ + y_0^2)(y_0 delta^+ + 2 y_0^2)^2]
                - [8 p^4 y_0^3 lambda^+]               / [e k (y_0 delta^+ + 2 y_0^2)^4]
                - [2 p^4 y_0 (delta^+)^2 lambda^+]      / [e k (y_0 delta^+ + 2 y_0^2)^4]
                - [8 p^4 y_0^2 delta^+ lambda^+]         / [e k (y_0 delta^+ + 2 y_0^2)^4]
                - [2 p^2 y_0 lambda^+]                / [e k (y_0 delta^+ + 2 y_0^2)^2]
                - [p^2 delta^+ lambda^+]              / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [12 p^4 phi]                        / [y_0^5]
        """
        y = self.y_0

        # Precompute denominators for the buy side:
        denom_buy1 = (y**2 - y * self.delta_minus)
        denom_buy2 = (2 * y**2 - y * self.delta_minus)  # Appears raised to power 2 or 4

        # Precompute denominators for the sell side:
        denom_sell1 = (y * self.delta_plus + y**2)
        denom_sell2 = (y * self.delta_plus + 2 * y**2)  # Appears raised to power 2 or 4

        # Terms for the buy side (using lambda^- = self.int_buy):
        term1 = (2 * self.p4 * self.delta_minus**2 * self.int_buy) / (self.e * self.kappa * denom_buy1 * (denom_buy2**2))
        term2 = (8 * self.p4 * y**2 * self.delta_minus * self.int_buy) / (self.e * self.kappa * (denom_buy2**4))
        term3 = (-4 * self.p4 * y * self.delta_minus * self.int_buy) / (self.e * self.kappa * denom_buy1 * (denom_buy2**2))
        term4 = (-8 * self.p4 * y**3 * self.int_buy) / (self.e * self.kappa * (denom_buy2**4))
        term5 = (-2 * self.p4 * y * self.delta_minus**2 * self.int_buy) / (self.e * self.kappa * (denom_buy2**4))
        term6 = (2 * self.depth * y * self.int_buy) / (self.e * self.kappa * (denom_buy2**2))
        term7 = (- self.depth * self.delta_minus * self.int_buy) / (self.e * self.kappa * (denom_buy2**2))

        # Terms for the sell side (using lambda^+ = self.int_sell):
        term8  = (-2 * self.p4 * self.delta_plus**2 * self.int_sell) / (self.e * self.kappa * denom_sell1 * (denom_sell2**2))
        term9  = (-4 * self.p4 * y * self.delta_plus * self.int_sell) / (self.e * self.kappa * denom_sell1 * (denom_sell2**2))
        term10 = (-8 * self.p4 * y**3 * self.int_sell) / (self.e * self.kappa * (denom_sell2**4))
        term11 = (-2 * self.p4 * y * self.delta_plus**2 * self.int_sell) / (self.e * self.kappa * (denom_sell2**4))
        term12 = (-8 * self.p4 * y**2 * self.delta_plus * self.int_sell) / (self.e * self.kappa * (denom_sell2**4))
        term13 = (-2 * self.depth * y * self.int_sell) / (self.e * self.kappa * (denom_sell2**2))
        term14 = (- self.depth * self.delta_plus * self.int_sell) / (self.e * self.kappa * (denom_sell2**2))

        # Penalty term:
        term15 = (12 * self.p4 * self.pen_const) / (y**5)

        psi3 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + \
            term8 + term9 + term10 + term11 + term12 + term13 + term14 + term15

        return psi3

    def compute_psi_4(self):
        """
    Computes psi_4 using the new expression:
    
      psi_4 = [4 p^2 y_0 delta^- lambda^- - 2 p^2 (delta^-)^2 lambda^-] / [e k (2 y_0^2 - y_0 delta^-)^2]
            + [2 p^2 (delta^+)^2 lambda^+ + 4 p^2 y_0 delta^+ lambda^+] / [e k (y_0 delta^+ + 2 y_0^2)^2]
            - [4 p^2 phi] / [y_0^3]
        """
        y = self.y_0

        # Denominator for the buy side:
        denom_buy = (2 * y**2 - y * self.delta_minus)**2
        # Denominator for the sell side:
        denom_sell = (y * self.delta_plus + 2 * y**2)**2

        # Buy-side term:
        term_buy = (4 * self.depth * y * self.delta_minus * self.int_buy - 
                    2 * self.depth * self.delta_minus**2 * self.int_buy) / (self.e * self.kappa * denom_buy)
        
        # Sell-side term:
        term_sell = (2 * self.depth * self.delta_plus**2 * self.int_sell + 
                    4 * self.depth * y * self.delta_plus * self.int_sell) / (self.e * self.kappa * denom_sell)
        
        # Penalty term:
        term_pen = (4 * self.depth * self.pen_const) / (y**3)
        
        return term_buy + term_sell - term_pen

    def compute_psi_5(self):
        """
        Computes psi_5 using the new expression:

        psi_5 = -[2 p^2 (delta^-)^3 lambda^-]      / [e k (2 y_0^2 - y_0 delta^-)^2]
                + [4 p^2 (delta^-)^2 lambda^-]          / [e k (y_0^2 - y_0 delta^-)]
                + [8 p^2 y_0^2 delta^- lambda^-]         / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [2 delta^- lambda^-]                   / [e k]
                + [4 p^2 (delta^+)^2 lambda^+]           / [e k (y_0 delta^+ + y_0^2)]
                + [8 p^2 y_0^2 delta^+ lambda^+]          / [e k (y_0 delta^+ + 2 y_0^2)^2]
                - [2 p^2 (delta^+)^3 lambda^+]           / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [2 delta^+ lambda^+]                   / [e k]
        """
        y = self.y_0

        # Precompute denominators for the buy side:
        denom_buy_1 = (2 * y**2 - y * self.delta_minus)**2  # used in terms 1 and 3 (buy side)
        denom_buy_2 = (y**2 - y * self.delta_minus)           # used in term 2 (buy side)
        
        # Precompute denominators for the sell side:
        denom_sell_1 = (y * self.delta_plus + 2 * y**2)**2     # used in terms 6 and 7 (sell side)
        denom_sell_2 = (y * self.delta_plus + y**2)            # used in term 5 (sell side)

        # Buy side terms (with lambda^- = self.int_buy and delta^- = self.delta_minus):
        term1 = - (2 * self.depth * self.delta_minus**3 * self.int_buy) / (self.e * self.kappa * denom_buy_1)
        term2 =   (4 * self.depth * self.delta_minus**2 * self.int_buy) / (self.e * self.kappa * denom_buy_2)
        term3 =   (8 * self.depth * y**2 * self.delta_minus * self.int_buy) / (self.e * self.kappa * denom_buy_1)
        term4 =   - (2 * self.delta_minus * self.int_buy) / (self.e * self.kappa)

        # Sell side terms (with lambda^+ = self.int_sell and delta^+ = self.delta_plus):
        term5 =   (4 * self.depth * self.delta_plus**2 * self.int_sell) / (self.e * self.kappa * denom_sell_2)
        term6 =   (8 * self.depth * y**2 * self.delta_plus * self.int_sell) / (self.e * self.kappa * denom_sell_1)
        term7 =   - (2 * self.depth * self.delta_plus**3 * self.int_sell) / (self.e * self.kappa * denom_sell_1)
        term8 =   (2 * self.delta_plus * self.int_sell) / (self.e * self.kappa)

        psi5 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
        return psi5

    def compute_psi_6(self):
        """
        psi_6 = -4 (delta^-)^2 int_buy / (e kappa) - 4 (delta^+)^2 int_sell / (e kappa)
        """
        return (
            -(4 * (self.delta_minus**2) * self.int_buy)/(self.e * self.kappa)
            - (4 * (self.delta_plus**2)  * self.int_sell)/(self.e * self.kappa)
        )

    def compute_psi_7(self):
        """
        psi_7 = 4 (delta^+)^3 int_sell / (e kappa) - 4 (delta^-)^3 int_buy / (e kappa)
        """
        return (
            (4 * (self.delta_plus**3)  * self.int_sell)/(self.e * self.kappa)
            - (4 * (self.delta_minus**3) * self.int_buy)/(self.e * self.kappa)
        )

    def compute_psi_8(self):
        """
        psi_8 = 4 (delta^-)^2 int_buy / (e kappa) + 4 (delta^+)^2 int_sell / (e kappa)
        """
        return (
            (4 * (self.delta_minus**2) * self.int_buy)/(self.e * self.kappa)
            + (4 * (self.delta_plus**2)  * self.int_sell)/(self.e * self.kappa)
        )

    def compute_psi_9(self):
        """
        Computes psi_9 given by:
        
        psi_9 = [2 p^2 (delta^-)^2 lambda^- - 4 p^2 y_0 delta^- lambda^-] 
                / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [2 p^2 (delta^+)^2 lambda^+ + 4 p^2 y_0 delta^+ lambda^+]
                / [e k (y_0 delta^+ + 2 y_0^2)^2]
        """
        y = self.y_0
        
        # Buy-side denominator: (2 y_0^2 - y_0 delta^-)^2
        denom_buy = (2 * y**2 - y * self.delta_minus)**2
        # Sell-side denominator: (y_0 delta^+ + 2 y_0^2)^2
        denom_sell = (y * self.delta_plus + 2 * y**2)**2
        
        # Buy-side contribution:
        psi9_buy = (2 * self.depth * self.delta_minus**2 * self.int_buy - 
                    4 * self.depth * y * self.delta_minus * self.int_buy) \
                    / (self.e * self.kappa * denom_buy)
        
        # Sell-side contribution:
        psi9_sell = - (2 * self.depth * self.delta_plus**2 * self.int_sell + 
                    4 * self.depth * y * self.delta_plus * self.int_sell) \
                    / (self.e * self.kappa * denom_sell)
        
        return psi9_buy + psi9_sell

    def compute_psi_10(self):
        """
        Computes psi_10 given by:
        
        psi_10 = [p^4 (delta^-)^2 lambda^-] / [e k (y_0^2 - y_0 delta^-)^2]
                + [4 p^4 y_0^2 delta^- lambda^-] / [e k (y_0^2-y_0 delta^-)(2 y_0^2-y_0 delta^-)^2]
                + [4 p^4 y_0^4 lambda^-] / [e k (2 y_0^2-y_0 delta^-)^4]
                + [p^4 y_0^2 (delta^-)^2 lambda^-] / [e k (2 y_0^2-y_0 delta^-)^4]
                - [2 p^4 y_0 (delta^-)^2 lambda^-] / [e k (y_0^2-y_0 delta^-)(2 y_0^2-y_0 delta^-)^2]
                - [4 p^4 y_0^3 delta^- lambda^-] / [e k (2 y_0^2-y_0 delta^-)^4]
                + [p^2 y_0 delta^- lambda^-] / [e k (2 y_0^2-y_0 delta^-)^2]
                - [p^2 delta^- lambda^-] / [e k (y_0^2-y_0 delta^-)]
                - [2 p^2 y_0^2 lambda^-] / [e k (2 y_0^2-y_0 delta^-)^2]
                + [lambda^-] / [e k]
                + [p^4 (delta^+)^2 lambda^+] / [e k (y_0 delta^+ + y_0^2)^2]
                + [2 p^4 y_0 (delta^+)^2 lambda^+] / [e k (y_0 delta^+ + y_0^2)(y_0 delta^+ + 2 y_0^2)^2]
                + [4 p^4 y_0^2 delta^+ lambda^+] / [e k (y_0 delta^+ + y_0^2)(y_0 delta^+ + 2 y_0^2)^2]
                + [4 p^4 y_0^4 lambda^+] / [e k (y_0 delta^+ + 2 y_0^2)^4]
                + [p^4 y_0^2 (delta^+)^2 lambda^+] / [e k (y_0 delta^+ + 2 y_0^2)^4]
                + [4 p^4 y_0^3 delta^+ lambda^+] / [e k (y_0 delta^+ + 2 y_0^2)^4]
                + [p^2 delta^+ lambda^+] / [e k (y_0 delta^+ + y_0^2)]
                + [2 p^2 y_0^2 lambda^+] / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [p^2 y_0 delta^+ lambda^+] / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [lambda^+] / [e k]
                - [9 p^4 phi] / [y_0^4]
        """
        y  = self.y_0
        dm = self.delta_minus
        dp = self.delta_plus

        # Precompute common denominators for the buy side (involving delta^-):
        D_b_a = (y**2 - y * dm)         # appears squared in some terms
        D_b_b = (2 * y**2 - y * dm)       # appears squared or to the 4th power

        # Precompute denominators for the sell side (involving delta^+):
        D_s_a = (y * dp + y**2)
        D_s_b = (y * dp + 2 * y**2)
        
        # --- Buy-side terms (using lambda^- = self.int_buy) ---
        term1  =  (self.p4 * dm**2 * self.int_buy) / (self.e * self.kappa * (D_b_a**2))
        term2  =  (4 * self.p4 * y**2 * dm * self.int_buy) / (self.e * self.kappa * (D_b_a * (D_b_b**2)))
        term3  =  (4 * self.p4 * y**4 * self.int_buy) / (self.e * self.kappa * (D_b_b**4))
        term4  =  (self.p4 * y**2 * dm**2 * self.int_buy) / (self.e * self.kappa * (D_b_b**4))
        term5  = - (2 * self.p4 * y * dm**2 * self.int_buy) / (self.e * self.kappa * (D_b_a * (D_b_b**2)))
        term6  = - (4 * self.p4 * y**3 * dm * self.int_buy) / (self.e * self.kappa * (D_b_b**4))
        term7  =  (self.depth * y * dm * self.int_buy) / (self.e * self.kappa * (D_b_b))
        term8  = - (self.depth * dm * self.int_buy) / (self.e * self.kappa * D_b_a)
        term9  = - (2 * self.depth * y**2 * self.int_buy) / (self.e * self.kappa * (D_b_b))
        term10 =  (self.int_buy) / (self.e * self.kappa)
        
        # --- Sell-side terms (using lambda^+ = self.int_sell) ---
        term11 =  (self.p4 * dp**2 * self.int_sell) / (self.e * self.kappa * (D_s_a**2))
        term12 =  (2 * self.p4 * y * dp**2 * self.int_sell) / (self.e * self.kappa * (D_s_a * (D_s_b**2)))
        term13 =  (4 * self.p4 * y**2 * dp * self.int_sell) / (self.e * self.kappa * (D_s_a * (D_s_b**2)))
        term14 =  (4 * self.p4 * y**4 * self.int_sell) / (self.e * self.kappa * (D_s_b**4))
        term15 =  (self.p4 * y**2 * dp**2 * self.int_sell) / (self.e * self.kappa * (D_s_b**4))
        term16 =  (4 * self.p4 * y**3 * dp * self.int_sell) / (self.e * self.kappa * (D_s_b**4))
        term17 =  (self.depth * dp * self.int_sell) / (self.e * self.kappa * D_s_a)
        term18 =  (2 * self.depth * y**2 * self.int_sell) / (self.e * self.kappa * (D_s_b))
        term19 =  (self.depth * y * dp * self.int_sell) / (self.e * self.kappa * (D_s_b))
        term20 =  (self.int_sell) / (self.e * self.kappa)
        
        # --- Penalty term ---
        term_pen = - (9 * self.p4 * self.pen_const) / (y**4)
        
        psi10 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 +
                term8 + term9 + term10 + term11 + term12 + term13 + term14 +
                term15 + term16 + term17 + term18 + term19 + term20 + term_pen)
        
        return psi10

    def compute_psi_11(self):
        """
        Computes psi_11 given by:
        
        psi_11 = [2 p^2 y_0 (delta^-)^2 lambda^-]    / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [2 p^2 (delta^-)^2 lambda^-]          / [e k (y_0^2 - y_0 delta^-)]
                - [4 p^2 y_0^2 delta^- lambda^-]         / [e k (2 y_0^2 - y_0 delta^-)^2]
                + [delta^- lambda^-]                     / [e k]
                - [2 p^2 (delta^+)^2 lambda^+]           / [e k (y_0 delta^+ + y_0^2)]
                - [2 p^2 y_0 (delta^+)^2 lambda^+]       / [e k (y_0 delta^+ + 2 y_0^2)^2]
                - [4 p^2 y_0^2 delta^+ lambda^+]          / [e k (y_0 delta^+ + 2 y_0^2)^2]
                - [delta^+ lambda^+]                     / [e k]
                + [6 p^2 phi] / [y_0^2]
        """
        y = self.y_0

        # Precompute denominators for the buy side:
        D_b1 = (2 * y**2 - y * self.delta_minus)   # appears squared in some terms
        D_b2 = (y**2 - y * self.delta_minus)
        
        # Precompute denominators for the sell side:
        D_s1 = (y * self.delta_plus + y**2)
        D_s2 = (y * self.delta_plus + 2 * y**2)
        
        # Buy-side terms (using lambda^- = self.int_buy)
        term1 = (2 * self.depth * y * self.delta_minus**2 * self.int_buy) / (self.e * self.kappa * (D_b1**2))
        term2 = - (2 * self.depth * self.delta_minus**2 * self.int_buy) / (self.e * self.kappa * D_b2)
        term3 = - (4 * self.depth * y**2 * self.delta_minus * self.int_buy) / (self.e * self.kappa * (D_b1**2))
        term4 = (self.delta_minus * self.int_buy) / (self.e * self.kappa)
        
        # Sell-side terms (using lambda^+ = self.int_sell)
        term5 = - (2 * self.depth * self.delta_plus**2 * self.int_sell) / (self.e * self.kappa * D_s1)
        term6 = - (2 * self.depth * y * self.delta_plus**2 * self.int_sell) / (self.e * self.kappa * (D_s2**2))
        term7 = - (4 * self.depth * y**2 * self.delta_plus * self.int_sell) / (self.e * self.kappa * (D_s2**2))
        term8 = - (self.delta_plus * self.int_sell) / (self.e * self.kappa)
        
        # Penalty term:
        term_pen = (6 * self.depth * self.pen_const) / (y**2)
        
        psi11 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term_pen
        return psi11

    def compute_psi_12(self):
        """
        psi_12 = (delta^-)^2 int_buy / (e kappa)
               + (delta^+)^2 int_sell / (e kappa)
               - pen_const
        """
        return (
            (self.delta_minus**2 * self.int_buy)/(self.e * self.kappa)
            + (self.delta_plus**2  * self.int_sell)/(self.e * self.kappa)
            - self.pen_const
        )

    def compute_psi_13(self):
        """
        Computes psi_13 according to:
        
        psi_13 = [2 p^2 y_0 (delta^-)^3 lambda^-] / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [2 p^2 (delta^-)^3 lambda^-]   / [e k (y_0^2 - y_0 delta^-)]
                - [4 p^2 y_0^2 (delta^-)^2 lambda^-] / [e k (2 y_0^2 - y_0 delta^-)^2]
                + [(delta^-)^2 lambda^-]           / [e k]
                + [2 p^2 (delta^+)^3 lambda^+]      / [e k (y_0 delta^+ + y_0^2)]
                + [2 p^2 y_0 (delta^+)^3 lambda^+]   / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [4 p^2 y_0^2 (delta^+)^2 lambda^+]  / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [(delta^+)^2 lambda^+]           / [e k]
        """
        y  = self.y_0
        dm = self.delta_minus
        dp = self.delta_plus

        # Precompute denominators for the buy side:
        D_b1 = (y**2 - y * dm)         # denominator from second term
        D_b2 = (2 * y**2 - y * dm)       # appears squared in first and third terms

        # Precompute denominators for the sell side:
        D_s1 = (y * dp + y**2)          # denominator for the 5th term
        D_s2 = (y * dp + 2 * y**2)        # appears squared in 6th and 7th terms

        # Buy-side terms (with lambda^- = self.int_buy):
        term1 = (2 * self.depth * y * dm**3 * self.int_buy) / (self.e * self.kappa * (D_b2**2))
        term2 = - (2 * self.depth * dm**3 * self.int_buy) / (self.e * self.kappa * D_b1)
        term3 = - (4 * self.depth * y**2 * dm**2 * self.int_buy) / (self.e * self.kappa * (D_b2**2))
        term4 = (dm**2 * self.int_buy) / (self.e * self.kappa)

        # Sell-side terms (with lambda^+ = self.int_sell):
        term5 = (2 * self.depth * dp**3 * self.int_sell) / (self.e * self.kappa * D_s1)
        term6 = (2 * self.depth * y * dp**3 * self.int_sell) / (self.e * self.kappa * (D_s2**2))
        term7 = (4 * self.depth * y**2 * dp**2 * self.int_sell) / (self.e * self.kappa * (D_s2**2))
        term8 = (dp**2 * self.int_sell) / (self.e * self.kappa)

        psi13 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
        return psi13

    def compute_psi_14(self):
        """
        psi_14 = 2 (delta^-)^3 int_buy /(e kappa)
               - 2 (delta^+)^3 int_sell/(e kappa)
        """
        return (
            (2 * (self.delta_minus**3) * self.int_buy)/(self.e * self.kappa)
            - (2 * (self.delta_plus**3)  * self.int_sell)/(self.e * self.kappa)
        )

    def compute_psi_15(self):
        """
        psi_15 = (delta^-)^4 int_buy /(e kappa)
               + (delta^+)^4 int_sell/(e kappa)
        """
        return (
            (self.delta_minus**4 * self.int_buy)/(self.e * self.kappa)
            + (self.delta_plus**4  * self.int_sell)/(self.e * self.kappa)
        )

    def compute_psi_16(self):
        """
        psi_16 = 2 (delta^+)^3 int_sell/(e kappa)
               - 2 (delta^-)^3 int_buy /(e kappa)
        """
        return (
            (2 * (self.delta_plus**3)  * self.int_sell)/(self.e * self.kappa)
            - (2 * (self.delta_minus**3) * self.int_buy)/(self.e * self.kappa)
        )

    def compute_psi_17(self):
        """
        Computes psi_17 given by:
        
        psi_17 = [2 p^2 y_0 (delta^-)^3 lambda^-]   / [e k (2 y_0^2 - y_0 delta^-)^2]
                - [2 p^2 (delta^-)^3 lambda^-]         / [e k (y_0^2 - y_0 delta^-)]
                - [4 p^2 y_0^2 (delta^-)^2 lambda^-]    / [e k (2 y_0^2 - y_0 delta^-)^2]
                + [(delta^-)^2 lambda^-]                / [e k]
                + [2 p^2 (delta^+)^3 lambda^+]          / [e k (y_0 delta^+ + y_0^2)]
                + [2 p^2 y_0 (delta^+)^3 lambda^+]       / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [4 p^2 y_0^2 (delta^+)^2 lambda^+]      / [e k (y_0 delta^+ + 2 y_0^2)^2]
                + [(delta^+)^2 lambda^+]                / [e k]
        """
        y  = self.y_0
        dm = self.delta_minus
        dp = self.delta_plus

        # Precompute denominators for the buy side:
        D_b1 = y**2 - y * dm          # Denom for the second term
        D_b2 = 2 * y**2 - y * dm       # Denom for first and third terms (squared)

        # Precompute denominators for the sell side:
        D_s1 = y * dp + y**2          # Denom for the 5th term
        D_s2 = y * dp + 2 * y**2       # Denom for 6th and 7th terms (squared)

        # Buy-side terms (with lambda^- = self.int_buy):
        term1 = (2 * self.depth * y * dm**3 * self.int_buy) / (self.e * self.kappa * (D_b2**2))
        term2 = - (2 * self.depth * dm**3 * self.int_buy) / (self.e * self.kappa * D_b1)
        term3 = - (4 * self.depth * y**2 * dm**2 * self.int_buy) / (self.e * self.kappa * (D_b2**2))
        term4 = (dm**2 * self.int_buy) / (self.e * self.kappa)

        # Sell-side terms (with lambda^+ = self.int_sell):
        term5 = (2 * self.depth * dp**3 * self.int_sell) / (self.e * self.kappa * D_s1)
        term6 = (2 * self.depth * y * dp**3 * self.int_sell) / (self.e * self.kappa * (D_s2**2))
        term7 = (4 * self.depth * y**2 * dp**2 * self.int_sell) / (self.e * self.kappa * (D_s2**2))
        term8 = (dp**2 * self.int_sell) / (self.e * self.kappa)

        psi17 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
        return psi17

    def compute_psi_18(self):
        """
        psi_18 = -2 (delta^-)^2 int_buy /(e kappa)
                -2 (delta^+)^2 int_sell/(e kappa)
        """
        return (
            -(2 * (self.delta_minus**2) * self.int_buy)/(self.e * self.kappa)
            - (2 * (self.delta_plus**2)  * self.int_sell)/(self.e * self.kappa)
        )

    def compute_psi_19(self):
        """
        psi_19 = (delta^-)^2 int_buy /(e kappa)
               + (delta^+)^2 int_sell/(e kappa)
        """
        return (
            (self.delta_minus**2 * self.int_buy)/(self.e * self.kappa)
            + (self.delta_plus**2  * self.int_sell)/(self.e * self.kappa)
        )
    
    def solve_riccati(self):
        Nt = 1000
        t_min = 0

        def riccati_equation(t,A):
            return -(self.compute_psi_0() + self.compute_psi_1() * A + self.compute_psi_2() * (A**2))
        
        _ts = np.linspace(0,1,Nt +1)
    
  
        sol = solve_ivp(
            riccati_equation,
            [1,0],
            [0],
            t_eval = _ts[::-1]
        )
    
        t_sol = sol.t[::-1]
        A_sol = sol.y[0][::-1]
    
        return t_sol, A_sol
    
    def solve_system_ODE(self):
        Nt = 1000
        ts_asc = np.linspace(0, 1, Nt + 1)
        
        Gt = lambda t, q: np.array([
                # 1) A'(t)
                -(
                    self.compute_psi_0()
                    + self.compute_psi_1() * q[0]
                    + self.compute_psi_2() * (q[0] ** 2)
                ),
                # 2) b0'(t)
                -(
                    self.compute_psi_9() * q[1]
                    + self.compute_psi_8() * q[0] * q[1]
                    + self.compute_psi_5() * q[0]
                    + self.compute_psi_7() * (q[0] ** 2)
                    + self.compute_psi_3()
                ),
                # 3) b1'(t)
                -(
                    self.compute_psi_9() * q[2]
                    + self.compute_psi_8() * q[0] * q[2]
                    + self.compute_psi_6() * q[0]
                    + self.compute_psi_4()
                ),
                # 4) c0'(t)
                -(
                    self.compute_psi_16() * q[0] * q[1]
                    + self.compute_psi_17() * q[1]
                    + self.compute_psi_19() * (q[1] ** 2)
                    + self.compute_psi_10()
                    + self.compute_psi_13() * q[0]
                    + self.compute_psi_15() * (q[0] ** 2)
                    + self.sigma**2 * q[5]
                ),
                # 5) c1'(t)
                -(
                    2 * self.compute_psi_19() * q[1] * q[2]
                    + self.compute_psi_18() * q[1]
                    + self.compute_psi_16() * q[0] * q[2]
                    + self.compute_psi_17() * q[2]
                    + self.compute_psi_11()
                    + self.compute_psi_14() * q[0]
                ),
                # 6) c2'(t)
                -(
                    self.compute_psi_18() * q[2]
                    + self.compute_psi_19() * (q[2] ** 2)
                    + self.compute_psi_12()
                ),
            ])
        
        # We'll integrate *backward* from [self.T .. 0].
        # Because we want the solver to evaluate times in descending order,
        # pass t_eval=ts_asc[::-1], which goes from 1 down to 0.
        sol = solve_ivp(
            Gt,
            [self.T, 0],
            y0=np.array([0, 0, 0, 0, 0, 0]),  # A(T)=b0(T)=...=c2(T)=0
            t_eval=ts_asc[::-1]
        )
        
        # The solver returns sol.t in descending order matching t_eval,
        # i.e., from self.T down to 0. We'll reverse so final results
        # go from 0..1 in ascending order.
        t_sol_desc = sol.t
        q_sol_desc = sol.y  # shape (6, #points)
        
        t_sol = t_sol_desc[::-1]       # now ascending 0..1
        q_sol = q_sol_desc[:,::-1]    # match the reversed time dimension
        
        return t_sol, q_sol
    
    def _calculate_g_t(self,s):
        t_sol,q_sol = self.solve_system_ODE()
        
        A = q_sol[0]
        B = q_sol[1] + s*q_sol[2]
        C = q_sol[3] + q_sol[4]*s + q_sol[5] * (s**2)

        A_2d = A[:, None]  # (N_t,1)
        B_2d = B[:, None]
        C_2d = C[:, None]
        y_2d = self.y_grid[None, :]  # (1,N_y)

        # 4) The broadcasting expression
        g = (y_2d**2)*A_2d + (y_2d)*B_2d + C_2d
        return t_sol,g
    
    def _calculate_fees_t(self,t,s): # Compute the optimal fees
        t_sol, g = self._calculate_g_t(s)
        index = np.where(t_sol == 0.5)[0]
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1./self.kappa + g[index,i] - g[index,i+1])/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1./self.kappa + g[index,i] - g[index,i-1])/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        return p, m
    

class F_AMM:
    def __init__(self,int_sell, int_buy, kappa, oracleprice, depth, y_grid, y_0, T =1., pen_const=0., sigma = 10):
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

        # volatility
        self.sigma = sigma

    def level_fct(self,y): # We assume CPMM
        return self.depth / y
    
    def der_level_fct(self,y): # Returns the derivative of the level function
        return - self.depth/(y**2)
    
    def delta_buy(self, y, i): #Compute the trading size for buying
        indicator_buy = np.where(i - 1 >= 0, 1, 0)
        return y[i] - y[i-1*indicator_buy]
    
    def delta_sell(self, y, i): #Compute the trading size for selling
        indicator_sell = np.where(i + 1 < self.dim, 1, 0)
        return y[i+1*indicator_sell] - y[i]
    
    def _calculate_matrix_t(self,t,s): # Compute the matrix A
        A_matrix = np.zeros((self.dim,self.dim))
        for i in range(self.dim): 
            quantity = self.y_grid[i]  
            A_matrix[i,i] = - self.kappa*self.pen_const*(-self.der_level_fct(quantity) - s)**2
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+1]
                A_matrix[i,i+1] = self.int_sell * np.exp(-self.kappa * s*self.delta_sell(self.y_grid,i) -1 + self.kappa*(self.level_fct(quantity) - self.level_fct(quantity_P1)))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i,i-1] = self.int_buy * np.exp(self.kappa * s*self.delta_buy(self.y_grid,i) -1 - self.kappa*(self.level_fct(quantity_M1) - self.level_fct(quantity)))
        return A_matrix
    
    def _calculate_omega_t(self,t,s): # Compute the function omega 
        A_matrix = np.zeros((self.dim,self.dim))
        vector = np.ones((self.dim,1))
        A_matrix = self._calculate_matrix_t(t,s)
        return np.matmul(expm(A_matrix*(self.T-t)), vector)

    def _calculate_v_t(self,t,s):  # Compute the function v 
        omega_function = self._calculate_omega_t(t,s)
        return ( 1 / self.kappa) * np.log(omega_function)
    
    def _calculate_fees_t(self,t,s): # Compute the optimal fees
        v_qs = self._calculate_v_t(t,s)
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1./self.kappa + v_qs[i,0] - v_qs[i+1,0])/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1./self.kappa + v_qs[i,0] - v_qs[i-1,0])/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        return p, m
    
    def get_arrival(self,stoc_intensity_sell,stoc_intensity_buy,num_simulations,dt): # Given intensity compute if there is a jump or not
        unif_s = np.random.uniform(size=num_simulations)
        unif_b = np.random.uniform(size=num_simulations)
        return unif_s < 1. - np.exp(-stoc_intensity_sell * dt), unif_b < 1. - np.exp(-stoc_intensity_buy * dt)
    
    def get_linear_fees(self,t,s): # Compute the linear fees 
        p, m = self._calculate_fees_t(t,s)
        lin_p = np.zeros((self.dim))
        lin_m = np.zeros((self.dim))
        min_idx = self.dim//2 - 1
        max_idx = self.dim//2 + 1
        slope_p = (p[max_idx] - p[min_idx]) / (self.y_grid[max_idx] - self.y_grid[min_idx])
        slope_m = (m[max_idx] - m[min_idx]) / (self.y_grid[max_idx] - self.y_grid[min_idx])
        for i,q in enumerate(self.y_grid):
            lin_p[i] = q * slope_p + p[max_idx] - slope_p* self.y_grid[max_idx]
            lin_m[i] = q * slope_m + m[max_idx] - slope_m* self.y_grid[max_idx]
        #lin_p[-1] = "NaN"
        #lin_m[0] = "NaN"
        return lin_p,lin_m,slope_p, slope_m
    
    def compute_intensities(self, p, m, idx_quantity,s): # Compute the intensities
        indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
        indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        stoch_int_sell = self.int_sell * np.exp( self.kappa * ((1. - p[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1*indicator_sell])) - s * self.delta_sell(self.y_grid, idx_quantity)) )
        stoch_int_buy = self.int_buy * np.exp( -self.kappa * ((1. + m[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity-1*indicator_buy]) - self.level_fct(self.y_grid[idx_quantity])) - s * self.delta_buy(self.y_grid, idx_quantity) ) )
        return stoch_int_sell, stoch_int_buy
     
    def compute_cash_step(self, p, m, idx_quantity, sell_order, buy_order):
        indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
        indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        cash_step = p[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1*indicator_sell])) * sell_order.astype(int)  \
                        + m[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity-1*indicator_buy]) - self.level_fct(self.y_grid[idx_quantity])) * buy_order.astype(int) 
        return cash_step
        
    def simulate_PnL(self, nsims, Nt, strategy, const=0, seed = 123, return_trajectory = False):
        np.random.seed(seed=seed)
        dt = self.T/Nt
        timesteps = np.linspace(0, self.T, Nt+1)
        cash = np.zeros((nsims))
        n_sell_order = np.zeros((nsims))
        n_buy_order = np.zeros((nsims))
        idx_quantity = np.full(nsims, self.dim // 2, dtype=int)
        dW = np.random.normal(0, np.sqrt(dt), Nt)  # Brownian increments
        W = np.hstack((0, np.cumsum(dW)))  # Start W at 0
        St = self.oracleprice + self.sigma * W
        if return_trajectory:
            traj_quantity = np.zeros((nsims, Nt+1))
            traj_quantity[:,0] = self.y_grid[[self.dim // 2]] 
        stoch_int_sell = np.zeros((nsims))
        stoch_int_buy = np.zeros((nsims))
        for it, t in enumerate(timesteps[:-1]):
            if strategy == "Optimal":
                p, m = self._calculate_fees_t(t,St[it])
            if strategy == "Linear":
                p, m,_,_ = self.get_linear_fees(t,St[it])
            if strategy == "Constant":
                p_opt, m_opt = self._calculate_fees_t(t,St[it])
                c = np.round((p_opt[20] + m_opt[20])/2,2)
                cp = c + c*0.05
                cm = c - c*0.05
                if const == 0:
                    p = c*np.ones((self.dim))
                    m = c*np.ones((self.dim))
                if const == 5:
                    p = cp*np.ones((self.dim))
                    m = cp*np.ones((self.dim))
                if const == -5:
                    p = cm*np.ones((self.dim))
                    m = cm*np.ones((self.dim))
            indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
            indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        
            stoch_int_sell, stoch_int_buy = self.compute_intensities(p, m, idx_quantity,St[it])
            sell_order, buy_order = self.get_arrival(stoch_int_sell, stoch_int_buy, nsims, dt)
            
            
            sell_order = (sell_order & indicator_sell).astype(int)
            buy_order = (buy_order & indicator_buy).astype(int)
            
            cash += self.compute_cash_step(p, m, idx_quantity, sell_order, buy_order)
            
            idx_quantity += sell_order.astype(int) - buy_order.astype(int)
            n_sell_order += sell_order.astype(int)
            n_buy_order += buy_order.astype(int)
            
            if return_trajectory:
                traj_quantity[:,it+1] = self.y_grid[idx_quantity]
        if return_trajectory:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order, traj_quantity, St)
        else:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order)
        
    def _calculate_fees_k_0_t(self, t): # Compute the optimal fees
        A_matrix = np.zeros((self.dim,self.dim))
        vector = np.ones((self.dim,1))
        for i in range(self.dim): # Define the matrix A
            quantity = self.y_grid[i]  
            A_matrix[i,i] = 0
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+1]
                A_matrix[i,i+1] = self.int_sell * np.exp(-1)
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i,i-1] = self.int_buy * np.exp(-1)
        omega = np.matmul(expm(A_matrix*(self.T-t)), vector)
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1. + np.log(omega[i,0]) - np.log(omega[i+1,0]))/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1. + np.log(omega[i,0]) - np.log(omega[i-1,0]))/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        p[-1] = np.NaN
        m[0] = np.NaN
        return p, m
    
        

    