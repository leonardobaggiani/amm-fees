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
        self.sigma = 0

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
    def __init__(self,int_sell, int_buy, kappa, oracleprice, depth, y_grid, y_0, T =1., pen_const=0., sigma = 0.2, delta_minus = 0.5, delta_plus = 0.5):
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

        self.delta_minus = delta_minus
        self.delta_plus = delta_plus
        
        self.e = math.e  # Base of natural logarithm
        self.p4 = self.depth**2  # p^4 = (p^2)^2 = depth^2
    
    def level_fct(self,y): # We assume CPMM
        return self.depth / y
    
    def der_level_fct(self,y): # Returns the derivative of the level function
        return - self.depth/(y**2)

    def compute_psi_0(self):
        # Shorthand assignments:
        y0 = self.y_0         # y₀
        p4 = self.p4          # p^4
        phi = self.pen_const  # φ
        k = self.kappa        # k
        e = self.e            # e
        delta_m = self.delta_minus  # δ⁻
        delta_p = self.delta_plus   # δ⁺
        lambda_m = self.int_buy     # λ⁻
        lambda_p = self.int_sell    # λ⁺

        # Denominators
        denom_buy = (y0**2 - y0 * delta_m)**4
        denom_sell = (y0 * delta_p + y0**2)**4

        # Terms corresponding to the LaTeX formula:
        term1 = - (4 * p4 * phi) / (y0**6)
        term2 = (2 * k * p4 * y0**2 * (delta_m**2) * lambda_m) / (e * denom_buy)
        term3 = - (2 * k * p4 * y0 * (delta_m**3) * lambda_m) / (e * denom_buy)
        term4 = (k * p4 * (delta_m**4) * lambda_m) / (2 * e * denom_buy)
        term5 = (2 * k * p4 * y0**2 * (delta_p**2) * lambda_p) / (e * denom_sell)
        term6 = (2 * k * p4 * y0 * (delta_p**3) * lambda_p) / (e * denom_sell)
        term7 = (k * p4 * (delta_p**4) * lambda_p) / (2 * e * denom_sell)

        return term1 + term2 + term3 + term4 + term5 + term6 + term7


    def compute_psi_1(self):
        
        # Shorthand assignments:
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators:
        denom_buy = (y0**2 - y0 * delta_m)**2
        denom_sell = (y0**2 + y0 * delta_p)**2

        # Terms as per the LaTeX expression:
        term1 = (2 * k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
        term2 = - (4 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
        term3 = - (2 * k * p2 * (delta_p**3) * lambda_p) / (e * denom_sell)
        term4 = - (4 * k * p2 * y0 * (delta_p**2) * lambda_p) / (e * denom_sell)

        return term1 + term2 + term3 + term4


    def compute_psi_2(self):
        # LaTeX for Ψ₂:
        # Ψ₂ = ( -2·k·δ⁻·λ⁻/(E) + 2·k·δ⁺·λ⁺/(E) )
        e  = self.e
        k  = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell

        term1 = (2 * k * (δm**2) * λm) / e
        term2 = (2 * k * (δp**2) * λp) / e
        return term1 + term2


    def compute_psi_3(self):
        # Shorthand assignments:
        y0 = self.y_0              # y₀
        p4 = self.p4               # p^4
        p2 = self.depth            # p^2
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺
        phi = self.pen_const       # φ

        # Denominators for the buy side:
        denom_buy_3 = (y0**2 - y0 * delta_m)**3
        denom_buy_4 = (y0**2 - y0 * delta_m)**4
        denom_buy_2 = (y0**2 - y0 * delta_m)**2

        # Denominators for the sell side:
        denom_sell_3 = (y0 * delta_p + y0**2)**3
        denom_sell_4 = (y0 * delta_p + y0**2)**4
        denom_sell_2 = (y0 * delta_p + y0**2)**2

        # Buy-side terms:
        term1 = (k * p4 * (delta_m**3) * lambda_m) / (e * denom_buy_3)
        term2 = (4 * k * p4 * y0**2 * (delta_m**3) * lambda_m) / (e * denom_buy_4)
        term3 = - (2 * k * p4 * y0 * (delta_m**2) * lambda_m) / (e * denom_buy_3)
        term4 = - (k * p4 * y0 * (delta_m**4) * lambda_m) / (e * denom_buy_4)
        term5 = - (4 * k * p4 * y0**3 * (delta_m**2) * lambda_m) / (e * denom_buy_4)
        term6 = (2 * p2 * y0 * delta_m * lambda_m) / (e * denom_buy_2)
        term7 = - (p2 * (delta_m**2) * lambda_m) / (e * denom_buy_2)

        # Sell-side terms:
        term8 = - (k * p4 * (delta_p**3) * lambda_p) / (e * denom_sell_3)
        term9 = - (2 * k * p4 * y0 * (delta_p**2) * lambda_p) / (e * denom_sell_3)
        term10 = - (k * p4 * y0 * (delta_p**4) * lambda_p) / (e * denom_sell_4)
        term11 = - (4 * k * p4 * y0**2 * (delta_p**3) * lambda_p) / (e * denom_sell_4)
        term12 = - (4 * k * p4 * y0**3 * (delta_p**2) * lambda_p) / (e * denom_sell_4)
        term13 = - (p2 * (delta_p**2) * lambda_p) / (e * denom_sell_2)
        term14 = - (2 * p2 * y0 * delta_p * lambda_p) / (e * denom_sell_2)

        # The phi term:
        term15 = (12 * p4 * phi) / (y0**5)

        return (term1 + term2 + term3 + term4 + term5 +
                term6 + term7 + term8 + term9 + term10 +
                term11 + term12 + term13 + term14 + term15)


    def compute_psi_4(self):
        # Shorthand assignments:
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators for the buy side:
        denom_buy_1 = (y0**2 - y0 * delta_m)
        denom_buy_2 = denom_buy_1**2

        # Denominators for the sell side:
        denom_sell_1 = (y0**2 + y0 * delta_p)
        denom_sell_2 = denom_sell_1**2

        # Terms as per the LaTeX expression:
        term1 = - (k * p2 * lambda_m * (delta_m**4)) / (e * denom_buy_2)
        term2 = (2 * k * p2 * lambda_m * (delta_m**2)) / (e * denom_buy_1)
        term3 = (4 * k * p2 * y0**2 * lambda_m * (delta_m**2)) / (e * denom_buy_2)
        term4 = - (2 * lambda_m * delta_m) / e
        term5 = (2 * lambda_p * delta_p) / e
        term6 = (2 * k * p2 * (delta_p**2) * lambda_p) / (e * denom_sell_1)
        term7 = - (k * p2 * (delta_p**4) * lambda_p) / (e * denom_sell_2)
        term8 = (4 * k * p2 * y0**2 * (delta_p**2) * lambda_p) / (e * denom_sell_2)

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8



    def compute_psi_5(self):
        # LaTeX for Ψ₅:
        # Ψ₅ = ( -2·k·(δ⁺)³·λ⁺ + 2·k·(δ⁻)³·λ⁻ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (2 * k * (δp**3) * λp) / e
        term2 = - (2 * k * (δm**3) * λm) / e
        return term1 + term2


    def compute_psi_6(self):
        # LaTeX for Ψ₆:
        # Ψ₆ = ( -2·k·(δ⁻)·λ⁻ + 2·k·(δ⁺)·λ⁺ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (2 * k * (δm**2) * λm) / e
        term2 = (2 * k * (δp**2) * λp) / e
        return term1 + term2


    def compute_psi_7(self):
        # Shorthand assignments:
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators:
        denom_buy = (y0**2 - y0 * delta_m)**2
        denom_sell = (y0**2 + y0 * delta_p)**2

        term1 = (k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
        term2 = - (2 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
        term3 = - (k * p2 * lambda_p * (delta_p**3)) / (e * denom_sell)
        term4 = - (2 * k * p2 * y0 * lambda_p * (delta_p**2)) / (e * denom_sell)

        return term1 + term2 + term3 + term4


    def compute_psi_8(self):
        # Shorthand assignments:
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺
        phi = self.pen_const       # φ

        # Denominators:
        denom_buy = (y0**2 - y0 * delta_m)**2
        # For the sell side, we use (y0*delta_p + y0^2)^2 (which equals y0^2 + y0*delta_p squared)
        denom_sell = (y0 * delta_p + y0**2)**2

        term1 = - (k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
        term2 = (2 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
        term3 = (k * p2 * lambda_p * (delta_p**3)) / (e * denom_sell)
        term4 = (2 * k * p2 * y0 * lambda_p * (delta_p**2)) / (e * denom_sell)
        term5 = - (4 * p2 * phi) / (y0**3)

        return term1 + term2 + term3 + term4 + term5


    def compute_psi_9(self):
        # LaTeX for Ψ₉:
        # Ψ₉ = ( -2·k·(δ⁻)²·λ⁻ + 2·k·(δ⁺)²·λ⁺ )/E
        e  = self.e
        k  = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (-2 * k * (δm**2) * λm) / e
        term2 = (-2 * k * (δp**2) * λp) / e
        return term1 + term2


    def compute_psi_10(self):
        # LaTeX for Ψ₁₀:
        # Ψ₁₀ = ( 2·k·(δ⁻)²·λ⁻ - 2·k·(δ⁺)²·λ⁺ )/E
        e  = self.e
        k  = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (2 * k * (δm**2) * λm) / e
        term2 = (2 * k * (δp**2) * λp) / e
        return term1 + term2


    def compute_psi_11(self):
        """
        Psi11 =
        ( k * p^2 * λ^- * (δ^-)^3 ) / ( e * (y0^2 - y0*δ^-)^2 )
        - ( 2 * k * p^2 * y0 * λ^- * (δ^-)^2 ) / ( e * (y0^2 - y0*δ^-)^2 )
        - ( k * p^2 * (δ^+)^3 * λ^+ ) / ( e * (y0^2 + y0*δ^+)^2 )
        - ( 2 * k * p^2 * y0 * (δ^+)^2 * λ^+ ) / ( e * (y0^2 + y0*δ^+)^2 )
        """
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        denom_buy = (y0**2 - y0 * delta_m)**2
        denom_sell = (y0**2 + y0 * delta_p)**2

        term1 = (k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
        term2 = - (2 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
        term3 = - (k * p2 * lambda_p * (delta_p**3)) / (e * denom_sell)
        term4 = - (2 * k * p2 * y0 * lambda_p * (delta_p**2)) / (e * denom_sell)

        return term1 + term2 + term3 + term4


    def compute_psi_12(self):
        """
        Psi12 =
        [ k * (δ^-)^2 * λ^- * p^4 ] / [ 2 e (y0^2 - y0 δ^-)^2 ]
        - [ k * y0 * (δ^-)^3 * λ^- * p^4 ] / [ e (y0^2 - y0 δ^-)^3 ]
        + [ 2 k * y0^2 * (δ^-)^2 * λ^- * p^4 ] / [ e (y0^2 - y0 δ^-)^3 ]
        + [ k * y0^2 * (δ^-)^4 * λ^- * p^4 ] / [ 2 e (y0^2 - y0 δ^-)^4 ]
        - [ 2 k * y0^3 * (δ^-)^3 * λ^- * p^4 ] / [ e (y0^2 - y0 δ^-)^4 ]
        + [ 2 k * y0^4 * (δ^-)^2 * λ^- * p^4 ] / [ e (y0^2 - y0 δ^-)^4 ]
        + [ k * (δ^+)^2 * λ^+ * p^4 ] / [ 2 e (y0^2 + y0 δ^+)^2 ]
        + [ k * y0 * (δ^+)^3 * λ^+ * p^4 ] / [ e (y0^2 + y0 δ^+)^3 ]
        + [ 2 k * y0^2 * (δ^+)^2 * λ^+ * p^4 ] / [ e (y0^2 + y0 δ^+)^3 ]
        + [ k * y0^2 * (δ^+)^4 * λ^+ * p^4 ] / [ 2 e (y0^2 + y0 δ^+)^4 ]
        + [ 2 k * y0^3 * (δ^+)^3 * λ^+ * p^4 ] / [ e (y0^2 + y0 δ^+)^4 ]
        + [ 2 k * y0^4 * (δ^+)^2 * λ^+ * p^4 ] / [ e (y0^2 + y0 δ^+)^4 ]
        - [ 9 φ p^4 ] / [ y0^4 ]
        - [ δ^- λ^- p^2 ] / [ e (y0^2 - y0 δ^-) ]
        + [ y0 (δ^-)^2 λ^- p^2 ] / [ e (y0^2 - y0 δ^-)^2 ]
        - [ 2 y0^2 δ^- λ^- p^2 ] / [ e (y0^2 - y0 δ^-)^2 ]
        + [ δ^+ λ^+ p^2 ] / [ e (y0^2 + y0 δ^+) ]
        + [ y0 (δ^+)^2 λ^+ p^2 ] / [ e (y0^2 + y0 δ^+)^2 ]
        + [ 2 y0^2 δ^+ λ^+ p^2 ] / [ e (y0^2 + y0 δ^+)^2 ]
        + (λ^-)/(e k)
        + (λ^+)/(e k)
        + [c0'(t)]  (ignored)
        """
        y0 = self.y_0              # y₀
        p4 = self.p4               # p^4
        p2 = self.depth            # p^2
        k = self.kappa             # k
        e = self.e                 # e
        phi = self.pen_const       # φ
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators for the buy side:
        denom_buy_1 = (y0**2 - y0 * delta_m)         # power 1
        denom_buy_2 = denom_buy_1**2                  # power 2
        denom_buy_3 = denom_buy_1**3                  # power 3
        denom_buy_4 = denom_buy_1**4                  # power 4

        # Denominators for the sell side:
        denom_sell_1 = (y0**2 + y0 * delta_p)         # power 1
        denom_sell_2 = denom_sell_1**2                # power 2
        denom_sell_3 = denom_sell_1**3                # power 3
        denom_sell_4 = denom_sell_1**4                # power 4

        # Buy-side terms:
        A = (k * (delta_m**2) * lambda_m * p4) / (2 * e * denom_buy_2)
        B = - (k * y0 * (delta_m**3) * lambda_m * p4) / (e * denom_buy_3)
        C = (2 * k * y0**2 * (delta_m**2) * lambda_m * p4) / (e * denom_buy_3)
        D = (k * y0**2 * (delta_m**4) * lambda_m * p4) / (2 * e * denom_buy_4)
        E = - (2 * k * y0**3 * (delta_m**3) * lambda_m * p4) / (e * denom_buy_4)
        F = (2 * k * y0**4 * (delta_m**2) * lambda_m * p4) / (e * denom_buy_4)

        # Sell-side terms:
        G = (k * (delta_p**2) * lambda_p * p4) / (2 * e * denom_sell_2)
        H = (k * y0 * (delta_p**3) * lambda_p * p4) / (e * denom_sell_3)
        I = (2 * k * y0**2 * (delta_p**2) * lambda_p * p4) / (e * denom_sell_3)
        J = (k * y0**2 * (delta_p**4) * lambda_p * p4) / (2 * e * denom_sell_4)
        K = (2 * k * y0**3 * (delta_p**3) * lambda_p * p4) / (e * denom_sell_4)
        L = (2 * k * y0**4 * (delta_p**2) * lambda_p * p4) / (e * denom_sell_4)

        M = - (9 * phi * p4) / (y0**4)

        # Buy-side p2 terms:
        N = - (delta_m * lambda_m * p2) / (e * denom_buy_1)
        O = (y0 * (delta_m**2) * lambda_m * p2) / (e * denom_buy_2)
        P = - (2 * y0**2 * delta_m * lambda_m * p2) / (e * denom_buy_2)

        # Sell-side p2 terms:
        Q = (delta_p * lambda_p * p2) / (e * denom_sell_1)
        R = (y0 * (delta_p**2) * lambda_p * p2) / (e * denom_sell_2)
        S = (2 * y0**2 * delta_p * lambda_p * p2) / (e * denom_sell_2)

        # Additional terms:
        T = lambda_m / (e * k)
        U = lambda_p / (e * k)

        # Ignore the derivative term c0'(t)

        return (A + B + C + D + E + F +
                G + H + I + J + K + L +
                M + N + O + P + Q + R + S +
                T + U)


    def compute_psi_13(self):
        """
        Psi13 =
        ( k * p^2 * y0 * λ^- * (δ^-)^4 ) / ( e * (y0^2 - y0 δ^-)^2 )
        - ( k * p^2 * λ^- * (δ^-)^3 ) / ( e * (y0^2 - y0 δ^-) )
        - ( 2 k * p^2 * y0^2 * λ^- * (δ^-)^3 ) / ( e * (y0^2 - y0 δ^-)^2 )
        + ( λ^- * (δ^-)^2 ) / e
        + ( (δ^+)^2 * λ^+ ) / e
        + ( k * p^2 * (δ^+)^3 * λ^+ ) / ( e * (y0^2 + y0 δ^+) )
        + ( k * p^2 * y0 * (δ^+)^4 * λ^+ ) / ( e * (y0^2 + y0 δ^+)^2 )
        + ( 2 k * p^2 * y0^2 * (δ^+)^3 * λ^+ ) / ( e * (y0^2 + y0 δ^+)^2 )
        """
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators for the buy side:
        denom_buy_1 = (y0**2 - y0 * delta_m)       # power 1
        denom_buy_2 = denom_buy_1**2                # power 2

        # Denominators for the sell side:
        denom_sell_1 = (y0**2 + y0 * delta_p)       # power 1
        denom_sell_2 = denom_sell_1**2              # power 2

        term1 = (k * p2 * y0 * lambda_m * (delta_m**4)) / (e * denom_buy_2)
        term2 = - (k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy_1)
        term3 = - (2 * k * p2 * y0**2 * lambda_m * (delta_m**3)) / (e * denom_buy_2)
        term4 = (lambda_m * (delta_m**2)) / e
        term5 = (lambda_p * (delta_p**2)) / e
        term6 = (k * p2 * lambda_p * (delta_p**3)) / (e * denom_sell_1)
        term7 = (k * p2 * y0 * lambda_p * (delta_p**4)) / (e * denom_sell_2)
        term8 = (2 * k * p2 * y0**2 * lambda_p * (delta_p**3)) / (e * denom_sell_2)

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


    def compute_psi_14(self):
        # LaTeX for Ψ₁₄:
        # Ψ₁₄ = ( -4·k·δ⁻·λ⁻ + 4·k·δ⁺·λ⁺ )/(2·E)
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (k * (δm**4) * λm) / (2 * e)
        term2 = (k * (δp**4) * λp) / (2 * e)
        return term1 + term2


    def compute_psi_15(self):
        # LaTeX for Ψ₁₅:
        # Ψ₁₅ = ( -3·k·δ⁻·λ⁻ + 3·k·δ⁺·λ⁺ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = -(k * (δm**3) * λm) / e
        term2 = (k * (δp**3) * λp) / e
        return term1 + term2


    def compute_psi_16(self):
        """
        Psi16 =
        - ( k * p^2 * y0 * λ^- * (δ^-)^3 ) / ( e * (y0^2 - y0 δ^-)^2 )
        +   ( k * p^2 * λ^- * (δ^-)^2 ) / ( e * (y0^2 - y0 δ^-) )
        +   ( 2 k * p^2 * y0^2 * λ^- * (δ^-)^2 ) / ( e * (y0^2 - y0 δ^-)^2 )
        -   ( λ^- * δ^- ) / e
        +   ( δ^+ * λ^+ ) / e
        +   ( k * p^2 * (δ^+)^2 * λ^+ ) / ( e * (y0^2 + y0 δ^+) )
        +   ( k * p^2 * y0 * (δ^+)^3 * λ^+ ) / ( e * (y0^2 + y0 δ^+)^2 )
        +   ( 2 k * p^2 * y0^2 * (δ^+)^2 * λ^+ ) / ( e * (y0^2 + y0 δ^+)^2 )
        """
        # Shorthand assignments
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators for the buy side:
        denom_buy_1 = (y0**2 - y0 * delta_m)        # power 1
        denom_buy_2 = denom_buy_1**2                 # power 2

        # Denominators for the sell side:
        # Note: (y0^2 + y0 δ^+) is equivalent to (y0 δ^+ + y0^2)
        denom_sell_1 = (y0**2 + y0 * delta_p)        # power 1
        denom_sell_2 = denom_sell_1**2               # power 2

        term1 = - (k * p2 * y0 * lambda_m * (delta_m**3)) / (e * denom_buy_2)
        term2 =   (k * p2 * lambda_m * (delta_m**2)) / (e * denom_buy_1)
        term3 =   (2 * k * p2 * y0**2 * lambda_m * (delta_m**2)) / (e * denom_buy_2)
        term4 = - (lambda_m * delta_m) / e
        term5 =   (delta_p * lambda_p) / e
        term6 =   (k * p2 * (delta_p**2) * lambda_p) / (e * denom_sell_1)
        term7 =   (k * p2 * y0 * (delta_p**3) * lambda_p) / (e * denom_sell_2)
        term8 =   (2 * k * p2 * y0**2 * (delta_p**2) * lambda_p) / (e * denom_sell_2)

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


    def compute_psi_17(self):
        # LaTeX for Ψ₁₇:
        # Ψ₁₇ = ( -2·k·(δ⁻)·λ⁻ + 2·k·(δ⁺)·λ⁺ )/(2·E)
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (k * (δm**2) * λm) / (2 * e)
        term2 = (k * (δp**2) * λp) / (2 * e)
        return term1 + term2


    def compute_psi_18(self):
        
        """
        Psi18 =
        (6 p^2 φ)/y0^2
        + ( k * p^2 * y0 * λ^- * (δ^-)^3 ) / ( e * (y0^2 - y0 δ^-)^2 )
        - ( k * p^2 * λ^- * (δ^-)^2 ) / ( e * (y0^2 - y0 δ^-) )
        - ( 2 k * p^2 * y0^2 * λ^- * (δ^-)^2 ) / ( e * (y0^2 - y0 δ^-)^2 )
        + ( λ^- * δ^- ) / e
        - ( k * p^2 * (δ^+)^2 * λ^+ ) / ( e * (y0^2 + y0 δ^+) )
        - ( k * p^2 * y0 * (δ^+)^3 * λ^+ ) / ( e * (y0^2 + y0 δ^+)^2 )
        - ( 2 k * p^2 * y0^2 * (δ^+)^2 * λ^+ ) / ( e * (y0^2 + y0 δ^+)^2 )
        - ( δ^+ * λ^+ ) / e
        """
        # Shorthand assignments
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        phi = self.pen_const       # φ
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators for the buy side:
        denom_buy_1 = (y0**2 - y0 * delta_m)
        denom_buy_2 = denom_buy_1**2

        # Denominators for the sell side:
        denom_sell_1 = (y0**2 + y0 * delta_p)   # equivalent to (y0 δ^+ + y0^2)
        denom_sell_2 = denom_sell_1**2

        termA = (6 * p2 * phi) / (y0**2)
        termB = (k * p2 * y0 * lambda_m * (delta_m**3)) / (e * denom_buy_2)
        termC = - (k * p2 * lambda_m * (delta_m**2)) / (e * denom_buy_1)
        termD = - (2 * k * p2 * y0**2 * lambda_m * (delta_m**2)) / (e * denom_buy_2)
        termE =   (lambda_m * delta_m) / e

        termF = - (k * p2 * (delta_p**2) * lambda_p) / (e * denom_sell_1)
        termG = - (k * p2 * y0 * (delta_p**3) * lambda_p) / (e * denom_sell_2)
        termH = - (2 * k * p2 * y0**2 * (delta_p**2) * lambda_p) / (e * denom_sell_2)
        termI = - (delta_p * lambda_p) / e

        return termA + termB + termC + termD + termE + termF + termG + termH + termI



    def compute_psi_19(self):
        # LaTeX for Ψ₁₉:
        # Ψ₁₉ = ( k·δ⁻·λ⁻ - k·δ⁺·λ⁺ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        return (k * (δm**3) * λm - k * (δp**3) * λp) / e


    def compute_psi_20(self):
        # LaTeX for Ψ₂₀:
        # Ψ₂₀ = ( k·δ⁺·λ⁺ - k·δ⁻·λ⁻ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        return (-k * (δm**3) * λm + k * (δp**3) * λp) / e


    def compute_psi_21(self):
        # LaTeX for Ψ₂₁:
        # Ψ₂₁ = - ( k·δ⁻·λ⁻ + k·δ⁺·λ⁺ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        return (-k * (δm**2) * λm - k * (δp**2) * λp) / e


    def compute_psi_22(self):
        # LaTeX for Ψ₂₂:
        # Ψ₂₂ = ( k·δ⁻·λ⁻ + k·δ⁺·λ⁺ )/E
        e = self.e
        k = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        return (k * (δm**2) * λm + k * (δp**2) * λp) / e


    def compute_psi_23(self):
        # Shorthand assignments:
        y0 = self.y_0              # y₀
        p2 = self.depth            # p²
        k = self.kappa             # k
        e = self.e                 # e
        delta_m = self.delta_minus # δ⁻
        delta_p = self.delta_plus  # δ⁺
        lambda_m = self.int_buy    # λ⁻
        lambda_p = self.int_sell   # λ⁺

        # Denominators for the buy side:
        denom_buy_1 = (y0**2 - y0 * delta_m)
        denom_buy_2 = denom_buy_1**2

        # Denominators for the sell side:
        denom_sell_1 = (y0**2 + y0 * delta_p)
        denom_sell_2 = denom_sell_1**2

        term1 = - (k * p2 * y0 * lambda_m * (delta_m**3)) / (e * denom_buy_2)
        term2 =   (k * p2 * lambda_m * (delta_m**2)) / (e * denom_buy_1)
        term3 =   (2 * k * p2 * y0**2 * lambda_m * (delta_m**2)) / (e * denom_buy_2)
        term4 = - (lambda_m * delta_m) / e
        term5 =   (delta_p * lambda_p) / e
        term6 =   (k * p2 * (delta_p**2) * lambda_p) / (e * denom_sell_1)
        term7 =   (k * p2 * y0 * (delta_p**3) * lambda_p) / (e * denom_sell_2)
        term8 =   (2 * k * p2 * y0**2 * (delta_p**2) * lambda_p) / (e * denom_sell_2)

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


    def compute_psi_24(self):
        # LaTeX for Ψ₂₄:
        # Ψ₂₄ = - φ + ( (δ⁻)²·λ⁻ )/(2·E·k) + ( (δ⁺)²·λ⁺ )/(2·E·k)
        φ  = self.pen_const
        e  = self.e
        k  = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = - φ
        term2 = (δm**2 * λm*k) / (2 * e)
        term3 = (δp**2 * λp*k) / (2 * e)
        return term1 + term2 + term3


    def compute_psi_25(self):
        # LaTeX for Ψ₂₅:
        # Ψ₂₅ = - ( (δ⁻)²·λ⁻ )/(E·k) - ( (δ⁺)²·λ⁺ )/(E·k)
        e  = self.e
        k  = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = - (δm**2 * λm*k) / (e )
        term2 = - (δp**2 * λp*k) / (e )
        return term1 + term2


    def compute_psi_26(self):
        # LaTeX for Ψ₂₆:
        # Ψ₂₆ = ( (δ⁻)²·λ⁻ )/(2·E·k) + ( (δ⁺)²·λ⁺ )/(2·E·k)
        e  = self.e
        k  = self.kappa
        δm = self.delta_minus
        δp = self.delta_plus
        λm = self.int_buy
        λp = self.int_sell
        term1 = (δm**2 * λm*k) / (2*e )
        term2 = (δp**2 * λp*k) / (2*e )
        return term1 + term2
    
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
                self.compute_psi_7() * q[1]
                + self.compute_psi_6() * q[0] * q[1]
                + self.compute_psi_4() * q[0]
                + self.compute_psi_5() * (q[0] ** 2)
                + self.compute_psi_3()
            ),
            # 3) b1'(t)
            -(
                self.compute_psi_11() * q[2]
                + self.compute_psi_10() * q[0] * q[2]
                + self.compute_psi_9() * q[0]
                + self.compute_psi_8()
            ),
            # 4) c0'(t)
            -(
                self.compute_psi_15() * q[0] * q[1]  
                + self.compute_psi_16() * q[1]
                + self.compute_psi_17() * (q[1] ** 2)
                + self.compute_psi_12()
                + self.compute_psi_13() * q[0]
                + self.compute_psi_14() * (q[0] ** 2)
                + self.sigma**2 * q[5]
            ),
            # 5) c1'(t)
            -(
                self.compute_psi_22() * q[2] * q[1]  
                + self.compute_psi_23() * q[2]
                + self.compute_psi_20() * q[0] * q[2]
                + self.compute_psi_21() * q[1]
                + self.compute_psi_18()
                + self.compute_psi_19() * q[0]
            ),
            # 6) c2'(t)
            -(
                self.compute_psi_25() * q[2]
                + self.compute_psi_26() * (q[2] ** 2)
                + self.compute_psi_24()
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
    
    def _calculate_g_t(self, s):
        t_sol, q_sol = self.solve_system_ODE()

        A = q_sol[0]
        B = s*q_sol[2] + q_sol[1]   # note the swap: b0 + s*b1
        C = q_sol[3] + s * q_sol[4] + (s ** 2) * q_sol[5]

        A_2d = A[:, None]
        B_2d = B[:, None]
        C_2d = C[:, None]
        y_2d = self.y_grid[None, :]

        g = (y_2d ** 2) * A_2d + y_2d * B_2d + C_2d
        return t_sol, g
    
    def _calculate_fees_t(self,t,s): # Compute the optimal fees
        t_sol, q_sol = self.solve_system_ODE()
        index = np.where(t_sol == t)[0]
        A = q_sol[0][index]
        B = s*q_sol[2][index] + q_sol[1][index]   # note the swap: b0 + s*b1
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1./self.kappa - 2*quantity*self.delta_plus*A - (A**2)*self.delta_plus**2 - self.delta_plus*B)/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1./self.kappa + 2*quantity*self.delta_minus*A - (A**2)*self.delta_minus**2 + self.delta_minus*B)/(self.level_fct(quantity_M1) - self.level_fct(quantity))
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
    
        

    