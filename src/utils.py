import numpy as np
from scipy.linalg import expm
import math
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply
import torch
from scipy.interpolate import interp1d
from importlib import reload
from . import constants as const

reload(const)

class AMM:
    def __init__(self,int_sell, int_buy, kappa, oracleprice, depth, y_grid, y_0, sigma = 0.2, T =1., pen_const=0.,
                 Nt = 1000, Sgrid: np.ndarray = np.linspace(90, 110, 100)):
            
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

            self.Nt = Nt
            self.timesteps = np.linspace(0, self.T, self.Nt+1)
            self.Sgrid = Sgrid

            self.e = math.e  # Base of natural logarithm
            self.p4 = self.depth**2  # p^4 = (p^2)^2 = depth^2

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
    
    def _calculate_matrix_A(self,St): # Compute the matrix A
        St = np.atleast_1d(St)
        A_matrix = np.zeros((self.dim,self.dim,len(St)))
        for i in range(self.dim): 
            quantity = self.y_grid[i]                  
            A_matrix[i,i] = - self.kappa*self.pen_const*(-self.der_level_fct(quantity) - St)**2
            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i+1]
                A_matrix[i,i+1] = self.int_sell * np.exp(-self.kappa * St*self.delta_sell(self.y_grid,i) -1 + self.kappa*(self.level_fct(quantity) - self.level_fct(quantity_P1)))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                A_matrix[i,i-1] = self.int_buy * np.exp(self.kappa * St*self.delta_buy(self.y_grid,i) -1 - self.kappa*(self.level_fct(quantity_M1) - self.level_fct(quantity)))
        return A_matrix
    
    def _calculate_omega_t(self, t, St):
        # 1) get A in shape (dim, dim, T)
        A_np = self._calculate_matrix_A(St)  # shape: (dim, dim, len(St))

        # 2) move time axis to the front → (T, dim, dim)
        A_batch = np.moveaxis(A_np, 2, 0)

        # 3) convert to PyTorch tensor
        A = torch.from_numpy(A_batch).double()  # shape: (T, dim, dim)

        # 4) time step and constant vector
        dt  = self.T - t
        vec = torch.ones((self.dim, 1), dtype=torch.double)  # shape: (dim, 1)
        vec_batch = vec.unsqueeze(0).expand(A.shape[0], -1, -1)  # shape: (T, dim, 1)

        # 5) batched matrix exponential and multiplication
        expA = torch.matrix_exp(A * dt)             # shape: (T, dim, dim)
        omega = torch.bmm(expA, vec_batch)          # shape: (T, dim, 1)

        return omega.squeeze(-1).numpy()            # shape: (T, dim)

    def _calculate_v_t(self, t, St):
        omega = self._calculate_omega_t(t, St)   # (T, dim)
        return (1.0 / self.kappa) * np.log(omega)
    
    def _calculate_fees_first_approx_t(self, t,St): # Compute the optimal fees
        v_qs = self._calculate_v_t(t,St)
        St = np.atleast_1d(St)
        p = np.ones((self.dim,len(St)))
        m = np.ones((self.dim,len(St)))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i,:] = (1./self.kappa + v_qs[:,i] - v_qs[:,i+1])/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i,:] = (1./self.kappa + v_qs[:,i] - v_qs[:,i-1])/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        return p, m
    
    def get_linear_fees(self,t): # Compute the linear fees 
        p, m = self._calculate_fees_first_approx_t(t,self.oracleprice)
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
    
    def _calculate_fees_first_approx_t_k_0(self, t): # Compute the optimal fees in the case k=0
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
        p[-1] = np.nan
        m[0] = np.nan
        return p, m
    
    def solve_system_ODE(self):

        ts_asc = np.linspace(0, self.T, self.Nt + 1)
        
        Gt = lambda t, q: np.array([
            # 1) A'(t)
            -(
                const.compute_psi_0(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell)
                + const.compute_psi_1(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0]
                + const.compute_psi_2(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * (q[0] ** 2)
            ),
            # 2) b0'(t)
            -(
                const.compute_psi_7(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[1]
                + const.compute_psi_6(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0] * q[1]
                + const.compute_psi_4(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0]
                + const.compute_psi_5(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * (q[0] ** 2)
                + const.compute_psi_3(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell)
            ),
            # 3) b1'(t)
            -(
                const.compute_psi_11(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[2]
                + const.compute_psi_10(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0] * q[2]
                + const.compute_psi_9(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0]
                + const.compute_psi_8(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell)
            ),
            # 4) c0'(t)
            -(
                const.compute_psi_15(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0] * q[1]  
                + const.compute_psi_16(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[1]
                + const.compute_psi_17(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * (q[1] ** 2)
                + const.compute_psi_12(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell)
                + const.compute_psi_13(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0]
                + const.compute_psi_14(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * (q[0] ** 2)
                + self.sigma**2 * q[5]
            ),
            # 5) c1'(t)
            -(
                const.compute_psi_22(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[2] * q[1]  
                + const.compute_psi_23(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[2]
                + const.compute_psi_20(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0] * q[2]
                + const.compute_psi_21(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[1]
                + const.compute_psi_18(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell)
                + const.compute_psi_19(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[0]
            ),
            # 6) c2'(t)
            -(
                const.compute_psi_25(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * q[2]
                + const.compute_psi_26(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell) * (q[2] ** 2)
                + const.compute_psi_24(self.y_0,self.p4,self.depth,self.pen_const,self.kappa,self.e,self.delta_buy,self.delta_sell,self.y_grid,self.int_buy,self.int_sell)
            ),
        ])
        

        sol = solve_ivp(
            Gt,
            [self.T, 0],
            y0=np.array([0, 0, 0, 0, 0, 0]),  # A(T)=b0(T)=...=c2(T)=0
            t_eval=ts_asc[::-1]
        )
        
    
        t_sol_desc = sol.t
        q_sol_desc = sol.y  # shape (6, #points)
        
        t_sol = t_sol_desc[::-1]       # now ascending 0..1
        q_sol = q_sol_desc[:,::-1]    
        
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

    def _calculate_fees_second_approx_t(self, t, St):  # Compute the optimal fees for all St
        St = np.asarray(St)  # Ensure it's a NumPy array
        St = np.atleast_1d(St)
        n = len(St)
        
        t_sol, q_sol = self.solve_system_ODE()
        index = np.where(t_sol == t)[0]
        if len(index) == 0:
            raise ValueError(f"Time {t} not found in t_sol.")
        index = index[0]

        # Coefficients at time t (scalars)
        A = q_sol[0][index]
        B = St * q_sol[2][index] + q_sol[1][index]  # shape (n,)

        # Initialize fee matrices
        p = np.ones((self.dim, n))
        m = np.ones((self.dim, n))

        # Precompute deltas
        delta_sell = self.delta_sell(self.y_grid, 1)
        delta_buy = self.delta_buy(self.y_grid, 1)

        for i in range(self.dim):
            quantity = self.y_grid[i]

            if i < self.dim - 1:
                quantity_P1 = self.y_grid[i + 1]
                denom = self.level_fct(quantity) - self.level_fct(quantity_P1)
                numer = (
                    1. / self.kappa
                    - 2 * quantity * delta_sell * A
                    - (A ** 2) * delta_sell ** 2
                    - delta_sell * B  # shape (n,)
                )
                p[i, :] = numer / denom  # vectorized over St

            if i > 0:
                quantity_M1 = self.y_grid[i - 1]
                denom = self.level_fct(quantity_M1) - self.level_fct(quantity)
                numer = (
                    1. / self.kappa
                    + 2 * quantity * delta_buy * A
                    - (A ** 2) * delta_buy ** 2
                    + delta_buy * B  # shape (n,)
                )
                m[i, :] = numer / denom  # vectorized over St

        return p, m  # shape (dim, n)
    
    def _calculate_fees_second_approx_t_k0(self,t,s): # Compute the optimal fees
        p = np.ones((self.dim))
        m = np.ones((self.dim))
        for i in range(self.dim):
            quantity = self.y_grid[i]
            if i < self.dim -1:
                quantity_P1 = self.y_grid[i+1]
                p[i] = (1.)/(self.level_fct(quantity) - self.level_fct(quantity_P1))
            if i > 0:
                quantity_M1 = self.y_grid[i-1]
                m[i] = (1.)/(self.level_fct(quantity_M1) - self.level_fct(quantity))
        return p, m
    
    def get_arrival_first(self,stoc_intensity_sell,stoc_intensity_buy,num_simulations,dt): # Given intensity compute if there is a jump or not
        unif_s = np.random.uniform(size=num_simulations)
        unif_b = np.random.uniform(size=num_simulations)
        return unif_s < 1. - np.exp(-stoc_intensity_sell * dt), unif_b < 1. - np.exp(-stoc_intensity_buy * dt)
    
    def compute_intensities_first(self, p, m, idx_quantity): # Compute the intensities
        indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
        indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        stoch_int_sell = self.int_sell * np.exp( self.kappa * ((1. - p[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1*indicator_sell])) - self.oracleprice * self.delta_sell(self.y_grid, idx_quantity)) )
        stoch_int_buy = self.int_buy * np.exp( -self.kappa * ((1. + m[idx_quantity]) * (self.level_fct(self.y_grid[idx_quantity-1*indicator_buy]) - self.level_fct(self.y_grid[idx_quantity])) - self.oracleprice * self.delta_buy(self.y_grid, idx_quantity) ) )
        return stoch_int_sell, stoch_int_buy
     
    def compute_cash_step_first(self, p, m, idx_quantity, sell_order, buy_order):
        indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
        indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        cash_step = p[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity]) - self.level_fct(self.y_grid[idx_quantity+1*indicator_sell])) * sell_order.astype(int)  \
                        + m[idx_quantity] * (self.level_fct(self.y_grid[idx_quantity-1*indicator_buy]) - self.level_fct(self.y_grid[idx_quantity])) * buy_order.astype(int) 
        return cash_step
        
    def simulate_PnL_first(self, strategy, c=0.01,nsims = 100_000, seed = 123, return_trajectory = False):
        np.random.seed(seed=seed)
        dt = self.T/self.Nt
        timesteps = np.linspace(0, self.T, self.Nt+1)
        cash = np.zeros((nsims))
        n_sell_order = np.zeros((nsims))
        n_buy_order = np.zeros((nsims))
        idx_quantity = np.full(nsims, self.dim // 2, dtype=int)
        if return_trajectory:
            traj_quantity = np.zeros((nsims, self.Nt+1))
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
                p, m = self._calculate_fees_first_approx_t(t,self.oracleprice)
                p = p.ravel()
                m = m.ravel()
            if strategy == "Linear":
                p, m,_,_ = self.get_linear_fees(t)
            indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
            indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        
            stoch_int_sell, stoch_int_buy = self.compute_intensities_first(p, m, idx_quantity)
            sell_order, buy_order = self.get_arrival_first(stoch_int_sell, stoch_int_buy, nsims, dt)
            
            
            sell_order = (sell_order & indicator_sell).astype(int)
            buy_order = (buy_order & indicator_buy).astype(int)
            
            cash += self.compute_cash_step_first(p, m, idx_quantity, sell_order, buy_order)
            
            idx_quantity += sell_order.astype(int) - buy_order.astype(int)
            n_sell_order += sell_order.astype(int)
            n_buy_order += buy_order.astype(int)
            
            if return_trajectory:
                traj_quantity[:,it+1] = self.y_grid[idx_quantity]
        if return_trajectory:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order, traj_quantity)
        else:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order)
        
    def get_arrival_second(self,stoc_intensity_sell,stoc_intensity_buy,num_simulations,dt): # Given intensity compute if there is a jump or not
        unif_s = np.random.uniform(size=num_simulations)
        unif_b = np.random.uniform(size=num_simulations)
        return unif_s < 1. - np.exp(-stoc_intensity_sell * dt), unif_b < 1. - np.exp(-stoc_intensity_buy * dt)
    
    def compute_intensities_second(self, p, m, idx_quantity, St):

        # 1) Allowed moves
        can_buy  = (idx_quantity - 1 >= 0).astype(int)       # (nsims,)
        can_sell = (idx_quantity + 1 < self.dim).astype(int) # (nsims,)

        # 2) Precompute level values of the grid
        level_vals = self.level_fct(self.y_grid)            # (dim,)

        # 3) Level jumps
        sell_idx = idx_quantity + can_sell
        Δsell   = level_vals[idx_quantity] - level_vals[sell_idx]

        buy_idx  = idx_quantity - can_buy
        Δbuy    = level_vals[buy_idx] - level_vals[idx_quantity]

        # 4) Intensities, fully elementwise
        stoch_int_sell = self.int_sell * np.exp(
            self.kappa * ( (1.0 - p) * Δsell - St * self.delta_sell(self.y_grid, idx_quantity) )
        )

        stoch_int_buy  = self.int_buy  * np.exp(
            -self.kappa * ( (1.0 + m) * Δbuy  - St * self.delta_buy(self.y_grid, idx_quantity) )
        )

        return stoch_int_sell, stoch_int_buy
     
    def compute_cash_step_second(self, p, m, idx_quantity, sell_order, buy_order):
        can_sell = (idx_quantity + 1 < self.dim)
        can_buy  = (idx_quantity - 1 >= 0)

        levels = self.level_fct(self.y_grid)  # shape (dim,)

        # Set default deltas to zero
        delta_sell = np.zeros_like(p)
        delta_buy  = np.zeros_like(m)

        # Safely compute delta_sell where possible
        valid_sell = can_sell
        next_idx = idx_quantity[valid_sell] + 1
        delta_sell[valid_sell] = levels[idx_quantity[valid_sell]] - levels[next_idx]

        # Same for delta_buy
        valid_buy = can_buy
        prev_idx = idx_quantity[valid_buy] - 1
        delta_buy[valid_buy] = levels[prev_idx] - levels[idx_quantity[valid_buy]]

        # Now compute cash flow
        cash_sell = p * delta_sell * sell_order
        cash_buy  = m * delta_buy  * buy_order

        return cash_sell + cash_buy
    
    def _calculate_fast_fees_first_approx_t(self, it, St,p_precomputed,m_precomputed): # Compute the optimal fees
        p = np.zeros((self.dim, len(St)))
        m = np.zeros((self.dim, len(St)))
        p_interp = interp1d(self.Sgrid, p_precomputed[it, :, :], axis=1, bounds_error=False, fill_value="extrapolate")
        m_interp = interp1d(self.Sgrid, m_precomputed[it, :, :], axis=1, bounds_error=False, fill_value="extrapolate")

        # Evaluate at desired points
        p = p_interp(St)  # shape: (len(St), dim)
        m = m_interp(St)
        return p, m
    
    def simulate_PnL_second(self, strategy, c=0.01, nsims= 100_000, seed = 123, return_trajectory = False):
        np.random.seed(seed=seed)
        dt = self.T/self.Nt

        # Brownian motion simulation: shape (nsims, Nt)
        dW = np.random.normal(0, np.sqrt(dt), (nsims, self.Nt))
        W = np.concatenate([np.zeros((nsims, 1)), np.cumsum(dW, axis=1)], axis=1)  # shape (nsims, Nt+1)
        St = self.oracleprice + self.sigma * W  # shape (nsims, Nt+1)

        # Precompute m and p for the above grid and timesteps
        p_precomputed = np.zeros((self.Nt+1, self.dim, len(self.Sgrid)))
        m_precomputed = np.zeros((self.Nt+1, self.dim, len(self.Sgrid)))
        for i, t in tqdm(enumerate(self.timesteps)):
            p_precomputed[i,:,:], m_precomputed[i,:,:] = self._calculate_fees_first_approx_t(t, self.Sgrid)

        cash = np.zeros((nsims))
        n_sell_order = np.zeros((nsims))
        n_buy_order = np.zeros((nsims))
        idx_quantity = np.full(nsims, self.dim // 2, dtype=int)
        if return_trajectory:
            traj_quantity = np.zeros((nsims, self.Nt+1))
            traj_quantity[:,0] = self.y_grid[[self.dim // 2]] 
        stoch_int_sell = np.zeros((nsims))
        stoch_int_buy = np.zeros((nsims))
        if strategy == "Constant":
            p_mat = c * np.ones((self.dim, nsims))
            m_mat = c * np.ones((self.dim, nsims))
            sims = np.arange(nsims)
            p = p_mat[idx_quantity, sims]  # shape: (nsims,)
            m = m_mat[idx_quantity, sims]
        for it, t in enumerate(tqdm(self.timesteps[:-1], desc="Simulating PnL")):
            if strategy == "First Approximation":
                p, m = self._calculate_fast_fees_first_approx_t(it,St[:,it],p_precomputed,m_precomputed)
                # collapse to one fee per sim
                sims = np.arange(nsims)
                p = p[idx_quantity, sims]      # → (nsims,)
                m = m[idx_quantity, sims]
            if strategy == "Second Approximation":
                p, m = self._calculate_fees_second_approx_t(t,St[:,it])
                # collapse to one fee per sim
                sims = np.arange(nsims)
                p = p[idx_quantity, sims]      # → (nsims,)
                m = m[idx_quantity, sims]
            indicator_buy = np.where(idx_quantity - 1 >= 0, 1, 0)
            indicator_sell = np.where(idx_quantity + 1 < self.dim, 1, 0)
        
            stoch_int_sell, stoch_int_buy = self.compute_intensities_second(p, m, idx_quantity, St[:,it])
            sell_order, buy_order = self.get_arrival_second(stoch_int_sell, stoch_int_buy, nsims, dt)
            
            
            sell_order = (sell_order & indicator_sell).astype(int)
            buy_order = (buy_order & indicator_buy).astype(int)
            
            cash += self.compute_cash_step_second(p, m, idx_quantity, sell_order, buy_order)
            
            idx_quantity += sell_order.astype(int) - buy_order.astype(int)
            n_sell_order += sell_order.astype(int)
            n_buy_order += buy_order.astype(int)
            
            if return_trajectory:
                traj_quantity[:,it+1] = self.y_grid[idx_quantity]
        if return_trajectory:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order, traj_quantity)
        else:
            return (cash, self.y_grid[idx_quantity], n_sell_order, n_buy_order)