

def compute_psi_0(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # Shorthand assignments:
    y0 = y_0         # y₀
    p4 = p4          # p^4
    phi = pen_const  # φ
    k = kappa        # k
    e = e            # e
    delta_m = delta_buy(y_grid, 1)  # δ⁻
    delta_p = delta_sell(y_grid, 1)   # δ⁺
    lambda_m = int_buy     # λ⁻
    lambda_p = int_sell    # λ⁺

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

def compute_psi_1(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    
    # Shorthand assignments:
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1)  # δ⁻
    delta_p = delta_sell(y_grid, 1) # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

    # Denominators:
    denom_buy = (y0**2 - y0 * delta_m)**2
    denom_sell = (y0**2 + y0 * delta_p)**2

    # Terms as per the LaTeX expression:
    term1 = (2 * k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
    term2 = - (4 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
    term3 = - (2 * k * p2 * (delta_p**3) * lambda_p) / (e * denom_sell)
    term4 = - (4 * k * p2 * y0 * (delta_p**2) * lambda_p) / (e * denom_sell)

    return term1 + term2 + term3 + term4

def compute_psi_2(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂:
    # Ψ₂ = ( -2·k·δ⁻·λ⁻/(E) + 2·k·δ⁺·λ⁺/(E) )
    e  = e
    k  = kappa
    delta_m = delta_buy(y_grid, 1)  # δ⁻
    delta_p = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell

    term1 = (2 * k * (delta_m**2) * λm) / e
    term2 = (2 * k * (delta_p**2) * λp) / e
    return term1 + term2

def compute_psi_3(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # Shorthand assignments:
    y0 = y_0              # y₀
    p4 = p4               # p^4
    p2 = depth            # p^2
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺
    phi = pen_const       # φ

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

def compute_psi_4(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # Shorthand assignments:
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

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

def compute_psi_5(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₅:
    # Ψ₅ = ( -2·k·(δ⁺)³·λ⁺ + 2·k·(δ⁻)³·λ⁻ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (2 * k * (δp**3) * λp) / e
    term2 = - (2 * k * (δm**3) * λm) / e
    return term1 + term2

def compute_psi_6(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₆:
    # Ψ₆ = ( -2·k·(δ⁻)·λ⁻ + 2·k·(δ⁺)·λ⁺ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (2 * k * (δm**2) * λm) / e
    term2 = (2 * k * (δp**2) * λp) / e
    return term1 + term2

def compute_psi_7(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # Shorthand assignments:
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

    # Denominators:
    denom_buy = (y0**2 - y0 * delta_m)**2
    denom_sell = (y0**2 + y0 * delta_p)**2

    term1 = (k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
    term2 = - (2 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
    term3 = - (k * p2 * lambda_p * (delta_p**3)) / (e * denom_sell)
    term4 = - (2 * k * p2 * y0 * lambda_p * (delta_p**2)) / (e * denom_sell)

    return term1 + term2 + term3 + term4

def compute_psi_8(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # Shorthand assignments:
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺
    phi = pen_const       # φ

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

def compute_psi_9(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₉:
    # Ψ₉ = ( -2·k·(δ⁻)²·λ⁻ + 2·k·(δ⁺)²·λ⁺ )/E
    e  = e
    k  = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (-2 * k * (δm**2) * λm) / e
    term2 = (-2 * k * (δp**2) * λp) / e
    return term1 + term2

def compute_psi_10(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₁₀:
    # Ψ₁₀ = ( 2·k·(δ⁻)²·λ⁻ - 2·k·(δ⁺)²·λ⁺ )/E
    e  = e
    k  = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (2 * k * (δm**2) * λm) / e
    term2 = (2 * k * (δp**2) * λp) / e
    return term1 + term2

def compute_psi_11(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    """
    Psi11 =
    ( k * p^2 * λ^- * (δ^-)^3 ) / ( e * (y0^2 - y0*δ^-)^2 )
    - ( 2 * k * p^2 * y0 * λ^- * (δ^-)^2 ) / ( e * (y0^2 - y0*δ^-)^2 )
    - ( k * p^2 * (δ^+)^3 * λ^+ ) / ( e * (y0^2 + y0*δ^+)^2 )
    - ( 2 * k * p^2 * y0 * (δ^+)^2 * λ^+ ) / ( e * (y0^2 + y0*δ^+)^2 )
    """
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

    denom_buy = (y0**2 - y0 * delta_m)**2
    denom_sell = (y0**2 + y0 * delta_p)**2

    term1 = (k * p2 * lambda_m * (delta_m**3)) / (e * denom_buy)
    term2 = - (2 * k * p2 * y0 * lambda_m * (delta_m**2)) / (e * denom_buy)
    term3 = - (k * p2 * lambda_p * (delta_p**3)) / (e * denom_sell)
    term4 = - (2 * k * p2 * y0 * lambda_p * (delta_p**2)) / (e * denom_sell)

    return term1 + term2 + term3 + term4

def compute_psi_12(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
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
    y0 = y_0              # y₀
    p4 = p4               # p^4
    p2 = depth            # p^2
    k = kappa             # k
    e = e                 # e
    phi = pen_const       # φ
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

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

def compute_psi_13(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
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
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

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

def compute_psi_14(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₁₄:
    # Ψ₁₄ = ( -4·k·δ⁻·λ⁻ + 4·k·δ⁺·λ⁺ )/(2·E)
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (k * (δm**4) * λm) / (2 * e)
    term2 = (k * (δp**4) * λp) / (2 * e)
    return term1 + term2

def compute_psi_15(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₁₅:
    # Ψ₁₅ = ( -3·k·δ⁻·λ⁻ + 3·k·δ⁺·λ⁺ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = -(k * (δm**3) * λm) / e
    term2 = (k * (δp**3) * λp) / e
    return term1 + term2

def compute_psi_16(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
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
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

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

def compute_psi_17(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₁₇:
    # Ψ₁₇ = ( -2·k·(δ⁻)·λ⁻ + 2·k·(δ⁺)·λ⁺ )/(2·E)
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (k * (δm**2) * λm) / (2 * e)
    term2 = (k * (δp**2) * λp) / (2 * e)
    return term1 + term2

def compute_psi_18(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    
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
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    phi = pen_const       # φ
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

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

def compute_psi_19(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₁₉:
    # Ψ₁₉ = ( k·δ⁻·λ⁻ - k·δ⁺·λ⁺ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    return (k * (δm**3) * λm - k * (δp**3) * λp) / e

def compute_psi_20(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂₀:
    # Ψ₂₀ = ( k·δ⁺·λ⁺ - k·δ⁻·λ⁻ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    return (-k * (δm**3) * λm + k * (δp**3) * λp) / e

def compute_psi_21(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂₁:
    # Ψ₂₁ = - ( k·δ⁻·λ⁻ + k·δ⁺·λ⁺ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    return (-k * (δm**2) * λm - k * (δp**2) * λp) / e

def compute_psi_22(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂₂:
    # Ψ₂₂ = ( k·δ⁻·λ⁻ + k·δ⁺·λ⁺ )/E
    e = e
    k = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    return (k * (δm**2) * λm + k * (δp**2) * λp) / e

def compute_psi_23(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # Shorthand assignments:
    y0 = y_0              # y₀
    p2 = depth            # p²
    k = kappa             # k
    e = e                 # e
    delta_m = delta_buy(y_grid, 1) # δ⁻
    delta_p = delta_sell(y_grid, 1)  # δ⁺
    lambda_m = int_buy    # λ⁻
    lambda_p = int_sell   # λ⁺

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

def compute_psi_24(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂₄:
    # Ψ₂₄ = - φ + ( (δ⁻)²·λ⁻ )/(2·E·k) + ( (δ⁺)²·λ⁺ )/(2·E·k)
    φ  = pen_const
    e  = e
    k  = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = - φ
    term2 = (δm**2 * λm*k) / (2 * e)
    term3 = (δp**2 * λp*k) / (2 * e)
    return term1 + term2 + term3

def compute_psi_25(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂₅:
    # Ψ₂₅ = - ( (δ⁻)²·λ⁻ )/(E·k) - ( (δ⁺)²·λ⁺ )/(E·k)
    e  = e
    k  = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = - (δm**2 * λm*k) / (e )
    term2 = - (δp**2 * λp*k) / (e )
    return term1 + term2

def compute_psi_26(y_0,p4,depth,pen_const,kappa,e,delta_buy,delta_sell,y_grid,int_buy,int_sell):
    # LaTeX for Ψ₂₆:
    # Ψ₂₆ = ( (δ⁻)²·λ⁻ )/(2·E·k) + ( (δ⁺)²·λ⁺ )/(2·E·k)
    e  = e
    k  = kappa
    δm = delta_buy(y_grid, 1)
    δp = delta_sell(y_grid, 1)
    λm = int_buy
    λp = int_sell
    term1 = (δm**2 * λm*k) / (2*e )
    term2 = (δp**2 * λp*k) / (2*e )
    return term1 + term2
