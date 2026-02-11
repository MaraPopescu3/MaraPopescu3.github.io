import numpy as np
from dcr_model.utils import generate_correlated_brownian

class GeometricBrownianMotion:
    def __init__(self, S0, rate_model, sigma, rho=0):
        """
        Initialize the GBM model with stochastic drift.

        Parameters:
        - S0: initial stock price
        - rate_model: model_hull_white instance (must have simulate method)
        - sigma: constant volatility
        """
        self.S0 = S0
        self.rate_model = rate_model
        self.sigma = sigma
        self.rho = rho
        self.Z = generate_correlated_brownian(10000000, rho=rho)

    def simulate(self, n_paths, T_max, N):
        """
        Simulate GBM paths with stochastic drift from Hull-White model.
        """
        
        dt = T_max / N
        times = np.linspace(0, T_max, N + 1)
    
        # Generate Brownian increments
        Z = generate_correlated_brownian(n_paths*N, rho=self.rho)
        Z1 = Z[0].reshape(n_paths, N)
        Z2 = Z[1].reshape(n_paths, N)
        Z1 -= Z1.mean(axis=0, keepdims=True)
        Z1 /= Z1.std(axis=0, keepdims=True)
        Z2 -= Z2.mean(axis=0, keepdims=True)
        Z2 /= Z2.std(axis=0, keepdims=True)
        
        # Simulate short rate paths from Hull-White model
        r_paths, _ = self.rate_model.simulate(n_paths, T_max, N, r0=0, Z=Z1)
        
        # Initialize stock price paths
        S_paths = np.zeros((n_paths, N + 1))
        S_paths[:, 0] = self.S0

        for i in range(1, N + 1):
            diffusion = self.sigma * Z2[:, i - 1] * np.sqrt(dt)
            S_paths[:, i] = S_paths[:, i - 1]/self.rate_model.PtT(r_paths, times, times[i-1], times[i]) \
            * np.exp(- 0.5 * self.sigma**2 * dt + diffusion)

        return S_paths, r_paths, times

    def simulate_nested(self, S_paths, r_paths, times, t, T_max, dt, N_inner):
        #print("Simulate Nested GBM paths ...")
        n_steps = int(round((T_max - t)/dt))
        N_outer = S_paths.shape[0]
        t_idx = np.searchsorted(times, t)
        
        historical_S = S_paths[:, :t_idx + 1]
        S0_array = S_paths[:, t_idx]

        #print("Sampling Wiener Process...")
        Z1, Z2 = self.Z
        idx = np.random.randint(0, 1000000)
        idx = np.array(list(range(idx, idx + N_inner*n_steps)))
        idx %= 1000000
        Z1 = Z1[idx].reshape(N_inner, n_steps)
        Z2 = Z2[idx].reshape(N_inner, n_steps)
        Z1 -= Z1.mean(axis=0, keepdims=True)
        Z1 /= Z1.std(axis=0, keepdims=True)
        Z2 -= Z2.mean(axis=0, keepdims=True)
        Z2 /= Z2.std(axis=0, keepdims=True)
        
        r_paths_nested, nested_times = self.rate_model.simulate_nested(r_paths, times, t, dt, n_steps, N_inner, Z=Z1)
        #print("Simulate Nested Rates Finished!")
        nested_S_all = np.zeros((N_outer * N_inner, n_steps+1))
        
        # Simulate inner paths for each outer path
        for i, S0 in enumerate(S0_array):
            # Simulate short rate path and generate nested GBM from S0
            nested_S_all[i*N_inner:(i+1)*N_inner,0] = S0_array[i]

        
        for i in range(1, n_steps+1):
            t_prev = nested_times[t_idx+i-1]
            t_curr = nested_times[t_idx+i]
            fwd = self.rate_model.PtT(r_paths_nested, nested_times, t_prev, t_curr)
            nested_S_all[:, i] = nested_S_all[:, i - 1] / fwd * np.exp(-0.5 * self.sigma**2 * dt + \
                                                                       self.sigma * np.sqrt(dt) * \
                                                                       np.hstack([Z2[:,i-1] for _ in range(N_outer)]))
      
        #print("Simulate Nested Price Finished!")
    
        # Concatenate historical part to each nested path
        historical_tiled = np.repeat(historical_S, N_inner, axis=0)
        S_paths_nested = np.concatenate([historical_tiled, nested_S_all[:,1:]], axis=1)
    
        return S_paths_nested, r_paths_nested, nested_times