import numpy as np

class HullWhiteModel:
    def __init__(self, a, sigma, term_structure):
        self.a = a
        self.sigma = sigma
        self.term_structure = term_structure
        self.Z = np.random.randn(1000000)

    def simulate(self, n_paths, T_max, N, r0, t0 = 0, Z = None):
        def alpha(t):
          t = np.asarray(t)
          fwd = self.term_structure.f0t(t)
          adj = (self.sigma**2) / (2 * self.a ** 2) * (1 - np.exp(-self.a * t))**2
          return fwd + adj
        dt = T_max / N
        times = np.linspace(0, T_max, N + 1) + t0
        r_paths = np.zeros((n_paths, N + 1))
        r_paths[:, 0] = r0 - alpha(t0)
        alpha_vals = alpha(times)
        exp_a_dt = np.exp(-self.a * dt)
        sqrt_term = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
        # Resample for Wiener Process and Correct Error
        if Z is None:
            idx = np.random.randint(0, 1000000)
            idx = np.array(list(range(idx, idx + n_paths*N)))
            idx %= 1000000
            Z = self.Z[idx].reshape(n_paths, N)
            Z -= Z.mean(axis=0, keepdims=True)
            Z /= Z.std(axis=0, keepdims=True)
        else:
            assert Z.shape == (n_paths, N)
        for i in range(1, N + 1):
            r_paths[:, i] = (
                r_paths[:, i - 1] * exp_a_dt
                + sqrt_term * Z[:, i - 1]
            )
        r_paths += alpha_vals
        return r_paths, times

    def simulate_nested(self, r_paths, times, t, dt, n_steps, N_inner, Z=None):
        #print("Simulate Nested Rate Paths ...")
        N_outer = r_paths.shape[0]
        t_index = np.searchsorted(times, t)
    
        # Extract the historical part up to and including time t
        historical_paths = r_paths[:, :t_index + 1]  # shape (N_outer, t_index+1)
        historical_times = times[:t_index + 1]
    
        # Initial rate for each outer path at time t
        r0_array = r_paths[:, t_index]  # shape (N_outer,)
    
        # Prepare inner simulation grid and shape
        inner_times = np.linspace(t + dt, t + dt * n_steps, n_steps)
    
        # Run nested simulation for each outer path
        if Z is None:
            r_nested, _ = self.simulate(N_outer*N_inner, T_max=dt * n_steps, N=n_steps, r0=np.repeat(r0_array,N_inner), t0=t)
        else:
            r_nested, _ = self.simulate(N_outer*N_inner, T_max=dt * n_steps, N=n_steps, r0=np.repeat(r0_array,N_inner), t0=t, 
                                        Z=np.vstack([Z for _ in range(N_outer)]))
            
        r_inner_all = r_nested[:,1:]
        
        # Tile historical paths and concatenate
        historical_tiled = np.repeat(historical_paths, N_inner, axis=0)  # shape (N_outer * N_inner, t_index+1)
        nested_r_paths = np.concatenate([historical_tiled, r_inner_all], axis=1)
    
        # Combine time grid
        nested_times = np.concatenate([historical_times, inner_times])
    
        return nested_r_paths, nested_times

    def numeraire(self, r_paths, times, t, T):
      t_idx = np.searchsorted(times, t)
      T_idx = np.searchsorted(times, T)
      r_slice = r_paths[:, t_idx:T_idx + 1]
      t_slice = times[t_idx:T_idx + 1]

      # Apply trapezoidal rule to each path
      integral_r = np.trapz(r_slice, t_slice, axis=1)
      X = np.exp(-integral_r)

      return X

    def numeraire_closed(self, r_paths, times, t, T):
      return self.PtT(r_paths, times, t, T)

    def PtT(self, r_paths, times, t, T):
      a = self.a
      sigma = self.sigma
      ts = self.term_structure

      t_idx = np.searchsorted(times, t)
      r_t = r_paths[:, t_idx]

      B = (1 - np.exp(-a * (T - t))) / a
      P0_T = ts.P0t(T)
      P0_t = ts.P0t(t)
      f0_t = ts.f0t(t)

      A = (P0_T / P0_t) * np.exp(
          B * f0_t - (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2
      )

      return A * np.exp(-B * r_t)




    

    

    

    
