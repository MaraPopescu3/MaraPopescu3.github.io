import numpy as np
from tqdm import tqdm 

class DownInPutOption:
    def __init__(self, strike, barrier, maturity, position=1):
        """
        Initialize the Down-and-In Put Option.

        Parameters:
        - strike: float, strike price (K)
        - barrier: float, barrier level (H)
        - maturity: float, option maturity in years (T)
        - position: int, +1 for long, -1 for short position (default -1)
        """
        self.strike = strike
        self.barrier = barrier
        self.maturity = maturity
        self.position = position

    def price(self, gbm_model, n_paths=10000, N=1000):
        """
        Price the Down-and-In Put option via Monte Carlo simulation.

        Parameters:
        - gbm_model: GeometricBrownianMotion instance
        - n_paths: int, number of simulated paths
        - N: int, number of time steps

        Returns:
        - float: Present value of the option
        """
        
        S_paths, r_paths, times = gbm_model.simulate(n_paths, self.maturity, N)
        knocked_in = np.any(S_paths <= self.barrier, axis=1)
        S_T = S_paths[:, -1]
        payoff = np.where(knocked_in, np.maximum(self.strike - S_T, 0.0), 0.0)
        discount = gbm_model.rate_model.term_structure.P0t(self.maturity)
        return self.position * np.mean(payoff) * discount

    def fva(self, gbm_model, sF=0.004, n_paths=10000, T_max = 1.5, N_nested=100, outer_dt=1/360, inner_dt=1/360):
        """
        Compute Funding Valuation Adjustment (FVA) using simulated expected exposures.

        Parameters:
        - gbm_model: GeometricBrownianMotion instance
        - sF: float, funding spread (e.g., 0.004 for 40 bps)
        - n_paths: int, number of simulated paths
        - N: int, number of time steps

        Returns:
        - float: FVA value
        """
        print("Simulating Outer Price Paths...")
        N = int(round(T_max/outer_dt))
        S_paths, r_paths, times = gbm_model.simulate(n_paths, self.maturity, N)
        dt = times[1] - times[0]
        n_steps = len(times) 

        fva = 0.0
        print("Iterating over times...")
        for i in tqdm(range(1, n_steps)):
            t = times[i]
            if t<T_max:
                S_paths_nested, r_paths_nested, times_nested = gbm_model.simulate_nested(S_paths, r_paths, times, t=t, T_max=T_max, dt=inner_dt, N_inner=N_nested)
                S_T = S_paths_nested[:, -1]
                knocked_in = np.any(S_paths_nested <= self.barrier, axis=1)
                payoff_t = np.where(knocked_in, np.maximum(self.strike - S_T, 0.0), 0.0)
                exposure = np.maximum(payoff_t, 0.0)
                df = gbm_model.rate_model.PtT(r_paths_nested, times_nested, t, times[-1])
                fva += np.mean(df * exposure) * dt
                #print(fva)
            else:
                S_T = S_paths[:, -1]
                knocked_in = np.any(S_paths <= self.barrier, axis=1)
                payoff_t = np.where(knocked_in, np.maximum(self.strike - S_T, 0.0), 0.0)
                exposure = np.maximum(payoff_t, 0.0)
                df = gbm_model.rate_model.PtT(r_paths, times, t, times[-1])
                fva += np.mean(df * exposure) * dt
                #print(fva)

        print(-self.position * sF * fva)

        return -self.position * sF * fva
