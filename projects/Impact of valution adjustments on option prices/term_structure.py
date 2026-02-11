import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

class TermStructure:
    
    def __init__(self, tenors, zero_rates, r0=0.0, interpolate_method='CUBIC'):
        tenors = np.array(tenors, dtype=float)
        zero_rates = np.array(zero_rates, dtype=float)
        # Enforce r(0)
        self.tenors = np.insert(tenors, 0, 0.0)
        self.zero_rates = np.insert(zero_rates, 0, r0)
        
        # Interpolate the log - discount factor:
        if interpolate_method=='CUBIC':
            self.log_discount = CubicSpline(
                self.tenors,
                -self.zero_rates * self.tenors,
                bc_type='natural',
                extrapolate=True
            )
        elif interpolate_method=='PCHIP':
            self.log_discount = PchipInterpolator(
                self.tenors,
                -self.zero_rates * self.tenors,
                extrapolate=True
            )
        
            
    def P0t(self, t):
        return np.exp(self.log_discount(t))

    def r(self, t):
        t = np.asarray(t)
        r_vals = np.empty_like(t)
        mask = t < 1e-6
        r_vals[mask] = self.zero_rates[0]
        r_vals[~mask] = -self.log_discount(t[~mask]) / t[~mask]
        return r_vals if r_vals.ndim > 0 else r_vals.item()

    def f0t(self, t):
        t = np.asarray(t)
        return np.where(t < 1e-6, self.zero_rates[0], -self.log_discount.derivative()(t))

    def forward_rate(self, t0, t1):
        dt = t1 - t0
        if dt < 1e-8:
            return self.f0t(t0)
        P0 = self.P0t(t0)
        P1 = self.P0t(t1)
        return (P0 / P1 - 1) / dt


