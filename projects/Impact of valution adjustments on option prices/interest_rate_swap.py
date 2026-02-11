import numpy as np
from tqdm import tqdm

class InterestRateSwap:
    def __init__(self, fixed_rate, maturity, notional=1.0, freq=2):
        self.fixed_rate = fixed_rate
        self.maturity = maturity
        self.notional = notional
        self.freq = freq
        self.dt = 1 / freq
        next_pay = self.maturity % self.dt if self.maturity % self.dt != 0 else self.dt
        self.payment_times = np.arange(next_pay, self.maturity + self.dt, self.dt)

    def price(self, term_structure):
        """
        Compute the fair price of the IRS for fix receiver (under single-curve framework).
        """
        fixed_leg_value = 0.0
        floating_leg_value = 0.0

        t_prev = 0
        for t in self.payment_times:
            dt = t - t_prev
            discount = term_structure.P0t(t)
            fixed_leg_value += self.fixed_rate * dt * discount
            fwd_rate = term_structure.forward_rate(t_prev, t)
            floating_leg_value += dt * fwd_rate * discount
            t_prev = t

        npv = self.notional * (fixed_leg_value - floating_leg_value)
        
        return npv

    def par_rate(self, term_structure):
        """
        Compute the fair fixed rate that makes the swap NPV zero from an arbitrary valuation time.
        """
        t_prev = 0

        numerator = 0.0  # floating leg value
        denominator = 0.0  # discounted fixed accruals

        for t in self.payment_times:
            dt = t - t_prev
            discount = term_structure.P0t(t)
            fwd_rate = term_structure.forward_rate(t_prev,t)
            numerator += dt * fwd_rate * discount
            denominator += dt * discount
            t_prev = t

        return numerator / denominator

    def mtm(self, r_path, times, t, hw_model):
      """
      Compute the MtM value of the swap at time t given short rate path r(t),
      using Hull-White bond pricing.
      """
      future_payments = self.payment_times[self.payment_times > t]
      if len(future_payments) == 0:
          return 0.0

      fixed_leg = 0.0
      float_leg = 0.0
      term_structure = hw_model.term_structure

      t_prev = t

      for T_k in future_payments:
          dt = T_k - t_prev
          # Discount using Hull-White PtT
          P_t_T = hw_model.PtT(r_path, times, t, T_k)
          # Forward rate from t_prev to T_k based on initial term structure
          P_t_tprev = hw_model.PtT(r_path, times, t, t_prev) if t_prev > t else 1.0
          fwd_rate = (P_t_tprev / P_t_T - 1) / dt
          fixed_leg += self.fixed_rate * dt * P_t_T
          float_leg += fwd_rate * dt * P_t_T

          t_prev = T_k

      return self.notional * (fixed_leg - float_leg)

    def simulate_mtm_paths(self, hw_model, n_paths, dt_sim=1/12):
      """
      Efficiently simulate MtM(t) paths of the interest rate swap under Hull-White model.
      Forward rates are computed dynamically at each simulation time using model-implied bond prices.
      """
      T_max = self.maturity
      r_paths, times = hw_model.simulate(n_paths, T_max, int(T_max / dt_sim), hw_model.term_structure.r(0), t0=0)
      n_steps = len(times)
      mtm_matrix = np.zeros((n_paths, n_steps))

      future_payments = self.payment_times

      for i, t in tqdm(enumerate(times), total=n_steps):
          valid_indices = np.where(future_payments > t)[0]
          t_prev = t

          for j in valid_indices:
              T_k = future_payments[j]
              dt = T_k - t_prev
              if dt == 0:
                  continue

              # Hull-White model bond prices
              P_t_T = hw_model.PtT(r_paths, times, t, T_k)
              P_t_tprev = hw_model.PtT(r_paths, times, t, t_prev) if t_prev > t else np.ones(len(r_paths))

              # Model-consistent forward rate
              fwd_rate = (P_t_tprev / P_t_T - 1) / dt

              # Fixed and floating legs
              fixed = self.fixed_rate * dt * P_t_T
              floating = fwd_rate * dt * P_t_T

              mtm_matrix[:, i] += self.notional * (fixed - floating)
              t_prev = T_k

      return mtm_matrix, times

    def compute_EE_PFE(self, mtm_matrix, times, fixing_dates, confidence=0.99):
      """
      Compute Expected Exposure (EE) and Potential Future Exposure (PFE) from MtM matrix,
      selecting only the rows in `times` that match `fixing_dates`.
      """
      # Get indices of fixing_dates in the full times array
      fixing_indices = [np.searchsorted(times, fd) for fd in fixing_dates]

      ee = []
      pfe = []

      for idx in fixing_indices:
          exposure = np.maximum(mtm_matrix[:, idx], 0)
          ee.append(np.mean(exposure))
          pfe.append(np.quantile(exposure, confidence))

      return np.array(ee), np.array(pfe)


    def compute_EE_PFE_collateralized(self, hw_model, fixing_dates, n_paths=10000, mpor_days=10, min_margin=0.0, confidence=0.99):
      """
      Compute Expected Exposure (EE) and Potential Future Exposure (PFE) for a collateralized swap using MtM
      changes over MPOR periods.
      Parameters:
      - swap: InterestRateSwap instance
      - hw_model: model_hull_white instance
      - fixing_dates: list of repricing times
      - mpor_days: integer, margin period of risk in days (default 10)
      - min_margin: float, minimum initial margin (default 0.005)
      """
      dt_sim = 1 / 360  # daily steps
      mpor_horizon = int(mpor_days)  # 10-day window
      T_max = self.maturity

      # Simulate MtM paths daily
      mtm_matrix, times = self.simulate_mtm_paths(hw_model, n_paths=n_paths, dt_sim=dt_sim)

      fixing_indices = [np.searchsorted(times, t) for t in fixing_dates]

      ee = []
      pfe = []

      for idx in fixing_indices:
          if idx == mtm_matrix.shape[1]-1:
            ee.append(0)
            pfe.append(0)
            break
          end_idx = min(idx + mpor_horizon + 1, mtm_matrix.shape[1])
          mtm_t = mtm_matrix[:, idx].reshape(-1, 1)
          mtm_window = mtm_matrix[:, idx: end_idx]
          protected_level = np.maximum(mtm_t, min_margin)
          exposures = np.maximum(mtm_window - protected_level, 0)
          worst_exposure = np.max(exposures, axis=1)
          ee.append(np.mean(worst_exposure))
          pfe.append(np.quantile(worst_exposure, confidence))

      return np.array(ee), np.array(pfe)

    def compute_EE_PFE_collateralized_nested(self, hw_model, fixing_dates, mpor_days=10,
                                         confidence=0.99, n_outer_paths=1000, n_inner_paths=1000):
   
        dt_outer = 1 / 12
        T_max = self.maturity
        n_steps_outer = int(T_max / dt_outer)
    
        # Simulate outer paths
        outer_paths, outer_times = hw_model.simulate(n_outer_paths, T_max, n_steps_outer, r0=hw_model.term_structure.r(0))
    
        ee = []
        pfe = []
    
        for i, t_fix in tqdm(enumerate(fixing_dates), total=len(fixing_dates)):
            if t_fix >= T_max:
                ee.append(0.0)
                pfe.append(0.0)
                continue
    
            idx_fix = np.searchsorted(outer_times, t_fix)
    
            # Define inner simulation parameters
            T_inner = min(t_fix + mpor_days / 360, T_max)
            n_steps_inner = int(round((T_inner - t_fix) * 360))
            dt_inner = 1/360
    
            # Run nested simulation using the new method
            nested_paths, nested_times = hw_model.simulate_nested(
                r_paths=outer_paths,
                times=outer_times,
                t=t_fix,
                dt=dt_inner,
                n_steps=n_steps_inner,
                N_inner=n_inner_paths
            )
    
            # Compute MtM over nested window
            n_nested = nested_paths.shape[0]
            mtm_matrix = np.zeros((n_nested, n_steps_inner + 1))
            future_payments = self.payment_times
    
            idx_fix_nested = np.searchsorted(nested_times, t_fix)
    
            for ii, t in enumerate(nested_times[idx_fix_nested:]):
                valid_indices = np.where(future_payments > t)[0]
                t_prev = t
    
                for jj in valid_indices:
                    T_k = future_payments[jj]
                    dt = T_k - t_prev
                    if dt <= 0:
                        continue
    
                    P_t_T = hw_model.PtT(nested_paths, nested_times, t, T_k)
                    P_t_tprev = hw_model.PtT(nested_paths, nested_times, t, t_prev) if t_prev > t else np.ones(n_nested)
                    fwd_rate = (P_t_tprev / P_t_T - 1) / dt
    
                    fixed = self.fixed_rate * dt * P_t_T
                    floating = fwd_rate * dt * P_t_T
                    mtm_matrix[:, ii] += self.notional * (fixed - floating)
    
                    t_prev = T_k
    
            # Calculate MPOR exposure over nested window
            initial_mtm = mtm_matrix[:, 0].reshape(-1, 1)
            mp_exposures = np.maximum(mtm_matrix - initial_mtm, 0.0)
            max_exposure_per_path = np.max(mp_exposures, axis=1)
    
            ee.append(np.mean(max_exposure_per_path))
            pfe.append(np.quantile(max_exposure_per_path, confidence))
    
        return np.array(ee), np.array(pfe)

    def compute_cva(self, fixing_dates, ee, term_structure, cds_spreads_bps, cds_tenors, lgd=0.4):
      """
      Compute CVA from Expected Exposure profile and CDS spread curve (using piecewise constant hazard rate model).
      Parameters:
      - fixing_dates: list or np.ndarray of times t_i
      - ee: list or np.ndarray of expected exposures EE(t_i)
      - term_structure: term_structure object with method P0t(t)
      - cds_spreads_bps: list of CDS spreads in basis points
      - cds_tenors: list of corresponding CDS maturity times in years
      - lgd: Loss Given Default (e.g., 0.4)
      Returns:
      - float: CVA price
      """
      cds_spreads = np.array(cds_spreads_bps) / 10000  # Convert bps to decimals
      cds_tenors = np.array(cds_tenors)
      hazard_rates = cds_spreads / lgd

      def hazard_interp(t):
          for i, T in enumerate(cds_tenors):
              if t <= T:
                  return hazard_rates[i]
          return hazard_rates[-1]  # Flat extrapolation

      cva = 0.0
      Q = 1.0  # Initial survival probability
      t_prev = 0.0

      for i, t in enumerate(fixing_dates):
          dt = t - t_prev
          h = hazard_interp(t)
          discount = term_structure.P0t(t)
          dQ = Q * (1 - np.exp(-h * dt))  # Marginal default probability
          cva += ee[i] * discount * dQ
          Q *= np.exp(-h * dt)  # Update survival
          t_prev = t

      return lgd * cva

    def compute_fva(self, fixing_dates, ee, term_structure, funding_spread_bps):
      """
      Compute FVA (Funding Valuation Adjustment) from Expected Exposure profile.

      Parameters:
      - fixing_dates: list or np.ndarray of times t_i
      - ee: list or np.ndarray of expected exposures EE(t_i)
      - term_structure: term_structure object with method P0t(t)
      - funding_spread_bps: float, funding spread in basis points (e.g., 40 for 0.4%)

      Returns:
      - float: FVA price
      """
      sf = funding_spread_bps / 10000  # Convert bps to decimal
      fva = 0.0
      t_prev = 0.0

      for i, t in enumerate(fixing_dates):
          dt = t - t_prev
          discount = term_structure.P0t(t)
          fva += ee[i] * discount * sf * dt
          t_prev = t

      return fva
