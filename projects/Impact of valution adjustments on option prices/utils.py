
import numpy as np
import matplotlib.pyplot as plt

def test_model_consistency(model, T_list, n_paths=10000, N=120, numeraire='numerical'):
    """
    Test Hull-White model consistency with the initial zero curve:
    E[N(0,T)] ≈ P(0,T)

    Parameters:
    - model: instance of model_hull_white
    - T_list: list of maturities to test (in years)
    - n_paths: number of Monte Carlo paths
    - N: number of time steps

    Prints comparison for each T
    """

    T_max = max(T_list)
    r_paths, times = model.simulate(n_paths, T_max, N, model.term_structure.r(0))

    for T in T_list:
        T_idx = np.searchsorted(times, T)
        if numeraire == 'numerical':
            D_sim = model.numeraire(r_paths, times, t=0, T=T)
        elif numeraire == 'closed':
            D_sim = model.numeraire_closed(r_paths, times, t=0, T=T)
        D_mean = np.mean(D_sim)
        P0T = model.term_structure.P0t(T)

        print(f"T = {T:.2f} yr | Simulated E[DF] = {D_mean:.6f} | P(0,T) = {P0T:.6f} | Abs Error = {abs(D_mean - P0T):.6e}")

def plot_mtm_paths_and_histograms(mtm_matrix, times, fixing_dates, n_paths_to_plot=1000, n_cols=4):
    """
    Plot MtM paths and histograms of MtM distributions at selected fixing dates.

    Parameters:
    - mtm_matrix: np.ndarray (n_paths x n_steps), simulated MtM paths
    - times: np.ndarray of simulation time grid
    - fixing_dates: list or array of fixing times to select
    - n_paths_to_plot: int, number of paths to plot (default: 100)
    - n_cols: int, number of subplot columns for histograms
    """
    fixing_dates = np.array(fixing_dates)
    time_indices = [np.searchsorted(times, t) for t in fixing_dates]
    selected_times = times[time_indices]
    mtm_subset = mtm_matrix[:, time_indices]

    # ---- Plot MtM paths ----
    plt.figure(figsize=(12, 6))
    for i in range(min(n_paths_to_plot, mtm_subset.shape[0])):
        plt.plot(selected_times, mtm_subset[i], lw=0.6, alpha=0.6)
    plt.title('MtM Paths of IRS at Fixing Dates')
    plt.xlabel('Time (Years)')
    plt.ylabel('MtM Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ---- Plot Histograms ----
    n_dates = len(fixing_dates)
    n_rows = int(np.ceil(n_dates / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
    axes = axes.flatten()

    for i, idx in enumerate(time_indices):
        ax = axes[i]
        ax.hist(mtm_matrix[:, idx], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'MtM @ t={fixing_dates[i]:.2f} yr')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axvline(0, color='black', lw=1)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Distribution of MtM Values at Selected Fixing Dates", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_EE_PFE_comparison(fixing_dates, ee_unsecured, pfe_unsecured,
                           ee_collateralized, pfe_collateralized):
    """
    Plot Expected Exposure (EE) and Potential Future Exposure (PFE) profiles
    for both unsecured and collateralized cases.

    Parameters:
    - fixing_dates: list or np.ndarray of repricing dates
    - ee_unsecured: array of EE values without collateral
    - pfe_unsecured: array of PFE values without collateral
    - ee_collateralized: array of EE values with collateral
    - pfe_collateralized: array of PFE values with collateral
    """
    plt.figure(figsize=(10, 6))

    # EE curves
    plt.plot(fixing_dates, ee_unsecured, label='EE (Unsecured)', color='darkblue', linewidth=2)
    plt.plot(fixing_dates, ee_collateralized, label='EE (Collateralized)', color='orangered', linestyle='--', linewidth=2)

    # PFE curves
    plt.plot(fixing_dates, pfe_unsecured, label='PFE (Unsecured)', color='royalblue', linestyle='-', linewidth=1.5)
    plt.plot(fixing_dates, pfe_collateralized, label='PFE (Collateralized)', color='tomato', linestyle='--', linewidth=1.5)

    plt.title('Expected & Potential Future Exposure (EE & PFE)', fontsize=14)
    plt.xlabel('Time (Years)', fontsize=12)
    plt.ylabel('Exposure', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_gbm_paths(S_paths, times, n_paths_to_plot=100):
    """
    Plot GBM stock price paths.

    Parameters:
    - S_paths: np.ndarray of shape (n_paths, len(times))
    - times: np.ndarray of time grid
    - n_paths_to_plot: int, number of paths to plot
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(n_paths_to_plot, S_paths.shape[0])):
        plt.plot(times, S_paths[i], lw=0.8, alpha=0.7)

    plt.title('Simulated Geometric Brownian Motion Stock Paths (100)', fontsize=14)
    plt.xlabel('Time (Years)', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def generate_correlated_brownian(size, rho=0):
    """
    Generate two correlated Brownian motion increments using multivariate normal.

    Parameters:
    - n_paths: number of simulated paths
    - n_steps: number of time steps
    - rho: correlation coefficient between the two Brownian motions
    - dt: time step size

    Returns:
    - Z1: np.ndarray of shape (n_paths, n_steps), Brownian increments for process 1
    - Z2: np.ndarray of shape (n_paths, n_steps), Brownian increments for process 2
    """
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]  # covariance matrix

    # Sample all paths at once
    correlated_normals = np.random.multivariate_normal(mean, cov, size=size)

    # Split into two processes and scale by sqrt(dt)
    Z1 = correlated_normals[:,0]
    Z2 = correlated_normals[:,1]

    return Z1, Z2




