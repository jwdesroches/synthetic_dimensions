# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import sys
import os

from synth_dim_model import *

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data_folder"))
sys.path.append(parent_dir)
sys.path.append(data_folder_path) 

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def plot_time_evolution(N, M, sign_V, results, times, J_V_ratios, mu_V_ratios, plot_probability=True, plot_gap=True, plot_overlaps=True, plot_sigma=True, plot_ground_state_manifold_overlaps = True):
    """
    Plots the time evolution of various observables of a quantum system.
    
    Depending on the flags provided, this function generates plots for:
      - State probabilities (population in each eigenstate)
      - Scaled energy gaps between the instantaneous eigenstates and the ground state
      - Real and imaginary parts of the overlaps between the time-evolved state and the instantaneous eigenstates
      - A synthetic observable σ calculated from the time-evolved wavefunctions
    
    The plots are annotated using the control parameters J/|V| and μ/|V|.
    
    Parameters:
    N (int): Number of particles or spins in the system.
    M (int): Local Hilbert space dimension.
    results (tuple): A tuple containing outputs from `simulate_hamiltonian_time_evolution`, specifically:
                     (energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies).
    times (array-like): Array of time values corresponding to the simulation.
    J_V_ratios (array-like): Array of J/|V| ratios used in the evolution.
    mu_V_ratios (array-like): Array of μ/|V| ratios used in the evolution.
    plot_probability (bool, optional): If True, plots the state probabilities. Default is True.
    plot_gap (bool, optional): If True, plots the scaled energy gap. Default is True.
    plot_overlaps (bool, optional): If True, plots the real and imaginary parts of state overlaps. Default is True.
    plot_sigma (bool, optional): If True, plots the synthetic observable σ. Default is True.
    
    Returns:
    None
    """
    
    energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies, ground_state_manifold_overlaps = results
    energies = energies * 1/N
    true_energies = true_energies * 1/N
    colors = get_cmap("gist_rainbow", M**N)
    
    if sign_V == "positive":
        sign_V_string = r"$V>0$"
    else:
        sign_V_string = r"$V<0$"
    
    if plot_probability:
        fig, ax = plt.subplots()
        for index in range(M**N):
            if index == 0:
                ax.plot(times, state_probabilities[:, index], color="k", label="Ground State")
            elif index == 1:
                ax.plot(times, state_probabilities[:, index], color=colors(index), label="1st Excited State")
            elif index == 2:
                ax.plot(times, state_probabilities[:, index], color=colors(index), label="2nd Excited State") 
            else: 
                ax.plot(times, state_probabilities[:, index], color=colors(index))
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"State Probabilities: $N={N}$, $M={M}$, {sign_V_string}, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_xlabel("Time")
        ax.set_ylabel("State Probability")
        ax.grid()
        fig.tight_layout()
        
    if plot_gap:
        fig, ax = plt.subplots()
        for index in range(M**N):
            ax.plot(times, true_energies[:, index] - true_energies[:, 0], color=colors(index))   
        ax.plot(times, energies - true_energies[:, 0], color="k", label="Time Evolved State")
        ax.legend(loc="upper center")
        ax.set_title(f"Scaled Energy Gap: $N={N}$, $M={M}$, {sign_V_string}, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("Scaled Energy [$E/N|V| = \epsilon/|V|$]")
        fig.tight_layout()
    
    if plot_overlaps:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        for index in range(M**N):
            if index == 0:
                ax1.plot(times, np.real(state_overlaps[:, 0]), '.', color="k")
                ax2.plot(times, np.imag(state_overlaps[:, 0]), '.', color="k")
            else:    
                ax1.plot(times, np.real(state_overlaps[:, index]), '.', color=colors(index))
                ax2.plot(times, np.imag(state_overlaps[:, index]), '.', color=colors(index))
            ax1.set_ylabel("$\Re$ Component")
            ax2.set_ylabel("$\Im$ Component")
        ax2.set_xlabel("Time [$t/|V|$]")
        fig.suptitle(f"State Overlap: $N={N}$, $M={M}$, {sign_V_string}, $(J/|V|)_f = {J_V_ratios[-1]}$")
        fig.tight_layout()
     
    if plot_sigma:
        fig, ax = plt.subplots()
        states, _ = enumerate_states(N, M)
        sigmas = []
        for wavefunction in time_evolved_wavefunctions:
            sigmas += [sigma_ij(0, 1, wavefunction=wavefunction, states=states, N=N, M=M) / M]
        ax.plot(times, sigmas, "-k")
        ax.set_title(f"Time Evolved $\sigma$: $N={N}$, $M={M}$, {sign_V_string}, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_ylabel("$\sigma^{01}/M$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.grid()
        
    if plot_ground_state_manifold_overlaps:
        fig, ax = plt.subplots()
        ax.plot(times, ground_state_manifold_overlaps, '-k')
        ax.set_title(f"Ground State Manifold Overlap: $N={N}$, $M={M}$, {sign_V_string}, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel(r"Ground State Manifold Overlap [$\langle \psi | P_D | \psi \rangle$]")
        ax.set_ylim(-0.1, 1.1)
        ax.grid()
        
# --------------------------------------------------------------------------------------------------------------------------------------------

def plot_data(N, M, sign_V="positive", gap_or_sigma="energy_gap", include_path=False, mu_V_ratios=None, J_V_ratios=None, times=None):
    """
    Plots simulation data loaded from a CSV file for either the energy gap or a synthetic distance (σ).
    
    The function constructs the filename based on the provided parameters, loads the data (assumed to be in a grid),
    reshapes it appropriately, and then generates a pseudocolor plot using `pcolormesh`. Depending on the value of 
    `gap_or_sigma`, the data may be transformed (e.g., logarithmically for the energy gap). Optionally, a control path 
    defined by the arrays `J_V_ratios`, `mu_V_ratios`, and `times` can be overlaid on the plot.
    
    Parameters:
    -----------
    N (int): Number of sites.
    M (int): Number of synthetic levels per site.
    sign_V (str, optional): Sign indicator for V. Use "positive" for V > 0 and "negative" for V < 0. Default is "positive".
    gap_or_sigma (str, optional): Specifies the type of data to plot. Use "energy_gap" to plot the energy gap data or any other value for synthetic distance (σ). Default is "energy_gap".
    include_path (bool, optional): If True, overlays a control path on the plot using `J_V_ratios`, `mu_V_ratios`, and `times`. Default is False.
    mu_V_ratios (array-like, optional): Array of μ/|V| ratios for the control path overlay (required if include_path is True).
    J_V_ratios (array-like, optional): Array of J/|V| ratios for the control path overlay (required if include_path is True).
    times (array-like, optional): Array of time values corresponding to the control path, used for color mapping (required if include_path is True).
    
    Returns:
    --------
    None
        Displays the generated plot.
    """
    
    # Construct the filename based on the provided parameters.
    filename = f"{gap_or_sigma}_V_{sign_V}_N={N}_M={M}.csv"
    
    # Construct the full path to the CSV file.
    file_path = os.path.join(data_folder_path, filename)

    # Check if the file exists.
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load the CSV file.
    try:
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    # The CSV file columns are assumed to be: mu_V_ratio, J_V_ratio, value.
    # Determine the unique coordinate values.
    unique_mu = np.unique(data[:, 0])
    unique_J = np.unique(data[:, 1])
    
    # Reshape the values into a grid of shape (len(unique_J), len(unique_mu)).
    Z = data[:, 2].reshape(len(unique_J), len(unique_mu))
    
    # Transform data if plotting the energy gap.
    if gap_or_sigma == "energy_gap":
        # Avoid division by zero in case any gap value is zero.
        with np.errstate(divide='ignore'):
            Z = np.log(1 / Z)
        plot_title = "Energy Gap"
        color_label = r"$\log(1/\Delta E)$"
    else:
        plot_title = "Synthetic Distance"
        color_label = r"$\sigma/M$"
    
    # Create a meshgrid for the pcolormesh plot (x-axis: J/|V|, y-axis: μ/|V|).
    J_grid, mu_grid = np.meshgrid(unique_J, unique_mu, indexing='ij')
    
    # Generate the plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = plt.pcolormesh(J_grid, mu_grid, Z, shading='auto', cmap='plasma')
    
    # Optionally overlay the control path.
    if include_path:
        from matplotlib.collections import LineCollection

        points = np.array([J_V_ratios, mu_V_ratios]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='gist_rainbow', norm=plt.Normalize(times.min(), times.max()))
        lc.set_array(times)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, label='t/|V|')
        ax.set_xlabel("J/|V|")
        ax.set_ylabel("μ/|V|")
    
    ax.set_ylim(0, 5)
    ax.set_xlim(-5, 5)
    
    sign_str = r"$V > 0$" if sign_V == "positive" else r"$V < 0$"
    plt.title(f"{plot_title}: $N={N}$, $M={M}$, {sign_str}")
    plt.xlabel(r"$J/|V|$")
    plt.ylabel(r"$\mu/|V|$")
    plt.colorbar(pcm, label=color_label)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------