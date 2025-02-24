# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.optimize import minimize
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data_folder"))
sys.path.append(parent_dir)
sys.path.append(data_folder_path) 

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def enumerate_states(N, M):
    """
    Enumerates all possible states of a system with N lattice sites and M synthetic levels.
    
    Parameters:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    
    Returns:
    list: A list of all possible states represented as lists of integers.
    list: A list of formatted state strings for easier visualization.
    """
    
    if N <= 0 or M <= 0:
        return [], []

    states = []
    current_state = [0] * N

    while True:
        states.append(current_state.copy())

        for i in range(N-1, -1, -1):
            if current_state[i] < M-1:
                current_state[i] += 1
                break
            else:
                current_state[i] = 0
        else:
            break

    formatted_states = ["|" + ",".join(map(str, state)) + ">" for state in states]
    return states, formatted_states

# --------------------------------------------------------------------------------------------------------------------------------------------

def create_H_key(formatted_states):
    """
    Create a matrix H_key where each element is a formatted string combination of state indices, representing the 
    action of the Hamiltonian operator between states.
    
    Parameters:
    formatted_states (list): List of formatted state strings, where each state is represented in the form "|x_1, ..., x_N>".
    
    Returns:
    np.ndarray: H_key matrix of size (M^N x M^N), where each element is a formatted string "<state_x|H|state_y>".
    """
    
    M_pow_N = len(formatted_states)
    H_key = np.empty((M_pow_N, M_pow_N), dtype=object)

    for x in range(M_pow_N):
        for y in range(M_pow_N):
            # construct formatted string "<state_x|H|state_y>"
            H_key[x, y] = "<" + formatted_states[x][::-1][1:] + "H" + formatted_states[y]

    return H_key

# --------------------------------------------------------------------------------------------------------------------------------------------

def exact_diagonalize(H, use_sparse = False, k = 1):
    """
    Diagonalizes a Hermitian matrix using numpy's `eigh()` method.

    Parameters:
    H (np.ndarray): Hermitian matrix to diagonalize.
    verbose (bool): If True, prints the eigenvalues and eigenvectors.
    check_reconstruction (bool): If True, checks whether the matrix is faithfully reconstructed.

    Returns:
    np.ndarray: Eigenvalues of the matrix.
    list: List of eigenvectors of the matrix.
    """
    
    if not np.allclose(np.conjugate(H.T), H):
            print("The matrix is not Hermitian.")
            return None, None
    
    if use_sparse:
        eigenvalues, eigenvectors = eigsh(H, k=k)
        return eigenvalues, [eigenvectors[:, i] for i in range(k)]
        
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        return eigenvalues, [eigenvectors[:, i] for i in range(H.shape[0])]
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_hamiltonian(N, M, J, V):
    """
    Constructs the J-V Hamiltonian matrix.

    Parameters:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    J (float): Tunneling coefficient.
    V (float): Interaction strength.

    Returns:
    np.ndarray: Hamiltonian matrix of size (M^N x M^N).
    """
    
    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    M_powers = np.array([M**i for i in range(N)])

    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])

    def state_to_index(state):
        return np.dot(state, M_powers[::-1])

    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            for n in range(1, M):
                if state[j] == n:
                    new_state = state.copy()
                    new_state[j] = n - 1
                    beta = state_to_index(new_state)
                    H[alpha, beta] -= J
                    H[beta, alpha] -= J  

    for alpha in range(dim):
        state = index_to_state(alpha)
        for i in range(N - 1):
            j = i + 1
            for n in range(1, M):
                if state[i] == n and state[j] == n - 1:
                    new_state = state.copy()
                    new_state[i], new_state[j] = n - 1, n
                    beta = state_to_index(new_state)
                    H[alpha, beta] += V
                    H[beta, alpha] += V  

    return H

# --------------------------------------------------------------------------------------------------------------------------------------------

def sigma_ij(i, j, wavefunction, states, N, M):
    """
    Computes the sigma value, representing the difference in synthetic dimension space between sites i and j, 
    weighted by the wavefunction. Serves as an order parameter for the quantum string-gas transition.

    Parameters:
    i (int): Index of the first site.
    j (int): Index of the second site.
    ground_state_wavefunction (np.ndarray): Ground state wavefunction coefficients for each state.
    states (list of lists): List of all states, where each state is represented as a list of occupation numbers.
    N (int): Number of sites.
    M (int): Number of synthetic levels (states per site).

    Returns:
    float: The sigma value, summing the occupation differences weighted by the ground state wavefunction.
    """
    
    sigma = 0
    dim = M**N
    
    for m in range(M):
        for n in range(M):
            for k in range(dim):
                if states[k][i] == m:  
                    if states[k][j] == n: 
                        sigma += abs(m - n) * np.linalg.norm(wavefunction[k])**2
                    else:
                        sigma += 0
                else:
                    sigma += 0
    
    return sigma

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_rescaled_hamiltonian(N, M, V, mu_V_ratio, J_V_ratio):
    """
    Constructs a rescaled J-V-mu Hamiltonian matrix with N sites and M states per site, incorporating 
    chemical potential, tunneling, and interaction terms. The Hamiltonian is normalized 
    by the absolute value of V to produce H_tilde. Uses open boundary conditions.

    Parameters:
    N (int): Number of sites in the system.
    M (int): Number of states per site.
    V (float): Interaction strength.
    mu_V_ratio (float): Ratio of the chemical potential (mu) to the interaction strength (V).
    J_V_ratio (float): Ratio of the tunneling parameter (J) to the interaction strength (V).

    Returns:
    np.ndarray: The rescaled Hamiltonian matrix H_tilde (normalized by |V|).
    """
    mu = mu_V_ratio * abs(V)
    J = J_V_ratio * abs(V)
    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Precompute powers of M for faster state-to-index conversion
    M_powers = np.array([M**i for i in range(N)])

    # Helper function to convert a state index to a state representation (array of states)
    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])
    
    # Helper function to convert a state representation (array of states) back to an index
    def state_to_index(state):
        return np.dot(state, M_powers[::-1])

    # Apply the chemical potential term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            if state[j] == 0:
                H[alpha, alpha] -= mu

    # Apply the tunneling term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            for n in range(1, M):
                if state[j] == n:
                    new_state = state.copy()
                    new_state[j] = n - 1
                    beta = state_to_index(new_state)
                    H[alpha, beta] -= J
                    H[beta, alpha] -= J  # Ensure Hermitian symmetry

    # Apply the interaction term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for i in range(N - 1):
            j = i + 1
            for n in range(1, M):
                if state[i] == n and state[j] == n - 1:
                    new_state = state.copy()
                    new_state[i], new_state[j] = n - 1, n
                    beta = state_to_index(new_state)
                    H[alpha, beta] += V
                    H[beta, alpha] += V  # Ensure Hermitian symmetry
                    
    # Rescale H to H_tilde by dividing by |V|
    H_tilde = H / abs(V)
        
    return H_tilde

# ------------------------------------------------------------------------------------------------------------------------------------------

def evolve_wavefunction(psi, H, dt, hbar=1.0):
    """
    Evolves the wavefunction psi under the Hamiltonian H using a unitary time evolution operator U = exp(-iH * dt / hbar).
    This function applies the time evolution over a time step dt.
    
    Parameters:
    psi (np.ndarray): The current wavefunction as a column vector.
    H (np.ndarray): The Hamiltonian matrix representing the system at the current time step.
    dt (float): Time step for the evolution.
    hbar (float, optional): Reduced Planck's constant (default is 1.0).

    Returns:
    np.ndarray: The evolved wavefunction after the time step dt.
    """
    U = expm(-1j * H * dt)
    psi = np.dot(U, psi)
    return psi

# --------------------------------------------------------------------------------------------------------------------------------------------

def simulate_hamiltonian_time_evolution(hamiltonians, times, initial_state=None):
    """To do: documentation."""
    
    n_excited_states = len(hamiltonians[0])
    initial_hamiltonian = hamiltonians[0]
    
    if initial_state is None:
        _, eigenvectors_0 = exact_diagonalize(initial_hamiltonian)
        psi_0 = eigenvectors_0[0]
    else:
        psi_0 = initial_state

    energies = []
    time_evolved_wavefunctions = []
    state_probabilities = []
    state_overlaps = []
    true_energies = []

    psi = psi_0.copy()
    for idx, instantaneous_hamiltonian in enumerate(hamiltonians):
        
        if idx > 1:
            dt = times[idx] - times[idx - 1]
        else:
            dt = times[idx]
            
        eigenvalues, eigenvectors = exact_diagonalize(instantaneous_hamiltonian)
        true_energies.append(eigenvalues)    
        
        psi = evolve_wavefunction(psi, instantaneous_hamiltonian, dt)
        psi = psi / np.linalg.norm(psi)  
        
        time_evolved_wavefunctions.append(psi)
        
        energy = np.real(np.conj(psi).T @ instantaneous_hamiltonian @ psi)
        energies.append(energy)
        
        overlap = [np.dot(np.conj(eigenvectors[i]).T, psi) for i in range(n_excited_states)] 
        probability = [np.abs(np.conj(eigenvectors[i]).T @ psi)**2 for i in range(n_excited_states)]
               
        state_probabilities.append(probability)
        state_overlaps.append(overlap)

    energies = np.array(energies)
    time_evolved_wavefunctions = np.array(time_evolved_wavefunctions)
    state_probabilities = np.array(state_probabilities)
    state_overlaps = np.array(state_overlaps)
    true_energies = np.array(true_energies)
        
    return energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies

# --------------------------------------------------------------------------------------------------------------------------------------------
def plot_time_evolution(N, M, results, times, J_V_ratios, mu_V_ratios, plot_probability = True, plot_gap = True, plot_overlaps = True, plot_sigma = True):
    """To do: documentation."""
    
    energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies = results
    energies = energies * 1/N
    true_energies = true_energies * 1/N
    colors = get_cmap("gist_rainbow", M**N)
    
    if plot_probability == True:
        fig, ax = plt.subplots()
        for index in range(M**N):
            if index == 0:
                ax.plot(times, state_probabilities[:,index], color = "k", label = "Ground State")
            elif index == 1:
                ax.plot(times, state_probabilities[:,index], color = colors(index), label = "1st Excited State")
            elif index == 2:
                ax.plot(times, state_probabilities[:,index], color = colors(index), label = "2nd Excited State") 
            else: 
                ax.plot(times, state_probabilities[:,index], color = colors(index))
        ax.set_ylim(-0.1,1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        fig.subplots_adjust(bottom=0.4)
        ax.set_title(f"State Probabilities: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("State Probability")
        ax.grid()
        fig.tight_layout()
        
    
    if plot_gap == True:
        fig, ax = plt.subplots()
        for index in range(M**N):
            ax.plot(times, true_energies[:,index]-true_energies[:,0], color = colors(index))   
        ax.plot(times, energies-true_energies[:,0], color = "k", label = "Time Evolved State")
        ax.legend(loc = "upper center")
        ax.set_title(f"Scaled Energy Gap: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("Scaled Energy [$E/N|V| = \epsilon/|V|$]")
        fig.tight_layout()
    

    if plot_overlaps == True:
        fig, (ax1,ax2) = plt.subplots(nrows = 2, sharex=True)
        for index in range(M**N):
            if index == 0:
                ax1.plot(times, np.real(state_overlaps[:,0]), '.', color = "k")
                ax2.plot(times, np.imag(state_overlaps[:,0]), '.', color = "k")
            else:    
                ax1.plot(times, np.real(state_overlaps[:,index]), '.', color = colors(index))
                ax2.plot(times, np.imag(state_overlaps[:,index]), '.', color = colors(index))
            ax1.set_ylabel("$\Re$ Component")
            ax2.set_ylabel("$\Im$ Component")
        ax2.set_xlabel("Time [$t/|V|$]")
        fig.suptitle(f"State Overlap: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratios[-1]}$")
        fig.tight_layout()
     

    if plot_sigma == True:
        fig, ax = plt.subplots()
        states, _ = enumerate_states(N, M)
        sigmas = []
        for wavefunction in time_evolved_wavefunctions:
            sigmas += [sigma_ij(0, 1, wavefunction = wavefunction, states = states, N=N, M=M)/M]
        ax.plot(times, sigmas, "-k")
        ax.set_title(f"Time Evolved $\sigma$: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratios[-1]}$")
        ax.set_ylabel("$\sigma^{01}/M$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.grid()

# --------------------------------------------------------------------------------------------------------------------------------------------

def create_piecewise_linear_paths_opt_times(N, M, T, dt, V, J_V_init, J_V_final,mu_V_init, mu_V_final, num_control_points):
    # To do: documentation.
    """
    Constructs J/V and μ/V paths using piecewise linear interpolation with optimized control points.
    In addition to optimizing the intermediate control values, this function optimizes the intermediate times.
    
    The control points (for J/V, μ/V, and times) have fixed endpoints:
      t[0] = 0, t[-1] = T,
      J[0] = J_V_init, J[-1] = J_V_final,
      μ[0] = mu_V_init, μ[-1] = mu_V_final.
    
    The optimization variables are:
      - J_intermediate: control values for J/V (length = num_control_points - 2)
      - mu_intermediate: control values for μ/V (length = num_control_points - 2)
      - t_intermediate: control times (length = num_control_points - 2)
         (with constraints: 0 < t_1 < t_2 < ... < t_{N-2} < T)
    
    Returns:
      times_dense   : Dense time grid (from 0 to T with step dt).
      J_V_path      : Optimized J/V values evaluated on times_dense.
      mu_V_path     : Optimized μ/V values evaluated on times_dense.
      obj_value     : Final objective value (ground-state infidelity + penalty).
      opt_params    : Optimized parameter vector.
      t_control_opt : Optimized full control times (with endpoints).
      J_control_opt : Optimized full control J/V values (with endpoints).
      mu_control_opt: Optimized full control μ/V values (with endpoints).
    """
    # Create a dense time grid for evaluation.
    times_dense = np.arange(0, T + dt, dt)
    
    # Total number of control points.
    n_points = num_control_points
    n_int = n_points - 2  # number of intermediate (free) control points

    # Initial guesses for the intermediate control values (linearly spaced between endpoints).
    J_initial_guess = np.linspace(J_V_init, J_V_final, n_points)[1:-1]
    mu_initial_guess = np.linspace(mu_V_init, mu_V_final, n_points)[1:-1]
    # Initial guess for the intermediate times: linearly spaced between 0 and T.
    t_initial_guess = np.linspace(0, T, n_points)[1:-1]

    # Combine into a single vector: first J, then mu, then times.
    x0 = np.concatenate((J_initial_guess, mu_initial_guess, t_initial_guess))
    
    # A small buffer to ensure strict ordering of times.
    eps = 1e-3

    # Define constraints for the times portion.
    # We require:
    #   t_intermediate[0] >= eps,
    #   t_intermediate[i] - t_intermediate[i-1] >= eps for i=1,...,n_int-1,
    #   T - t_intermediate[-1] >= eps.
    cons = []
    # Constraint for first intermediate time:
    cons.append({'type': 'ineq', 'fun': lambda x: x[2*n_int] - eps})
    # Constraints for ordering among intermediate times:
    for i in range(1, n_int):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*n_int + i] - x[2*n_int + i - 1] - eps
        })
    # Constraint for last intermediate time:
    cons.append({'type': 'ineq', 'fun': lambda x: T - x[2*n_int + n_int - 1] - eps})

    # Objective function: compute the piecewise linear paths,
    # build Hamiltonians at each dense time, simulate time evolution,
    # and return the ground-state infidelity plus a penalty for negative μ/V.
    def objective(x):
        # Unpack the optimization vector.
        J_int = x[:n_int]
        mu_int = x[n_int:2*n_int]
        t_int = x[2*n_int:3*n_int]
        
        # Construct full control arrays (with endpoints fixed).
        J_control = np.concatenate(([J_V_init], J_int, [J_V_final]))
        mu_control = np.concatenate(([mu_V_init], mu_int, [mu_V_final]))
        t_control = np.concatenate(([0.0], t_int, [T]))
        
        # Build the dense paths via linear interpolation.
        J_path_dense = np.interp(times_dense, t_control, J_control)
        mu_path_dense = np.interp(times_dense, t_control, mu_control)
        
        # Add a penalty if μ/V becomes negative anywhere.
        penalty = np.sum(np.abs(np.minimum(0, mu_path_dense)))
        
        # Build Hamiltonians at each dense time.
        hamiltonians = []
        for i, t in enumerate(times_dense):
            ham = construct_rescaled_hamiltonian(N, M, V,
                                                 mu_V_ratio=mu_path_dense[i],
                                                 J_V_ratio=J_path_dense[i])
            hamiltonians.append(ham)
        
        # Simulate time evolution.
        energies, _, state_probabilities, _, _ = simulate_hamiltonian_time_evolution(hamiltonians, times_dense)
        ground_state_fidelity = state_probabilities[-1, 0]
        ground_state_infidelity = 1 - ground_state_fidelity
        
        return ground_state_infidelity + penalty

    # Run the optimization with SLSQP so that constraints can be enforced.
    result = minimize(objective, x0, method='SLSQP', constraints=cons)
    opt_params = result.x

    # Extract optimized intermediate values.
    J_int_opt = opt_params[:n_int]
    mu_int_opt = opt_params[n_int:2*n_int]
    t_int_opt = opt_params[2*n_int:3*n_int]

    # Construct full control arrays.
    J_control_opt = np.concatenate(([J_V_init], J_int_opt, [J_V_final]))
    mu_control_opt = np.concatenate(([mu_V_init], mu_int_opt, [mu_V_final]))
    t_control_opt = np.concatenate(([0.0], t_int_opt, [T]))

    # Compute the dense paths with the optimized control points.
    J_V_path = np.interp(times_dense, t_control_opt, J_control_opt)
    mu_V_path = np.interp(times_dense, t_control_opt, mu_control_opt)
    
    obj_value = result.fun
    return (times_dense, J_V_path, mu_V_path, obj_value, opt_params,
            t_control_opt, J_control_opt, mu_control_opt)
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def plot_data(N, M, sign_V="positive", gap_or_sigma="energy_gap", include_path = False, mu_V_ratios = None, J_V_ratios = None, times = None):
    """
    Reads the CSV file corresponding to the chosen V sign and quantity, and plots the data using pcolormesh.
    
    Parameters:
      sign_V (str): Either "positive" for V>0 or "negative" for V<0.
      gap_or_sigma (str): Either "energy_gap" or "sigma".
    """
     # Construct the filename
    filename = f"{gap_or_sigma}_V_{sign_V}_N={N}_M={M}.csv"
    
    # Construct the full path to the CSV file
    file_path = os.path.join(data_folder_path, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load the CSV file
    try:
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    # The CSV file columns are: mu_V_ratio, J_V_ratio, value.
    # Determine the unique coordinate values.
    unique_mu = np.unique(data[:, 0])
    unique_J = np.unique(data[:, 1])
    
    # The data were saved in a grid of shape (len(J), len(mu)).
    # Reshape the values accordingly.
    Z = data[:, 2].reshape(len(unique_J), len(unique_mu))
    
    # If plotting the energy gap, apply the same transformation as in your original code.
    if gap_or_sigma == "energy_gap":
        # Avoid division by zero if any gap is zero.
        with np.errstate(divide='ignore'):
            Z = np.log(1 / Z)
        plot_title = "Energy Gap"
        color_label = r"$\log(1/\Delta E)$"
    else:
        plot_title = "Synthetic Distance"
        color_label = r"$\sigma/M$"
    
    # Create a meshgrid for pcolormesh.
    # Note: In the original simulation, the x-axis corresponds to J/|V| and the y-axis to mu/|V|.
    # Here, we set up the grid accordingly.
    # Since the CSV file was built from meshgrid(mu_V, J_V) with shape (len(J_V), len(mu_V)),
    # we create the grid with indexing 'ij'.
    J_grid, mu_grid = np.meshgrid(unique_J, unique_mu, indexing='ij')
    
    # Generate the plot.
    fig, ax = plt.subplots(figsize = (8,6))
    pcm = plt.pcolormesh(J_grid, mu_grid, Z, shading='auto', cmap='plasma')
    
    if include_path is not None:
        from matplotlib.collections import LineCollection

        points = np.array([J_V_ratios, mu_V_ratios]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='gist_rainbow', norm=plt.Normalize(times.min(), times.max()))

        lc.set_array(times)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, label='Time [t/|V|]')

        ax.set_xlabel("J/|V|")
        ax.set_ylabel("μ/|V|")
    
    ax.set_ylim(0,10)
    ax.set_xlim(-10,10)
    
    sign_str = r"$V > 0$" if sign_V == "positive" else r"$V < 0$"
    plt.title(f"{plot_title}: {sign_str}", fontsize=14)
    plt.xlabel(r"$J/|V|$", fontsize=12)
    plt.ylabel(r"$\mu/|V|$", fontsize=12)
    plt.colorbar(pcm, label=color_label)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------
